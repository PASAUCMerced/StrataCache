import stratacache.config as config
import json

from typing import Tuple

from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import queue

import copy
import threading
import time

import logging

logger = logging.getLogger(__name__)


class StrataTierType(Enum):
    GPU = -1
    CPU = 0
    CXL = 1
    NIXL = 2
    DISK = 3

    UNKNOWN = 99


@dataclass
class StrataTierStats:
    # Operation counts
    total_ops: int = 0
    total_read_ops: int = 0
    total_write_ops: int = 0
    total_delete_ops: int = 0

    # Cache payload stats
    byte_read: int = 0
    byte_written: int = 0
    byte_delete: int = 0

    read_latency: int = 0
    max_read_latency: int = 0
    write_latency: int = 0
    max_write_latency: int = 0
    delete_latency: int = 0
    max_delete_latency: int = 0

    read_throughput: float = 0.0
    write_throughput: float = 0.0
    delete_throughput: float = 0.0


class StrataTierTelemetry(ABC):
    def __init__(self, tier: StrataTierType, telemetry: "StrataTelemetry"):
        self._lock = threading.RLock()
        self._tier = tier
        telemetry.register_tier(tier, self)

    def get_stats(self) -> StrataTierStats:
        with self._lock:
            return copy.deepcopy(self._stats)


@dataclass
class StrataSystemStats:
    # Operation counts
    total_ops: int = 0
    total_read_ops: int = 0
    total_write_ops: int = 0
    total_delete_ops: int = 0

    # Cache payload stats
    byte_read: int = 0
    byte_written: int = 0
    byte_delete: int = 0

    read_latency: int = 0
    max_read_latency: int = 0
    write_latency: int = 0
    max_write_latency: int = 0
    delete_latency: int = 0
    max_delete_latency: int = 0

    read_throughput: float = 0.0
    write_throughput: float = 0.0
    delete_throughput: float = 0.0


class StrataTelemetry:
    """StrataTelemetry collects telemetry data for StrataCache.
    It provides telemetry from two perspectives: system-overall and by-tier.
    """

    def __init__(self):
        self._system_stats_lock = threading.RLock()
        self._system_stats = StrataSystemStats()
        self._per_tier_telemetry: dict[StrataTierType, StrataTierTelemetry] = {}
        self._message_queue = queue.Queue()
        self._last_tick = 0
        self._exporter_server = None  # set when uvicorn is started

        self._initialize_backends()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        if config.get_config().telemetry_export:
            self._exporter = threading.Thread(target=self._export_server, daemon=True)
            self._exporter.start()
            import atexit

            atexit.register(self._shutdown_export_server)

    _instance = None
    _instance_lock = threading.Lock()

    @staticmethod
    def get_or_create() -> "StrataTelemetry":
        if StrataTelemetry._instance is None:
            with StrataTelemetry._instance_lock:
                if StrataTelemetry._instance is None:
                    StrataTelemetry._instance = StrataTelemetry()
        return StrataTelemetry._instance

    def on_tier_op_async(self, tier: StrataTierType, op_type: str, **kwargs):
        self._message_queue.put_nowait((tier, op_type, kwargs))

    def get_system_stats(self) -> StrataSystemStats:
        with self._system_stats_lock:
            return copy.deepcopy(self._system_stats)

    def get_tier_stats(self, tier: StrataTierType) -> StrataTierStats:
        telemetry = self._per_tier_telemetry.get(tier, None)
        if telemetry is None:
            raise ValueError(f"Invalid tier type: {tier}")
        return telemetry.get_stats()

    def get_stats(
        self,
    ) -> Tuple[StrataSystemStats, dict[StrataTierType, StrataTierStats]]:
        """Return a tuple of system stats and per-tier stats dictionary."""
        system_stats = self.get_system_stats()
        per_tier_stats = {
            tier: telemetry.get_stats()
            for tier, telemetry in self._per_tier_telemetry.items()
        }
        return system_stats, per_tier_stats

    def register_tier(self, tier: StrataTierType, telemetry: StrataTierTelemetry):
        logging.info(f"Registering telemetry for tier: {tier}")
        self._per_tier_telemetry[tier] = telemetry

    def _initialize_backends(self):
        """Initialize all telemetry backends."""
        try:
            from stratacache.telemetry.backend import create_telemetry_backends

            create_telemetry_backends(self)
        except Exception:
            logger.exception("Failed to initialize telemetry backends")

    def _process_ops(
        self, ops: dict[StrataTierType, list[Tuple[str, dict]]], elapse: float
    ):
        for tier, tier_ops in ops.items():
            telemetry = self._per_tier_telemetry.get(tier, None)
            if telemetry is None:
                logger.warning(f"No telemetry registered for tier: {tier}")
                continue

            all_read_ops = [op for op in tier_ops if op[0] == "load"]
            all_write_ops = [op for op in tier_ops if op[0] == "store"]
            all_delete_ops = [op for op in tier_ops if op[0] == "delete"]

            byte_read = sum(op[1].get("size", 0) for op in all_read_ops)
            byte_written = sum(op[1].get("size", 0) for op in all_write_ops)
            byte_deleted = sum(op[1].get("size", 0) for op in all_delete_ops)

            read_latency = (
                sum(op[1].get("latency_us", 0) for op in all_read_ops)
                / len(all_read_ops)
                if all_read_ops
                else 0
            )
            max_read_latency = max(
                (op[1].get("latency_us", 0) for op in all_read_ops), default=0
            )
            write_latency = (
                sum(op[1].get("latency_us", 0) for op in all_write_ops)
                / len(all_write_ops)
                if all_write_ops
                else 0
            )
            max_write_latency = max(
                (op[1].get("latency_us", 0) for op in all_write_ops), default=0
            )
            delete_latency = (
                sum(op[1].get("latency_us", 0) for op in all_delete_ops)
                / len(all_delete_ops)
                if all_delete_ops
                else 0
            )
            max_delete_latency = max(
                (op[1].get("latency_us", 0) for op in all_delete_ops), default=0
            )

            read_throughput = (
                sum(op[1].get("size", 0) for op in all_read_ops) / elapse
                if elapse > 0
                else 0
            )
            write_throughput = (
                sum(op[1].get("size", 0) for op in all_write_ops) / elapse
                if elapse > 0
                else 0
            )
            delete_throughput = (
                sum(op[1].get("size", 0) for op in all_delete_ops) / elapse
                if elapse > 0
                else 0
            )

            with telemetry._lock:
                telemetry._stats.total_read_ops += len(all_read_ops)
                telemetry._stats.total_write_ops += len(all_write_ops)
                telemetry._stats.total_delete_ops += len(all_delete_ops)
                telemetry._stats.byte_read += byte_read
                telemetry._stats.byte_written += byte_written
                telemetry._stats.byte_delete += byte_deleted

                telemetry._stats.read_latency = read_latency
                telemetry._stats.max_read_latency = max_read_latency
                telemetry._stats.write_latency = write_latency
                telemetry._stats.max_write_latency = max_write_latency
                telemetry._stats.delete_latency = delete_latency
                telemetry._stats.max_delete_latency = max_delete_latency
                telemetry._stats.read_throughput = read_throughput
                telemetry._stats.write_throughput = write_throughput
                telemetry._stats.delete_throughput = delete_throughput

        # Calculate system-level stats
        with self._system_stats_lock:
            all_read_ops = [
                op for tier_ops in ops.values() for op in tier_ops if op[0] == "load"
            ]
            all_write_ops = [
                op for tier_ops in ops.values() for op in tier_ops if op[0] == "store"
            ]
            all_delete_ops = [
                op for tier_ops in ops.values() for op in tier_ops if op[0] == "delete"
            ]

            self._system_stats.total_read_ops += len(all_read_ops)
            self._system_stats.total_write_ops += len(all_write_ops)
            self._system_stats.total_delete_ops += len(all_delete_ops)
            self._system_stats.byte_read += sum(
                op[1].get("size", 0) for op in all_read_ops
            )
            self._system_stats.byte_written += sum(
                op[1].get("size", 0) for op in all_write_ops
            )
            self._system_stats.byte_delete += sum(
                op[1].get("size", 0) for op in all_delete_ops
            )

            self._system_stats.read_latency = (
                sum(op[1].get("latency_us", 0) for op in all_read_ops)
                / len(all_read_ops)
                if all_read_ops
                else 0
            )
            self._system_stats.max_read_latency = max(
                (op[1].get("latency_us", 0) for op in all_read_ops), default=0
            )
            self._system_stats.write_latency = (
                sum(op[1].get("latency_us", 0) for op in all_write_ops)
                / len(all_write_ops)
                if all_write_ops
                else 0
            )
            self._system_stats.max_write_latency = max(
                (op[1].get("latency_us", 0) for op in all_write_ops), default=0
            )
            self._system_stats.delete_latency = (
                sum(op[1].get("latency_us", 0) for op in all_delete_ops)
                / len(all_delete_ops)
                if all_delete_ops
                else 0
            )
            self._system_stats.max_delete_latency = max(
                (op[1].get("latency_us", 0) for op in all_delete_ops), default=0
            )
            self._system_stats.read_throughput = (
                sum(op[1].get("size", 0) for op in all_read_ops) / elapse
                if elapse > 0
                else 0
            )
            self._system_stats.write_throughput = (
                sum(op[1].get("size", 0) for op in all_write_ops) / elapse
                if elapse > 0
                else 0
            )
            self._system_stats.delete_throughput = (
                sum(op[1].get("size", 0) for op in all_delete_ops) / elapse
                if elapse > 0
                else 0
            )

    def _drain_all(self) -> dict[StrataTierType, list[Tuple[str, dict]]]:
        """Drain all pending telemetry messages from the queue."""
        tiers: dict[StrataTierType, list[Tuple[str, dict]]] = {}
        while True:
            try:
                tier, op_type, kwargs = self._message_queue.get_nowait()
                if tier not in tiers:
                    tiers[tier] = []
                tiers[tier].append((op_type, kwargs))
            except queue.Empty:
                break
        return tiers

    def _loop(self):
        CALC_INTERVAL = 1.0  # seconds
        next_tick_time = time.time() + CALC_INTERVAL
        self._last_tick = time.time()
        while True:
            elapse = time.time() - self._last_tick
            self._process_ops(self._drain_all(), elapse)

            self._last_tick = time.time()

            # self._print_stats()

            sleep_time = next_tick_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            next_tick_time += CALC_INTERVAL

    def _print_stats(self):
        stats = {
            "system": self.get_system_stats().__dict__,
            "tiers": {
                tier.name: tier_telemetry.get_stats().__dict__
                for tier, tier_telemetry in self._per_tier_telemetry.items()
            },
        }
        logger.info(json.dumps(stats, indent=2))

    def _export_server(self):
        from stratacache.telemetry.metrics import app
        import uvicorn

        # IMPORTANT: install_signal_handlers=False so uvicorn does NOT
        # intercept SIGINT/SIGTERM. With its default config it grabs both
        # signals on whichever thread it lives in and starts a graceful
        # shutdown that waits for active connections to drain - if a
        # Prometheus scraper keeps polling /metrics, the process never
        # exits. Letting signals reach the host process means Ctrl+C on
        # vLLM kills the engine-core children cleanly.
        config_obj = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=6954,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config_obj)
        server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self._exporter_server = server
        try:
            server.run()
        except Exception:  # noqa: BLE001
            logger.exception("StrataCache telemetry exporter crashed")

    def _shutdown_export_server(self) -> None:
        srv = self._exporter_server
        if srv is None:
            return
        try:
            srv.should_exit = True
        except Exception:  # noqa: BLE001
            pass
