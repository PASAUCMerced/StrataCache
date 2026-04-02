from stratacache.telemetry.telemetry import (
    StrataTierStats,
    StrataTierTelemetry,
    StrataTierType,
)
from stratacache.telemetry.time_wheel import TimeWheel, TimeWheelEvent
from stratacache.telemetry.utils import human_readable_size

from dataclasses import dataclass
import queue
import logging

import threading
import time
from cupti import cupti

CALC_INTERVAL = 1

logger = logging.getLogger(__name__)

_gpu_telemetry_instance: "StrataGPUTelemetry | None" = None

# Pre-resolve enum int values once to avoid per-activity object creation.
_ALLOC_OP: int = cupti.ActivityMemoryOperationType.ALLOCATION.value
_RELEASE_OP: int = cupti.ActivityMemoryOperationType.RELEASE.value
_DEVICE_KIND: int = cupti.ActivityMemoryKind.DEVICE.value
_DEVICE_STATIC_KIND: int = cupti.ActivityMemoryKind.DEVICE_STATIC.value
_DEVICE_KINDS = (_DEVICE_KIND, _DEVICE_STATIC_KIND)
_PINNED_KIND: int = cupti.ActivityMemoryKind.PINNED.value
_MEMORY2_KIND = cupti.ActivityKind.MEMORY2.value
_SYNCHRONIZATION_KIND = cupti.ActivityKind.SYNCHRONIZATION.value
_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT = (
    cupti.ActivitySynchronizationType.STREAM_WAIT_EVENT.value
)


def get_activity_kind_name(activity_kind: int):
    kind = cupti.ActivityKind(activity_kind)
    return kind.name


def get_synchronization_type_name(sync_type: int):
    sync_type_enum = cupti.ActivitySynchronizationType(sync_type)
    return sync_type_enum.name


@dataclass
class StrataGPUStats(StrataTierStats):
    alloc_bytes: int = 0
    free_bytes: int = 0
    pin_bytes: int = 0

    alloc_speed: float = 0.0
    free_speed: float = 0.0
    pin_speed: float = 0.0

    dtd_throughput: float = 0.0
    max_dtd_latency: int = 0
    dtd_latency: int = 0
    htd_throughput: float = 0.0
    max_htd_latency: int = 0
    htd_latency: int = 0
    dth_throughput: float = 0.0
    max_dth_latency: int = 0
    dth_latency: int = 0
    hth_throughput: float = 0.0
    hth_latency: int = 0
    max_hth_latency: int = 0

    sync_duration: int = 0
    max_sync_duration: int = 0


def _func_buffer_requested() -> None:
    buffer_size = 32 * 1024 * 1024  # 32MB buffer
    max_num_records = 0
    return buffer_size, max_num_records


def _func_buffer_completed(activities: list) -> None:
    """CUPTI callback — non-blocking batch enqueue into the telemetry instance."""
    telemetry = _gpu_telemetry_instance
    if telemetry is None:
        return
    telemetry._activity_queue.put_nowait(activities)


class StrataGPUTelemetry(StrataTierTelemetry):
    def __init__(self, telemetry):
        super().__init__(StrataTierType.GPU, telemetry)
        self._stats = StrataGPUStats()
        self._lock = threading.Lock()
        self._activity_queue: queue.SimpleQueue = queue.SimpleQueue()

        global _gpu_telemetry_instance
        _gpu_telemetry_instance = self

        self._enable_cupti()

        logger.info("Starting GPU telemetry thread")
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _enable_cupti(self):
        logger.info("Registering CUPTI activities")
        try:
            cupti.activity_register_callbacks(
                _func_buffer_requested, _func_buffer_completed
            )
            # cupti.activity_enable(
            #     cupti.ActivityKind.MEMCPY
            # )  # MEMCPY hurt performance significantly, so we enable specific copy kinds instead
            cupti.activity_enable(cupti.ActivityKind.MEMORY2)
            cupti.activity_enable(cupti.ActivityKind.SYNCHRONIZATION)
        except cupti.cuptiError as e:
            raise RuntimeError(f"Failed to register or enable CUPTI activities: {e}")

    def _disable_cupti(self):
        logger.info("Disabling CUPTI activities")
        try:
            cupti.activity_disable(cupti.ActivityKind.MEMCPY)
            cupti.activity_disable(cupti.ActivityKind.MEMORY2)
            cupti.activity_disable(cupti.ActivityKind.SYNCHRONIZATION)
        except cupti.cuptiError as e:
            raise RuntimeError(f"Failed to disable CUPTI activities: {e}")

    def _flush(self):
        cupti.activity_flush_all(1)

    def _process_activity(self, activities: list, elapse: float) -> None:
        # Memcpy
        memcpy_activities = [
            a for a in activities if a.kind == cupti.ActivityKind.MEMCPY.value
        ]
        memcpy_dtd_activities = [
            a
            for a in memcpy_activities
            if getattr(a, "copy_kind", None) == cupti.ActivityMemcpyKind.DTOD.value
        ]
        memcpy_htd_activities = [
            a
            for a in memcpy_activities
            if getattr(a, "copy_kind", None) == cupti.ActivityMemcpyKind.HTOD.value
        ]
        memcpy_dth_activities = [
            a
            for a in memcpy_activities
            if getattr(a, "copy_kind", None) == cupti.ActivityMemcpyKind.DTOH.value
        ]
        memcpy_hth_activities = [
            a
            for a in memcpy_activities
            if getattr(a, "copy_kind", None) == cupti.ActivityMemcpyKind.HTOH.value
        ]

        dtd_throughput = sum(a.bytes for a in memcpy_dtd_activities) / elapse
        dtd_latency = (
            sum(a.end - a.start for a in memcpy_dtd_activities)
            / len(memcpy_dtd_activities)
            if memcpy_dtd_activities
            else 0.0
        )
        max_dtd_latency = max(
            (a.end - a.start for a in memcpy_dtd_activities), default=0
        )
        htd_throughput = sum(a.bytes for a in memcpy_htd_activities) / elapse
        htd_latency = (
            sum(a.end - a.start for a in memcpy_htd_activities)
            / len(memcpy_htd_activities)
            if memcpy_htd_activities
            else 0.0
        )
        max_htd_latency = max(
            (a.end - a.start for a in memcpy_htd_activities), default=0
        )
        dth_throughput = sum(a.bytes for a in memcpy_dth_activities) / elapse
        dth_latency = (
            sum(a.end - a.start for a in memcpy_dth_activities)
            / len(memcpy_dth_activities)
            if memcpy_dth_activities
            else 0.0
        )
        max_dth_latency = max(
            (a.end - a.start for a in memcpy_dth_activities), default=0
        )
        hth_throughput = sum(a.bytes for a in memcpy_hth_activities) / elapse
        hth_latency = (
            sum(a.end - a.start for a in memcpy_hth_activities)
            / len(memcpy_hth_activities)
            if memcpy_hth_activities
            else 0.0
        )
        max_hth_latency = max(
            (a.end - a.start for a in memcpy_hth_activities), default=0
        )

        # Alloc and free
        mem_activities = [a for a in activities if a.kind == _MEMORY2_KIND]
        mem_device_activities = [
            a
            for a in mem_activities
            if getattr(a, "memory_kind", None) in _DEVICE_KINDS
        ]
        mem_pinned_activities = [
            a for a in mem_activities if getattr(a, "memory_kind", None) == _PINNED_KIND
        ]
        alloc_bytes = sum(
            a.bytes
            for a in mem_device_activities
            if getattr(a, "memory_operation_type", None) == _ALLOC_OP
        )
        free_bytes = sum(
            a.bytes
            for a in mem_device_activities
            if getattr(a, "memory_operation_type", None) == _RELEASE_OP
        )
        pin_bytes = sum(
            a.bytes
            for a in mem_pinned_activities
            if getattr(a, "memory_operation_type", None) == _ALLOC_OP
        )
        alloc_speed = alloc_bytes / elapse
        free_speed = free_bytes / elapse
        pin_speed = pin_bytes / elapse

        # Sync
        sync_activities = [a for a in activities if a.kind == _SYNCHRONIZATION_KIND]
        sync_duration = (
            sum(a.end - a.start for a in sync_activities) / len(sync_activities)
            if sync_activities
            else 0.0
        )
        max_sync_duration = max((a.end - a.start for a in sync_activities), default=0)

        with self._lock:
            self._stats.alloc_bytes = alloc_bytes
            self._stats.free_bytes = free_bytes
            self._stats.pin_bytes = pin_bytes
            self._stats.alloc_speed = alloc_speed
            self._stats.free_speed = free_speed
            self._stats.pin_speed = pin_speed
            self._stats.max_sync_duration = max_sync_duration
            self._stats.sync_duration = sync_duration
            self._stats.dtd_throughput = dtd_throughput
            self._stats.dtd_latency = dtd_latency
            self._stats.max_dtd_latency = max_dtd_latency
            self._stats.htd_throughput = htd_throughput
            self._stats.htd_latency = htd_latency
            self._stats.max_htd_latency = max_htd_latency
            self._stats.dth_throughput = dth_throughput
            self._stats.dth_latency = dth_latency
            self._stats.max_dth_latency = max_dth_latency
            self._stats.hth_throughput = hth_throughput
            self._stats.hth_latency = hth_latency
            self._stats.max_hth_latency = max_hth_latency

    def _drain_all(self) -> list:
        all_activities = []
        while True:
            try:
                batch = self._activity_queue.get_nowait()
                all_activities.extend(batch)
            except queue.Empty:
                break
        return all_activities

    def _loop(self):
        next_tick_time = time.time() + CALC_INTERVAL
        last_tick = time.time()
        while True:
            self._flush()
            elapse = time.time() - last_tick

            all_activities = self._drain_all()
            self._process_activity(all_activities, elapse)

            last_tick = time.time()

            sleep_time = next_tick_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            next_tick_time += CALC_INTERVAL
