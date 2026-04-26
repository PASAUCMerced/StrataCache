from stratacache.telemetry.telemetry import (
    StrataTierStats,
    StrataTierTelemetry,
    StrataTierType,
)
from stratacache.telemetry.utils import human_readable_size

from dataclasses import dataclass, field
import threading
import logging
import time

try:
    import stratacache.pcm as pcm  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    pcm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CALC_INTERVAL = 1.0
MEM_BANDWIDTH_BASE = 1000


@dataclass
class StrataCPUStats(StrataTierStats):
    mem_read_bw: float = 0.0
    mem_write_bw: float = 0.0

    pcie_read_bw: float = 0.0
    pcie_write_bw: float = 0.0


class StrataCPUTelemetry(StrataTierTelemetry):
    def __init__(self, telemetry, pcm_events=None, lib_path=None):
        super().__init__(StrataTierType.CPU, telemetry)
        self._stats = StrataCPUStats()
        self._lock = threading.Lock()
        if pcm is None:
            logger.info(
                "stratacache.pcm extension is not built; CPU bandwidth "
                "metrics disabled. Rebuild with STRATACACHE_BUILD_PCM=1 "
                "to enable."
            )
            self._pcm_collector = None
            self._mem_bw_matrix = None
            self._pcie_bw_matrix = None
            return
        self._pcm_collector = pcm.PCMCollectorHandle()
        self._mem_bw_matrix = self._pcm_collector.get_mem_bandwidth_buffer()
        self._pcie_bw_matrix = self._pcm_collector.get_pcie_bandwidth_buffer()

        logger.info("Starting CPU telemetry thread")
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self):
        next_tick_time = time.time() + CALC_INTERVAL
        while True:
            self._pcm_collector.tick()
            read_bw = self._mem_bw_matrix[:, :, 0].sum()
            write_bw = self._mem_bw_matrix[:, :, 1].sum()
            pcie_read_bw = self._pcie_bw_matrix[:, 0].sum()
            pcie_write_bw = self._pcie_bw_matrix[:, 1].sum()

            with self._lock:
                self._stats.mem_read_bw = read_bw
                self._stats.mem_write_bw = write_bw
                self._stats.pcie_read_bw = pcie_read_bw
                self._stats.pcie_write_bw = pcie_write_bw

            sleep_time = next_tick_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            next_tick_time += CALC_INTERVAL
