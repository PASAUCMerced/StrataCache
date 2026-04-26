"""
System / topology detection helpers (B6, B9).

Best-effort: NUMA work uses libnuma via ctypes if available, else logs and
returns a no-op. Memory detection uses psutil if available, else parses
/proc/meminfo. Nothing here raises on missing dependencies.

Class names (`NUMADetector`, `SystemMemoryDetector`) mirror LMCache's
public API (Apache-2.0); the implementations here are independent.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# System memory
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemMemoryInfo:
    total_bytes: int
    available_bytes: int


class SystemMemoryDetector:
    @staticmethod
    def info() -> Optional[SystemMemoryInfo]:
        try:
            import psutil  # type: ignore[import-not-found]

            v = psutil.virtual_memory()
            return SystemMemoryInfo(
                total_bytes=int(v.total), available_bytes=int(v.available)
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            kv = {}
            for ln in lines:
                if ":" not in ln:
                    continue
                k, v = ln.split(":", 1)
                kv[k.strip()] = v.strip()
            total_kb = int(kv["MemTotal"].split()[0])
            avail_kb = int(kv.get("MemAvailable", kv["MemFree"]).split()[0])
            return SystemMemoryInfo(
                total_bytes=total_kb * 1024,
                available_bytes=avail_kb * 1024,
            )
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def get_available_memory_bytes() -> Optional[int]:
        info = SystemMemoryDetector.info()
        return None if info is None else info.available_bytes


# ----------------------------------------------------------------------------
# NUMA
# ----------------------------------------------------------------------------


_LIBNUMA_LOCK = threading.Lock()
_LIBNUMA: Optional[ctypes.CDLL] = None
_LIBNUMA_PROBED = False


def _load_libnuma() -> Optional[ctypes.CDLL]:
    global _LIBNUMA, _LIBNUMA_PROBED  # noqa: PLW0603
    with _LIBNUMA_LOCK:
        if _LIBNUMA_PROBED:
            return _LIBNUMA
        _LIBNUMA_PROBED = True
        try:
            path = ctypes.util.find_library("numa")
            if path is None:
                return None
            lib = ctypes.CDLL(path, use_errno=True)
            lib.numa_available.restype = ctypes.c_int
            if int(lib.numa_available()) < 0:
                return None
            lib.numa_max_node.restype = ctypes.c_int
            lib.numa_node_of_cpu.argtypes = [ctypes.c_int]
            lib.numa_node_of_cpu.restype = ctypes.c_int
            lib.numa_tonode_memory.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.numa_tonode_memory.restype = None
            _LIBNUMA = lib
            return lib
        except Exception:  # noqa: BLE001
            return None


@dataclass(frozen=True, slots=True)
class NumaTopology:
    available: bool
    max_node: int
    cpu_to_node: dict[int, int]


class NUMADetector:
    @staticmethod
    def topology() -> NumaTopology:
        lib = _load_libnuma()
        if lib is None:
            return NumaTopology(available=False, max_node=-1, cpu_to_node={})
        max_node = int(lib.numa_max_node())
        n_cpu = os.cpu_count() or 0
        cpu_to_node: dict[int, int] = {}
        for c in range(n_cpu):
            try:
                cpu_to_node[c] = int(lib.numa_node_of_cpu(c))
            except Exception:  # noqa: BLE001
                pass
        return NumaTopology(
            available=True, max_node=max_node, cpu_to_node=cpu_to_node
        )

    @staticmethod
    def node_for_current_cpu() -> Optional[int]:
        lib = _load_libnuma()
        if lib is None:
            return None
        # sched_getcpu via libc
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        if not hasattr(libc, "sched_getcpu"):
            return None
        libc.sched_getcpu.restype = ctypes.c_int
        cpu = int(libc.sched_getcpu())
        if cpu < 0:
            return None
        try:
            return int(lib.numa_node_of_cpu(cpu))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def bind_buffer_to_node(addr: int, length: int, node: int) -> bool:
        """
        Best-effort `numa_tonode_memory(addr, length, node)`. Returns True
        when libnuma is present and the call did not raise; False
        otherwise. Caller should not rely on success for correctness.
        """
        lib = _load_libnuma()
        if lib is None:
            return False
        try:
            lib.numa_tonode_memory(
                ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(node)
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception(
                "NUMADetector.bind_buffer_to_node failed (addr=0x%x len=%d node=%d)",
                addr,
                length,
                node,
            )
            return False
