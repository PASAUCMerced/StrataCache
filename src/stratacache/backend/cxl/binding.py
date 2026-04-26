from __future__ import annotations

import ctypes
import logging
import os
from ctypes import POINTER, Structure, c_char, c_char_p, c_int, c_size_t, c_uint32, c_uint64, c_uint8, c_void_p
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _load_lib() -> ctypes.CDLL:
    """
    Locate and load libcxl_shm.so.

    Default location is stratacache/csrc/cxl/libcxl_shm.so (produced by
    `make` in that directory). The CWD and the loader path are tried as
    fallbacks so a hand-installed system copy still works.
    """
    here = os.path.dirname(__file__)
    csrc_lib = os.path.abspath(
        os.path.join(here, "..", "..", "..", "..", "csrc", "cxl", "libcxl_shm.so")
    )
    candidates = [
        csrc_lib,
        os.path.join(os.getcwd(), "libcxl_shm.so"),
        "./libcxl_shm.so",
        "libcxl_shm.so",
    ]

    last_err: Optional[OSError] = None
    for p in candidates:
        try:
            lib = ctypes.CDLL(p)
            logger.info("Loaded libcxl_shm.so from %s", p)
            return lib
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load libcxl_shm.so. last error: {last_err}")


_lib = _load_lib()


# Constants from cxl_shm.h
CXL_SHM_ONAME_LEN = 20


class CxlShmObjMeta(Structure):
    _fields_ = [
        ("name", c_char * CXL_SHM_ONAME_LEN),
        ("offset", c_uint64),
        ("size", c_uint64),
        ("actual_size", c_uint64),
        ("in_use", c_uint8),
    ]


class CxlShmHnd(Structure):
    _fields_ = [
        ("obj", POINTER(CxlShmObjMeta)),
        ("mapped_addr", c_void_p),
    ]


# Function signatures
_lib.cxl_shm_init.argtypes = [c_int, c_int]
_lib.cxl_shm_init.restype = c_int

_lib.cxl_shm_finalize.argtypes = []
_lib.cxl_shm_finalize.restype = c_int

_lib.cxl_shm_create.argtypes = [c_char_p, c_size_t, c_size_t, POINTER(CxlShmHnd)]
_lib.cxl_shm_create.restype = c_int

_lib.cxl_shm_open_obj.argtypes = [c_char_p, POINTER(CxlShmHnd)]
_lib.cxl_shm_open_obj.restype = c_int

_lib.cxl_shm_close.argtypes = [POINTER(CxlShmHnd)]
_lib.cxl_shm_close.restype = c_int

_lib.cxl_shm_destroy_from_hnd.argtypes = [POINTER(CxlShmHnd)]
_lib.cxl_shm_destroy_from_hnd.restype = c_int

_lib.clflush_region_with_mfence.argtypes = [c_void_p, c_size_t]
_lib.clflush_region_with_mfence.restype = c_int

_lib.cxl_shm_reset_metadata.argtypes = []
_lib.cxl_shm_reset_metadata.restype = c_int

_lib.cxl_shm_debug_count_in_use.argtypes = [
    POINTER(c_uint64),
    POINTER(c_uint64),
    POINTER(c_uint64),
    POINTER(c_uint64),
    POINTER(c_uint64),
    POINTER(c_uint32),
]
_lib.cxl_shm_debug_count_in_use.restype = c_int


class CxlShm:
    def __init__(self, *, num_procs: int = 1, rank: int = 0) -> None:
        self._num_procs = int(num_procs)
        self._rank = int(rank)
        self._initialized = False

    def init(self) -> None:
        rc = _lib.cxl_shm_init(self._num_procs, self._rank)
        if rc != 0:
            raise RuntimeError(f"cxl_shm_init failed rc={rc}")
        self._initialized = True

    def finalize(self) -> None:
        if not self._initialized:
            return
        rc = _lib.cxl_shm_finalize()
        if rc != 0:
            raise RuntimeError(f"cxl_shm_finalize failed rc={rc}")
        self._initialized = False

    def reset_metadata(self) -> None:
        rc = _lib.cxl_shm_reset_metadata()
        if rc != 0:
            raise RuntimeError(f"cxl_shm_reset_metadata failed rc={rc}")

    def create(self, name: str, alloc_size: int, actual_size: int) -> CxlShmHnd:
        hnd = CxlShmHnd()
        rc = _lib.cxl_shm_create(name.encode("utf-8"), int(alloc_size), int(actual_size), ctypes.byref(hnd))
        if rc != 0:
            raise RuntimeError(f"cxl_shm_create failed rc={rc}")
        return hnd

    def open(self, name: str) -> CxlShmHnd:
        hnd = CxlShmHnd()
        rc = _lib.cxl_shm_open_obj(name.encode("utf-8"), ctypes.byref(hnd))
        if rc != 0:
            raise KeyError(name)
        return hnd

    def close(self, hnd: CxlShmHnd) -> None:
        rc = _lib.cxl_shm_close(ctypes.byref(hnd))
        if rc != 0:
            raise RuntimeError(f"cxl_shm_close failed rc={rc}")

    def destroy(self, hnd: CxlShmHnd) -> None:
        rc = _lib.cxl_shm_destroy_from_hnd(ctypes.byref(hnd))
        if rc != 0:
            raise RuntimeError(f"cxl_shm_destroy_from_hnd failed rc={rc}")

    def flush(self, addr: c_void_p, size: int) -> None:
        rc = _lib.clflush_region_with_mfence(addr, int(size))
        if rc != 0:
            raise RuntimeError(f"clflush_region_with_mfence failed rc={rc}")

    def debug_count_in_use(self) -> Optional[Tuple[int, int, int, int, int, int]]:
        reachable_in_use = c_uint64(0)
        reachable_max = c_uint64(0)
        total_in_use = c_uint64(0)
        total_max = c_uint64(0)
        curr_offset = c_uint64(0)
        bucket_levels = c_uint32(0)
        rc = _lib.cxl_shm_debug_count_in_use(
            ctypes.byref(reachable_in_use),
            ctypes.byref(reachable_max),
            ctypes.byref(total_in_use),
            ctypes.byref(total_max),
            ctypes.byref(curr_offset),
            ctypes.byref(bucket_levels),
        )
        if rc != 0:
            return None
        return (
            int(reachable_in_use.value),
            int(reachable_max.value),
            int(total_in_use.value),
            int(total_max.value),
            int(curr_offset.value),
            int(bucket_levels.value),
        )

