"""
CXL allocator: thin wrapper around the native cxl_shm_* primitives.

Holds the lifecycle of the underlying CXL DAX mapping and exposes
allocation / lookup / destroy operations keyed by short, ASCII names.
The byte layout of each object is the caller's responsibility.

cxl_memory.py composes this with the record codec to actually store
MemoryObjs.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional

from stratacache.backend.cxl.binding import CXL_SHM_ONAME_LEN, CxlShm, CxlShmHnd
from stratacache.core.keycodec import KeyCodec


def _align_up(n: int, align: int) -> int:
    if align <= 1:
        return n
    return (n + (align - 1)) // align * align


@dataclass(frozen=True, slots=True)
class CxlConfig:
    num_procs: int = 1
    rank: int = 0
    dax_device: Optional[str] = None
    reset_metadata_on_init: bool = False
    alloc_align: int = 64
    max_bytes: Optional[int] = None  # best-effort accounting


class CxlAllocator:
    """
    Owns the CxlShm session and the conventions for naming objects.
    """

    def __init__(self, config: CxlConfig) -> None:
        self._cfg = config
        # The C lib reads the DAX device path from this env var.
        if config.dax_device is not None:
            os.environ["STRATACACHE_CXL_DAX_DEVICE"] = config.dax_device
        self._cxl = CxlShm(num_procs=config.num_procs, rank=config.rank)
        self._cxl.init()
        if config.reset_metadata_on_init:
            self._cxl.reset_metadata()

    @property
    def config(self) -> CxlConfig:
        return self._cfg

    def derive_name(self, artifact_id_str: str) -> str:
        """
        Map a logical ArtifactId string to a deterministic CXL object name
        within CXL's 20-byte limit.
        """
        if 0 < len(artifact_id_str) <= CXL_SHM_ONAME_LEN and artifact_id_str.isascii():
            return artifact_id_str
        return KeyCodec.short_hash_name(
            artifact_id_str, prefix="H", hex_chars=16, max_len=CXL_SHM_ONAME_LEN
        )

    def open(self, name: str) -> Optional[CxlShmHnd]:
        try:
            return self._cxl.open(name)
        except KeyError:
            return None

    def close(self, hnd: CxlShmHnd) -> None:
        self._cxl.close(hnd)

    def create(self, name: str, *, actual_size: int) -> tuple[CxlShmHnd, int]:
        alloc_size = _align_up(actual_size, self._cfg.alloc_align)
        hnd = self._cxl.create(name, alloc_size=alloc_size, actual_size=actual_size)
        return hnd, alloc_size

    def destroy(self, hnd: CxlShmHnd) -> int:
        """Destroy an object referenced by handle and return its alloc_size."""
        obj = hnd.obj.contents
        alloc_size = int(getattr(obj, "alloc_size", 0))
        self._cxl.destroy(hnd)
        return alloc_size

    def write(self, hnd: CxlShmHnd, data: bytes) -> None:
        ctypes.memmove(hnd.mapped_addr, data, len(data))
        self._cxl.flush(hnd.mapped_addr, len(data))

    def read(self, hnd: CxlShmHnd) -> bytes:
        obj = hnd.obj.contents
        actual = int(getattr(obj, "actual_size", 0))
        if actual <= 0:
            raise RuntimeError(
                f"CXL object has invalid actual_size={actual}"
            )
        return ctypes.string_at(hnd.mapped_addr, actual)
