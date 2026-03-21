from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from stratacache.backend.base import BackendStats, MemoryLayer
from stratacache.backend.cxl.binding import CXL_SHM_ONAME_LEN, CxlShm
from stratacache.core.artifact import ArtifactId, ArtifactMeta
from stratacache.core.errors import ArtifactNotFound, BackendError
from stratacache.core.keycodec import KeyCodec
from stratacache.core.record_codec import decode_record, encode_record


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


class CxlMemoryLayer(MemoryLayer):
    """
    CXL store backed by `cxl_shm.c` DAX mapping.

    Each stored object is a single CXL shm object keyed by a short deterministic
    name derived from ArtifactId (CXL name max length is 20 bytes).
    """

    def __init__(self, *, config: CxlConfig = CxlConfig(), store_name: str = "cxl") -> None:
        self._name = store_name
        self._cfg = config
        self._lock = threading.RLock()
        self._cxl = CxlShm(num_procs=config.num_procs, rank=config.rank)

        # cxl_shm.c reads DAX device from env var; keep ours separate from lmcache.
        if config.dax_device is not None:
            os.environ["LMCACHE_CXL_DAX_DEVICE"] = config.dax_device

        self._cxl.init()
        if config.reset_metadata_on_init:
            self._cxl.reset_metadata()

        self._bytes_used = 0  # best-effort: only updated for this process' puts

    @property
    def name(self) -> str:
        return self._name

    def _cxl_name(self, artifact_id: ArtifactId) -> str:
        # Prefer plain ids if short, else stable hash.
        s = str(artifact_id)
        if 0 < len(s) <= CXL_SHM_ONAME_LEN and s.isascii():
            return s
        return KeyCodec.short_hash_name(s, prefix="H", hex_chars=16, max_len=CXL_SHM_ONAME_LEN)

    def exists(self, artifact_id: ArtifactId) -> bool:
        name = self._cxl_name(artifact_id)
        with self._lock:
            try:
                hnd = self._cxl.open(name)
            except KeyError:
                return False
            try:
                return True
            finally:
                self._cxl.close(hnd)

    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]:
        name = self._cxl_name(artifact_id)
        with self._lock:
            try:
                hnd = self._cxl.open(name)
            except KeyError as e:
                raise ArtifactNotFound(str(artifact_id)) from e
            try:
                obj = hnd.obj.contents  # type: ignore[union-attr]
                actual = int(getattr(obj, "actual_size", 0))
                if actual <= 0:
                    raise BackendError(f"CXL object has invalid actual_size={actual} for name={name}")
                buf = ctypes.string_at(hnd.mapped_addr, actual)
                payload, meta = decode_record(buf)
                return payload, meta
            finally:
                self._cxl.close(hnd)

    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> int:
        name = self._cxl_name(artifact_id)
        record = encode_record(payload, meta)
        actual_size = len(record)
        alloc_size = _align_up(actual_size, self._cfg.alloc_align)
        
        released_size = 0

        with self._lock:
            # v0.1 behavior: overwrite by best-effort delete then create.
            try:
                old = self._cxl.open(name)
            except KeyError:
                old = None
            if old is not None:
                try:
                    # Get replacement size delta from old object to accumulate release size.
                    obj = old.obj.contents
                    old_alloc_size = int(getattr(obj, "alloc_size", 0))
                    if old_alloc_size > 0:
                        released_size += old_alloc_size

                    self._cxl.destroy(old)
                finally:
                    self._cxl.close(old)

            hnd = self._cxl.create(name, alloc_size=alloc_size, actual_size=actual_size)
            try:
                ctypes.memmove(hnd.mapped_addr, record, actual_size)
                self._cxl.flush(hnd.mapped_addr, actual_size)
            finally:
                self._cxl.close(hnd)

            self._bytes_used += alloc_size
            self._bytes_used -= released_size
        return released_size
            

    def delete(self, artifact_id: ArtifactId) -> int:
        name = self._cxl_name(artifact_id)
        with self._lock:
            try:
                hnd = self._cxl.open(name)
            except KeyError:
                return 0
            try:
                # Get size from object to accumulate release size.
                obj = hnd.obj.contents
                alloc_size = int(getattr(obj, "alloc_size", 0))
                self._cxl.destroy(hnd)
                self._bytes_used -= alloc_size if alloc_size > 0 else 0
                return alloc_size
            finally:
                self._cxl.close(hnd)

    def stats(self) -> BackendStats:
        with self._lock:
            return BackendStats(
                items=-1,  # unknown without enumerating (not supported by cxl_shm.c)
                bytes_used=int(self._bytes_used),
                bytes_capacity=self._cfg.max_bytes,
            )


# Backward-compatible alias (v0.1)
CxlStore = CxlMemoryLayer

