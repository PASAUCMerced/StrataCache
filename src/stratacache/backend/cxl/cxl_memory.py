from __future__ import annotations

import threading
from typing import Optional

from stratacache.backend.base import BackendStats, MemoryLayer
from stratacache.backend.cxl.cxl_allocator import CxlAllocator, CxlConfig
from stratacache.core.artifact import ArtifactId
from stratacache.core.errors import ArtifactNotFound, BackendError
from stratacache.core.memory_obj import BytesMemoryObj, MemoryObj
from stratacache.core.record_codec import decode_record, encode_record


class CxlMemoryLayer(MemoryLayer):
    """
    CXL store backed by `cxl_shm.c` DAX mapping.

    Each stored object is a single CXL shm object keyed by a short
    deterministic name derived from ArtifactId. Because CXL stores opaque
    bytes per object, we serialize MemoryObj -> (record bytes via
    encode_record) and the reverse on read; the resulting in-memory shape
    is BytesMemoryObj.

    NOTE: this asymmetry with the CPU backend (which can keep tensor-typed
    MemoryObjs) is by design: see ARCHITECTURE.md §3 / discussion in
    tmp/lmcache_gap_analysis.md.
    """

    def __init__(
        self,
        *,
        config: CxlConfig = CxlConfig(),
        store_name: str = "cxl",
        allocator: Optional[CxlAllocator] = None,
    ) -> None:
        self._name = store_name
        self._lock = threading.RLock()
        self._allocator = allocator or CxlAllocator(config)
        self._bytes_used = 0  # best-effort: only updated for this process' puts

    @property
    def name(self) -> str:
        return self._name

    def _name_of(self, artifact_id: ArtifactId) -> str:
        return self._allocator.derive_name(str(artifact_id))

    def exists(self, artifact_id: ArtifactId) -> bool:
        name = self._name_of(artifact_id)
        with self._lock:
            hnd = self._allocator.open(name)
            if hnd is None:
                return False
            self._allocator.close(hnd)
            return True

    def get(self, artifact_id: ArtifactId) -> MemoryObj:
        name = self._name_of(artifact_id)
        with self._lock:
            hnd = self._allocator.open(name)
            if hnd is None:
                raise ArtifactNotFound(str(artifact_id))
            try:
                buf = self._allocator.read(hnd)
            except RuntimeError as e:
                raise BackendError(str(e)) from e
            finally:
                self._allocator.close(hnd)
        payload, meta = decode_record(buf)
        return BytesMemoryObj(payload, meta)

    def put(self, artifact_id: ArtifactId, memory_obj: MemoryObj) -> int:
        name = self._name_of(artifact_id)
        # Materialize bytes via the codec for CXL persistence.
        record = encode_record(memory_obj.byte_array, memory_obj.metadata.artifact_meta)
        actual_size = len(record)

        released_size = 0
        with self._lock:
            old = self._allocator.open(name)
            if old is not None:
                try:
                    released_size += self._allocator.destroy(old)
                finally:
                    self._allocator.close(old)

            hnd, alloc_size = self._allocator.create(name, actual_size=actual_size)
            try:
                self._allocator.write(hnd, record)
            finally:
                self._allocator.close(hnd)

            self._bytes_used += alloc_size
            self._bytes_used -= released_size
        return released_size

    def delete(self, artifact_id: ArtifactId) -> int:
        name = self._name_of(artifact_id)
        with self._lock:
            hnd = self._allocator.open(name)
            if hnd is None:
                return 0
            try:
                released = self._allocator.destroy(hnd)
            finally:
                self._allocator.close(hnd)
            self._bytes_used -= released if released > 0 else 0
            return released

    def stats(self) -> BackendStats:
        with self._lock:
            return BackendStats(
                items=-1,  # unknown without enumerating (not supported by cxl_shm.c)
                bytes_used=int(self._bytes_used),
                bytes_capacity=self._allocator.config.max_bytes,
            )


# Backward-compatible alias (v0.1)
CxlStore = CxlMemoryLayer
