from __future__ import annotations

import logging
import threading
from typing import Optional

from stratacache.backend.base import BackendEventSink, BackendStats, MemoryLayer
from stratacache.backend.cxl.cxl_allocator import CxlAllocator, CxlConfig
from stratacache.core.artifact import ArtifactId, ArtifactMeta
from stratacache.core.errors import ArtifactNotFound, BackendError
from stratacache.core.memory_obj import BytesMemoryObj, MemoryObj, TensorMemoryObj


logger = logging.getLogger(__name__)


class CxlMemoryLayer(MemoryLayer):
    """
    CXL store backed by `cxl_shm.c` DAX mapping.

    Each stored object is a single CXL shm object keyed by a short
    deterministic name derived from ArtifactId. The CXL C library
    provides only its own bookkeeping in metadata slots (name / offset /
    size / actual_size / in_use); user-level ArtifactMeta is NOT
    persisted.

    On read, callers may pass `dtype` / `shape` hints to get a typed
    `TensorMemoryObj` view; otherwise we return a `BytesMemoryObj` with
    empty meta and let the caller decide how to interpret the bytes.
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
        self._event_sink: Optional[BackendEventSink] = None

    def set_event_sink(self, sink: Optional[BackendEventSink]) -> None:
        self._event_sink = sink

    def _emit(self, op: str, aid: ArtifactId, sz: int) -> None:
        sink = self._event_sink
        if sink is None:
            return
        try:
            sink(op, aid, sz, self._name)
        except Exception:  # noqa: BLE001
            logger.exception("BackendEventSink raised; ignoring.")

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

    def get(
        self,
        artifact_id: ArtifactId,
        *,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> MemoryObj:
        name = self._name_of(artifact_id)
        with self._lock:
            hnd = self._allocator.open(name)
            if hnd is None:
                raise ArtifactNotFound(str(artifact_id))
            try:
                payload = self._allocator.read(hnd)
            except RuntimeError as e:
                raise BackendError(str(e)) from e
            finally:
                self._allocator.close(hnd)

        if dtype is not None and shape is not None:
            return self._typed_from_bytes(payload, dtype=dtype, shape=shape)
        # Bytes return: caller is responsible for reshaping. ArtifactMeta is
        # empty because CXL doesn't persist user-meta.
        return BytesMemoryObj(payload, ArtifactMeta())

    @staticmethod
    def _typed_from_bytes(
        payload: bytes, *, dtype: str, shape: tuple[int, ...]
    ) -> MemoryObj:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            # Torch missing: degrade to raw bytes view.
            return BytesMemoryObj(payload, ArtifactMeta())
        dt = getattr(torch, str(dtype), None)
        if dt is None:
            raise BackendError(f"unsupported dtype hint: {dtype}")
        u8 = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        tensor = u8.view(dt).reshape(tuple(int(x) for x in shape)).clone()
        return TensorMemoryObj(tensor, ArtifactMeta())

    def put(self, artifact_id: ArtifactId, memory_obj: MemoryObj) -> int:
        name = self._name_of(artifact_id)
        # Raw payload only; user-meta is NOT persisted on CXL by design.
        payload = memory_obj.byte_array
        actual_size = len(payload)

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
                self._allocator.write(hnd, payload)
            finally:
                self._allocator.close(hnd)

            self._bytes_used += alloc_size
            self._bytes_used -= released_size
        if released_size > 0:
            self._emit("remove", artifact_id, released_size)
        self._emit("store", artifact_id, actual_size)
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
        if released > 0:
            self._emit("remove", artifact_id, released)
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
