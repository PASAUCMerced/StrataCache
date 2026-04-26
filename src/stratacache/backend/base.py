from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from stratacache.core.artifact import ArtifactId
from stratacache.core.errors import ArtifactNotFound
from stratacache.core.memory_obj import MemoryObj


@dataclass(frozen=True, slots=True)
class BackendStats:
    items: int
    bytes_used: int
    bytes_capacity: Optional[int] = None


# Type for backend-level admit/evict event sinks (B11/A11). Keeps the
# backend layer type-agnostic: payload identity is just (op, ArtifactId,
# size, layer_name). KV-specific translation lives in artifacts/kv/.
BackendEventSink = Callable[[str, ArtifactId, int, str], None]


class MemoryLayer(ABC):
    """
    Minimal memory layer contract.

    Payload type at this boundary is `MemoryObj` (type-agnostic). Backends
    that internally serialize (e.g. CXL) call into the appropriate codec
    themselves.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def exists(self, artifact_id: ArtifactId) -> bool: ...

    @abstractmethod
    def get(
        self,
        artifact_id: ArtifactId,
        *,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> MemoryObj:
        """
        Fetch a stored MemoryObj.

        `dtype` and `shape` are optional caller-supplied hints used by
        backends that don't persist user-meta (notably CXL: only logical
        byte size is recoverable from the C library, so dtype/shape must
        come from the caller's schema cache). Backends that store typed
        objects directly (CPU layer) ignore the hints.
        """
        ...

    @abstractmethod
    def put(self, artifact_id: ArtifactId, memory_obj: MemoryObj) -> int: ...

    @abstractmethod
    def delete(self, artifact_id: ArtifactId) -> int: ...

    @abstractmethod
    def stats(self) -> BackendStats: ...

    # ---- Batched API (B7) -------------------------------------------------
    # Default implementations loop over the single-item methods; backends
    # that can amortize lock acquisition or I/O override these.

    def batched_exists(
        self, artifact_ids: Sequence[ArtifactId]
    ) -> list[bool]:
        return [self.exists(a) for a in artifact_ids]

    def batched_get(
        self,
        artifact_ids: Sequence[ArtifactId],
        *,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> list[Optional[MemoryObj]]:
        out: list[Optional[MemoryObj]] = []
        for a in artifact_ids:
            try:
                out.append(self.get(a, dtype=dtype, shape=shape))
            except ArtifactNotFound:
                out.append(None)
        return out

    def batched_put(
        self,
        items: Iterable[tuple[ArtifactId, MemoryObj]],
    ) -> int:
        total_released = 0
        for aid, mo in items:
            total_released += self.put(aid, mo)
        return total_released

    def batched_delete(self, artifact_ids: Sequence[ArtifactId]) -> int:
        total = 0
        for a in artifact_ids:
            total += self.delete(a)
        return total

    # ---- Event sink protocol (A11) ---------------------------------------
    # Default no-op; backends that emit events override.

    def set_event_sink(self, sink: Optional[BackendEventSink]) -> None:
        return None


# Backward-compatible aliases (v0.1)
StorageBackend = MemoryLayer
LayerStats = BackendStats
