from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from stratacache.core.artifact import ArtifactId
from stratacache.core.memory_obj import MemoryObj


@dataclass(frozen=True, slots=True)
class BackendStats:
    items: int
    bytes_used: int
    bytes_capacity: Optional[int] = None


class MemoryLayer(ABC):
    """
    Minimal memory layer contract.

    Payload type at this boundary is `MemoryObj` (type-agnostic). The
    previous v0.1 contract that used raw `bytes` is gone; backends that
    internally serialize (e.g. CXL) call into the appropriate codec
    themselves.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def exists(self, artifact_id: ArtifactId) -> bool:
        """Return True iff `artifact_id` is currently stored in this layer."""
        ...

    @abstractmethod
    def get(self, artifact_id: ArtifactId) -> MemoryObj:
        """
        Retrieve the stored MemoryObj. Raises ArtifactNotFound if absent.
        """
        ...

    @abstractmethod
    def put(self, artifact_id: ArtifactId, memory_obj: MemoryObj) -> int:
        """
        Store the MemoryObj. Returns the number of bytes that this put
        released (sum of replaced-key size + LRU-evicted sizes); 0 if
        nothing was freed.
        """
        ...

    @abstractmethod
    def delete(self, artifact_id: ArtifactId) -> int:
        """Remove the MemoryObj. Returns the bytes released."""
        ...

    @abstractmethod
    def stats(self) -> BackendStats: ...


# Backward-compatible aliases (v0.1)
StorageBackend = MemoryLayer
LayerStats = BackendStats
