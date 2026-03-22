from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from stratacache.core.artifact import ArtifactId, ArtifactMeta


@dataclass(frozen=True, slots=True)
class BackendStats:
    items: int
    bytes_used: int
    bytes_capacity: Optional[int] = None


class MemoryLayer(ABC):
    """
    Minimal memory layer contract.

    v0.1 uses bytes payloads and JSON-like metadata (encoded by the caller).
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def exists(self, artifact_id: ArtifactId) -> bool:
        """_summary_
            Check if an artifact exists in the store.
        Args:
            artifact_id (ArtifactId): Artifact ID
        Returns:
            bool: True if the artifact exists, False otherwise.
        """
        ...

    @abstractmethod
    def get(self, artifact_id: ArtifactId) -> Tuple[bytes, ArtifactMeta]:
        """_summary_
            Get an artifact from the store.
        Args:
            artifact_id (ArtifactId): Artifact ID
        Returns:
            Tuple[bytes, ArtifactMeta]: A tuple of (payload, metadata)
        """
        ...

    @abstractmethod
    def put(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> int:
        """_summary_
            Put an artifact into the store. If this operation releases a block, return the size of the released block.
        Args:
            artifact_id (ArtifactId): Artifact ID
            payload (bytes): Data payload
            meta (ArtifactMeta): Artifact metadata
        Returns:
            int: Total released block size. If no block is released, return 0.
        """
        ...

    @abstractmethod
    def delete(self, artifact_id: ArtifactId) -> int:
        """_summary_
            Delete an artifact from the store, and return the released block size.
        Args:
            artifact_id (ArtifactId): Artifact ID
        Returns:
            int: Released block size.
        """
        ...

    @abstractmethod
    def stats(self) -> BackendStats: ...


# Backward-compatible aliases (v0.1)
StorageBackend = MemoryLayer
LayerStats = BackendStats

