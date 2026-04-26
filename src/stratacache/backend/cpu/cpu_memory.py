from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

from stratacache.backend.base import BackendStats, MemoryLayer
from stratacache.backend.cpu.cpu_allocator import CpuAllocator
from stratacache.core.artifact import ArtifactId
from stratacache.core.errors import ArtifactNotFound
from stratacache.core.memory_obj import MemoryObj


class CpuMemoryLayer(MemoryLayer):
    """
    In-process memory store with best-effort LRU by bytes.

    Phase 1: payloads are MemoryObj (was: bytes). The eviction strategy is
    unchanged from v0.1 - sum of MemoryObj.get_size() is compared against
    capacity, oldest entries pop first.
    """

    def __init__(
        self,
        *,
        capacity_bytes: Optional[int] = None,
        store_name: str = "cpu",
        allocator: Optional[CpuAllocator] = None,
    ) -> None:
        self._name = store_name
        self._capacity_bytes = capacity_bytes
        self._allocator = allocator or CpuAllocator(capacity_bytes=capacity_bytes)
        self._lock = threading.RLock()
        self._lru: "OrderedDict[str, MemoryObj]" = OrderedDict()
        self._bytes_used = 0

    @property
    def name(self) -> str:
        return self._name

    def exists(self, artifact_id: ArtifactId) -> bool:
        k = str(artifact_id)
        with self._lock:
            return k in self._lru

    def get(self, artifact_id: ArtifactId) -> MemoryObj:
        k = str(artifact_id)
        with self._lock:
            mo = self._lru.get(k)
            if mo is None:
                raise ArtifactNotFound(k)
            # Touch LRU
            self._lru.move_to_end(k, last=True)
            return mo

    def put(self, artifact_id: ArtifactId, memory_obj: MemoryObj) -> int:
        k = str(artifact_id)
        with self._lock:
            old_size = 0
            old = self._lru.get(k)
            if old is not None:
                old_size = old.get_size()
                self._bytes_used -= old_size
                self._lru.pop(k, None)

            self._lru[k] = memory_obj
            self._bytes_used += memory_obj.get_size()
            self._lru.move_to_end(k, last=True)

            evicted_size = self._evict_if_needed()
            return old_size + evicted_size

    def delete(self, artifact_id: ArtifactId) -> int:
        k = str(artifact_id)
        with self._lock:
            mo = self._lru.pop(k, None)
            if mo is not None:
                size = mo.get_size()
                self._bytes_used -= size
                return size
            return 0

    def stats(self) -> BackendStats:
        with self._lock:
            return BackendStats(
                items=len(self._lru),
                bytes_used=int(self._bytes_used),
                bytes_capacity=self._capacity_bytes,
            )

    def _evict_if_needed(self) -> int:
        if self._capacity_bytes is None:
            return 0

        released_size = 0
        while self._bytes_used > self._capacity_bytes and self._lru:
            _, mo = self._lru.popitem(last=False)
            sz = mo.get_size()
            released_size += sz
            self._bytes_used -= sz
        return released_size


# Backward-compatible alias (v0.1)
CpuStore = CpuMemoryLayer
