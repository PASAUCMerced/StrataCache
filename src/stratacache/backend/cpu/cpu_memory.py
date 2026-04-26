from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Optional, Sequence

from stratacache.backend.base import (
    BackendEventSink,
    BackendStats,
    MemoryLayer,
)
from stratacache.backend.cpu.cpu_allocator import CpuAllocator, MemorySlot
from stratacache.core.artifact import ArtifactId
from stratacache.core.errors import ArtifactNotFound
from stratacache.core.memory_obj import MemoryObj


logger = logging.getLogger(__name__)


class CpuMemoryLayer(MemoryLayer):
    """
    In-process memory store with best-effort LRU by bytes.

    Phase 2/3 features:
    - Optional pinned slab via CpuAllocator (B1/B2).
    - Ref-count / pin aware eviction (B3).
    - Allocate-then-evict-then-retry loop with `busy_loop` backpressure (B8).
    - Type-agnostic admit/evict event emission (A11).
    - Batched op overrides (B7) - same lock acquired once per batch.
    """

    def __init__(
        self,
        *,
        capacity_bytes: Optional[int] = None,
        store_name: str = "cpu",
        allocator: Optional[CpuAllocator] = None,
        pin_memory: bool = False,
        evict_retry_sleep_s: float = 0.01,
        evict_retry_max_s: float = 1.0,
    ) -> None:
        self._name = store_name
        self._capacity_bytes = capacity_bytes
        if allocator is None:
            slab_cap = capacity_bytes if pin_memory else None
            allocator = CpuAllocator(capacity_bytes=slab_cap, pin_memory=pin_memory)
        self._allocator = allocator
        self._lock = threading.Lock()
        self._lru: "OrderedDict[str, MemoryObj]" = OrderedDict()
        self._bytes_used = 0
        self._evict_skip = 0
        self._evict_retry_sleep_s = float(evict_retry_sleep_s)
        self._evict_retry_max_s = float(evict_retry_max_s)
        self._event_sink: Optional[BackendEventSink] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def allocator(self) -> CpuAllocator:
        return self._allocator

    def set_event_sink(self, sink: Optional[BackendEventSink]) -> None:
        self._event_sink = sink

    # ---- single-item API ---------------------------------------------------

    def exists(self, artifact_id: ArtifactId) -> bool:
        k = str(artifact_id)
        with self._lock:
            return k in self._lru

    def get(
        self,
        artifact_id: ArtifactId,
        *,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> MemoryObj:
        # CPU layer holds typed MemoryObjs directly; hints unused here.
        del dtype, shape
        k = str(artifact_id)
        with self._lock:
            mo = self._lru.get(k)
            if mo is None:
                raise ArtifactNotFound(k)
            self._lru.move_to_end(k, last=True)
            return mo

    def put(self, artifact_id: ArtifactId, memory_obj: MemoryObj) -> int:
        return self._put_one(artifact_id, memory_obj, fire_events=True)

    def delete(self, artifact_id: ArtifactId) -> int:
        return self._delete_one(artifact_id, fire_events=True)

    def stats(self) -> BackendStats:
        with self._lock:
            return BackendStats(
                items=len(self._lru),
                bytes_used=int(self._bytes_used),
                bytes_capacity=self._capacity_bytes,
            )

    # ---- batched overrides (B7) -------------------------------------------

    def batched_exists(self, artifact_ids: Sequence[ArtifactId]) -> list[bool]:
        with self._lock:
            return [str(a) in self._lru for a in artifact_ids]

    def batched_get(
        self,
        artifact_ids: Sequence[ArtifactId],
        *,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> list[Optional[MemoryObj]]:
        del dtype, shape
        out: list[Optional[MemoryObj]] = []
        with self._lock:
            for a in artifact_ids:
                k = str(a)
                mo = self._lru.get(k)
                if mo is None:
                    out.append(None)
                    continue
                self._lru.move_to_end(k, last=True)
                out.append(mo)
        return out

    def batched_put(self, items) -> int:
        total = 0
        events: list[tuple[str, ArtifactId, int, str]] = []
        with self._lock:
            for aid, mo in items:
                total += self._put_locked(aid, mo, events)
        # Notify outside the lock (B12).
        self._fan_events(events)
        return total

    def batched_delete(self, artifact_ids: Sequence[ArtifactId]) -> int:
        total = 0
        events: list[tuple[str, ArtifactId, int, str]] = []
        with self._lock:
            for aid in artifact_ids:
                total += self._delete_locked(aid, events)
        self._fan_events(events)
        return total

    # ---- B8: allocate-with-eviction --------------------------------------

    def allocate_slot(
        self,
        nbytes: int,
        *,
        busy_loop: bool = True,
        timeout_s: Optional[float] = None,
    ) -> Optional[MemorySlot]:
        """
        Try to obtain a slab slot of `nbytes`. If the underlying allocator
        is OOM, evict candidates from the LRU and retry. With
        `busy_loop=True`, sleep/retry until either a slot is obtained, the
        timeout (default `evict_retry_max_s`) is reached, or no
        evict candidates remain. With `busy_loop=False`, give up after one
        eviction round.

        Used by adapters that want to pre-reserve a buffer before calling
        `cpu_memory_obj_from_tensor`. The new MemoryObj is NOT inserted
        into the layer; callers do that via `put` in the usual way.
        """
        if not self._allocator.has_slab:
            return None

        deadline = (
            None
            if timeout_s is None
            else time.monotonic() + max(0.0, float(timeout_s))
        )
        if timeout_s is None:
            deadline = time.monotonic() + self._evict_retry_max_s

        evicted_events: list[tuple[str, ArtifactId, int, str]] = []
        try:
            while True:
                slot = self._allocator.try_allocate(nbytes)
                if slot is not None:
                    return slot

                # Try to free at least `nbytes` worth of evictable space.
                with self._lock:
                    freed = self._evict_until(nbytes, evicted_events)

                if freed == 0:
                    # Nothing evictable. If single-shot, give up.
                    if not busy_loop:
                        return None
                    # Otherwise back off and retry; the obstruction may
                    # clear once consumers release pins/refs.
                    if deadline is not None and time.monotonic() >= deadline:
                        return None
                    time.sleep(self._evict_retry_sleep_s)
                    continue
        finally:
            self._fan_events(evicted_events)

    # ---- internal: locked helpers ----------------------------------------

    def _put_one(
        self, artifact_id: ArtifactId, memory_obj: MemoryObj, *, fire_events: bool
    ) -> int:
        events: list[tuple[str, ArtifactId, int, str]] = []
        with self._lock:
            released = self._put_locked(artifact_id, memory_obj, events)
        if fire_events:
            self._fan_events(events)
        return released

    def _delete_one(
        self, artifact_id: ArtifactId, *, fire_events: bool
    ) -> int:
        events: list[tuple[str, ArtifactId, int, str]] = []
        with self._lock:
            released = self._delete_locked(artifact_id, events)
        if fire_events:
            self._fan_events(events)
        return released

    def _put_locked(
        self,
        artifact_id: ArtifactId,
        memory_obj: MemoryObj,
        events: list,
    ) -> int:
        k = str(artifact_id)
        old_size = 0
        old = self._lru.get(k)
        if old is not None:
            old_size = old.get_size()
            self._bytes_used -= old_size
            self._lru.pop(k, None)
            old.ref_count_down()
            events.append(("remove", artifact_id, old_size, self._name))

        self._lru[k] = memory_obj
        self._bytes_used += memory_obj.get_size()
        self._lru.move_to_end(k, last=True)
        events.append(("store", artifact_id, memory_obj.get_size(), self._name))

        evicted_size = self._evict_if_needed(events)
        return old_size + evicted_size

    def _delete_locked(
        self, artifact_id: ArtifactId, events: list
    ) -> int:
        k = str(artifact_id)
        mo = self._lru.pop(k, None)
        if mo is None:
            return 0
        size = mo.get_size()
        self._bytes_used -= size
        mo.ref_count_down()
        events.append(("remove", artifact_id, size, self._name))
        return size

    def _evict_if_needed(self, events: list) -> int:
        if self._capacity_bytes is None:
            return 0
        released = 0
        while self._bytes_used > self._capacity_bytes:
            victim_key = self._next_evict_candidate()
            if victim_key is None:
                logger.warning(
                    "CpuMemoryLayer(%s): cannot evict to reach capacity "
                    "(%d/%d bytes used); all remaining entries are pinned.",
                    self._name,
                    self._bytes_used,
                    self._capacity_bytes,
                )
                break
            mo = self._lru.pop(victim_key)
            sz = mo.get_size()
            released += sz
            self._bytes_used -= sz
            mo.ref_count_down()
            events.append(("remove", ArtifactId(victim_key), sz, self._name))
        return released

    def _evict_until(self, target_bytes: int, events: list) -> int:
        """
        Force eviction up to `target_bytes` of payload regardless of the
        capacity gauge. Used by `allocate_slot` to make room for an
        explicit alloc request.
        """
        released = 0
        while released < target_bytes:
            victim_key = self._next_evict_candidate()
            if victim_key is None:
                break
            mo = self._lru.pop(victim_key)
            sz = mo.get_size()
            released += sz
            self._bytes_used -= sz
            mo.ref_count_down()
            events.append(("remove", ArtifactId(victim_key), sz, self._name))
        return released

    def _next_evict_candidate(self) -> Optional[str]:
        # Walk LRU oldest-first, skip non-evictable.
        for key, mo in self._lru.items():
            if mo.can_evict():
                return key
            self._evict_skip += 1
        return None

    def _fan_events(
        self, events: list[tuple[str, ArtifactId, int, str]]
    ) -> None:
        sink = self._event_sink
        if sink is None or not events:
            return
        for op, aid, sz, layer in events:
            try:
                sink(op, aid, sz, layer)
            except Exception:  # noqa: BLE001
                logger.exception("BackendEventSink raised; ignoring.")


# Backward-compatible alias (v0.1)
CpuStore = CpuMemoryLayer
