"""
LazyCpuAllocator: starts with a small pinned slab and grows in a
background thread up to `capacity_bytes`.

Design (mirrors LMCache `LazyMixedMemoryAllocator` /
`AsyncMemoryExpander`, Apache-2.0):

- Maintain a list of `CpuAllocator` segments. Each segment owns one
  contiguous host buffer with its own free list.
- `try_allocate(nbytes)` walks segments in insertion order; first fit by
  segment.
- A background thread expands by appending one new segment at a time,
  each of `growth_step_bytes`, until total capacity reaches the target.
- Expansion sleeps `growth_pause_s` between segments to avoid blocking
  the engine boot path.

Limitations (intentional, for first cut):
- Coalescing happens within a segment, not across segments. A composite
  free list spanning segments would allow cross-segment merge; not done yet.
- Eviction loop in CpuMemoryLayer doesn't know about segments; it just
  asks for a slot and gets None when all current segments are full.

Falls back to a single eager segment when torch/sortedcontainers/CUDA
are unavailable.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from stratacache.backend.cpu.cpu_allocator import (
    CpuAllocator,
    CpuAllocatorStats,
    MemorySlot,
)


logger = logging.getLogger(__name__)


class LazyCpuAllocator:
    """
    Composite allocator made of independently-allocated `CpuAllocator`
    segments, expanded on a background thread.

    Public surface mirrors `CpuAllocator` so it can be passed wherever a
    `CpuAllocator` is expected.
    """

    def __init__(
        self,
        *,
        capacity_bytes: int,
        initial_bytes: Optional[int] = None,
        growth_step_bytes: Optional[int] = None,
        growth_pause_s: float = 0.05,
        pin_memory: bool = False,
        align: int = 256,
        autostart: bool = True,
    ) -> None:
        capacity_bytes = int(capacity_bytes)
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        initial = int(initial_bytes if initial_bytes is not None else min(
            capacity_bytes, max(64 * 1024 * 1024, capacity_bytes // 16)
        ))
        step = int(growth_step_bytes if growth_step_bytes is not None else max(
            64 * 1024 * 1024, capacity_bytes // 16
        ))
        self._target = capacity_bytes
        self._step = max(1, step)
        self._growth_pause_s = float(growth_pause_s)
        self._pin_memory = bool(pin_memory)
        self._align = int(align)

        self._segments: list[CpuAllocator] = []
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Initial eager segment.
        self._segments.append(
            CpuAllocator(
                capacity_bytes=min(initial, capacity_bytes),
                pin_memory=self._pin_memory,
                align=self._align,
            )
        )

        if autostart and self._current_capacity() < self._target:
            self.start()

    # ---- public API mirroring CpuAllocator ------------------------------

    @property
    def capacity_bytes(self) -> int:
        return self._target

    @property
    def has_slab(self) -> bool:
        return any(s.has_slab for s in self._segments)

    @property
    def pin_memory(self) -> bool:
        return self._pin_memory

    def try_allocate(self, nbytes: int) -> Optional[MemorySlot]:
        with self._lock:
            segments = list(self._segments)
        for seg in segments:
            slot = seg.try_allocate(nbytes)
            if slot is not None:
                return slot
        return None

    def free(self, slot: MemorySlot) -> None:
        slot.free()

    def stats(self) -> CpuAllocatorStats:
        with self._lock:
            segments = list(self._segments)
        cap = sum(s.capacity_bytes for s in segments)
        in_use = 0
        free = 0
        failed = 0
        for s in segments:
            st = s.stats()
            in_use += st.bytes_in_use
            free += st.bytes_free
            failed += st.num_alloc_failed
        return CpuAllocatorStats(
            capacity_bytes=cap,
            bytes_in_use=in_use,
            bytes_free=free,
            num_alloc_failed=failed,
            pin_memory=self._pin_memory,
        )

    # ---- expander control -----------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._expand_loop, name="cpu-lazy-expander", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def wait_until_full(self, timeout_s: Optional[float] = None) -> bool:
        """Test/diagnostic helper: block until target capacity is reached."""
        deadline = (
            None if timeout_s is None else time.monotonic() + timeout_s
        )
        while self._current_capacity() < self._target:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(self._growth_pause_s)
        return True

    # ---- internal --------------------------------------------------------

    def _current_capacity(self) -> int:
        with self._lock:
            return sum(s.capacity_bytes for s in self._segments)

    def _expand_loop(self) -> None:
        while not self._stop_evt.is_set():
            cur = self._current_capacity()
            if cur >= self._target:
                return
            grow = min(self._step, self._target - cur)
            try:
                seg = CpuAllocator(
                    capacity_bytes=grow,
                    pin_memory=self._pin_memory,
                    align=self._align,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "LazyCpuAllocator: failed to grow by %d bytes; aborting "
                    "background expansion.",
                    grow,
                )
                return
            with self._lock:
                self._segments.append(seg)
            if self._stop_evt.wait(self._growth_pause_s):
                return
