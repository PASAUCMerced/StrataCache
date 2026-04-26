"""
CPU host-memory allocator.

Phase 2: a real allocator that owns a pre-allocated host slab (optionally
page-locked / pinned for fast H2D), and hands out byte-range slots tracked
by a coalescing free list.

Acknowledgement: shape and free-list discipline follow LMCache's
`TensorMemoryAllocator` and `MixedMemoryAllocator`
(lmcache/v1/memory_management.py, Apache-2.0).

What is implemented (this commit):
- Single contiguous slab as `torch.Tensor` of dtype=uint8 on the CPU.
  pin_memory=True when `pin_memory=True` is passed AND torch.cuda is
  available; falls back to pageable otherwise.
- Address-keyed sorted free list with neighbour coalescing on free.
- 4KB-aligned allocation by default.
- Returns `MemorySlot` handles describing (offset, length, owner) so
  callers (typically `CpuMemoryLayer` + `TensorMemoryObj`) can construct
  views without copies.
- Stats for telemetry (`bytes_capacity`, `bytes_allocated`,
  `evict_failed_count`).

What is NOT yet implemented:
- NUMA placement (B6).
- Lazy / async expansion (B5).
- Paged-fixed-size variant (`PagedTensorMemoryAllocator`).
- Concurrent-allocate backpressure with `busy_loop` (B8) - this allocator
  is a single-shot `try_allocate`; the eviction loop lives in
  `CpuMemoryLayer`.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

try:
    # SortedList gives us O(log n) free-list ops.
    # Avoid the hard dependency for tests that only need bytes-mode.
    from sortedcontainers import SortedList  # type: ignore[import-not-found]
    _HAS_SORTEDCONTAINERS = True
except Exception:  # noqa: BLE001
    SortedList = None  # type: ignore[assignment]
    _HAS_SORTEDCONTAINERS = False


logger = logging.getLogger(__name__)


_DEFAULT_ALIGN = 256  # bytes; slab slot quantum


def _align_up(n: int, align: int) -> int:
    if align <= 1:
        return n
    return (n + (align - 1)) // align * align


def clamp_capacity_to_system(
    requested_bytes: int,
    *,
    reserve_bytes: int = 1 * 1024 * 1024 * 1024,
) -> int:
    """
    Return min(requested_bytes, system_available - reserve_bytes). When
    system memory cannot be detected, return `requested_bytes` unchanged
    (B9). Logs a warning when clamping kicks in.
    """
    try:
        from stratacache.system_detection import SystemMemoryDetector

        avail = SystemMemoryDetector.get_available_memory_bytes()
    except Exception:  # noqa: BLE001
        avail = None
    if avail is None:
        return int(requested_bytes)
    headroom = max(0, int(avail) - int(reserve_bytes))
    if requested_bytes > headroom:
        logger.warning(
            "CpuAllocator: requested capacity %.2f GiB exceeds available "
            "%.2f GiB minus reserve %.2f GiB; clamping to %.2f GiB.",
            requested_bytes / (1 << 30),
            avail / (1 << 30),
            reserve_bytes / (1 << 30),
            headroom / (1 << 30),
        )
        return headroom
    return int(requested_bytes)


@dataclass
class MemorySlot:
    """
    A handle to a contiguous byte range inside a CpuAllocator's slab.

    `tensor_view()` returns a uint8 view (no copy) of that range so callers
    can build typed `torch.frombuffer`-style views on top.
    """

    offset: int
    length: int
    aligned_length: int
    allocator: "CpuAllocator"
    # Set to False once free() has run, so the slot can't be released twice.
    live: bool = field(default=True, repr=False)

    def tensor_view(self) -> Any:
        if torch is None:
            raise RuntimeError("torch is required to use tensor_view()")
        return self.allocator._slab_view(self.offset, self.length)

    def free(self) -> None:
        if not self.live:
            return
        self.allocator._free_slot(self)
        self.live = False


@dataclass(frozen=True, slots=True)
class CpuAllocatorStats:
    capacity_bytes: int
    bytes_in_use: int
    bytes_free: int
    num_alloc_failed: int
    pin_memory: bool


class _FreeListBase:
    """Tiny free-list interface so we can swap implementations."""

    def add(self, offset: int, length: int) -> None: ...
    def take(self, length: int) -> Optional[int]:
        """Return the offset of a free range >= length, or None. Splits."""
        ...
    def __iter__(self): ...
    def total_free(self) -> int: ...


class _SortedFreeList(_FreeListBase):
    """Address-keyed list of (offset, length) tuples with neighbour merging."""

    def __init__(self, capacity: int) -> None:
        if not _HAS_SORTEDCONTAINERS:
            raise RuntimeError(
                "sortedcontainers is required for the slab free list; "
                "install with `pip install sortedcontainers`."
            )
        # Sort by offset.
        self._ranges = SortedList(key=lambda t: t[0])
        self._ranges.add((0, capacity))
        self._free_bytes = capacity

    def total_free(self) -> int:
        return self._free_bytes

    def add(self, offset: int, length: int) -> None:
        # Insert and coalesce with neighbours.
        idx = self._ranges.bisect_key_left(offset)

        # Try merge-with-prev.
        merged_offset = offset
        merged_length = length
        if idx > 0:
            prev = self._ranges[idx - 1]
            if prev[0] + prev[1] == offset:
                merged_offset = prev[0]
                merged_length += prev[1]
                self._ranges.pop(idx - 1)
                idx -= 1

        # Try merge-with-next.
        if idx < len(self._ranges):
            nxt = self._ranges[idx]
            if merged_offset + merged_length == nxt[0]:
                merged_length += nxt[1]
                self._ranges.pop(idx)

        self._ranges.add((merged_offset, merged_length))
        self._free_bytes += length

    def take(self, length: int) -> Optional[int]:
        # First-fit by address: cheap and good enough when most allocations
        # are roughly the same size (KV chunks).
        for i, (off, ln) in enumerate(self._ranges):
            if ln >= length:
                self._ranges.pop(i)
                if ln > length:
                    self._ranges.add((off + length, ln - length))
                self._free_bytes -= length
                return off
        return None

    def __iter__(self):
        return iter(self._ranges)


class CpuAllocator:
    """
    Owns a pre-allocated host slab and hands out byte slots from a free list.

    Phase 2 contract:
    - `try_allocate(nbytes)` returns a `MemorySlot` or None on OOM.
    - `slot.free()` (or `allocator.free(slot)`) releases the slot;
      coalesces with neighbours.
    - Telemetry exposed via `stats()`.
    """

    def __init__(
        self,
        *,
        capacity_bytes: Optional[int] = None,
        pin_memory: bool = False,
        align: int = _DEFAULT_ALIGN,
        numa_node: Optional[int] = None,
    ) -> None:
        if capacity_bytes is None or capacity_bytes <= 0:
            # No-slab mode: behaves like a sink. CpuMemoryLayer will fall
            # back to dict-of-references mode.
            self._capacity = 0
            self._slab = None
            self._free_list: Optional[_FreeListBase] = None
            self._pin_memory_effective = False
            self._numa_node = None
        else:
            if torch is None:
                raise RuntimeError(
                    "torch is required to allocate a CPU slab; install torch "
                    "or pass capacity_bytes=None to disable the slab."
                )
            cap = _align_up(int(capacity_bytes), align)
            pin_requested = bool(pin_memory) and bool(torch.cuda.is_available())
            if pin_memory and not pin_requested:
                logger.warning(
                    "CpuAllocator: pin_memory=True requested but CUDA is "
                    "unavailable; falling back to pageable host memory."
                )

            slab, pin_effective = self._try_alloc_slab(cap, pin_requested)
            self._slab = slab
            self._capacity = cap if slab is not None else 0
            self._free_list = _SortedFreeList(cap) if slab is not None else None
            self._pin_memory_effective = pin_effective
            self._numa_node = numa_node
            if slab is not None and numa_node is not None:
                self._maybe_bind_numa(numa_node)

        self._align = int(align)
        self._lock = threading.RLock()
        self._num_alloc_failed = 0
        self._bytes_in_use = 0

    @staticmethod
    def _try_alloc_slab(
        cap: int, pin_requested: bool
    ) -> tuple[Optional[Any], bool]:
        """
        Try to allocate the slab, with two graceful fallbacks:
        - pinned host alloc OOM -> retry as pageable
        - pageable alloc OOM    -> return (None, False) so the layer
          degrades to dict-of-references mode instead of killing the
          host process.
        """
        # Pass 1: honour pin_requested.
        try:
            slab = torch.empty(cap, dtype=torch.uint8, pin_memory=pin_requested)
            return slab, pin_requested
        except (RuntimeError, MemoryError) as e:
            if pin_requested:
                logger.warning(
                    "CpuAllocator: failed to pin %.2f GiB of host memory (%s); "
                    "retrying without pin_memory. Consider lowering "
                    "`cpu_capacity_gb` or enabling `use_lazy_allocator`.",
                    cap / (1 << 30),
                    e,
                )
            else:
                logger.warning(
                    "CpuAllocator: failed to allocate %.2f GiB of host memory "
                    "(%s).",
                    cap / (1 << 30),
                    e,
                )

        # Pass 2: pageable retry.
        try:
            slab = torch.empty(cap, dtype=torch.uint8, pin_memory=False)
            return slab, False
        except (RuntimeError, MemoryError) as e:
            logger.error(
                "CpuAllocator: pageable host alloc of %.2f GiB also failed (%s); "
                "running without a slab. Layer will degrade to "
                "dict-of-references mode.",
                cap / (1 << 30),
                e,
            )
            return None, False

    def _maybe_bind_numa(self, node: int) -> None:
        """Best-effort NUMA bind for the slab. No-op when libnuma is missing."""
        if self._slab is None:
            return
        try:
            from stratacache.system_detection import NUMADetector

            addr = int(self._slab.data_ptr())
            ok = NUMADetector.bind_buffer_to_node(
                addr, int(self._capacity), int(node)
            )
            if not ok:
                logger.info(
                    "CpuAllocator: NUMA bind to node=%d skipped "
                    "(libnuma unavailable or returned error).",
                    node,
                )
        except Exception:  # noqa: BLE001
            logger.exception("CpuAllocator: NUMA bind raised; ignoring.")

    # ---- introspection ----

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def has_slab(self) -> bool:
        return self._slab is not None

    @property
    def pin_memory(self) -> bool:
        return self._pin_memory_effective

    def stats(self) -> CpuAllocatorStats:
        with self._lock:
            free = (
                self._free_list.total_free() if self._free_list is not None else 0
            )
            return CpuAllocatorStats(
                capacity_bytes=self._capacity,
                bytes_in_use=self._bytes_in_use,
                bytes_free=free,
                num_alloc_failed=self._num_alloc_failed,
                pin_memory=self._pin_memory_effective,
            )

    # ---- core API ----

    def try_allocate(self, nbytes: int) -> Optional[MemorySlot]:
        """
        Allocate a slot of `nbytes` (rounded up to alignment), or None on OOM.
        """
        if self._free_list is None or nbytes <= 0:
            return None
        aligned = _align_up(int(nbytes), self._align)
        with self._lock:
            offset = self._free_list.take(aligned)
            if offset is None:
                self._num_alloc_failed += 1
                return None
            self._bytes_in_use += aligned
            return MemorySlot(
                offset=offset,
                length=int(nbytes),
                aligned_length=aligned,
                allocator=self,
            )

    def free(self, slot: MemorySlot) -> None:
        slot.free()

    # ---- internal helpers used by MemorySlot ----

    def _free_slot(self, slot: MemorySlot) -> None:
        if self._free_list is None:
            return
        with self._lock:
            self._free_list.add(slot.offset, slot.aligned_length)
            self._bytes_in_use -= slot.aligned_length

    def _slab_view(self, offset: int, length: int) -> Any:
        """Return a uint8 tensor view (no copy) of [offset, offset+length)."""
        if self._slab is None:
            raise RuntimeError("CpuAllocator has no slab; cannot return view")
        return self._slab.narrow(0, offset, length)
