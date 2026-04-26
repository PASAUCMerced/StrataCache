"""
CPU-side allocator.

Phase 1 (this commit): a no-op allocator that wraps the existing semantics
of "store whatever MemoryObj the caller hands us in an OrderedDict and rely
on Python's GC for free". Concrete allocate / free / pin / unpin behaviour
will be implemented when we port the LMCache slab + free-list allocator
(B1, B2, B3, B5, B6 in tmp/lmcache_gap_analysis.md).

Keeping this file as a stub today serves two purposes:
- Establish the `backend/cpu/` directory layout you asked for, mirroring
  what `backend/cxl/` already has.
- Give us a single place to land the slab logic later without touching
  `cpu_memory.py` again.
"""
from __future__ import annotations

from typing import Optional


class CpuAllocator:
    """
    Placeholder CPU allocator.

    Today, `CpuMemoryLayer` does not actually consult an allocator: it
    accepts a MemoryObj produced by the caller (e.g. wrapping bytes or a
    torch tensor) and stores the reference. This class exists so that
    `cpu_memory.py` can be wired against an injectable allocator dependency
    from day one; the real implementation will replace this file.
    """

    def __init__(self, *, capacity_bytes: Optional[int] = None) -> None:
        self._capacity_bytes = capacity_bytes

    @property
    def capacity_bytes(self) -> Optional[int]:
        return self._capacity_bytes
