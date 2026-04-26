"""
Deprecated: CPU backend moved to `stratacache.backend.cpu`.

Re-exports kept for backward compatibility; new code should import from
`stratacache.backend.cpu` (which exposes `CpuMemoryLayer`, `CpuAllocator`).
"""
from stratacache.backend.cpu.cpu_memory import CpuMemoryLayer, CpuStore

__all__ = ["CpuMemoryLayer", "CpuStore"]
