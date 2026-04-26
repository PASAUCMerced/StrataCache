"""
Deprecated: split into `cxl_memory.py` (layer) and `cxl_allocator.py`
(native allocator wrapper). Re-exports kept for backward compatibility.
"""
from stratacache.backend.cxl.cxl_allocator import CxlAllocator, CxlConfig
from stratacache.backend.cxl.cxl_memory import CxlMemoryLayer, CxlStore

__all__ = ["CxlMemoryLayer", "CxlStore", "CxlAllocator", "CxlConfig"]
