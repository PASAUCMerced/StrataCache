"""
Deprecated re-export shim.

Type-specific key builders moved to artifacts/<family>/key_builder.py to
keep `core/` artifact-type-agnostic. This module re-exports the originals
for backward compatibility; new code should import from `stratacache.artifacts`.
"""
from __future__ import annotations

from stratacache.artifacts.kv.key_builder import build_kv_chunk_id
from stratacache.artifacts.params.key_builder import build_param_chunk_id

__all__ = ["build_kv_chunk_id", "build_param_chunk_id"]
