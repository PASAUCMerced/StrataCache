"""
KV-cache specific abstractions consumed by inference-engine adapters.
"""
from stratacache.artifacts.kv.key_builder import build_kv_chunk_id
from stratacache.artifacts.kv.token_database import (
    CacheEngineKey,
    ChunkSpec,
    ChunkedTokenDatabase,
    boundary_prefix_hashes,
)

__all__ = [
    "build_kv_chunk_id",
    "CacheEngineKey",
    "ChunkSpec",
    "ChunkedTokenDatabase",
    "boundary_prefix_hashes",
]
