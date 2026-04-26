"""
KV-cache specific abstractions consumed by inference-engine adapters
(currently adapters/vllm/, future adapters/sglang/).
"""
from stratacache.artifacts.kv.key_builder import build_kv_chunk_id

__all__ = ["build_kv_chunk_id"]
