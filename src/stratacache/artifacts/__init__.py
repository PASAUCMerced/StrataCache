"""
Type-specific high-level abstractions that sit *above* the type-agnostic
storage plane (core/, memory/, backend/, tiering/, engine/).

Each subpackage owns the semantics of one artifact family:
- artifacts.kv: KV cache reuse for inference engines.
- artifacts.params: model parameter chunk offload/prefetch.

These modules may import from core/, engine/ etc., but the reverse direction
is forbidden: storage-plane code must never know which artifact family a
payload belongs to.
"""
