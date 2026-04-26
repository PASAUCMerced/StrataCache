"""
Placeholder for KV admit/evict events surfaced to inference engines.

Will subscribe to the type-agnostic backend event stream and translate to
vLLM-shaped CacheStoreEvent / CacheRemoveEvent. See
tmp/lmcache_gap_analysis.md (A11).
"""
