from __future__ import annotations


def run() -> None:
    from stratacache.artifacts.kv.kv_events import (
        CacheRemoveEvent,
        CacheStoreEvent,
        KVEventTranslator,
    )
    from stratacache.backend.cpu import CpuMemoryLayer
    from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
    from stratacache.core.memory_obj import BytesMemoryObj
    from stratacache.engine import StorageEngine
    from stratacache.tiering.policy import LinkPolicy

    eng = StorageEngine.from_tiers(
        tiers=[CpuMemoryLayer(store_name="cpu", capacity_bytes=120)],
        links=[],
        enable_writeback_worker=False,
    )
    try:
        translator = KVEventTranslator()
        eng.set_event_sink(translator.on_backend_event)

        meta = ArtifactMeta(artifact_type=ArtifactType.KV_BLOCKS)
        # KV-shaped key (matches token_database / build_kv_chunk_id grammar):
        kv_key = ArtifactId("vllm013:m:tp=1:rank=0:ph=deadbeef:chunk_end=64:bundle=bundleT")
        eng.store(kv_key, BytesMemoryObj(b"x" * 50, meta))

        # Non-KV key should be ignored by the translator.
        param_key = ArtifactId("sglang:m:rev=r1:param:layer=0:unit=attn:dtype=float16:chunk=0")
        eng.store(param_key, BytesMemoryObj(b"y" * 50, meta))

        # Trigger an eviction by inserting another KV chunk that won't fit.
        kv_key2 = ArtifactId("vllm013:m:tp=1:rank=0:ph=cafebabe:chunk_end=128:bundle=bundleT")
        eng.store(kv_key2, BytesMemoryObj(b"z" * 50, meta))

        events = translator.drain_events()
        store_events = [e for e in events if isinstance(e, CacheStoreEvent)]
        remove_events = [e for e in events if isinstance(e, CacheRemoveEvent)]
        # Should have at least the two KV stores. Param chunk is filtered out.
        assert len(store_events) >= 2
        # An eviction must have happened (kv_key got dropped).
        assert any(e.artifact_id == str(kv_key) for e in remove_events)
        # Translator parsed chunk_end + ph correctly.
        assert any(e.chunk_end == 64 for e in store_events)
        assert any(e.prefix_hash == "deadbeef" for e in store_events)
    finally:
        eng.close()
