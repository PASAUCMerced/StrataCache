from __future__ import annotations


def run() -> None:
    from stratacache.artifacts.kv import (
        ChunkedTokenDatabase,
        boundary_prefix_hashes,
    )

    db = ChunkedTokenDatabase(
        chunk_size=4,
        engine_tag="vllm013",
        model_tag="m",
        tp=1,
        rank=0,
        save_partial_chunks=True,
    )

    toks = [101, 102, 103, 104, 105, 106]
    specs = list(db.process_tokens(toks))
    # 4-token full chunk + 2-token tail = 2 specs.
    assert len(specs) == 2
    assert specs[0].start == 0 and specs[0].end == 4
    assert specs[1].start == 4 and specs[1].end == 6

    # Same prefix gives identical chunk_end=4 hash on rerun.
    again = list(db.process_tokens(toks[:4]))
    assert again[0].key.prefix_hash == specs[0].key.prefix_hash

    # Same prefix should match the standalone helper.
    bh = boundary_prefix_hashes(toks, [4])
    assert bh[4] == specs[0].key.prefix_hash

    # to_artifact_id roundtrip carries the prefix hash and chunk_end.
    aid_str = str(specs[0].key.to_artifact_id())
    assert "ph=" in aid_str
    assert "chunk_end=4" in aid_str

    # Mask: skip emission for tokens already cached.
    cached_specs = list(db.process_tokens(toks, mask_prefix_tokens=4))
    # Only the tail chunk should be emitted.
    assert len(cached_specs) == 1
    assert cached_specs[0].start == 4 and cached_specs[0].end == 6
    # The tail's chained hash matches what we got without the mask.
    assert cached_specs[0].key.prefix_hash == specs[1].key.prefix_hash
