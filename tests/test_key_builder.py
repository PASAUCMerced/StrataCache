from __future__ import annotations

from stratacache.artifacts.kv import build_kv_chunk_id
from stratacache.artifacts.params import build_param_chunk_id


def run() -> None:
    kv_id = build_kv_chunk_id(
        engine_tag="vllm013",
        model_tag="Qwen2.5-7B",
        tp=2,
        rank=1,
        prefix_hash="abcd",
        chunk_end=256,
        bundle="bundleT",
    )
    param_id = build_param_chunk_id(
        engine_tag="sglang",
        model_tag="Qwen2.5-7B",
        revision="rev42",
        layer_idx=3,
        unit="attn",
        dtype="float16",
        chunk_idx=0,
    )

    assert str(kv_id) != str(param_id)
    assert "bundle=bundleT" in str(kv_id)
    assert ":param:" in str(param_id)
