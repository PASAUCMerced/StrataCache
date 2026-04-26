from __future__ import annotations


def run() -> None:
    try:
        import torch
    except Exception:
        return

    from stratacache.adapters.torch.parameter_client import ParameterStoreClient
    from stratacache.backend.cpu import CpuMemoryLayer
    from stratacache.engine import AccessMode, StorageEngine

    eng = StorageEngine.from_tiers(
        tiers=[CpuMemoryLayer(store_name="cpu")],
        links=[],
        enable_writeback_worker=False,
    )
    try:
        client = ParameterStoreClient(
            eng,
            engine_tag="sglang",
            model_tag="wan-video",
            revision="r1",
        )

        t = torch.randn(128, dtype=torch.float16)
        aid = client.put_chunk(
            layer_idx=1,
            unit="attn1",
            chunk_idx=0,
            tensor=t,
            medium="cpu",
            mode=AccessMode.EXACT,
        )

        c = client.has_chunk(
            layer_idx=1,
            unit="attn1",
            dtype="float16",
            chunk_idx=0,
            medium="cpu",
            mode=AccessMode.EXACT,
        )
        assert c.exists
        assert c.hit_medium == "cpu"

        out = client.get_chunk(
            layer_idx=1,
            unit="attn1",
            dtype="float16",
            chunk_idx=0,
            mode=AccessMode.EXACT,
            medium="cpu",
            promote=False,
        )
        assert out.dtype == t.dtype
        assert tuple(out.shape) == tuple(t.shape)
        assert torch.allclose(out.cpu(), t.cpu())
        assert "rev=r1" in str(aid)
    finally:
        eng.close()
