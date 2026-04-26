from __future__ import annotations


def run() -> None:
    try:
        import torch
    except Exception:
        # Optional test: only runs when torch is installed.
        return

    from stratacache.backend.cpu import CpuMemoryLayer
    from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
    from stratacache.core.memory_obj import BytesMemoryObj
    from stratacache.tiering.chain import TierChain
    from stratacache.adapters.vllm.connector_v1 import (
        _decode_tensor_stable,
        _encode_tensor_stable,
        _gather_by_slots,
        _scatter_by_slots,
    )

    chain = TierChain(tiers=[CpuMemoryLayer(store_name="cpu")], links=[], enable_writeback_worker=False)
    try:
        kv = torch.randn(2, 2, 4, 1, 2)  # [2, B, S, H, D]
        slots = torch.tensor([0, 2, 7], dtype=torch.long)
        gathered = _gather_by_slots(kv, slots).cpu()
        payload = _encode_tensor_stable(gathered)

        aid = ArtifactId("test:kv:layer0")
        meta = ArtifactMeta(artifact_type=ArtifactType.KV_BLOCKS)
        chain.store(aid, BytesMemoryObj(payload, meta))
        fr = chain.fetch(aid, promote=False)
        out_payload = fr.memory_obj.byte_array
        out = _decode_tensor_stable(out_payload, device=torch.device("cpu"))

        kv2 = torch.zeros_like(kv)
        _scatter_by_slots(kv2, slots, out)
        flat2 = kv2.reshape(2, -1, *kv2.shape[3:])
        flat1 = kv.reshape(2, -1, *kv.shape[3:])
        assert torch.allclose(flat1[:, slots], flat2[:, slots])
    finally:
        chain.close()
