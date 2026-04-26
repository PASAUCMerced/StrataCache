from __future__ import annotations

from stratacache.backend.cpu import CpuMemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
from stratacache.core.memory_obj import BytesMemoryObj
from stratacache.engine import AccessMode, StorageEngine
from stratacache.tiering.chain import TierChain
from stratacache.tiering.policy import LinkPolicy


def run() -> None:
    l0 = CpuMemoryLayer(store_name="cpu")
    l1 = CpuMemoryLayer(store_name="cxl")
    chain = TierChain(tiers=[l0, l1], links=[LinkPolicy.WRITE_THROUGH], enable_writeback_worker=False)
    eng = StorageEngine(chain)
    try:
        aid = ArtifactId("eng:chain")
        meta = ArtifactMeta(artifact_type=ArtifactType.CUSTOM)

        # chain mode: writes head and follows link semantics.
        eng.store(aid, BytesMemoryObj(b"abc", meta))
        assert l0.get(aid).byte_array == b"abc"
        assert l1.get(aid).byte_array == b"abc"

        c0 = eng.contains(aid)
        assert c0.exists
        assert c0.hit_tier == 0
        assert c0.hit_medium == "cpu"

        # exact mode on target medium only.
        aid2 = ArtifactId("eng:exact")
        eng.store(aid2, BytesMemoryObj(b"z", meta), medium="cxl", mode=AccessMode.EXACT)
        assert l1.get(aid2).byte_array == b"z"
        try:
            l0.get(aid2)
            assert False, "expected miss in cpu for exact write to cxl"
        except Exception:
            pass

        lr = eng.load(aid2, medium="cxl", mode=AccessMode.EXACT, promote=False)
        assert lr.memory_obj.byte_array == b"z"
        assert lr.hit_medium == "cxl"

        # prefer mode should fallback to chain scan when preferred medium misses.
        c1 = eng.contains(aid, medium="cxl", mode=AccessMode.PREFER)
        assert c1.exists

        # delete single medium should not remove from all tiers.
        eng.delete(aid, medium="cpu")
        try:
            l0.get(aid)
            assert False, "expected cpu miss after medium delete"
        except Exception:
            pass
        assert l1.get(aid).byte_array == b"abc"
    finally:
        eng.close()
