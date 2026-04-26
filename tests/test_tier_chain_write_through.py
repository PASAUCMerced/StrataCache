from __future__ import annotations

from stratacache.backend.cpu import CpuMemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
from stratacache.core.memory_obj import BytesMemoryObj
from stratacache.tiering.chain import TierChain
from stratacache.tiering.policy import LinkPolicy


def run() -> None:
    l0 = CpuMemoryLayer(store_name="l0")
    l1 = CpuMemoryLayer(store_name="l1")
    chain = TierChain(tiers=[l0, l1], links=[LinkPolicy.WRITE_THROUGH], enable_writeback_worker=False)
    try:
        aid = ArtifactId("t:wt")
        meta = ArtifactMeta(artifact_type=ArtifactType.CUSTOM, attrs={"x": 1})
        chain.store(aid, BytesMemoryObj(b"abc", meta))

        assert l0.get(aid).byte_array == b"abc"
        assert l1.get(aid).byte_array == b"abc"
    finally:
        chain.close()
