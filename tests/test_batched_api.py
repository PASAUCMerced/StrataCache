from __future__ import annotations


def run() -> None:
    from stratacache.backend.cpu import CpuMemoryLayer
    from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
    from stratacache.core.memory_obj import BytesMemoryObj

    layer = CpuMemoryLayer(store_name="cpu", capacity_bytes=10_000)
    meta = ArtifactMeta(artifact_type=ArtifactType.CUSTOM)
    items = [(ArtifactId(f"k{i}"), BytesMemoryObj(f"v{i}".encode(), meta)) for i in range(5)]

    layer.batched_put(items)
    aids = [aid for aid, _ in items]

    assert layer.batched_exists(aids) == [True] * 5
    got = layer.batched_get(aids)
    for i, mo in enumerate(got):
        assert mo is not None
        assert mo.byte_array == f"v{i}".encode()

    layer.batched_delete(aids[:2])
    assert layer.batched_exists(aids) == [False, False, True, True, True]

    # Missing keys come back as None.
    miss = layer.batched_get([ArtifactId("nope")])
    assert miss == [None]
