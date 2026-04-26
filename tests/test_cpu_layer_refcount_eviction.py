from __future__ import annotations


def run() -> None:
    """
    Verify that the LRU eviction loop respects ref_count / pin_count.

    Pure-Python test: uses BytesMemoryObj so the test runs even without
    torch / sortedcontainers.
    """
    from stratacache.backend.cpu import CpuMemoryLayer
    from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
    from stratacache.core.memory_obj import BytesMemoryObj

    # Capacity 30 bytes; each entry 10 bytes. After 3 inserts we are at
    # capacity. Insert a 4th -> oldest must be evicted.
    layer = CpuMemoryLayer(store_name="cpu", capacity_bytes=30)

    aids = [ArtifactId(f"k{i}") for i in range(4)]
    meta = ArtifactMeta(artifact_type=ArtifactType.CUSTOM)
    for aid in aids[:3]:
        layer.put(aid, BytesMemoryObj(b"x" * 10, meta))
    assert layer.stats().bytes_used == 30

    # Borrow k0 (simulating an in-flight reader) and pin k1.
    mo0 = layer.get(aids[0])
    mo0.ref_count_up()
    mo1 = layer.get(aids[1])
    mo1.pin()

    # Now insert k3; eviction loop must skip k0 (ref_count=2) and k1
    # (pinned), then evict k2 instead, even though k2 is more recently used.
    layer.put(aids[3], BytesMemoryObj(b"y" * 10, meta))

    assert layer.exists(aids[0])
    assert layer.exists(aids[1])
    assert not layer.exists(aids[2])  # evicted
    assert layer.exists(aids[3])

    # Releasing the borrow + pin should make them evictable on the next put.
    mo0.ref_count_down()
    mo1.unpin()

    layer.put(ArtifactId("k4"), BytesMemoryObj(b"z" * 10, meta))
    # Now one of {k0, k1, k3} got evicted, layer is back to 3 entries.
    assert layer.stats().items == 3
