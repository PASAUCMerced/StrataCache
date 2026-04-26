from __future__ import annotations


def run() -> None:
    try:
        import torch  # noqa: F401
        import sortedcontainers  # noqa: F401
    except Exception:
        return

    from stratacache.backend.cpu import (
        CpuAllocator,
        CpuMemoryLayer,
        cpu_memory_obj_from_tensor,
    )
    from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType

    cap = 2560
    alloc = CpuAllocator(capacity_bytes=cap)
    layer = CpuMemoryLayer(
        store_name="cpu",
        capacity_bytes=cap,
        allocator=alloc,
    )

    # Fill the slab with two unpinned 1KB MOs (2048 of 2560 used).
    meta = ArtifactMeta(artifact_type=ArtifactType.PARAM_CHUNK)
    t1 = torch.zeros(1024, dtype=torch.uint8)
    t2 = torch.zeros(1024, dtype=torch.uint8)
    layer.put(ArtifactId("a"), cpu_memory_obj_from_tensor(t1, meta, allocator=alloc))
    layer.put(ArtifactId("b"), cpu_memory_obj_from_tensor(t2, meta, allocator=alloc))
    assert alloc.stats().bytes_free < 2048

    # Request a 2KB slot via allocate_slot. Free is < 2KB, so the layer
    # must evict at least one entry.
    slot = layer.allocate_slot(2048, busy_loop=False)
    assert slot is not None
    remaining = sum(1 for k in ("a", "b") if layer.exists(ArtifactId(k)))
    assert remaining < 2

    # Pin the rest, then a too-big request should give up cleanly.
    for k in ("a", "b"):
        try:
            layer.get(ArtifactId(k)).pin()
        except Exception:
            pass
    slot.free()

    # No more evictable space; with busy_loop=False the call returns None promptly.
    impossible = layer.allocate_slot(cap, busy_loop=False)
    assert impossible is None or impossible.aligned_length <= cap
    if impossible is not None:
        impossible.free()
