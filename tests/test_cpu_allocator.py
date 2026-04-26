from __future__ import annotations


def run() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        # Allocator slab requires torch.
        return
    try:
        import sortedcontainers  # noqa: F401
    except Exception:
        # Free list needs sortedcontainers; skip when unavailable.
        return

    from stratacache.backend.cpu.cpu_allocator import CpuAllocator

    cap = 4096
    alloc = CpuAllocator(capacity_bytes=cap, pin_memory=False, align=256)
    assert alloc.has_slab
    assert alloc.capacity_bytes == cap
    assert alloc.stats().bytes_in_use == 0

    # Allocate three slots; verify offsets are non-overlapping.
    s1 = alloc.try_allocate(700)
    s2 = alloc.try_allocate(1000)
    s3 = alloc.try_allocate(2000)
    assert s1 is not None and s2 is not None and s3 is not None
    assert s1.offset == 0
    assert s2.offset == s1.aligned_length
    assert s3.offset == s1.aligned_length + s2.aligned_length

    # OOM after capacity exhausted.
    s4 = alloc.try_allocate(2000)
    assert s4 is None
    assert alloc.stats().num_alloc_failed == 1

    # Free middle slot, then allocate exactly the freed size: should reuse.
    middle_off = s2.offset
    s2.free()
    s5 = alloc.try_allocate(s2.aligned_length)
    assert s5 is not None
    assert s5.offset == middle_off

    # Coalesce: free s3 + s5; total freed should equal s3 + s5 sizes.
    in_use_before = alloc.stats().bytes_in_use
    s3.free()
    s5.free()
    in_use_after = alloc.stats().bytes_in_use
    assert in_use_before - in_use_after == s3.aligned_length + s5.aligned_length

    # After free, allocate the merged region in one shot.
    s_big = alloc.try_allocate(s3.aligned_length + s5.aligned_length)
    assert s_big is not None
