from __future__ import annotations


def run() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        import sortedcontainers  # noqa: F401
    except Exception:
        return

    from stratacache.backend.cpu import (
        CpuAllocator,
        CpuMemoryLayer,
        cpu_memory_obj_from_tensor,
    )
    from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType

    # 64KB slab is plenty for the test tensors.
    alloc = CpuAllocator(capacity_bytes=64 * 1024, pin_memory=False)
    layer = CpuMemoryLayer(store_name="cpu", capacity_bytes=64 * 1024, allocator=alloc)

    src = torch.randn(128, dtype=torch.float16)
    meta = ArtifactMeta(artifact_type=ArtifactType.PARAM_CHUNK)
    mo = cpu_memory_obj_from_tensor(src, meta, allocator=alloc)

    # 1) Slab-backed: tensor view points into the slab.
    t = mo.tensor
    assert t is not None
    assert t.dtype == src.dtype
    assert tuple(t.shape) == tuple(src.shape)
    assert torch.allclose(t, src)
    # Backing storage should be the slab tensor (not a copy of `src`).
    assert t.untyped_storage().data_ptr() == alloc._slab.untyped_storage().data_ptr()  # type: ignore[union-attr]

    # 2) Round-trip through the layer; load returns the same MemoryObj.
    aid = ArtifactId("zc:1")
    bytes_in_use_before_put = alloc.stats().bytes_in_use
    assert bytes_in_use_before_put > 0
    layer.put(aid, mo)

    got = layer.get(aid)
    assert got is mo
    assert got.tensor is t

    # 3) Delete -> ref_count_down -> slot freed.
    layer.delete(aid)
    assert alloc.stats().bytes_in_use == 0

    # 4) byte_array still works (materializes from slab via uint8 view).
    mo2 = cpu_memory_obj_from_tensor(src, meta, allocator=alloc)
    bbuf = mo2.byte_array
    expected = src.detach().contiguous().view(torch.uint8).numpy().tobytes()
    assert bbuf == expected
    # Free the slot so the test leaves the allocator clean.
    mo2.ref_count_down()
    assert alloc.stats().bytes_in_use == 0

    # 5) Multi-dim source (KV-bundle shape) - regression check.
    # Previously copy_() failed because src.view(torch.uint8) is multi-D
    # while the slab view is 1-D and not broadcast-compatible.
    multi = torch.randn(3, 4, 5, dtype=torch.bfloat16)
    mo3 = cpu_memory_obj_from_tensor(multi, meta, allocator=alloc)
    assert mo3.tensor.shape == multi.shape
    assert torch.allclose(mo3.tensor, multi)
    mo3.ref_count_down()
    assert alloc.stats().bytes_in_use == 0
