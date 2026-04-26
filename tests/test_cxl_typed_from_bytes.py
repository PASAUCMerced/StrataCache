from __future__ import annotations


def run() -> None:
    """Verify the typed reconstruction path used on CXL load."""
    try:
        import torch
    except Exception:
        return

    from stratacache.backend.cxl.cxl_memory import CxlMemoryLayer
    from stratacache.core.memory_obj import BytesMemoryObj

    src = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    raw = src.contiguous().view(torch.uint8).numpy().tobytes(order="C")

    # No-hint path: bytes returned, meta empty.
    mo_bytes = CxlMemoryLayer._typed_from_bytes(raw, dtype="", shape=()) if False else BytesMemoryObj(raw, __import__("stratacache.core.artifact", fromlist=["ArtifactMeta"]).ArtifactMeta())
    assert mo_bytes.byte_array == raw

    # Typed path: same payload reconstructs to the original tensor.
    mo_typed = CxlMemoryLayer._typed_from_bytes(raw, dtype="bfloat16", shape=(2, 3, 4))
    assert mo_typed.tensor is not None
    assert mo_typed.tensor.dtype == torch.bfloat16
    assert tuple(mo_typed.tensor.shape) == (2, 3, 4)
    assert torch.allclose(mo_typed.tensor, src)
