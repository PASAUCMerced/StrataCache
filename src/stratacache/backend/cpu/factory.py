"""
Convenience factories that produce zero-copy TensorMemoryObjs backed by a
CpuAllocator slab. Adapters (vllm connector, parameter client) call these
to skip the GPU-tensor -> bytes -> dict-of-bytes -> bytes -> tensor path
that Phase 1 still uses.

Falls back gracefully:
- If `allocator` is None or has no slab, returns a TensorMemoryObj backed
  by a fresh CPU tensor (still avoids the bytes round-trip; just not
  pinned and not slab-pooled).
- If the slab is OOM, same fallback. Logged at debug.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from stratacache.backend.cpu.cpu_allocator import CpuAllocator
from stratacache.core.artifact import ArtifactMeta
from stratacache.core.memory_obj import TensorMemoryObj


logger = logging.getLogger(__name__)


def cpu_memory_obj_from_tensor(
    tensor: Any,  # torch.Tensor
    artifact_meta: ArtifactMeta,
    *,
    allocator: Optional[CpuAllocator] = None,
) -> TensorMemoryObj:
    """
    Materialize a CPU-resident TensorMemoryObj from `tensor`.

    Behaviour:
    - Detach + ensure contiguous.
    - If `tensor` is on GPU, copy to host (single H2D->H copy via
      torch.empty_like(...).copy_(non_blocking=True) when slab is pinned;
      otherwise plain .cpu()).
    - Wrap into a TensorMemoryObj. When slab-backed, attach a release
      callback that returns the slot to the allocator on ref_count==0.
    """
    import torch  # type: ignore[import-not-found]

    src = tensor.detach()
    if not src.is_contiguous():
        src = src.contiguous()

    nbytes = int(src.numel() * src.element_size())

    slot = None
    if allocator is not None and allocator.has_slab:
        slot = allocator.try_allocate(nbytes)
        if slot is None:
            logger.debug(
                "CpuAllocator slab OOM (%d bytes requested); "
                "falling back to non-slab TensorMemoryObj.",
                nbytes,
            )

    if slot is not None:
        view_u8 = slot.tensor_view()  # uint8, length=nbytes, slab-backed
        # Flatten src to a 1-D uint8 view so shapes match the slab view.
        # `tensor.view(torch.uint8)` keeps the leading dims and only doubles
        # the last (per element-size), which produces a multi-D view that
        # is NOT broadcast-compatible with the 1-D slab; `.reshape(-1)`
        # collapses it. Same trick applies for both CPU and GPU sources.
        src_u8 = src.view(torch.uint8).reshape(-1)
        pending_event: Any = None
        if src.device.type == "cpu":
            view_u8.copy_(src_u8)
        else:
            # Issue D2H on the source's current stream and record an event.
            # Do NOT host-block here: that costs ~1-2ms per chunk on the
            # forward critical path. Consumers that need host-visible bytes
            # (e.g. CXL write-through, byte_array) wait on the event lazily;
            # by then it has usually already drained.
            view_u8.copy_(src_u8, non_blocking=allocator.pin_memory)
            if allocator.pin_memory:
                pending_event = torch.cuda.Event()
                pending_event.record(torch.cuda.current_stream(src.device))

        # Re-interpret the slab range as the original dtype/shape, no copy.
        typed = view_u8.view(src.dtype).reshape(src.shape)
        mo = TensorMemoryObj(
            typed,
            artifact_meta,
            size=nbytes,
            release_callback=slot.free,
        )
        if pending_event is not None:
            mo.attach_pending_event(pending_event)
        return mo

    # Non-slab fallback: own a fresh CPU tensor (one host alloc + one copy).
    if src.device.type == "cpu":
        host = src.clone()
    else:
        host = src.cpu()
    return TensorMemoryObj(host, artifact_meta, size=nbytes)


def cpu_memory_obj_from_bytes(
    payload: bytes,
    artifact_meta: ArtifactMeta,
    *,
    dtype: str,
    shape: tuple[int, ...],
    allocator: Optional[CpuAllocator] = None,
) -> TensorMemoryObj:
    """
    Construct a TensorMemoryObj from a bytes blob with known dtype/shape.

    Used on the load path when a record arrived as bytes (e.g. from CXL)
    but downstream consumers want a typed tensor view. Avoids the
    `frombuffer + view + clone` triad that the legacy decoder does when a
    slab is available - we copy straight into the slab.
    """
    import torch  # type: ignore[import-not-found]

    nbytes = len(payload)
    dt = getattr(torch, str(dtype), None)
    if dt is None:
        raise ValueError(f"unsupported tensor dtype: {dtype}")

    slot = None
    if allocator is not None and allocator.has_slab:
        slot = allocator.try_allocate(nbytes)

    if slot is not None:
        view_u8 = slot.tensor_view()
        # frombuffer over `payload` gives us a non-owning view; copy into slab.
        src_u8 = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        view_u8.copy_(src_u8)
        typed = view_u8.view(dt).reshape(tuple(int(x) for x in shape))
        return TensorMemoryObj(
            typed,
            artifact_meta,
            size=nbytes,
            release_callback=slot.free,
        )

    # Non-slab fallback.
    src_u8 = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    typed = src_u8.view(dt).reshape(tuple(int(x) for x in shape)).clone()
    return TensorMemoryObj(typed, artifact_meta, size=nbytes)
