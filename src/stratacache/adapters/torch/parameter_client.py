from __future__ import annotations

from typing import Any

from stratacache.artifacts.params.key_builder import build_param_chunk_id
from stratacache.backend.cpu.factory import (
    cpu_memory_obj_from_bytes,
    cpu_memory_obj_from_tensor,
)
from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
from stratacache.engine.storage_engine import StorageEngine
from stratacache.engine.types import AccessMode, ContainsResult

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]


class ParameterStoreClient:
    """
    Thin helper for parameter chunk offload/prefetch on top of StorageEngine.

    Phase 2: store path produces a TensorMemoryObj backed by the head CPU
    allocator's slab when one is available (zero-copy into pinned host
    memory). Falls back to a non-slab TensorMemoryObj otherwise. The
    legacy bytes round-trip is gone.
    """

    def __init__(
        self,
        engine: StorageEngine,
        *,
        engine_tag: str = "sglang",
        model_tag: str,
        revision: str,
    ) -> None:
        self._engine = engine
        self._engine_tag = str(engine_tag)
        self._model_tag = str(model_tag)
        self._revision = str(revision)

    def chunk_id(
        self,
        *,
        layer_idx: int,
        unit: str,
        dtype: str,
        chunk_idx: int,
    ) -> ArtifactId:
        return build_param_chunk_id(
            engine_tag=self._engine_tag,
            model_tag=self._model_tag,
            revision=self._revision,
            layer_idx=layer_idx,
            unit=unit,
            dtype=dtype,
            chunk_idx=chunk_idx,
        )

    def put_chunk(
        self,
        *,
        layer_idx: int,
        unit: str,
        chunk_idx: int,
        tensor: "torch.Tensor",
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.CHAIN,
        meta_extra: dict[str, Any] | None = None,
    ) -> ArtifactId:
        _require_torch()
        dtype_name = _dtype_to_name(tensor.dtype)
        aid = self.chunk_id(
            layer_idx=layer_idx,
            unit=unit,
            dtype=dtype_name,
            chunk_idx=chunk_idx,
        )

        attrs = {
            "tensor_codec": "stable_raw",
            "tensor_dtype": dtype_name,
            "tensor_shape": list(tensor.shape),
            "layer_idx": int(layer_idx),
            "unit": str(unit),
            "chunk_idx": int(chunk_idx),
            "revision": self._revision,
        }
        if meta_extra:
            attrs.update(dict(meta_extra))

        meta = ArtifactMeta(artifact_type=ArtifactType.PARAM_CHUNK, attrs=attrs)
        memory_obj = cpu_memory_obj_from_tensor(
            tensor,
            meta,
            allocator=self._engine.get_cpu_allocator(),
        )
        self._engine.store(aid, memory_obj, medium=medium, mode=mode)
        return aid

    def get_chunk(
        self,
        *,
        layer_idx: int,
        unit: str,
        dtype: str,
        chunk_idx: int,
        shape: tuple[int, ...] | list[int] | None = None,
        device: "torch.device | str | None" = None,
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.PREFER,
        promote: bool = True,
    ) -> "torch.Tensor":
        _require_torch()
        aid = self.chunk_id(
            layer_idx=layer_idx,
            unit=unit,
            dtype=dtype,
            chunk_idx=chunk_idx,
        )
        # Pass dtype/shape as hints so a CXL hit can return a typed tensor
        # directly (CXL doesn't persist user-meta).
        shape_tuple = tuple(int(x) for x in shape) if shape is not None else None
        lr = self._engine.load(
            aid,
            medium=medium,
            mode=mode,
            promote=promote,
            dtype=dtype,
            shape=shape_tuple,
        )
        mo = lr.memory_obj
        # Fast path: TensorMemoryObj already has a typed view.
        t = mo.tensor
        if t is None:
            attrs = dict(mo.metadata.artifact_meta.attrs)
            t = _decode_tensor_raw(
                mo.byte_array,
                dtype_name=str(attrs.get("tensor_dtype", dtype)),
                shape=attrs.get("tensor_shape", shape_tuple or []),
            )
        # Always clone: the underlying buffer may be a slab/CXL-backed
        # view and the caller's tensor must outlive it.
        out = t.clone()
        if device is not None:
            out = out.to(device)
        return out

    def has_chunk(
        self,
        *,
        layer_idx: int,
        unit: str,
        dtype: str,
        chunk_idx: int,
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.PREFER,
    ) -> ContainsResult:
        aid = self.chunk_id(
            layer_idx=layer_idx,
            unit=unit,
            dtype=dtype,
            chunk_idx=chunk_idx,
        )
        return self._engine.contains(aid, medium=medium, mode=mode)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for ParameterStoreClient")


def _dtype_to_name(dtype: "torch.dtype") -> str:
    s = str(dtype)
    if s.startswith("torch."):
        return s.split(".", 1)[1]
    return s


_DTYPE_FROM_NAME: dict[str, Any] = {
    "float16": getattr(torch, "float16", None) if torch is not None else None,
    "bfloat16": getattr(torch, "bfloat16", None) if torch is not None else None,
    "float32": getattr(torch, "float32", None) if torch is not None else None,
    "float64": getattr(torch, "float64", None) if torch is not None else None,
    "int8": getattr(torch, "int8", None) if torch is not None else None,
    "int16": getattr(torch, "int16", None) if torch is not None else None,
    "int32": getattr(torch, "int32", None) if torch is not None else None,
    "int64": getattr(torch, "int64", None) if torch is not None else None,
    "uint8": getattr(torch, "uint8", None) if torch is not None else None,
    "bool": getattr(torch, "bool", None) if torch is not None else None,
}


def _decode_tensor_raw(
    payload: bytes,
    *,
    dtype_name: str,
    shape: list[int] | tuple[int, ...],
) -> "torch.Tensor":
    dtype = _DTYPE_FROM_NAME.get(str(dtype_name))
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype: {dtype_name}")
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"invalid tensor shape metadata: {shape}")
    out = torch.frombuffer(memoryview(payload), dtype=dtype)
    return out.clone().reshape(list(shape))
