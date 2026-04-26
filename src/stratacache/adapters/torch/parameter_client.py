from __future__ import annotations

from typing import Any

from stratacache.artifacts.params.key_builder import build_param_chunk_id
from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
from stratacache.core.memory_obj import BytesMemoryObj
from stratacache.engine.storage_engine import StorageEngine
from stratacache.engine.types import AccessMode, ContainsResult

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]


_DTYPE_FROM_NAME: dict[str, Any] = {
    "float16": getattr(torch, "float16", None),
    "bfloat16": getattr(torch, "bfloat16", None),
    "float32": getattr(torch, "float32", None),
    "float64": getattr(torch, "float64", None),
    "int8": getattr(torch, "int8", None),
    "int16": getattr(torch, "int16", None),
    "int32": getattr(torch, "int32", None),
    "int64": getattr(torch, "int64", None),
    "uint8": getattr(torch, "uint8", None),
    "bool": getattr(torch, "bool", None),
}


class ParameterStoreClient:
    """
    Thin helper for parameter chunk offload/prefetch on top of StorageEngine.

    NOTE: this phase keeps the v0.1 bytes-roundtrip path. Zero-copy via
    TensorMemoryObj is part of the later allocator port (B1/B2/B3).
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

        payload = _encode_tensor_raw(tensor)
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
        self._engine.store(
            aid,
            BytesMemoryObj(payload, meta),
            medium=medium,
            mode=mode,
        )
        return aid

    def get_chunk(
        self,
        *,
        layer_idx: int,
        unit: str,
        dtype: str,
        chunk_idx: int,
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
        lr = self._engine.load(aid, medium=medium, mode=mode, promote=promote)
        attrs = dict(lr.memory_obj.metadata.artifact_meta.attrs)
        return _decode_tensor_raw(
            lr.memory_obj.byte_array,
            dtype_name=str(attrs.get("tensor_dtype", dtype)),
            shape=attrs.get("tensor_shape", []),
            device=device,
        )

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


def _encode_tensor_raw(t: "torch.Tensor") -> bytes:
    cpu = t.detach().to(device="cpu", copy=True).contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def _decode_tensor_raw(
    payload: bytes,
    *,
    dtype_name: str,
    shape: list[int] | tuple[int, ...],
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    dtype = _DTYPE_FROM_NAME.get(str(dtype_name))
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype: {dtype_name}")
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"invalid tensor shape metadata: {shape}")
    out = torch.frombuffer(memoryview(payload), dtype=dtype)
    out = out.clone().reshape(list(shape))
    if device is not None:
        out = out.to(device)
    return out
