from __future__ import annotations

from stratacache.core.artifact import ArtifactId


def build_param_chunk_id(
    *,
    engine_tag: str,
    model_tag: str,
    revision: str | int,
    layer_idx: int,
    unit: str,
    dtype: str,
    chunk_idx: int,
) -> ArtifactId:
    return ArtifactId(
        f"{engine_tag}:{model_tag}:rev={revision}:"
        f"param:layer={int(layer_idx)}:unit={unit}:dtype={dtype}:chunk={int(chunk_idx)}"
    )
