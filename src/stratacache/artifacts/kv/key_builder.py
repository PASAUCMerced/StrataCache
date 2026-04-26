from __future__ import annotations

from stratacache.core.artifact import ArtifactId


def build_kv_chunk_id(
    *,
    engine_tag: str,
    model_tag: str,
    tp: int | None,
    rank: int | None,
    prefix_hash: str,
    chunk_end: int,
    layer_idx: int | None = None,
    bundle: str | None = None,
) -> ArtifactId:
    base = (
        f"{engine_tag}:{model_tag}:tp={_v(tp)}:rank={_v(rank)}:"
        f"ph={prefix_hash}:chunk_end={int(chunk_end)}"
    )
    if bundle is not None:
        return ArtifactId(f"{base}:bundle={bundle}")
    if layer_idx is not None:
        return ArtifactId(f"{base}:layer={int(layer_idx)}")
    return ArtifactId(base)


def _v(v: object) -> str:
    if v is None:
        return "na"
    return str(v)
