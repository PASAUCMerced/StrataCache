"""
ChunkedTokenDatabase: prefix-hash + chunk-key construction for KV cache.

Adapted from LMCache `lmcache/v1/token_database.py` (Apache-2.0). Uses
incremental hashing so the cost of computing N chunk hashes is O(N*chunk)
once, not O(N*N*chunk) as our previous per-call SHA-256 of the whole
prefix did.

`CacheEngineKey` is the canonical chunk identifier; `to_artifact_id()`
serialises it into the StrataCache `ArtifactId` namespace so the rest of
the storage plane keeps treating chunks as opaque keys.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

from stratacache.core.artifact import ArtifactId


@dataclass(frozen=True, slots=True)
class CacheEngineKey:
    """
    Canonical chunk key.

    Fields are deliberately small and stable so two processes computing
    the same prefix get the same key string.

    `prefix_hash` is a hex-encoded SHA-256 digest covering the chunk
    content + the previous chunk's `prefix_hash` (chained, vLLM-block-hash
    compatible style).
    """

    engine_tag: str
    model_tag: str
    tp: Optional[int]
    rank: Optional[int]
    chunk_end: int
    prefix_hash: str
    layer_idx: Optional[int] = None
    bundle: Optional[str] = None

    def to_artifact_id(self) -> ArtifactId:
        base = (
            f"{self.engine_tag}:{self.model_tag}"
            f":tp={_v(self.tp)}:rank={_v(self.rank)}"
            f":ph={self.prefix_hash}:chunk_end={int(self.chunk_end)}"
        )
        if self.bundle is not None:
            return ArtifactId(f"{base}:bundle={self.bundle}")
        if self.layer_idx is not None:
            return ArtifactId(f"{base}:layer={int(self.layer_idx)}")
        return ArtifactId(base)


def _v(v: object) -> str:
    return "na" if v is None else str(v)


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    """One chunk produced by `ChunkedTokenDatabase.process_tokens`."""

    start: int
    end: int
    key: CacheEngineKey


class ChunkedTokenDatabase:
    """
    Stateful database that yields one ChunkSpec per chunk boundary.

    Usage pattern:

        db = ChunkedTokenDatabase(
            chunk_size=256,
            engine_tag="vllm013",
            model_tag="Qwen2.5-7B",
            tp=2,
            rank=0,
        )
        for spec in db.process_tokens(token_ids, mask_prefix_tokens=cached):
            ...

    `mask_prefix_tokens` lets the caller skip rehashing the prefix already
    known to be cached: hashing resumes from `mask_prefix_tokens` using
    the seed digest computed for that prefix length.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        engine_tag: str,
        model_tag: str,
        tp: Optional[int] = None,
        rank: Optional[int] = None,
        save_partial_chunks: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = int(chunk_size)
        self._engine_tag = str(engine_tag)
        self._model_tag = str(model_tag)
        self._tp = tp
        self._rank = rank
        self._save_partial_chunks = bool(save_partial_chunks)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def process_tokens(
        self,
        token_ids: Sequence[int],
        *,
        mask_prefix_tokens: int = 0,
        layer_idx: Optional[int] = None,
        bundle: Optional[str] = None,
    ) -> Iterator[ChunkSpec]:
        """
        Yield ChunkSpec per chunk boundary.

        - Full chunks at multiples of chunk_size.
        - When save_partial_chunks=True, the trailing partial chunk
          (length < chunk_size) is also yielded.
        - `mask_prefix_tokens` skips re-emitting chunks fully contained in
          the masked prefix; the running hash is still advanced so later
          chunks have the same key as if the prefix had been re-hashed.
        """
        n = len(token_ids)
        cs = self._chunk_size
        if n == 0:
            return

        # We can't shortcut the running hash without rehashing the masked
        # prefix (the digest is content-derived); the mask only
        # suppresses *emission*, not work.
        h = hashlib.sha256()
        # Advance to the nearest chunk boundary that doesn't exceed mask.
        skip_emit_until = (mask_prefix_tokens // cs) * cs

        i = 0
        chunk_start = 0
        while i < n:
            j = min(i + cs, n)
            for t in token_ids[i:j]:
                h.update(int(t).to_bytes(4, "little", signed=False))
            chunk_end = j
            chunk_start = i
            is_full = (chunk_end - chunk_start) == cs
            is_tail = chunk_end == n and not is_full
            if not is_full and not is_tail:
                # Should not happen given our slicing, but defensive.
                break

            if chunk_end > skip_emit_until and (is_full or self._save_partial_chunks):
                ph = h.copy().hexdigest()
                yield ChunkSpec(
                    start=chunk_start,
                    end=chunk_end,
                    key=CacheEngineKey(
                        engine_tag=self._engine_tag,
                        model_tag=self._model_tag,
                        tp=self._tp,
                        rank=self._rank,
                        chunk_end=chunk_end,
                        prefix_hash=ph,
                        layer_idx=layer_idx,
                        bundle=bundle,
                    ),
                )
            i = j

    # ---- helpers used by adapters ----------------------------------------

    def chunk_boundaries(self, num_tokens: int) -> list[int]:
        cs = self._chunk_size
        ends = list(range(cs, (num_tokens // cs) * cs + 1, cs))
        if self._save_partial_chunks and (num_tokens % cs != 0) and num_tokens > 0:
            ends.append(num_tokens)
        return ends


def boundary_prefix_hashes(
    token_ids: Sequence[int], boundaries: Iterable[int]
) -> dict[int, str]:
    """
    Free function for callers that need just the boundary digests without
    constructing a database. Single incremental hasher; matches what
    `_prefix_hashes` in connector_v1 used to do but exposed at this layer
    so other adapters can share it.
    """
    want = set(int(b) for b in boundaries)
    out: dict[int, str] = {}
    if not want:
        return out
    h = hashlib.sha256()
    for i, t in enumerate(token_ids, start=1):
        h.update(int(t).to_bytes(4, "little", signed=False))
        if i in want:
            out[i] = h.hexdigest()
            if len(out) == len(want):
                break
    return out
