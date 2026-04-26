from __future__ import annotations

from collections.abc import Sequence

from stratacache.backend.base import MemoryLayer
from stratacache.core.artifact import ArtifactId
from stratacache.core.errors import ArtifactNotFound
from stratacache.core.memory_obj import MemoryObj
from stratacache.engine.types import AccessMode, ContainsResult, LoadResult
from stratacache.tiering.chain import TierChain
from stratacache.tiering.policy import LinkPolicy


class StorageEngine:
    """
    Public storage facade used by adapters and direct clients.

    It exposes a small, stable API (`store/load/contains/delete`) and hides
    tier traversal details behind `AccessMode`. Payloads at this boundary
    are MemoryObj.
    """

    def __init__(self, chain: TierChain) -> None:
        self._chain = chain

    @classmethod
    def from_tiers(
        cls,
        *,
        tiers: Sequence[MemoryLayer],
        links: Sequence[LinkPolicy],
        enable_writeback_worker: bool = True,
    ) -> "StorageEngine":
        chain = TierChain(
            tiers=tiers,
            links=links,
            enable_writeback_worker=enable_writeback_worker,
        )
        return cls(chain)

    @property
    def tier_names(self) -> list[str]:
        return self._chain.tier_names

    @property
    def chain(self) -> TierChain:
        """Expose the underlying TierChain for advanced/internal integrations."""
        return self._chain

    def close(self) -> None:
        self._chain.close()

    def store(
        self,
        artifact_id: ArtifactId,
        memory_obj: MemoryObj,
        *,
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.CHAIN,
    ) -> None:
        m = _normalize_mode(mode)
        if medium is None:
            self._chain.store(artifact_id, memory_obj)
            return

        if m == AccessMode.CHAIN:
            self._chain.store_at(medium, artifact_id, memory_obj, propagate=True)
            return
        if m == AccessMode.EXACT:
            self._chain.store_at(medium, artifact_id, memory_obj, propagate=False)
            return
        if m == AccessMode.PREFER:
            try:
                self._chain.store_at(medium, artifact_id, memory_obj, propagate=False)
            except ValueError:
                self._chain.store(artifact_id, memory_obj)
            return
        raise ValueError(f"unknown mode: {mode}")

    def load(
        self,
        artifact_id: ArtifactId,
        *,
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.CHAIN,
        promote: bool = True,
    ) -> LoadResult:
        m = _normalize_mode(mode)

        if medium is None or m == AccessMode.CHAIN:
            fr = self._chain.fetch(artifact_id, promote=promote)
            return LoadResult(
                memory_obj=fr.memory_obj,
                hit_tier=fr.hit_tier,
                hit_medium=self._chain.tier_names[fr.hit_tier],
            )

        if m == AccessMode.EXACT:
            fr = self._chain.fetch_from(medium, artifact_id, promote=promote)
            return LoadResult(
                memory_obj=fr.memory_obj,
                hit_tier=fr.hit_tier,
                hit_medium=self._chain.tier_names[fr.hit_tier],
            )

        if m == AccessMode.PREFER:
            try:
                fr = self._chain.fetch_from(medium, artifact_id, promote=promote)
            except (ArtifactNotFound, ValueError):
                fr = self._chain.fetch(artifact_id, promote=promote)
            return LoadResult(
                memory_obj=fr.memory_obj,
                hit_tier=fr.hit_tier,
                hit_medium=self._chain.tier_names[fr.hit_tier],
            )

        raise ValueError(f"unknown mode: {mode}")

    def contains(
        self,
        artifact_id: ArtifactId,
        *,
        medium: int | str | None = None,
        mode: AccessMode | str = AccessMode.CHAIN,
    ) -> ContainsResult:
        m = _normalize_mode(mode)

        if medium is None or m == AccessMode.CHAIN:
            hit = self._chain.exists(artifact_id)
            if hit is None:
                return ContainsResult(exists=False, hit_tier=None, hit_medium=None)
            return ContainsResult(
                exists=True,
                hit_tier=hit,
                hit_medium=self._chain.tier_names[hit],
            )

        if m == AccessMode.EXACT:
            try:
                ok = self._chain.exists_in(medium, artifact_id)
            except ValueError:
                ok = False
            if not ok:
                return ContainsResult(exists=False, hit_tier=None, hit_medium=None)
            idx = _resolve_tier(self._chain, medium)
            return ContainsResult(
                exists=True, hit_tier=idx, hit_medium=self._chain.tier_names[idx]
            )

        if m == AccessMode.PREFER:
            try:
                ok = self._chain.exists_in(medium, artifact_id)
            except ValueError:
                ok = False
            if ok:
                idx = _resolve_tier(self._chain, medium)
                return ContainsResult(
                    exists=True,
                    hit_tier=idx,
                    hit_medium=self._chain.tier_names[idx],
                )
            hit = self._chain.exists(artifact_id)
            if hit is None:
                return ContainsResult(exists=False, hit_tier=None, hit_medium=None)
            return ContainsResult(
                exists=True, hit_tier=hit, hit_medium=self._chain.tier_names[hit]
            )

        raise ValueError(f"unknown mode: {mode}")

    def delete(self, artifact_id: ArtifactId, *, medium: int | str | None = None) -> None:
        if medium is None:
            self._chain.delete(artifact_id)
            return
        self._chain.delete_from(medium, artifact_id)


def _normalize_mode(mode: AccessMode | str) -> AccessMode:
    if isinstance(mode, AccessMode):
        return mode
    s = str(mode).strip().lower()
    try:
        return AccessMode(s)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"unknown mode: {mode}") from e


def _resolve_tier(chain: TierChain, tier: int | str) -> int:
    if isinstance(tier, int):
        if tier < 0 or tier >= len(chain.tiers):
            raise ValueError(f"unknown tier: {tier}")
        return tier
    name = str(tier)
    for i, t in enumerate(chain.tiers):
        if t.name == name:
            return i
    raise ValueError(f"unknown tier: {tier}")
