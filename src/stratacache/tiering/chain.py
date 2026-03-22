from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from stratacache.backend.base import MemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta
from stratacache.core.errors import ArtifactNotFound
from stratacache.tiering.policy import LinkPolicy, StoreReason
from stratacache.writeback.manager import WritebackManager

from stratacache.telemetry.telemetry import StrataTelemetry, StrataTierType

import time

@dataclass(frozen=True, slots=True)
class FetchResult:
    payload: bytes
    meta: ArtifactMeta
    hit_tier: int


class TierChain:
    """
    Ordered tier chain with per-link write-through / write-back semantics.
    """

    def __init__(
        self,
        *,
        tiers: Sequence[MemoryLayer],
        links: Sequence[LinkPolicy],
        enable_writeback_worker: bool = True,
    ) -> None:
        if len(tiers) < 1:
            raise ValueError("tiers must be non-empty")
        if len(links) != max(0, len(tiers) - 1):
            raise ValueError("links must have len(tiers)-1 items")
        self._tiers = list(tiers)
        self._tier_name_to_idx: dict[str, int] = {}
        for i, tier in enumerate(self._tiers):
            if tier.name in self._tier_name_to_idx:
                raise ValueError(f"duplicate tier name: {tier.name}")
            self._tier_name_to_idx[tier.name] = i
        self._links = list(links)
        self._lock = threading.RLock()
        self._wb = WritebackManager(
            links=self._links,
            flush_hop=self._flush_hop,
            enable_worker=enable_writeback_worker,
        )
        
        self._telemetry = StrataTelemetry.get_or_create()


    @property
    def tiers(self) -> list[MemoryLayer]:
        return list(self._tiers)

    @property
    def links(self) -> list[LinkPolicy]:
        return list(self._links)

    @property
    def tier_names(self) -> list[str]:
        return [tier.name for tier in self._tiers]

    def close(self) -> None:
        self._wb.stop()
        
    def _resolve_tier_type(self, tier: int | str) -> StrataTierType:
        if tier == 0 or tier == "cpu":
            return StrataTierType.CPU
        if tier == 1 or tier == "cxl":
            return StrataTierType.CXL
        if tier == 2 or tier == "nixl":
            return StrataTierType.NIXL
        if tier == 3 or tier == "disk":
            return StrataTierType.DISK
        return StrataTierType.UNKNOWN

    def _resolve_tier_index(self, tier: int | str) -> int:
        if isinstance(tier, int):
            idx = int(tier)
        else:
            idx = self._tier_name_to_idx.get(str(tier), -1)
        if idx < 0 or idx >= len(self._tiers):
            raise ValueError(f"unknown tier: {tier}")
        return idx

    def exists(self, artifact_id: ArtifactId) -> Optional[int]:
        """
        Best-effort existence check across tiers (no payload read, no promotion).

        Returns:
            hit_tier index if present, else None.
        """
        with self._lock:
            for i, b in enumerate(self._tiers):
                try:
                    if b.exists(artifact_id):
                        return i
                except Exception:
                    # Backends should not throw, but keep exists() best-effort.
                    continue
        return None

    def exists_in(self, tier: int | str, artifact_id: ArtifactId) -> bool:
        idx = self._resolve_tier_index(tier)
        with self._lock:
            return self._tiers[idx].exists(artifact_id)

    def fetch(self, artifact_id: ArtifactId, *, promote: bool = True) -> FetchResult:
        """
        Read-through lookup from top to bottom.

        If promote=True, a hit in lower tiers is promoted to all upper tiers
        without producing dirty entries (promotion does not change persistence).
        """
        with self._lock:
            for i, b in enumerate(self._tiers):
                try:                    
                    payload, meta = self.tier_get_telemetry_wrap(i, b.get, artifact_id)
                except ArtifactNotFound:
                    continue

                if promote and i > 0:
                    for up in range(0, i):
                        self._put_direct(up, artifact_id, payload, meta, reason=StoreReason.PROMOTION)
                return FetchResult(payload=payload, meta=meta, hit_tier=i)
        raise ArtifactNotFound(str(artifact_id))

    def fetch_from(self, tier: int | str, artifact_id: ArtifactId, *, promote: bool = False) -> FetchResult:
        idx = self._resolve_tier_index(tier)
        with self._lock:
            payload, meta = self.tier_get_telemetry_wrap(idx, self._tiers[idx].get, artifact_id)
            if promote and idx > 0:
                for up in range(0, idx):
                    self._put_direct(up, artifact_id, payload, meta, reason=StoreReason.PROMOTION)
            return FetchResult(payload=payload, meta=meta, hit_tier=idx)

    def tier_get_telemetry_wrap(self, tier_index: int, func, *args, **kwargs) -> Tuple[bytes, ArtifactMeta]:
        """Wrap tier.get() to record telemetry data

        Args:
            tier_index (int): Tier
            func (_type_): Tier get function call

        Returns:
            Tuple[bytes, ArtifactMeta]: payload and meta returned by tier.get()
        """
        
        start_time = time.perf_counter()
        payload, meta = func(*args, **kwargs)
        end_time = time.perf_counter()
        tier_bytes_used = self._tiers[tier_index].stats().bytes_used
        self._telemetry.on_tier_op_async(
            tier=self._resolve_tier_type(tier_index),
            op_type="load",
            latency_us=(end_time - start_time) * 1_000_000,
            size=len(payload),
            tier_bytes_used=tier_bytes_used,
        )
        return payload, meta

    def store(self, artifact_id: ArtifactId, payload: bytes, meta: ArtifactMeta) -> None:
        """
        Store into the head tier and apply link semantics down the chain.
        """
        with self._lock:
            self._put_direct(0, artifact_id, payload, meta, reason=StoreReason.CLIENT_WRITE)
            self._propagate_after_write(0, artifact_id, payload, meta, reason=StoreReason.CLIENT_WRITE)

    def store_at(
        self,
        tier: int | str,
        artifact_id: ArtifactId,
        payload: bytes,
        meta: ArtifactMeta,
        *,
        propagate: bool = False,
    ) -> None:
        idx = self._resolve_tier_index(tier)
        with self._lock:
            self._put_direct(idx, artifact_id, payload, meta, reason=StoreReason.CLIENT_WRITE)
            if propagate:
                self._propagate_after_write(
                    idx,
                    artifact_id,
                    payload,
                    meta,
                    reason=StoreReason.CLIENT_WRITE,
                )

    def delete(self, artifact_id: ArtifactId) -> None:
        with self._lock:
            for i, b in enumerate(self._tiers):
                self.tier_delete_telemetry_wrap(i, b.delete, artifact_id)
            # Clear dirty markers for all possible upper tiers.
            for upper in range(len(self._links)):
                self._wb.clear_dirty(upper, artifact_id)

    def delete_from(self, tier: int | str, artifact_id: ArtifactId) -> None:
        idx = self._resolve_tier_index(tier)
        with self._lock:
            self.tier_delete_telemetry_wrap(idx, self._tiers[idx].delete, artifact_id)
            upper = idx
            if upper < len(self._links):
                self._wb.clear_dirty(upper, artifact_id)
            if upper > 0:
                self._wb.clear_dirty(upper - 1, artifact_id)
                
    def tier_delete_telemetry_wrap(self, tier_index: int, func, *args, **kwargs) -> None:
        """Wrap tier.delete() to record telemetry data.

        Args:
            tier_index (int): Tier
            func (_type_): Tier delete function call
        """
        start_time = time.perf_counter()
        released_size = func(*args, **kwargs)
        end_time = time.perf_counter()
        tier_bytes_used = self._tiers[tier_index].stats().bytes_used
        self._telemetry.on_tier_op_async(
            tier=self._resolve_tier_type(tier_index),
            op_type="delete",
            latency_us=(end_time - start_time) * 1_000_000,
            size=released_size,
            tier_bytes_used=tier_bytes_used,
        )

    def flush(self, artifact_id: Optional[ArtifactId] = None, *, max_items: Optional[int] = None) -> int:
        """
        Best-effort synchronous flush for write-back links.
        """
        with self._lock:
            if artifact_id is None:
                return self._wb.flush(None, max_items=max_items)

            # For a specific artifact, flush until stable so multi-hop write-back
            # chains can fully propagate in one call.
            total = 0
            max_rounds = max(1, len(self._links) + 2)
            for _ in range(max_rounds):
                n = self._wb.flush(artifact_id)
                total += n
                if n == 0:
                    break
            return total

    # ---- internals ----

    def _put_direct(
        self,
        tier_index: int,
        artifact_id: ArtifactId,
        payload: bytes,
        meta: ArtifactMeta,
        *,
        reason: StoreReason,
    ) -> None:
        self.tier_put_telemetry_wrap(tier_index, self._tiers[tier_index].put, artifact_id, payload, meta)
        # Promotions / flush writes should not mark dirty for upper tiers above them.
        _ = reason
        
    def tier_put_telemetry_wrap(self, tier_index: int, func, *args, **kwargs) -> None:
        """Wrap tier.put() to record telemetry data.

        Args:
            tier_index (int): Tier
            func (_type_): Tier put function call
        """
        start_time = time.perf_counter()
        released_size = func(*args, **kwargs)
        end_time = time.perf_counter()
        payload_size = len(args[1]) if len(args) > 1 else 0  # payload is the second arg
        tier_bytes_used = self._tiers[tier_index].stats().bytes_used
        self._telemetry.on_tier_op_async(
            tier=self._resolve_tier_type(tier_index),
            op_type="store",
            latency_us=(end_time - start_time) * 1_000_000,
            size=payload_size,
            released_size=released_size,  # Report released block size
            tier_bytes_used=tier_bytes_used,
        )

    def _propagate_after_write(
        self,
        tier_index: int,
        artifact_id: ArtifactId,
        payload: bytes,
        meta: ArtifactMeta,
        *,
        reason: StoreReason,
    ) -> None:
        """
        After a successful write to tier_index, apply downstream link semantics.
        """
        if tier_index >= len(self._links):
            return  # no lower tier

        policy = self._links[tier_index]
        if policy == LinkPolicy.WRITE_THROUGH:
            lower = tier_index + 1
            self._put_direct(lower, artifact_id, payload, meta, reason=reason)
            self._propagate_after_write(lower, artifact_id, payload, meta, reason=reason)
            return

        if policy == LinkPolicy.WRITE_BACK:
            # Mark this tier as dirty w.r.t the next tier.
            if reason in (StoreReason.CLIENT_WRITE, StoreReason.WRITEBACK_FLUSH):
                self._wb.mark_dirty(tier_index, artifact_id)
            return

        raise ValueError(f"unknown policy: {policy}")

    def _flush_hop(self, upper_tier: int, artifact_id: ArtifactId) -> None:
        """
        Flush one write-back hop: upper_tier -> upper_tier+1.
        This is called by WritebackManager.
        """
        with self._lock:
            if upper_tier < 0 or upper_tier >= len(self._links):
                return
            if self._links[upper_tier] != LinkPolicy.WRITE_BACK:
                self._wb.clear_dirty(upper_tier, artifact_id)
                return

            # If artifact is missing from upper tier, treat as clean (nothing to flush).
            try:
                payload, meta = self.tier_get_telemetry_wrap(upper_tier, self._tiers[upper_tier].get, artifact_id)
            except ArtifactNotFound:
                self._wb.clear_dirty(upper_tier, artifact_id)
                return

            lower = upper_tier + 1
            self._put_direct(lower, artifact_id, payload, meta, reason=StoreReason.WRITEBACK_FLUSH)
            self._wb.clear_dirty(upper_tier, artifact_id)

            # After updating the lower tier, apply downstream semantics starting from lower.
            self._propagate_after_write(
                lower, artifact_id, payload, meta, reason=StoreReason.WRITEBACK_FLUSH
            )
