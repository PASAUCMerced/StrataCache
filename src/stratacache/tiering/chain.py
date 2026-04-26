from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from stratacache.backend.base import MemoryLayer
from stratacache.core.artifact import ArtifactId
from stratacache.core.errors import ArtifactNotFound
from stratacache.core.memory_obj import MemoryObj
from stratacache.tiering.policy import LinkPolicy, StoreReason
from stratacache.writeback.manager import WritebackManager

from stratacache.telemetry.telemetry import StrataTelemetry, StrataTierType


@dataclass(frozen=True, slots=True)
class FetchResult:
    memory_obj: MemoryObj
    hit_tier: int


class TierChain:
    """
    Ordered tier chain with per-link write-through / write-back semantics.

    Payloads at this layer are MemoryObj (was: bytes + ArtifactMeta).
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

    def set_event_sink(self, sink) -> None:
        """Subscribe `sink` to admit/evict events from every tier (A11)."""
        for t in self._tiers:
            t.set_event_sink(sink)

    def _resolve_tier_type(self, tier: int | str) -> StrataTierType:
        if tier == -1 or tier == "gpu":
            return StrataTierType.GPU
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
        """Best-effort existence check across tiers."""
        with self._lock:
            for i, b in enumerate(self._tiers):
                try:
                    if b.exists(artifact_id):
                        return i
                except Exception:
                    continue
        return None

    def exists_in(self, tier: int | str, artifact_id: ArtifactId) -> bool:
        idx = self._resolve_tier_index(tier)
        with self._lock:
            return self._tiers[idx].exists(artifact_id)

    def fetch(
        self,
        artifact_id: ArtifactId,
        *,
        promote: bool = True,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> FetchResult:
        """Read-through lookup from top to bottom."""
        with self._lock:
            for i in range(len(self._tiers)):
                try:
                    mo = self._tier_get(i, artifact_id, dtype=dtype, shape=shape)
                except ArtifactNotFound:
                    continue

                if promote and i > 0:
                    for up in range(0, i):
                        self._put_direct(up, artifact_id, mo, reason=StoreReason.PROMOTION)
                return FetchResult(memory_obj=mo, hit_tier=i)
        raise ArtifactNotFound(str(artifact_id))

    def fetch_from(
        self,
        tier: int | str,
        artifact_id: ArtifactId,
        *,
        promote: bool = False,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> FetchResult:
        idx = self._resolve_tier_index(tier)
        with self._lock:
            mo = self._tier_get(idx, artifact_id, dtype=dtype, shape=shape)
            if promote and idx > 0:
                for up in range(0, idx):
                    self._put_direct(up, artifact_id, mo, reason=StoreReason.PROMOTION)
            return FetchResult(memory_obj=mo, hit_tier=idx)

    def store(self, artifact_id: ArtifactId, memory_obj: MemoryObj) -> None:
        """Store into the head tier and apply link semantics down the chain."""
        with self._lock:
            self._put_direct(0, artifact_id, memory_obj, reason=StoreReason.CLIENT_WRITE)
            self._propagate_after_write(
                0, artifact_id, memory_obj, reason=StoreReason.CLIENT_WRITE
            )

    def store_at(
        self,
        tier: int | str,
        artifact_id: ArtifactId,
        memory_obj: MemoryObj,
        *,
        propagate: bool = False,
    ) -> None:
        idx = self._resolve_tier_index(tier)
        with self._lock:
            self._put_direct(idx, artifact_id, memory_obj, reason=StoreReason.CLIENT_WRITE)
            if propagate:
                self._propagate_after_write(
                    idx, artifact_id, memory_obj, reason=StoreReason.CLIENT_WRITE
                )

    def delete(self, artifact_id: ArtifactId) -> None:
        with self._lock:
            for i in range(len(self._tiers)):
                self._tier_delete(i, artifact_id)
            for upper in range(len(self._links)):
                self._wb.clear_dirty(upper, artifact_id)

    def delete_from(self, tier: int | str, artifact_id: ArtifactId) -> None:
        idx = self._resolve_tier_index(tier)
        with self._lock:
            self._tier_delete(idx, artifact_id)
            upper = idx
            if upper < len(self._links):
                self._wb.clear_dirty(upper, artifact_id)
            if upper > 0:
                self._wb.clear_dirty(upper - 1, artifact_id)

    def flush(
        self,
        artifact_id: Optional[ArtifactId] = None,
        *,
        max_items: Optional[int] = None,
    ) -> int:
        """Best-effort synchronous flush for write-back links."""
        with self._lock:
            if artifact_id is None:
                return self._wb.flush(None, max_items=max_items)

            total = 0
            max_rounds = max(1, len(self._links) + 2)
            for _ in range(max_rounds):
                n = self._wb.flush(artifact_id)
                total += n
                if n == 0:
                    break
            return total

    # ---- internals ----

    def _tier_get(
        self,
        tier_index: int,
        artifact_id: ArtifactId,
        *,
        dtype: Optional[str] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> MemoryObj:
        start_time = time.perf_counter()
        mo = self._tiers[tier_index].get(artifact_id, dtype=dtype, shape=shape)
        end_time = time.perf_counter()
        self._telemetry.on_tier_op_async(
            tier=self._resolve_tier_type(tier_index),
            op_type="load",
            latency_us=(end_time - start_time) * 1_000_000,
            size=mo.get_size(),
        )
        return mo

    def _tier_delete(self, tier_index: int, artifact_id: ArtifactId) -> None:
        start_time = time.perf_counter()
        released_size = self._tiers[tier_index].delete(artifact_id)
        end_time = time.perf_counter()
        self._telemetry.on_tier_op_async(
            tier=self._resolve_tier_type(tier_index),
            op_type="delete",
            latency_us=(end_time - start_time) * 1_000_000,
            size=released_size,
        )

    def _put_direct(
        self,
        tier_index: int,
        artifact_id: ArtifactId,
        memory_obj: MemoryObj,
        *,
        reason: StoreReason,
    ) -> None:
        start_time = time.perf_counter()
        released_size = self._tiers[tier_index].put(artifact_id, memory_obj)
        end_time = time.perf_counter()
        self._telemetry.on_tier_op_async(
            tier=self._resolve_tier_type(tier_index),
            op_type="store",
            latency_us=(end_time - start_time) * 1_000_000,
            size=memory_obj.get_size(),
            released_size=released_size,
        )
        # Promotions / flush writes should not mark dirty for upper tiers above them.
        _ = reason

    def _propagate_after_write(
        self,
        tier_index: int,
        artifact_id: ArtifactId,
        memory_obj: MemoryObj,
        *,
        reason: StoreReason,
    ) -> None:
        if tier_index >= len(self._links):
            return  # no lower tier

        policy = self._links[tier_index]
        if policy == LinkPolicy.WRITE_THROUGH:
            lower = tier_index + 1
            self._put_direct(lower, artifact_id, memory_obj, reason=reason)
            self._propagate_after_write(lower, artifact_id, memory_obj, reason=reason)
            return

        if policy == LinkPolicy.WRITE_BACK:
            if reason in (StoreReason.CLIENT_WRITE, StoreReason.WRITEBACK_FLUSH):
                self._wb.mark_dirty(tier_index, artifact_id)
            return

        raise ValueError(f"unknown policy: {policy}")

    def _flush_hop(self, upper_tier: int, artifact_id: ArtifactId) -> None:
        """Flush one write-back hop: upper_tier -> upper_tier+1."""
        with self._lock:
            if upper_tier < 0 or upper_tier >= len(self._links):
                return
            if self._links[upper_tier] != LinkPolicy.WRITE_BACK:
                self._wb.clear_dirty(upper_tier, artifact_id)
                return

            try:
                mo = self._tier_get(upper_tier, artifact_id)
            except ArtifactNotFound:
                self._wb.clear_dirty(upper_tier, artifact_id)
                return

            lower = upper_tier + 1
            self._put_direct(lower, artifact_id, mo, reason=StoreReason.WRITEBACK_FLUSH)
            self._wb.clear_dirty(upper_tier, artifact_id)

            self._propagate_after_write(
                lower, artifact_id, mo, reason=StoreReason.WRITEBACK_FLUSH
            )
