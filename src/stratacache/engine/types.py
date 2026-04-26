from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stratacache.core.memory_obj import MemoryObj


class AccessMode(str, Enum):
    CHAIN = "chain"
    EXACT = "exact"
    PREFER = "prefer"


@dataclass(frozen=True, slots=True)
class LoadResult:
    memory_obj: MemoryObj
    hit_tier: int
    hit_medium: str


@dataclass(frozen=True, slots=True)
class ContainsResult:
    exists: bool
    hit_tier: int | None
    hit_medium: str | None
