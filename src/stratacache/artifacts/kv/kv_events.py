"""
KV-side translation of the type-agnostic backend admit/evict events into
vLLM-shaped CacheStore/CacheRemove events.

Backends emit `(op, ArtifactId, size, layer_name)`. KVEventTranslator
decodes the chunk key, filters non-KV artifacts, and forwards them to a
caller sink. The connector subscribes the translator to every backend in
the chain and drains its queue from `get_kv_events()`.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional

from stratacache.core.artifact import ArtifactId


logger = logging.getLogger(__name__)


# Match keys produced by build_kv_chunk_id / CacheEngineKey.to_artifact_id().
# Patterns:
#   <engine>:<model>:tp=<tp>:rank=<rank>:ph=<hex>:chunk_end=<n>[:layer=N | :bundle=...]
#   plus the legacy connector forms:
#   vllm013:<model>:chunk_end=<n>:ph=<hex>:[manifest|bundle|bundleT|layer=N]
_KV_KEY_PATTERNS = [
    re.compile(
        r":ph=(?P<ph>[0-9a-f]+):chunk_end=(?P<end>\d+)"
    ),
    re.compile(
        r":chunk_end=(?P<end>\d+):ph=(?P<ph>[0-9a-f]+)"
    ),
]


@dataclass(frozen=True, slots=True)
class CacheStoreEvent:
    artifact_id: str
    chunk_end: int
    prefix_hash: str
    layer_name: str
    size: int


@dataclass(frozen=True, slots=True)
class CacheRemoveEvent:
    artifact_id: str
    chunk_end: int
    prefix_hash: str
    layer_name: str
    size: int


def _parse_kv_key(s: str) -> Optional[tuple[int, str]]:
    for rx in _KV_KEY_PATTERNS:
        m = rx.search(s)
        if m:
            try:
                return int(m.group("end")), m.group("ph")
            except Exception:  # noqa: BLE001
                return None
    return None


class KVEventTranslator:
    """
    Subscribes to backend admit/evict events and translates KV ones into
    CacheStoreEvent / CacheRemoveEvent. Buffers them in a thread-safe
    queue that the connector drains via `drain_events`.

    Non-KV artifacts (e.g. parameter chunks) are silently ignored at this
    layer; they would have their own translator under `artifacts/params/`
    once needed.
    """

    def __init__(self, *, max_queue: int = 65536) -> None:
        self._lock = threading.Lock()
        self._max = int(max_queue)
        self._queue: list[object] = []  # Union[CacheStoreEvent, CacheRemoveEvent]
        self._dropped = 0

    def on_backend_event(
        self, op: str, aid: ArtifactId, size: int, layer_name: str
    ) -> None:
        parsed = _parse_kv_key(str(aid))
        if parsed is None:
            return  # not a KV chunk
        chunk_end, ph = parsed
        if op == "store":
            ev: object = CacheStoreEvent(
                artifact_id=str(aid),
                chunk_end=chunk_end,
                prefix_hash=ph,
                layer_name=layer_name,
                size=int(size),
            )
        elif op == "remove":
            ev = CacheRemoveEvent(
                artifact_id=str(aid),
                chunk_end=chunk_end,
                prefix_hash=ph,
                layer_name=layer_name,
                size=int(size),
            )
        else:
            return
        with self._lock:
            if len(self._queue) >= self._max:
                # Drop the oldest to keep the most recent events.
                self._queue.pop(0)
                self._dropped += 1
            self._queue.append(ev)

    def drain_events(self) -> list:
        with self._lock:
            out = self._queue
            self._queue = []
            return out

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped
