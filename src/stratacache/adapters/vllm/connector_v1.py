from __future__ import annotations

"""
vLLM v0.13.0 connector (KVConnector v1) - StrataCache implementation.

Stores generalized records via StrataCache's tiered storage plane.
"""

import hashlib
import io
import os
import re
import threading
import time
import json
import struct
import logging
import atexit
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from stratacache.backend.cpu import CpuMemoryLayer
from stratacache.core.artifact import ArtifactId, ArtifactMeta, ArtifactType
from stratacache.core.errors import ArtifactNotFound
from stratacache.core.memory_obj import BytesMemoryObj, MemoryObj
from stratacache.engine.storage_engine import StorageEngine
from stratacache.tiering.chain import TierChain
from stratacache.tiering.policy import LinkPolicy

try:  # optional dependency
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore[import-not-found]
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    KVConnectorBase_V1 = object  # type: ignore[assignment]
    KVConnectorMetadata = object  # type: ignore[assignment]
    KVConnectorRole = object  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

try:  # optional dependency for config file parsing
    import yaml  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    yaml = None  # type: ignore[assignment]


_LAYER_RE = re.compile(r"(\d+)")
logger = logging.getLogger(__name__)

# ------------------------------
# Process-wide TierChain singleton
# ------------------------------
_CHAIN_LOCK = threading.RLock()
_CHAIN_BY_KEY: dict[str, TierChain] = {}
_CHAIN_REFCOUNT: dict[str, int] = {}

# ------------------------------
# Process-wide request stats aggregator (best-effort; per-process)
# ------------------------------
_REQ_STATS_LOCK = threading.RLock()
_REQ_STATS_BY_ID: dict[str, dict[str, Any]] = {}
_REQ_STATS_CUM: dict[str, int] = {"total": 0, "gpu": 0}  # plus per-tier tokens dynamically


def _prof_enabled() -> bool:
    return _parse_bool(os.getenv("STRATACACHE_PROFILE", "0"))


_PROF_LOCK = threading.RLock()
_PROF: dict[str, dict[str, Any]] = {}
_PROF_SAMPLES_CAP = 2000


def _prof_record(name: str, dt_s: float) -> None:
    if not _prof_enabled():
        return
    with _PROF_LOCK:
        d = _PROF.get(name)
        if d is None:
            d = {"count": 0, "total_s": 0.0, "samples": []}
            _PROF[name] = d
        d["count"] = int(d["count"]) + 1
        d["total_s"] = float(d["total_s"]) + float(dt_s)
        s = d.get("samples")
        if isinstance(s, list) and len(s) < _PROF_SAMPLES_CAP:
            s.append(float(dt_s))


class _ProfTimer:
    __slots__ = ("_name", "_t0")

    def __init__(self, name: str):
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _prof_record(self._name, time.perf_counter() - self._t0)
        return False


def _prof_dump() -> None:
    if not _prof_enabled():
        return
    with _PROF_LOCK:
        items = list(_PROF.items())
    if not items:
        return

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        k = int(round((len(ys) - 1) * p))
        return float(ys[max(0, min(k, len(ys) - 1))])

    lines = []
    lines.append("StrataCache(vLLM) connector_profile(pid=%d):" % os.getpid())
    for name, d in sorted(items, key=lambda kv: float(kv[1].get("total_s", 0.0)), reverse=True):
        c = int(d.get("count", 0))
        tot = float(d.get("total_s", 0.0))
        avg = 0.0 if c <= 0 else tot / c
        ss = d.get("samples") if isinstance(d.get("samples"), list) else []
        p50 = pct(ss, 0.50)
        p95 = pct(ss, 0.95)
        lines.append(
            "  - %s: count=%d total_ms=%.2f avg_ms=%.3f p50_ms=%.3f p95_ms=%.3f"
            % (name, c, tot * 1000.0, avg * 1000.0, p50 * 1000.0, p95 * 1000.0)
        )
    logger.info("\n".join(lines))


_PROF_DUMPED = False


def _prof_dump_once(reason: str) -> None:
    global _PROF_DUMPED  # noqa: PLW0603
    if _PROF_DUMPED:
        return
    _PROF_DUMPED = True
    try:
        _prof_dump()
    except Exception:  # noqa: BLE001
        pass
    # Extra safety: if logging is shutting down, also emit to stderr.
    if _prof_enabled():
        try:
            sys.stderr.write(f"[stratacache] profile dump attempted ({reason})\\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass


_SIG_HOOKED = False


def _install_profile_signal_handlers() -> None:
    """
    vLLM may terminate EngineCore processes via SIGTERM, which bypasses atexit.
    Install best-effort signal hooks so we still dump profiling summary.
    """
    global _SIG_HOOKED  # noqa: PLW0603
    if _SIG_HOOKED:
        return
    _SIG_HOOKED = True

    def _handler(signum, frame):  # noqa: ARG001
        _prof_dump_once(f"signal:{signum}")
        # Let default behavior proceed by raising SystemExit.
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except Exception:  # noqa: BLE001
            pass


def _rs_get(req_id: str) -> dict[str, Any]:
    with _REQ_STATS_LOCK:
        d = _REQ_STATS_BY_ID.get(req_id)
        if d is None:
            d = {"total": 0, "gpu": 0, "tiers": {}, "set_once": False}
            _REQ_STATS_BY_ID[req_id] = d
        return d


def _rs_pop(req_id: str) -> Optional[dict[str, Any]]:
    with _REQ_STATS_LOCK:
        return _REQ_STATS_BY_ID.pop(req_id, None)


def _rs_add_cum(total: int, gpu: int, tiers: dict[str, int]) -> None:
    with _REQ_STATS_LOCK:
        _REQ_STATS_CUM["total"] = int(_REQ_STATS_CUM.get("total", 0)) + int(total)
        _REQ_STATS_CUM["gpu"] = int(_REQ_STATS_CUM.get("gpu", 0)) + int(gpu)
        for k, v in (tiers or {}).items():
            _REQ_STATS_CUM[k] = int(_REQ_STATS_CUM.get(k, 0)) + int(v)


def _rs_cum_rates() -> dict[str, float]:
    with _REQ_STATS_LOCK:
        tot = int(_REQ_STATS_CUM.get("total", 0))
        out: dict[str, float] = {}
        if tot <= 0:
            return out
        for k, v in _REQ_STATS_CUM.items():
            if k == "total":
                continue
            out[k] = float(int(v)) / float(tot)
        return out


# ------------------------------
# Bundle codec: store all layers for a chunk in one artifact
# ------------------------------
_BUNDLE_MAGIC = b"SCB0"
_BUNDLE_HDR = struct.Struct("<4sI")  # magic, json_len


def _infer_num_layers_from_vllm_config(vllm_config: Any) -> Optional[int]:
    """
    Best-effort extraction of transformer layer count from vLLM config.
    We avoid importing vLLM internals here; this is purely duck-typing.
    """
    mc = getattr(vllm_config, "model_config", None)
    if mc is None:
        return None
    # Common patterns in vLLM / HF configs.
    for obj in (mc, getattr(mc, "hf_config", None), getattr(mc, "model_config", None)):
        if obj is None:
            continue
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers", "num_transformer_layers"):
            try:
                v = getattr(obj, attr)
            except Exception:  # noqa: BLE001
                continue
            try:
                iv = int(v)
            except Exception:  # noqa: BLE001
                continue
            if iv > 0:
                return iv
    # Method-style accessors.
    for meth in ("get_num_layers", "num_layers"):
        fn = getattr(mc, meth, None)
        if callable(fn):
            try:
                iv = int(fn())
            except Exception:  # noqa: BLE001
                continue
            if iv > 0:
                return iv
    return None


def _encode_bundle(layer_payloads: dict[int, bytes]) -> bytes:
    """
    Encode {layer_idx: encoded_tensor_bytes} into a single payload.
    """
    items = []
    for li in sorted(layer_payloads.keys()):
        b = layer_payloads[int(li)]
        items.append({"layer": int(li), "len": int(len(b))})
    hdr = json.dumps({"items": items}, separators=(",", ":"), sort_keys=True).encode("utf-8")
    out = bytearray()
    out += _BUNDLE_HDR.pack(_BUNDLE_MAGIC, len(hdr))
    out += hdr
    for it in items:
        out += layer_payloads[int(it["layer"])]
    return bytes(out)


def _decode_bundle(payload: bytes) -> dict[int, bytes]:
    if len(payload) < _BUNDLE_HDR.size:
        raise ValueError("bundle buffer too short")
    magic, jlen = _BUNDLE_HDR.unpack_from(payload, 0)
    if magic != _BUNDLE_MAGIC:
        raise ValueError("bad bundle magic")
    j0 = _BUNDLE_HDR.size
    j1 = j0 + int(jlen)
    meta = json.loads(payload[j0:j1].decode("utf-8"))
    items = list(meta.get("items", []) or [])
    off = j1
    out: dict[int, bytes] = {}
    for it in items:
        li = int(it["layer"])
        ln = int(it["len"])
        out[li] = payload[off : off + ln]
        off += ln
    return out


def _chain_key_from_config(cfg: dict[str, Any], vllm_config: Any) -> str:
    # Only include knobs that affect storage identity/shape.
    mc = getattr(vllm_config, "model_config", None)
    model_tag = getattr(mc, "served_model_name", None) or getattr(mc, "model", None) or "model"
    pc = getattr(vllm_config, "parallel_config", None)
    tp = getattr(pc, "tensor_parallel_size", None)
    rank = getattr(pc, "rank", None)
    return json.dumps(
        {
            "model": str(model_tag),
            "tp": tp,
            "rank": rank,
            "use_cxl": str(cfg.get("use_cxl", False)),
            "writeback": str(cfg.get("writeback", False)),
            "cpu_cap_gb": str(cfg.get("cpu_capacity_gb", 60)),
            "cxl_dax": str(cfg.get("cxl_dax_device", "") or ""),
            "bundle_layers": str(cfg.get("bundle_layers", True)),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _get_or_create_chain(*, key: str, tiers: list[Any], links: list[LinkPolicy]) -> TierChain:
    with _CHAIN_LOCK:
        ch = _CHAIN_BY_KEY.get(key)
        if ch is None:
            ch = TierChain(tiers=tiers, links=links, enable_writeback_worker=True)
            _CHAIN_BY_KEY[key] = ch
            _CHAIN_REFCOUNT[key] = 0
        _CHAIN_REFCOUNT[key] = int(_CHAIN_REFCOUNT.get(key, 0)) + 1
        return ch


def _release_chain(key: str) -> None:
    with _CHAIN_LOCK:
        if key not in _CHAIN_REFCOUNT:
            return
        n = int(_CHAIN_REFCOUNT.get(key, 0)) - 1
        if n > 0:
            _CHAIN_REFCOUNT[key] = n
            return
        _CHAIN_REFCOUNT.pop(key, None)
        ch = _CHAIN_BY_KEY.pop(key, None)
        if ch is not None:
            ch.close()


@atexit.register
def _close_all_chains() -> None:
    # Dump profiling summary (if enabled) before teardown.
    try:
        _prof_dump_once("atexit")
    except Exception:  # noqa: BLE001
        pass
    with _CHAIN_LOCK:
        keys = list(_CHAIN_BY_KEY.keys())
    for k in keys:
        try:
            _release_chain(k)
        except Exception:
            pass


def _parse_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    return default


def _parse_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        return default


_CONNECTOR_DEFAULTS: dict[str, Any] = {
    "use_cxl": False,
    "writeback": False,
    "cpu_capacity_gb": 60,
    "chunk_size": 256,
    "bundle_layers": True,
    "tensor_codec": "stable",
    "tensor_header_in_payload": False,
    "log_stats": True,
    "debug": False,
    "save_partial_chunks": True,
    "log_every": 50,
    "log_min_interval_s": 2.0,
    "cxl_dax_device": None,
    "cxl_reset_metadata": False,
    # ---- Phase 3 features ----
    "use_pinned_slab": True,          # B1+B2: allocate pinned host slab
    "use_lazy_allocator": False,      # B5: background-grow the slab
    "lazy_initial_mb": 0,             # 0 -> auto
    "lazy_growth_step_mb": 0,         # 0 -> auto
    "use_layerwise_pipeline": False,  # A1+A2: paged connector + streams
    "use_token_database": False,      # A4: incremental ChunkedTokenDatabase
    "expose_kv_events": False,        # A11: surface CacheStore/Remove via get_kv_events
    "numa_node": -1,                  # B6: -1 = no bind, else node id
    "reserve_local_cpu_mb": 1024,     # B9: capacity clamp headroom
}


def _default_connector_config_path() -> Path:
    p = Path(__file__).resolve()
    candidates = [
        p.parents[2] / "config.yaml",       # installed package layout
        p.parents[4] / "config.yaml",       # source tree layout
        Path.cwd() / "stratacache" / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _read_yaml_config(path: Path, *, warn_missing_parser: bool) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        if warn_missing_parser:
            logger.warning("config file exists but PyYAML is unavailable: %s", path)
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to parse config file %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    # Accept either flat keys or nested:
    # stratacache:
    #   connector:
    #     ...
    node: dict[str, Any] = data
    sc = node.get("stratacache")
    if isinstance(sc, dict):
        node = sc
    conn = node.get("connector")
    if isinstance(conn, dict):
        node = conn
    return node


def _load_connector_config(extra: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_CONNECTOR_DEFAULTS)

    explicit_cfg_path = extra.get("stratacache.config_path") or extra.get("config_path")
    cfg_path_raw = explicit_cfg_path or _default_connector_config_path()
    cfg_path = Path(str(cfg_path_raw)).expanduser()
    file_cfg = _read_yaml_config(cfg_path, warn_missing_parser=bool(explicit_cfg_path))

    def pick(key: str) -> Any:
        if f"stratacache.{key}" in extra:
            return extra[f"stratacache.{key}"]
        if key in extra:
            return extra[key]
        if key in file_cfg:
            return file_cfg[key]
        return cfg[key]

    cfg["use_cxl"] = _parse_bool(pick("use_cxl"), bool(cfg["use_cxl"]))
    cfg["writeback"] = _parse_bool(pick("writeback"), bool(cfg["writeback"]))
    cfg["cpu_capacity_gb"] = _parse_int(pick("cpu_capacity_gb"), int(cfg["cpu_capacity_gb"]))
    cfg["chunk_size"] = _parse_int(pick("chunk_size"), int(cfg["chunk_size"]))
    cfg["bundle_layers"] = _parse_bool(pick("bundle_layers"), bool(cfg["bundle_layers"]))
    cfg["tensor_codec"] = str(pick("tensor_codec") or cfg["tensor_codec"]).lower()
    cfg["tensor_header_in_payload"] = _parse_bool(
        pick("tensor_header_in_payload"),
        bool(cfg["tensor_header_in_payload"]),
    )
    cfg["log_stats"] = _parse_bool(pick("log_stats"), bool(cfg["log_stats"]))
    cfg["debug"] = _parse_bool(pick("debug"), bool(cfg["debug"]))
    cfg["save_partial_chunks"] = _parse_bool(
        pick("save_partial_chunks"),
        bool(cfg["save_partial_chunks"]),
    )
    cfg["log_every"] = _parse_int(pick("log_every"), int(cfg["log_every"]))
    try:
        cfg["log_min_interval_s"] = float(pick("log_min_interval_s"))
    except Exception:  # noqa: BLE001
        cfg["log_min_interval_s"] = float(_CONNECTOR_DEFAULTS["log_min_interval_s"])
    dax = pick("cxl_dax_device")
    cfg["cxl_dax_device"] = None if dax in (None, "", "null") else str(dax)
    cfg["cxl_reset_metadata"] = _parse_bool(
        pick("cxl_reset_metadata"),
        bool(cfg["cxl_reset_metadata"]),
    )

    # Phase 3 toggles.
    cfg["use_pinned_slab"] = _parse_bool(pick("use_pinned_slab"), bool(cfg["use_pinned_slab"]))
    cfg["use_lazy_allocator"] = _parse_bool(pick("use_lazy_allocator"), bool(cfg["use_lazy_allocator"]))
    cfg["lazy_initial_mb"] = _parse_int(pick("lazy_initial_mb"), int(cfg["lazy_initial_mb"]))
    cfg["lazy_growth_step_mb"] = _parse_int(pick("lazy_growth_step_mb"), int(cfg["lazy_growth_step_mb"]))
    cfg["use_layerwise_pipeline"] = _parse_bool(pick("use_layerwise_pipeline"), bool(cfg["use_layerwise_pipeline"]))
    cfg["use_token_database"] = _parse_bool(pick("use_token_database"), bool(cfg["use_token_database"]))
    cfg["expose_kv_events"] = _parse_bool(pick("expose_kv_events"), bool(cfg["expose_kv_events"]))
    cfg["numa_node"] = _parse_int(pick("numa_node"), int(cfg["numa_node"]))
    cfg["reserve_local_cpu_mb"] = _parse_int(pick("reserve_local_cpu_mb"), int(cfg["reserve_local_cpu_mb"]))
    return cfg


def _as_token_list(x: Any) -> list[int]:
    """
    Best-effort conversion for vLLM token id containers:
    - list[int]
    - tuple[int]
    - torch.Tensor (1D)
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [int(t) for t in x]
    if isinstance(x, tuple):
        return [int(t) for t in x]
    if torch is not None:
        try:
            import torch as _torch  # type: ignore[import-not-found]

            if isinstance(x, _torch.Tensor):
                return [int(t) for t in x.flatten().tolist()]
        except Exception:  # noqa: BLE001
            pass
    return []


def _extract_token_ids(request: Any) -> list[int]:
    """
    vLLM request objects differ across versions/paths.

    Prefer `prompt_token_ids` for cross-request prefix reuse, then fall
    back to other token id containers.
    """
    # Prefer prompt_token_ids for cross-request prefix reuse. all_token_ids can
    # include generated tokens for in-flight requests, which breaks prompt-key
    # stability.
    for attr in ("prompt_token_ids", "prompt_token_ids_list", "all_token_ids"):
        if hasattr(request, attr):
            toks = _as_token_list(getattr(request, attr))
            if toks:
                return toks
    return []

def _extract_block_ids(blocks: Any) -> list[int]:
    """
    Extract block_ids from vLLM's KVCacheBlocks (v1) or similar containers.

    vLLM v0.13 uses `vllm.v1.core.kv_cache_manager.KVCacheBlocks` which exposes
    `get_block_ids()` returning `tuple[list[int], ...]` (kv_cache_groups).
    """
    if blocks is None:
        return []
    # KVCacheBlocks path
    if hasattr(blocks, "get_block_ids"):
        try:
            gids = blocks.get_block_ids(allow_none=True)  # type: ignore[attr-defined]
            if gids is None:
                return []
            # Only one KVCacheGroup is supported for connectors currently.
            if isinstance(gids, tuple) and len(gids) > 0:
                return [int(x) for x in gids[0]]
        except Exception:  # noqa: BLE001
            pass
    # Fallback: list/tuple of ints
    if isinstance(blocks, (list, tuple)):
        if blocks and isinstance(blocks[0], list):
            blocks = blocks[0]
        return [int(x) for x in blocks]
    # Fallback: try iter()
    try:
        return [int(x) for x in blocks]
    except Exception:  # noqa: BLE001
        return []


def _prompt_hash(token_ids: list[int]) -> str:
    h = hashlib.sha256()
    # stable little-endian packing to avoid string joins
    for t in token_ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()

def _prefix_hashes(token_ids: list[int], boundaries: list[int]) -> dict[int, str]:
    """
    Compute sha256 hex digests for token_ids prefixes at specified boundaries.
    Uses a single incremental hasher for efficiency.
    """
    want = set(boundaries)
    out: dict[int, str] = {}
    h = hashlib.sha256()
    for i, t in enumerate(token_ids, start=1):
        h.update(int(t).to_bytes(4, "little", signed=False))
        if i in want:
            out[i] = h.hexdigest()
            if len(out) == len(want):
                break
    return out


def _token_sig(token_ids: list[int]) -> tuple[int, int, int]:
    """
    Lightweight token-sequence signature for per-request scheduler caches.
    """
    n = len(token_ids)
    if n <= 0:
        return (0, 0, 0)
    return (n, int(token_ids[0]), int(token_ids[-1]))


def _layer_index(layer_name: str, fallback: int) -> int:
    m = list(_LAYER_RE.finditer(layer_name))
    if not m:
        return fallback
    return int(m[-1].group(1))


def _flatten_slots(kv_layer: "torch.Tensor") -> "torch.Tensor":
    """
    Return a view where the slot dimension is flattened to a single axis.

    Supports common vLLM shapes:
      - [num_blocks, block_size, ...] -> [num_slots, ...]
      - [2, num_blocks, block_size, ...] -> [2, num_slots, ...]
    """
    if kv_layer.dim() < 2:
        raise ValueError(f"Unsupported kv_layer ndim={kv_layer.dim()}")
    if kv_layer.dim() >= 3 and kv_layer.size(0) == 2:
        # [2, B, S, ...]
        b = int(kv_layer.size(1))
        s = int(kv_layer.size(2))
        return kv_layer.reshape(2, b * s, *kv_layer.shape[3:])
    # [B, S, ...]
    b = int(kv_layer.size(0))
    s = int(kv_layer.size(1))
    return kv_layer.reshape(b * s, *kv_layer.shape[2:])


def _gather_by_slots(kv_layer: "torch.Tensor", slot_mapping: "torch.Tensor") -> "torch.Tensor":
    """
    Gather KV rows by slot ids.

    Fast path: if slot_mapping is a contiguous increasing range on CPU (common for prefill),
    use slicing instead of index_select (much faster). Return a view here to avoid
    an extra device copy; callers that need contiguous bytes will materialize later.
    """
    flat = _flatten_slots(kv_layer)
    sm = slot_mapping
    if isinstance(sm, torch.Tensor) and sm.device.type == "cpu" and sm.numel() > 0:
        try:
            # Only when there are no masked slots (-1) and the mapping is contiguous.
            if int(sm.min().item()) >= 0:
                # contiguous increasing by 1
                if sm.numel() == 1 or bool(torch.all((sm[1:] - sm[:-1]) == 1).item()):
                    s0 = int(sm[0].item())
                    ln = int(sm.numel())
                    if flat.dim() >= 2 and flat.size(0) == 2:
                        return flat[:, s0 : s0 + ln]
                    return flat[s0 : s0 + ln]
        except Exception:  # noqa: BLE001
            pass

    slots = sm.to(device=kv_layer.device)
    mask = slots >= 0
    slots = slots[mask].to(dtype=torch.long)
    if flat.dim() >= 2 and flat.size(0) == 2:
        # [2, num_slots, ...]
        return flat.index_select(1, slots).contiguous()
    return flat.index_select(0, slots).contiguous()


def _scatter_by_slots(kv_layer: "torch.Tensor", slot_mapping: "torch.Tensor", gathered: "torch.Tensor") -> None:
    flat = _flatten_slots(kv_layer)
    sm = slot_mapping
    # Fast path: contiguous increasing mapping and no masked slots.
    if isinstance(sm, torch.Tensor) and sm.device.type == "cpu" and sm.numel() > 0:
        try:
            if int(sm.min().item()) >= 0:
                if sm.numel() == 1 or bool(torch.all((sm[1:] - sm[:-1]) == 1).item()):
                    s0 = int(sm[0].item())
                    ln = int(sm.numel())
                    g = gathered.to(device=kv_layer.device)
                    if flat.dim() >= 2 and flat.size(0) == 2:
                        # gathered may include full-length with masked slots; here no mask, so accept both
                        if g.size(1) == ln:
                            flat[:, s0 : s0 + ln] = g
                        else:
                            flat[:, s0 : s0 + ln] = g[:, :ln]
                    else:
                        if g.size(0) == ln:
                            flat[s0 : s0 + ln] = g
                        else:
                            flat[s0 : s0 + ln] = g[:ln]
                    return
        except Exception:  # noqa: BLE001
            pass

    slots = sm.to(device=kv_layer.device)
    mask = slots >= 0
    slots_pos = slots[mask].to(dtype=torch.long)

    # slot_mapping may contain invalid positions (-1) for tokens that vLLM
    # already has in its own prefix cache / not allocated in this step.
    # Support both cases:
    # - gathered is full-length (includes invalid positions)  -> filter by mask
    # - gathered is already filtered (length == mask.sum())   -> keep as-is
    if flat.dim() >= 2 and flat.size(0) == 2:
        g = gathered.to(device=kv_layer.device)
        if g.size(1) == mask.numel():
            g = g[:, mask]
        flat[:, slots_pos] = g
    else:
        g = gathered.to(device=kv_layer.device)
        if g.size(0) == mask.numel():
            g = g[mask]
        flat[slots_pos] = g


def _tensor_to_bytes(t: "torch.Tensor") -> bytes:
    # Deprecated in connector v0.2: keep for compatibility behind a switch.
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _bytes_to_tensor(b: bytes, device: "torch.device") -> "torch.Tensor":
    # Deprecated in connector v0.2: keep for compatibility behind a switch.
    buf = io.BytesIO(b)
    t = torch.load(buf, map_location="cpu")
    return t.to(device=device)


_TENSOR_MAGIC = b"SCT0"
_TENSOR_HDR = struct.Struct("<4sI")  # magic, json_len


def _encode_tensor_stable(t: "torch.Tensor") -> bytes:
    """
    Stable, versioned tensor codec (no pickle).

    Format:
      - magic: b"SCT0"
      - u32 json header length
      - json header: {"dtype": "...", "shape":[...]}
      - raw bytes: contiguous CPU tensor in C order
    """
    tcpu = t.detach().contiguous().cpu()
    header = {
        "dtype": str(tcpu.dtype).replace("torch.", ""),
        "shape": list(tcpu.shape),
    }
    hdr_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    # NOTE: do NOT use tcpu.numpy() directly: NumPy doesn't support bfloat16.
    # Instead, reinterpret the underlying bytes as uint8 (supported) and serialize.
    raw = tcpu.view(torch.uint8).numpy().tobytes(order="C")
    return _TENSOR_HDR.pack(_TENSOR_MAGIC, len(hdr_bytes)) + hdr_bytes + raw


def _encode_tensor_raw_payload(t: "torch.Tensor") -> tuple[bytes, str, list[int]]:
    """
    Headerless tensor payload: raw bytes only.
    dtype/shape are returned separately and should be stored in ArtifactMeta attrs.
    """
    tcpu = t.detach().contiguous().cpu()
    dtype_str = str(tcpu.dtype).replace("torch.", "")
    shape = [int(x) for x in tcpu.shape]
    raw = tcpu.view(torch.uint8).numpy().tobytes(order="C")
    return raw, dtype_str, shape


def _decode_tensor_stable(b: bytes, device: "torch.device") -> "torch.Tensor":
    if len(b) < _TENSOR_HDR.size:
        raise ValueError("tensor buffer too short")
    magic, jlen = _TENSOR_HDR.unpack_from(b, 0)
    if magic != _TENSOR_MAGIC:
        raise ValueError("bad tensor magic")
    j0 = _TENSOR_HDR.size
    j1 = j0 + int(jlen)
    header = json.loads(b[j0:j1].decode("utf-8"))
    dtype_str = header["dtype"]
    shape = tuple(int(x) for x in header["shape"])
    # map dtype string to torch dtype
    dt = getattr(torch, dtype_str, None)
    if dt is None:
        # common names: bfloat16, float16, float32, int64...
        raise ValueError(f"unsupported dtype: {dtype_str}")
    raw = b[j1:]
    # Create tensor from bytes.
    # We decode as uint8 then view into the desired dtype to support bfloat16.
    # Make a writable copy to avoid PyTorch warning about non-writable buffers.
    u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    t = u8.view(dt).reshape(shape).clone()  # clone to own the memory
    return t.to(device=device)


def _decode_tensor_raw_payload(
    raw: bytes,
    *,
    dtype_str: str,
    shape: list[int],
    device: "torch.device",
) -> "torch.Tensor":
    dt = getattr(torch, str(dtype_str), None)
    if dt is None:
        raise ValueError(f"unsupported dtype: {dtype_str}")
    u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    t = u8.view(dt).reshape(tuple(int(x) for x in shape)).clone()
    return t.to(device=device)


@dataclass
class StrataKVReq:
    req_id: str
    token_ids: list[int]
    slot_mapping: Any  # torch.Tensor (keep Any for safe import)
    # Total prefix length that is available from external cache (token count).
    # Worker will load the range beyond vLLM's already-cached prefix.
    tokens_to_load: int = 0
    # Prefix length already present in vLLM GPU cache at scheduling time.
    vllm_cached_tokens: int = 0
    tokens_to_save: int = 0
    # Full prompt hash (used only for debugging).
    prompt_hash: str = ""
    # Chunk boundaries that are expected to exist in cache (scheduler computed).
    chunk_ends_to_load: list[int] = field(default_factory=list)
    chunk_ends_to_save: list[int] = field(default_factory=list)
    # Prefix-hash map computed by the scheduler over `chunk_ends_to_load
    # ∪ chunk_ends_to_save`. Worker reuses this instead of re-running
    # SHA256 in start_load_kv / save_kv_layer.
    ph_map: dict[int, str] = field(default_factory=dict)


@dataclass
class StrataConnectorMetadata(KVConnectorMetadata):  # type: ignore[misc]
    requests: list[StrataKVReq] = field(default_factory=list)


class _StrataConnectorImpl:
    """
    Connector implementation that is shared between scheduler and workers.
    """

    def __init__(self, vllm_config: Any, role: Any, parent: Any, kv_cache_config: Any = None):
        if torch is None:
            raise RuntimeError("vllm/torch are required to use StrataCacheConnectorV1")
        self._vllm_config = vllm_config
        self._role = role
        self._parent = parent
        self._kv_cache_config = kv_cache_config
        self._lock = threading.RLock()
        # Install signal handlers once per process so profiling dumps even if vLLM
        # terminates EngineCore via SIGTERM (atexit may not run).
        _install_profile_signal_handlers()

        extra = getattr(vllm_config.kv_transfer_config, "kv_connector_extra_config", None) or {}
        cfg = _load_connector_config(extra)
        use_cxl = bool(cfg["use_cxl"])
        writeback = bool(cfg["writeback"])
        cpu_cap_gb = int(cfg["cpu_capacity_gb"])
        self._block_size = int(getattr(vllm_config.cache_config, "block_size", 16))
        self._chunk_size = int(cfg["chunk_size"])
        self._bundle_layers = bool(cfg["bundle_layers"])

        self._tensor_codec = str(cfg["tensor_codec"]).lower()
        # When disabled (default), stable codec stores raw tensor bytes and puts
        # dtype/shape in ArtifactMeta attrs, avoiding per-payload JSON header work.
        self._tensor_header_in_payload = bool(cfg["tensor_header_in_payload"])
        self._log_stats = bool(cfg["log_stats"])
        self._debug = bool(cfg["debug"])
        self._save_partial_chunks = bool(cfg["save_partial_chunks"])
        self._log_every = int(cfg["log_every"])
        self._log_min_interval_s = float(cfg["log_min_interval_s"])
        self._stats = {
            # Scheduler-side token-level accounting.
            "sched_calls": 0,
            "prompt_tokens_total": 0,
            # vLLM local prefix caching tokens (GPU-side cache hit tokens reported by scheduler).
            "gpu_hit_tokens_total": 0,
            "external_matched_tokens_total": 0,
            # tokens actually loaded from external cache (best-effort, measured on worker)
            "external_loaded_tokens_total": 0,
            # Chunk-level accounting.
            "manifest_hits": 0,
            "manifest_misses": 0,
            "stored_chunks": 0,
            "loaded_chunks": 0,
            "bytes_stored": 0,
            "bytes_loaded": 0,
        }
        self._last_log_t = 0.0
        self._last_logged_io_total = -1
        self._last_logged_sched_calls = -1

        # CPU tier: optionally back by a pinned slab (B1+B2), optionally
        # via the lazy/expanding allocator (B5), with capacity clamped to
        # what the OS can actually give us (B9), and optionally NUMA-bound
        # to the GPU's socket (B6). All toggles default to off so existing
        # workloads see the same shape.
        cpu_cap_bytes = cpu_cap_gb * 1024 * 1024 * 1024
        if cfg.get("use_pinned_slab"):
            from stratacache.backend.cpu import (
                CpuAllocator,
                CpuMemoryLayer as _CpuLayer,
                LazyCpuAllocator,
            )
            from stratacache.backend.cpu.cpu_allocator import (
                clamp_capacity_to_system,
            )

            reserve = int(cfg["reserve_local_cpu_mb"]) * 1024 * 1024
            effective_cap = clamp_capacity_to_system(
                cpu_cap_bytes, reserve_bytes=reserve
            )
            numa_node = int(cfg["numa_node"])
            numa_arg = numa_node if numa_node >= 0 else None
            if cfg.get("use_lazy_allocator"):
                init_b = int(cfg["lazy_initial_mb"]) * 1024 * 1024 or None
                step_b = int(cfg["lazy_growth_step_mb"]) * 1024 * 1024 or None
                allocator = LazyCpuAllocator(
                    capacity_bytes=effective_cap,
                    initial_bytes=init_b,
                    growth_step_bytes=step_b,
                    pin_memory=True,
                )
            else:
                allocator = CpuAllocator(
                    capacity_bytes=effective_cap,
                    pin_memory=True,
                    numa_node=numa_arg,
                )
            tiers = [
                _CpuLayer(
                    capacity_bytes=effective_cap,
                    store_name="cpu",
                    allocator=allocator,
                )
            ]
        else:
            tiers = [CpuMemoryLayer(capacity_bytes=cpu_cap_bytes, store_name="cpu")]

        links: list[LinkPolicy] = []
        if use_cxl:
            from stratacache.backend.cxl import CxlConfig, CxlMemoryLayer

            dax = cfg.get("cxl_dax_device")
            reset_md = bool(cfg["cxl_reset_metadata"])
            tiers.append(
                CxlMemoryLayer(
                    config=CxlConfig(dax_device=dax, reset_metadata_on_init=reset_md),
                    store_name="cxl",
                )
            )
            links.append(LinkPolicy.WRITE_BACK if writeback else LinkPolicy.WRITE_THROUGH)

        # vLLM may create multiple connector instances within the same process
        # (scheduler vs worker paths). Use a process-wide singleton TierChain so
        # in-process backends (CpuMemoryLayer) are shared and external matches can succeed.
        self._chain_key = _chain_key_from_config(cfg, vllm_config)
        self._chain = _get_or_create_chain(key=self._chain_key, tiers=tiers, links=links)
        self._engine = StorageEngine(self._chain)

        # ---- Phase 3 plug-ins ----
        self._token_db = None
        if cfg.get("use_token_database"):
            from stratacache.artifacts.kv.token_database import (
                ChunkedTokenDatabase,
            )

            mc = getattr(vllm_config, "model_config", None)
            model_tag = getattr(mc, "served_model_name", None) or getattr(mc, "model", None) or "model"
            pc = getattr(vllm_config, "parallel_config", None)
            self._token_db = ChunkedTokenDatabase(
                chunk_size=int(self._chunk_size),
                engine_tag="vllm013",
                model_tag=str(model_tag),
                tp=getattr(pc, "tensor_parallel_size", None),
                rank=getattr(pc, "rank", None),
                save_partial_chunks=bool(self._save_partial_chunks),
            )

        self._kv_event_translator = None
        if cfg.get("expose_kv_events"):
            from stratacache.artifacts.kv.kv_events import KVEventTranslator

            self._kv_event_translator = KVEventTranslator()
            self._engine.set_event_sink(self._kv_event_translator.on_backend_event)

        self._gpu_paged = None
        if cfg.get("use_layerwise_pipeline"):
            try:
                from stratacache.gpu import MultiLayerPagedConnector

                self._gpu_paged = MultiLayerPagedConnector()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "use_layerwise_pipeline=True but MultiLayerPagedConnector "
                    "could not be initialised; falling back to legacy path."
                )

        # Per-memory-layer IO attribution (by fetch hit tier and by write-through policy).
        self._tier_names = list(self._engine.tier_names)
        self._io_by_tier = {
            name: {
                "bytes_loaded": 0,
                "bytes_stored": 0,
                "chunks_loaded": 0,
                "chunks_stored": 0,
                "manifest_hits": 0,
                "manifest_misses": 0,
                # token-level attribution (best-effort)
                "tokens_loaded": 0,
                "tokens_stored": 0,
            }
            for name in self._tier_names
        }

        # scheduler-side bookkeeping
        self._alloc_blocks: dict[str, list[int]] = {}
        self._num_external_tokens: dict[str, int] = {}
        self._prompt_tokens: dict[str, list[int]] = {}
        # Cached prefix-hash map per request (populated once by
        # get_num_new_matched_tokens, salvaged in update_state_after_alloc,
        # forwarded to the worker via StrataKVReq.ph_map).
        self._ph_map_by_req: dict[str, dict[int, str]] = {}

        # worker-side cached references to vLLM KV layers (filled lazily from ForwardContext)
        self._kv_caches: dict[str, "torch.Tensor"] = {}
        self._expected_num_layers: Optional[int] = _infer_num_layers_from_vllm_config(vllm_config)

        # KV bundleT schema template populated on the first save so that the
        # CXL load path can reconstruct (dtype, full_shape) without the
        # CXL backend having to persist user-meta. See ARCHITECTURE.md
        # discussion of the codec-on-CXL question.
        # Layout: {"dtype": "bfloat16",
        #          "has_kv2": bool,           # True for [2, B, S, ...] KV layout
        #          "tail": tuple[int,...],   # per-layer dims AFTER the slot dim
        #          "elem_size": int}
        self._kv_template: Optional[dict[str, Any]] = None

        # Bundle buffer: (req_id, chunk_end) -> {layer_idx: bytes}
        self._bundle_buf: dict[tuple[str, int], dict[int, bytes]] = {}

        # Fast save path: enqueue per-layer references in save_kv_layer,
        # do heavy gather/encode/store once per step in wait_for_save.
        # Keyed by (req_id, chunk_end).
        self._pending_save: dict[tuple[str, int], dict[str, Any]] = {}
        # Track which layer indices have been enqueued for a given (req, end).
        self._pending_layers: dict[tuple[str, int], set[int]] = {}
        # Request-local index for pending keys to avoid O(N) scans on request_finished.
        self._pending_keys_by_req: dict[str, set[tuple[str, int]]] = {}
        # Per-request current-step chunk_ends to save (computed once on layer0),
        # so other layers can attach without recomputing boundaries/hashes.
        self._pending_ends_by_req: dict[str, list[int]] = {}

        # Scheduler-side slot_mapping cache to avoid rebuilding every step.
        self._slot_offs = torch.arange(0, int(self._block_size), dtype=torch.long)
        self._slot_mapping_by_req: dict[str, "torch.Tensor"] = {}
        self._slot_blocks_by_req: dict[str, list[int]] = {}
        # Scheduler-side matched-token incremental probe state by request.
        # Keeps confirmed contiguous chunk hits so we don't re-probe from chunk 0.
        self._match_state_by_req: dict[str, dict[str, Any]] = {}

        # worker-side save state
        self._seen_layers: dict[int, bool] = {}
        # Avoid re-saving the same prompt KV repeatedly (chunked prefill / decode steps).
        # Tracks the maximum token index saved so far for (request_id, layer_idx).
        self._saved_upto_by_req_layer: dict[tuple[str, int], int] = {}
        # Debug: prove which process calls which entrypoints (avoid spam).
        self._dbg_logged_sched_pid = False
        self._dbg_logged_worker_pid = False
        self._dbg_logged_load_meta = False
        self._dbg_logged_load_miss = False

        # Avoid repeated external loads across engine steps for the same request.
        self._loaded_or_attempted: set[str] = set()

    def _init_kv_caches_from_forward_context(self, forward_context: Any) -> None:
        """
        vLLM v0.13 ForwardContext does not pass KV tensors into connector
        hooks; discover them via forward_context.no_compile_layers[*].kv_cache.
        """
        ncl = getattr(forward_context, "no_compile_layers", None)
        if not isinstance(ncl, dict):
            return
        ve = getattr(forward_context, "virtual_engine", 0)
        for layer_name, attn_layer in ncl.items():
            if not hasattr(attn_layer, "kv_cache"):
                continue
            if layer_name in self._kv_caches:
                continue
            try:
                kv = attn_layer.kv_cache[ve]
            except Exception:  # noqa: BLE001
                continue
            # kv is the per-layer paged KV tensor (vLLM internal type).
            self._kv_caches[str(layer_name)] = kv
        if self._kv_caches and self._expected_num_layers is None:
            self._expected_num_layers = int(len(self._kv_caches))

    def _model_tag(self) -> str:
        mc = getattr(self._vllm_config, "model_config", None)
        return getattr(mc, "served_model_name", None) or getattr(mc, "model", None) or "model"

    def _ns_prefix(self, prompt_hash: str) -> str:
        # Keep it simple and deterministic.
        pc = getattr(self._vllm_config, "parallel_config", None)
        tp = getattr(pc, "tensor_parallel_size", None)
        rank = getattr(pc, "rank", None)
        return f"vllm013:{self._model_tag()}:tp={tp}:rank={rank}:h={prompt_hash}"

    def _chunk_manifest_id(self, prefix_hash: str, chunk_end: int) -> ArtifactId:
        # prefix_hash already identifies the prefix tokens; chunk_end is stored for sanity/debug.
        return ArtifactId(f"vllm013:{self._model_tag()}:chunk_end={chunk_end}:ph={prefix_hash}:manifest")

    def _chunk_bundle_id(self, prefix_hash: str, chunk_end: int) -> ArtifactId:
        return ArtifactId(f"vllm013:{self._model_tag()}:chunk_end={chunk_end}:ph={prefix_hash}:bundle")

    def _chunk_bundle_tensor_id(self, prefix_hash: str, chunk_end: int) -> ArtifactId:
        # New (faster) format: one tensor containing all layers for this chunk_end.
        return ArtifactId(f"vllm013:{self._model_tag()}:chunk_end={chunk_end}:ph={prefix_hash}:bundleT")

    def _chunk_layer_id(self, prefix_hash: str, chunk_end: int, layer_idx: int) -> ArtifactId:
        return ArtifactId(f"vllm013:{self._model_tag()}:chunk_end={chunk_end}:ph={prefix_hash}:layer={layer_idx}")

    def _encode_tensor(self, t: "torch.Tensor") -> tuple[bytes, dict[str, Any]]:
        if self._tensor_codec == "torchsave":
            return _tensor_to_bytes(t), {"tensor_codec": "torchsave"}
        if self._tensor_header_in_payload:
            return _encode_tensor_stable(t), {"tensor_codec": "stable"}
        raw, dtype_str, shape = _encode_tensor_raw_payload(t)
        return raw, {"tensor_codec": "stable_raw", "tensor_dtype": dtype_str, "tensor_shape": shape}

    def _maybe_update_kv_template(self, stacked: "torch.Tensor") -> None:
        """
        Cache the KV bundleT schema (dtype + per-layer shape minus slot dim)
        from the first stacked tensor we see. Used later on the load path
        to reconstruct shape from CXL `actual_size`.

        `stacked` shape:
        - K/V combined: [num_layers, 2, slots, *tail]
        - K/V flat:     [num_layers, slots, *tail]
        """
        if self._kv_template is not None:
            return
        if stacked.dim() < 2:
            return
        has_kv2 = stacked.dim() >= 3 and int(stacked.size(1)) == 2
        if has_kv2:
            tail = tuple(int(x) for x in stacked.shape[3:])
        else:
            tail = tuple(int(x) for x in stacked.shape[2:])
        self._kv_template = {
            "dtype": str(stacked.dtype).replace("torch.", ""),
            "has_kv2": has_kv2,
            "tail": tail,
            "elem_size": int(stacked.element_size()),
        }

    def _bundle_load_hint(
        self, exp: int, slots: int
    ) -> tuple[Optional[str], Optional[tuple[int, ...]]]:
        """
        Build (dtype, full_shape) hint for a bundleT load given the chunk's
        slot count. Returns (None, None) when no template is cached yet
        (cold-start with CXL hit but no prior save) - in that case the
        caller should fall back to bytes-mode and either reconstruct from
        meta.attrs (CPU hit path) or skip the chunk.
        """
        tmpl = self._kv_template
        if tmpl is None or exp <= 0 or slots <= 0:
            return None, None
        if tmpl["has_kv2"]:
            full_shape = (int(exp), 2, int(slots), *tmpl["tail"])
        else:
            full_shape = (int(exp), int(slots), *tmpl["tail"])
        return str(tmpl["dtype"]), tuple(int(x) for x in full_shape)

    def _make_kv_memory_obj_pipelined(
        self,
        layers_dict: dict[int, "torch.Tensor"],
        sm_slice: "torch.Tensor",
        base_attrs: dict[str, Any],
        exp: int,
    ) -> Optional[tuple["MemoryObj", int]]:
        """
        A1+A2 pipelined path: skip the per-layer Python `_gather_by_slots`
        + `torch.stack` and gather all `exp` layers into one slab-backed
        host buffer using `MultiLayerPagedConnector.gather_chunk` on a
        dedicated CUDA stream. Returns None when the pipeline is disabled
        or unable to run, in which case the caller should fall back to
        the legacy stack-based path.
        """
        if self._gpu_paged is None:
            return None
        if exp <= 0 or len(layers_dict) < exp:
            return None
        from stratacache.core.memory_obj import TensorMemoryObj

        layers_list = [layers_dict[i] for i in range(exp)]
        sample_flat = self._gpu_paged._flatten_slots(layers_list[0])
        rng = self._gpu_paged._maybe_contig_range(sm_slice)
        if rng is not None:
            s0, ln = rng
            if sample_flat.dim() >= 2 and sample_flat.size(0) == 2:
                per_shape: tuple[int, ...] = (2, ln, *sample_flat.shape[2:])
            else:
                per_shape = (ln, *sample_flat.shape[1:])
        else:
            try:
                n = int((sm_slice >= 0).sum().item())
            except Exception:  # noqa: BLE001
                return None
            if sample_flat.dim() >= 2 and sample_flat.size(0) == 2:
                per_shape = (2, n, *sample_flat.shape[2:])
            else:
                per_shape = (n, *sample_flat.shape[1:])

        dtype = sample_flat.dtype
        full_shape = (exp, *per_shape)
        from functools import reduce
        from operator import mul

        numel = reduce(mul, full_shape, 1)
        elem_size = sample_flat.element_size()
        nbytes = int(numel * elem_size)

        alloc = self._engine.get_cpu_allocator()
        slot = None
        if alloc is not None and alloc.has_slab:
            slot = alloc.try_allocate(nbytes)
            if slot is None:
                cpu_layer = self._chain.tiers[0]
                if hasattr(cpu_layer, "allocate_slot"):
                    slot = cpu_layer.allocate_slot(nbytes, busy_loop=False)
        if slot is not None:
            host = slot.tensor_view().view(dtype).reshape(full_shape)
            release_cb = slot.free
        else:
            pin = bool(alloc.pin_memory) if alloc is not None else False
            host = torch.empty(full_shape, dtype=dtype, pin_memory=pin)
            release_cb = None

        try:
            self._gpu_paged.gather_chunk(layers_list, sm_slice, host)
            self._gpu_paged.synchronize_store()
        except Exception:  # noqa: BLE001
            logger.exception("Pipelined gather failed; falling back to legacy path.")
            if slot is not None:
                slot.free()
            return None

        # Cache schema for the CXL load path.
        self._maybe_update_kv_template(host)

        attrs = dict(base_attrs)
        attrs.setdefault("tensor_codec", "stable_raw")
        attrs.setdefault("tensor_dtype", str(dtype).replace("torch.", ""))
        attrs.setdefault("tensor_shape", [int(x) for x in full_shape])
        meta = ArtifactMeta(artifact_type=ArtifactType.KV_BLOCKS, attrs=attrs)
        mo = TensorMemoryObj(
            host, meta, size=nbytes, release_callback=release_cb
        )
        return mo, nbytes

    def _make_kv_memory_obj(
        self, t: "torch.Tensor", base_attrs: dict[str, Any]
    ) -> tuple["MemoryObj", int]:
        # Cache KV bundleT schema for the CXL load path. `t` here is the
        # already-stacked tensor with shape (num_layers, ...).
        if int(base_attrs.get("bundle_layers", 0)) > 0:
            self._maybe_update_kv_template(t)

        """
        Produce a MemoryObj for a gathered KV tensor and return (memory_obj,
        size_bytes).

        Preferred path (default `stable_raw` codec): wraps `t` in a
        TensorMemoryObj backed by the head CPU allocator's slab when one is
        available; this is the zero-bytes-roundtrip path. Falls back to
        BytesMemoryObj for the `torchsave` and in-payload-header `stable`
        codecs which need a serialized representation on disk.
        """
        from stratacache.backend.cpu.factory import cpu_memory_obj_from_tensor

        if self._tensor_codec == "stable_raw" or (
            self._tensor_codec == "stable" and not self._tensor_header_in_payload
        ):
            attrs = dict(base_attrs)
            attrs.setdefault("tensor_codec", "stable_raw")
            attrs.setdefault("tensor_dtype", str(t.dtype).replace("torch.", ""))
            attrs.setdefault("tensor_shape", [int(x) for x in t.shape])
            meta = ArtifactMeta(artifact_type=ArtifactType.KV_BLOCKS, attrs=attrs)
            mo = cpu_memory_obj_from_tensor(
                t,
                meta,
                allocator=self._engine.get_cpu_allocator(),
            )
            return mo, mo.get_size()

        # Legacy / framed encoders still need a bytes blob.
        payload, tattrs = self._encode_tensor(t)
        attrs = dict(base_attrs)
        attrs.update(tattrs)
        meta = ArtifactMeta(artifact_type=ArtifactType.KV_BLOCKS, attrs=attrs)
        return BytesMemoryObj(payload, meta), len(payload)

    def _decode_tensor(self, b: bytes, device: "torch.device", meta: Optional[ArtifactMeta] = None) -> "torch.Tensor":
        attrs = dict(getattr(meta, "attrs", {}) or {}) if meta is not None else {}
        codec = str(attrs.get("tensor_codec", ""))
        if codec == "torchsave" or self._tensor_codec == "torchsave":
            return _bytes_to_tensor(b, device=device)
        if codec == "stable_raw":
            dtype_str = str(attrs.get("tensor_dtype", ""))
            shape = attrs.get("tensor_shape", [])
            if dtype_str and isinstance(shape, list):
                return _decode_tensor_raw_payload(b, dtype_str=dtype_str, shape=shape, device=device)
        # Backward compatibility: old stable payload with in-band header.
        return _decode_tensor_stable(b, device=device)

    def _try_pipelined_scatter_bundleT(
        self,
        mo: MemoryObj,
        layer_items: list[tuple[int, "torch.Tensor"]],
        sm_slice: "torch.Tensor",
    ) -> bool:
        """
        A1+A2 load fast path. When `mo` is a slab-backed bundleT
        TensorMemoryObj of shape [exp, ...per-layer...], scatter every
        layer into the GPU paged tensors via `MultiLayerPagedConnector`.

        Returns True on success; False (and the caller falls back to the
        legacy path) when the pipeline can't run.
        """
        if self._gpu_paged is None:
            return False
        host = mo.tensor
        if host is None:
            return False
        try:
            layers_list = [kv for _, kv in layer_items]
            self._gpu_paged.scatter_chunk(layers_list, sm_slice, host)
            self._gpu_paged.synchronize_load()
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Pipelined scatter failed; falling back to legacy path.")
            return False

    def _decode_tensor_from_mo(
        self, mo: MemoryObj, device: "torch.device"
    ) -> "torch.Tensor":
        """
        Zero-copy fast path: if the MemoryObj already exposes a typed
        tensor view (slab-backed TensorMemoryObj), copy it straight to
        `device`. Otherwise fall back to the bytes-codec decoder.
        """
        t = mo.tensor
        if t is not None:
            # Order the H2D back to GPU after any in-flight D2H from the
            # producer, but do it via stream-side wait so the host doesn't
            # block. When producer and consumer share the default stream
            # (the common case) this is effectively a no-op.
            mo.wait_pending_on_stream()
            # The slab tensor must outlive the device tensor; .to(device)
            # already produces a fresh device-side allocation.
            if t.device == device:
                return t.clone()
            return t.to(device, non_blocking=True)
        return self._decode_tensor(
            mo.byte_array,
            device=device,
            meta=mo.metadata.artifact_meta,
        )

    # ---------- scheduler side ----------

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> int:
        _t0_prof = time.perf_counter()
        if self._debug and not self._dbg_logged_sched_pid:
            logger.info(
                "StrataCache(vLLM/sched) entry pid=%d impl_id=%s chain_id=%s",
                os.getpid(),
                hex(id(self)),
                hex(id(self._chain)),
            )
            self._dbg_logged_sched_pid = True
        token_ids = _extract_token_ids(request)
        if not token_ids:
            _prof_record("scheduler.get_num_new_matched_tokens", time.perf_counter() - _t0_prof)
            return 0
        # Chunked prefix reuse:
        # Check cached KV at chunk boundaries; stop at first miss.
        n = len(token_ids)
        cs = max(1, int(self._chunk_size))
        boundaries = list(range(cs, (n // cs) * cs + 1, cs))
        # Optionally include the last partial chunk boundary.
        if self._save_partial_chunks and (n % cs != 0):
            boundaries.append(n)
        self._stats["sched_calls"] += 1
        self._stats["prompt_tokens_total"] += n
        self._stats["gpu_hit_tokens_total"] += int(num_computed_tokens)

        # Best-effort per-request gpu/total accounting (for req_finished log).
        rid = getattr(request, "request_id", None) or getattr(request, "req_id", None)
        if rid is not None:
            rd = _rs_get(str(rid))
            rd["total"] = max(int(rd.get("total", 0)), int(n))
            rd["gpu"] = max(int(rd.get("gpu", 0)), int(num_computed_tokens))

        if not boundaries:
            self._stats["manifest_misses"] += 1
            return 0

        rid_str = str(rid) if rid is not None else None
        sig = _token_sig(token_ids)
        st = self._match_state_by_req.get(rid_str) if rid_str is not None else None
        if (
            st is None
            or st.get("sig") != sig
            or int(st.get("chunk_size", 0)) != cs
            or list(st.get("boundaries", [])) != boundaries
        ):
            ph_map = _prefix_hashes(token_ids, boundaries)
            st = {
                "sig": sig,
                "chunk_size": int(cs),
                "boundaries": list(boundaries),
                "ph_map": ph_map,
                "hit_end": 0,
                "next_idx": 0,
                "hit_segs": [],
            }
            if rid_str is not None:
                self._match_state_by_req[rid_str] = st
        else:
            ph_map = dict(st.get("ph_map", {}) or {})

        cached_tokens = int(st.get("hit_end", 0))
        # Track which tier each chunk boundary hits in (for per-backend accounting).
        segs: list[tuple[str, int, int]] = list(st.get("hit_segs", []) or [])  # (tier_name, start, end)
        prev_end = int(cached_tokens)
        start_idx = int(st.get("next_idx", 0))
        for i in range(start_idx, len(boundaries)):
            end = int(boundaries[i])
            pref = ph_map.get(end)
            if pref is None:
                break
            # bundle_layers=True is the default and only writes bundleT;
            # bundle_layers=False writes per-layer artifacts. One contains
            # against the format we actually wrote is enough.
            if self._bundle_layers:
                lid0 = self._chunk_bundle_tensor_id(pref, end)
            else:
                lid0 = self._chunk_layer_id(pref, end, 0)
            hit_tier = self._engine.contains(lid0).hit_tier
            if hit_tier is None:
                self._stats["manifest_misses"] += 1
                st["next_idx"] = i
                break
            tier = self._tier_names[int(hit_tier)]
            self._stats["manifest_hits"] += 1
            self._io_by_tier[tier]["manifest_hits"] += 1
            cached_tokens = end
            segs.append((tier, int(prev_end), int(end)))
            prev_end = int(end)
            st["hit_end"] = int(end)
            st["next_idx"] = int(i + 1)
        st["hit_segs"] = segs

        if cached_tokens <= 0:
            _prof_record("scheduler.get_num_new_matched_tokens", time.perf_counter() - _t0_prof)
            return 0

        # Full prompt hit: subtract 1 token to force a logits recompute.
        if cached_tokens == n:
            cached_tokens = max(0, cached_tokens - 1)

        matched = max(0, int(cached_tokens) - int(num_computed_tokens))
        self._stats["external_matched_tokens_total"] += matched

        # Per-request hit breakdown (gpu + external tiers). Best-effort and set once.
        rid = getattr(request, "request_id", None) or getattr(request, "req_id", None)
        if rid is not None:
            rid = str(rid)
            rd = _rs_get(rid)
            # Record "once" to avoid later chunked-prefill updates skewing the per-request view.
            if not bool(rd.get("set_once", False)):
                rd["set_once"] = True
                rd["total"] = int(n)
                rd["gpu"] = int(num_computed_tokens)

                # External tokens are the part beyond what vLLM already has.
                # Use chunk-aligned start so the boundaries match what was stored.
                cs = max(1, int(self._chunk_size))
                ext_start = (int(num_computed_tokens) // cs) * cs
                ext_end = int(cached_tokens)
                tiers: dict[str, int] = {}
                if ext_end > ext_start:
                    for tname, s, e in segs:
                        a = max(int(s), int(ext_start))
                        b = min(int(e), int(ext_end))
                        if b > a:
                            tiers[tname] = int(tiers.get(tname, 0)) + int(b - a)
                rd["tiers"] = tiers

        _prof_record("scheduler.get_num_new_matched_tokens", time.perf_counter() - _t0_prof)
        return matched

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        req_id = getattr(request, "request_id", None) or getattr(request, "req_id", None)
        if req_id is None:
            return
        token_ids = _extract_token_ids(request)

        block_ids = _extract_block_ids(blocks)

        with self._lock:
            self._alloc_blocks[str(req_id)] = block_ids
            self._num_external_tokens[str(req_id)] = int(num_external_tokens)
            self._prompt_tokens[str(req_id)] = token_ids
            # Salvage the ph_map computed during get_num_new_matched_tokens
            # before we drop the probe state, so build_connector_meta can
            # forward it to the worker without re-hashing per step.
            cached_st = self._match_state_by_req.pop(str(req_id), None)
            if cached_st is not None:
                ph = cached_st.get("ph_map") or {}
                if ph:
                    self._ph_map_by_req[str(req_id)] = dict(ph)

        # No INFO logging here; per-request summary is emitted on request_finished.
        # Track total tokens best-effort.
        rd = _rs_get(str(req_id))
        rd["total"] = max(int(rd.get("total", 0)), int(len(token_ids)))

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        _t0_prof = time.perf_counter()
        meta = StrataConnectorMetadata()

        def _add_req(
            *,
            req_id: str,
            prompt_token_ids: list[int],
            block_ids: list[int],
            num_computed_tokens_before_step: int,
            num_scheduled_tokens_this_step: int,
            num_external_tokens_to_load: int,
        ) -> None:
            t0_req = time.perf_counter()
            if not prompt_token_ids or not block_ids:
                return

            # slot mapping is linear slots: block_id * block_size + offset
            t0_slot = time.perf_counter()
            block_size = int(self._block_size)
            blocks_seq = list(block_ids)
            # Cache slot_mapping by req_id and extend only when new blocks are appended.
            prev_blocks = self._slot_blocks_by_req.get(req_id)
            prev_sm = self._slot_mapping_by_req.get(req_id)
            if prev_blocks is not None and prev_sm is not None and prev_blocks == blocks_seq:
                slot_mapping = prev_sm
            else:
                # Try fast extend: new list is old prefix + appended unique blocks.
                if prev_blocks and prev_sm is not None and blocks_seq[: len(prev_blocks)] == prev_blocks:
                    new_blocks = blocks_seq[len(prev_blocks) :]
                    if new_blocks:
                        block_ids_t = torch.tensor(new_blocks, dtype=torch.long)
                        new_sm = (self._slot_offs.reshape(1, block_size) + block_ids_t.reshape(-1, 1) * block_size).flatten()
                        slot_mapping = torch.cat([prev_sm, new_sm], dim=0)
                    else:
                        slot_mapping = prev_sm
                else:
                    block_ids_t = torch.tensor(blocks_seq, dtype=torch.long)
                    slot_mapping = (self._slot_offs.reshape(1, block_size) + block_ids_t.reshape(-1, 1) * block_size).flatten()
                self._slot_blocks_by_req[req_id] = blocks_seq
                self._slot_mapping_by_req[req_id] = slot_mapping
            _prof_record("scheduler.build_connector_meta.add_req.slot_mapping", time.perf_counter() - t0_slot)

            # Save only KV that is actually computed/available in this step.
            t0_calc = time.perf_counter()
            computed_end = int(num_computed_tokens_before_step) + int(num_scheduled_tokens_this_step)
            save_cap = min(int(computed_end), len(prompt_token_ids), int(slot_mapping.numel()))
            if save_cap < 0:
                save_cap = 0

            cs = max(1, int(self._chunk_size))
            # Only attempt external loading when vLLM asked for it (num_external_tokens_to_load>0).
            # For scheduled_cached_reqs this is always 0, and repeatedly "loading"
            # would destroy latency. Saving still happens for chunked prefill.
            if int(num_external_tokens_to_load) > 0:
                # vLLM passes `num_external_tokens_to_load` as "additional tokens
                # to load beyond `num_computed_tokens_before_step`". Convert to a
                # total cached prefix length so the worker can skip what vLLM already has.
                vllm_cached = min(len(prompt_token_ids), max(0, int(num_computed_tokens_before_step) - int(num_external_tokens_to_load)))
                cached_total = vllm_cached + max(0, int(num_external_tokens_to_load))
                cached_total = min(int(cached_total), len(prompt_token_ids))
                load_ends = list(range(cs, (cached_total // cs) * cs + 1, cs))
                if self._save_partial_chunks and (cached_total % cs != 0) and cached_total > 0:
                    # get_num_new_matched_tokens subtracts 1 from a full-prompt hit to force
                    # logit recompute (e.g. cached_total=15159 but stored boundary is 15160=n).
                    # If cached_total+1 is a valid stored boundary (chunk-aligned or == n),
                    # use it so the worker fetches the artifact that was actually written.
                    n_prompt = len(prompt_token_ids)
                    candidate = int(cached_total)
                    if (candidate + 1) % cs == 0 or (candidate + 1) == n_prompt:
                        candidate = candidate + 1
                        cached_total = candidate  # keep tokens_to_load consistent with load_ends
                    load_ends.append(candidate)
            else:
                vllm_cached = 0
                cached_total = 0
                load_ends = []
            save_len = int(save_cap)

            # Chunk boundaries for saving: allow full chunks, plus optional last partial boundary.
            save_ends = list(range(cs, (save_len // cs) * cs + 1, cs))
            if self._save_partial_chunks and (save_len % cs != 0) and save_len > 0:
                save_ends.append(save_len)
            _prof_record("scheduler.build_connector_meta.add_req.compute_fields", time.perf_counter() - t0_calc)

            # Forward whatever ph_map was salvaged from
            # `get_num_new_matched_tokens` (kept across steps in
            # `_ph_map_by_req`). Do NOT pre-compute missing boundaries
            # here - this method runs per active req per step and a full
            # SHA256 sweep over the prompt would dominate scheduler
            # latency. Worker recomputes once if needed.
            cached_ph = self._ph_map_by_req.get(str(req_id), {})

            t0_meta = time.perf_counter()
            meta.requests.append(
                StrataKVReq(
                    req_id=str(req_id),
                    token_ids=prompt_token_ids,
                    slot_mapping=slot_mapping[: len(prompt_token_ids)],
                    tokens_to_load=int(cached_total),
                    vllm_cached_tokens=int(vllm_cached),
                    tokens_to_save=save_len,
                    # prompt_hash is currently unused on worker hot path.
                    prompt_hash="",
                    chunk_ends_to_load=load_ends,
                    chunk_ends_to_save=save_ends,
                    ph_map=cached_ph,
                )
            )
            _prof_record("scheduler.build_connector_meta.add_req.append_meta", time.perf_counter() - t0_meta)
            _prof_record("scheduler.build_connector_meta.add_req.total", time.perf_counter() - t0_req)

        # In vLLM v1 with chunked prefill, prompt KV is computed across multiple
        # steps while the request lives in scheduled_cached_reqs. We must include
        # both scheduled_new and scheduled_cached here.
        scheduled_new = getattr(scheduler_output, "scheduled_new_reqs", []) or []
        num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}

        t0_new = time.perf_counter()
        for r in scheduled_new:
            rid = getattr(r, "req_id", None) or getattr(r, "request_id", None)
            if rid is None:
                continue
            rid = str(rid)
            with self._lock:
                prompt_token_ids = self._prompt_tokens.get(rid, [])
                ext = int(self._num_external_tokens.get(rid, 0))
            # Prefer block ids from scheduler output if present (vLLM provides it).
            block_ids = []
            blk = getattr(r, "block_ids", None)
            if blk:
                # vLLM uses ( [block_ids], ) for 1 kv-cache-group.
                if isinstance(blk, tuple) and len(blk) > 0:
                    block_ids = [int(x) for x in (blk[0] or [])]
            if not block_ids:
                with self._lock:
                    block_ids = list(self._alloc_blocks.get(rid, []))
            computed_before = int(getattr(r, "num_computed_tokens", 0) or 0)
            scheduled_now = int(num_scheduled_tokens.get(rid, 0) or 0)
            _add_req(
                req_id=rid,
                prompt_token_ids=prompt_token_ids,
                block_ids=block_ids,
                num_computed_tokens_before_step=computed_before,
                num_scheduled_tokens_this_step=scheduled_now,
                num_external_tokens_to_load=ext,
            )
        _prof_record("scheduler.build_connector_meta.loop_new", time.perf_counter() - t0_new)

        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            t0_cached = time.perf_counter()
            req_ids = list(getattr(cached, "req_ids", []) or [])
            new_block_ids = list(getattr(cached, "new_block_ids", []) or [])
            num_comp = list(getattr(cached, "num_computed_tokens", []) or [])
            all_token_ids = getattr(cached, "all_token_ids", {}) or {}
            for i, rid in enumerate(req_ids):
                rid = str(rid)
                with self._lock:
                    prompt_token_ids = self._prompt_tokens.get(rid, [])
                if not prompt_token_ids:
                    # Fallback: vLLM provides all_token_ids for requests not scheduled
                    # in prev step. This may include output tokens; we keep it best-effort.
                    prompt_token_ids = list(all_token_ids.get(rid, []) or [])
                blk = new_block_ids[i] if i < len(new_block_ids) else None
                # For cached/running requests, vLLM only reports *new* block ids.
                # We must combine them with the existing allocation to build a
                # slot_mapping that covers the full prompt prefix.
                with self._lock:
                    base = list(self._alloc_blocks.get(rid, []))
                block_ids: list[int] = []
                if blk:
                    if isinstance(blk, tuple) and len(blk) > 0 and blk[0]:
                        block_ids = [int(x) for x in blk[0]]
                    elif isinstance(blk, list) and blk and isinstance(blk[0], list):
                        block_ids = [int(x) for x in blk[0]]
                    elif isinstance(blk, list):
                        block_ids = [int(x) for x in blk]
                if base:
                    # Append new blocks (if any) while preserving order.
                    if not block_ids:
                        block_ids = base
                    else:
                        seen = set(int(x) for x in base)
                        merged = list(int(x) for x in base)
                        for x in block_ids:
                            xi = int(x)
                            if xi not in seen:
                                merged.append(xi)
                                seen.add(xi)
                        block_ids = merged
                # Persist merged block ids for future steps.
                if block_ids:
                    with self._lock:
                        self._alloc_blocks[rid] = list(block_ids)
                computed_before = int(num_comp[i]) if i < len(num_comp) else 0
                scheduled_now = int(num_scheduled_tokens.get(rid, 0) or 0)
                # No external load for cached/running requests; external is only
                # decided at request start (num_computed_tokens==0).
                _add_req(
                    req_id=rid,
                    prompt_token_ids=prompt_token_ids,
                    block_ids=block_ids,
                    num_computed_tokens_before_step=computed_before,
                    num_scheduled_tokens_this_step=scheduled_now,
                    num_external_tokens_to_load=0,
                )
            _prof_record("scheduler.build_connector_meta.loop_cached", time.perf_counter() - t0_cached)
        _prof_record("scheduler.build_connector_meta", time.perf_counter() - _t0_prof)
        return meta

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        _t0_prof = time.perf_counter()
        _ = block_ids
        # Best-effort cleanup of per-request save watermarks.
        rid = getattr(request, "request_id", None) or getattr(request, "req_id", None)
        if rid is not None:
            rid = str(rid)
            # Emit one concise per-request stats line (and update cumulative).
            if self._log_stats:
                rs = _rs_pop(rid) or {}
                total = int(rs.get("total", 0))
                gpu = int(rs.get("gpu", 0))
                tiers: dict[str, int] = dict(rs.get("tiers", {}) or {})
                # Ensure we always print known tiers, even if 0 (for stable parsing).
                for tname in getattr(self, "_tier_names", []) or []:
                    tiers.setdefault(str(tname), 0)
                ext = sum(int(v) for v in tiers.values())
                # Keep invariants sane.
                if total > 0:
                    gpu = max(0, min(gpu, total))
                    if gpu + ext > total:
                        ext = max(0, total - gpu)
                        # Clamp tiers proportionally is overkill; just cap total ext.
                _rs_add_cum(total=total, gpu=gpu, tiers=tiers)
                rates = _rs_cum_rates()
                parts = [f"gpu={gpu}"]
                for tname in sorted(tiers.keys()):
                    parts.append(f"{tname}={int(tiers[tname])}")
                logger.info(
                    "StrataCache(vLLM) req_done: req=%s tok=%d %s | cum_hit_rate: %s",
                    rid,
                    total,
                    " ".join(parts),
                    " ".join(f"{k}={v:.4f}" for k, v in sorted(rates.items())),
                )
            # Remove all layers for this request.
            self._saved_upto_by_req_layer.pop((rid, -1), None)
            n_layers = int(self._expected_num_layers or 0)
            if n_layers > 0:
                for li in range(n_layers):
                    self._saved_upto_by_req_layer.pop((rid, li), None)
            # Cleanup any pending bundles.
            req_pending = self._pending_keys_by_req.pop(rid, set())
            for pkey in req_pending:
                self._pending_save.pop(pkey, None)
                self._pending_layers.pop(pkey, None)
            self._pending_ends_by_req.pop(rid, None)
            self._slot_mapping_by_req.pop(rid, None)
            self._slot_blocks_by_req.pop(rid, None)
            # Scheduler-side state cleanup to keep per-request maps bounded.
            with self._lock:
                self._prompt_tokens.pop(rid, None)
                self._alloc_blocks.pop(rid, None)
                self._num_external_tokens.pop(rid, None)
                self._match_state_by_req.pop(rid, None)
                self._ph_map_by_req.pop(rid, None)
            self._loaded_or_attempted.discard(rid)
        _prof_record("scheduler.request_finished", time.perf_counter() - _t0_prof)
        return False, None

    # ---------- worker side ----------

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        _t0_prof = time.perf_counter()
        if self._debug and not self._dbg_logged_worker_pid:
            logger.info(
                "StrataCache(vLLM/worker) entry pid=%d impl_id=%s chain_id=%s",
                os.getpid(),
                hex(id(self)),
                hex(id(self._chain)),
            )
            self._dbg_logged_worker_pid = True
        connector_meta = None
        if hasattr(self._parent, "_get_connector_metadata"):
            connector_meta = self._parent._get_connector_metadata()
        if connector_meta is None:
            # vLLM has used a few kwarg names across minor versions.
            connector_meta = (
                kwargs.get("connector_metadata")
                or kwargs.get("kv_connector_metadata")
                or kwargs.get("kv_connector_meta")
                or kwargs.get("connector_meta")
            )
        if self._debug and not self._dbg_logged_load_meta:
            keys = sorted(list(kwargs.keys()))
            mtype = type(connector_meta).__name__ if connector_meta is not None else "None"
            nreq = len(getattr(connector_meta, "requests", []) or []) if connector_meta is not None else 0
            logger.debug(
                "StrataCache(vLLM) load_meta(pid=%d): meta_type=%s nreq=%d kw=%s",
                os.getpid(),
                mtype,
                int(nreq),
                ",".join(keys),
            )
            self._dbg_logged_load_meta = True
        if not isinstance(connector_meta, StrataConnectorMetadata):
            _prof_record("worker.start_load_kv", time.perf_counter() - _t0_prof)
            return
        # Normal logging is via periodic stats; debug may log per-request IO.

        kv_caches = kwargs.get("kv_caches")
        if kv_caches is None:
            # vLLM v0.13: discover KV layers from ForwardContext.
            if len(self._kv_caches) == 0:
                self._init_kv_caches_from_forward_context(forward_context)
            if len(self._kv_caches) > 0:
                kv_caches = self._kv_caches
            else:
                # Best-effort legacy fallbacks.
                kv_caches = getattr(forward_context, "kv_caches", None) or getattr(forward_context, "kv_cache", None)
        if kv_caches is None:
            _prof_record("worker.start_load_kv", time.perf_counter() - _t0_prof)
            return

        # kv_caches may be list[Tensor] or dict[str, Tensor]
        for req in connector_meta.requests:
            cached_total = int(req.tokens_to_load)
            if cached_total <= 0:
                continue
            # Only attempt external load once per request id (avoid repeated
            # work across steps; huge latency impact).
            if req.req_id in self._loaded_or_attempted:
                continue
            vllm_cached = max(0, int(getattr(req, "vllm_cached_tokens", 0)))
            cs = max(1, int(self._chunk_size))
            # IMPORTANT: vLLM may already have a prefix in GPU cache (its own
            # prefix cache). Only load the range beyond that prefix, aligned
            # down to chunk boundary.
            load_start = (vllm_cached // cs) * cs
            if load_start >= cached_total:
                continue

            # Load full chunks and (optionally) a last partial chunk at e==cached_total.
            chunk_ends = [
                e
                for e in (req.chunk_ends_to_load or [])
                if e <= cached_total and e > load_start and (e % cs == 0 or (self._save_partial_chunks and e == cached_total))
            ]
            if not chunk_ends:
                continue
            # Reuse scheduler-computed ph_map; only fall back to a local
            # SHA256 sweep for boundaries the scheduler didn't pre-hash.
            ph_map = {e: req.ph_map[e] for e in chunk_ends if e in req.ph_map}
            missing = [e for e in chunk_ends if e not in ph_map]
            if missing:
                ph_map.update(_prefix_hashes(req.token_ids, missing))

            # We'll count "actual loaded tokens" best-effort based on layer0 progress.
            loaded_tokens_for_req = 0

            # ---- bundle path (fast): one fetch per chunk_end, then scatter per layer ----
            if self._bundle_layers:
                # Prepare layer views
                if isinstance(kv_caches, dict):
                    layer_items = [( _layer_index(str(ln), 0), kv) for ln, kv in kv_caches.items()]
                else:
                    layer_items = list(enumerate(list(kv_caches)))
                layer_items.sort(key=lambda x: x[0])

                chunk_start = int(load_start)
                for end in chunk_ends:
                    t_fetch0 = time.perf_counter()
                    pref = ph_map.get(end)
                    if pref is None:
                        break
                    sm_slice = req.slot_mapping[chunk_start:end]
                    if int(sm_slice.numel()) != int(end - chunk_start):
                        break
                    # Prefer new bundleT tensor format.
                    aid = self._chunk_bundle_tensor_id(pref, end)
                    # Build a (dtype, shape) hint from the cached KV
                    # template so a CXL hit can come back as a typed
                    # TensorMemoryObj (CXL doesn't persist user-meta).
                    hint_dtype, hint_shape = self._bundle_load_hint(
                        int(self._expected_num_layers or 0),
                        int(end - chunk_start),
                    )
                    try:
                        fr = self._engine.load(
                            aid,
                            promote=False,
                            dtype=hint_dtype,
                            shape=hint_shape,
                        )
                        _prof_record("worker.start_load_kv.bundle.fetch", time.perf_counter() - t_fetch0)
                        t_dec0 = time.perf_counter()
                        stacked = self._decode_tensor_from_mo(fr.memory_obj, device=layer_items[0][1].device)
                        _prof_record("worker.start_load_kv.bundleT.decode_tensor", time.perf_counter() - t_dec0)
                    except ArtifactNotFound:
                        # bundleT is the only KV format we write under
                        # bundle_layers=True; on miss, just stop the chain.
                        break
                    except Exception:  # noqa: BLE001
                        break

                    tier = self._tier_names[fr.hit_tier]
                    # A1+A2 fast scatter: one stream-overlapped batch.
                    if not self._try_pipelined_scatter_bundleT(
                        fr.memory_obj, layer_items, sm_slice
                    ):
                        # Legacy: per-layer Python scatter from stacked tensor.
                        for layer_idx, kv_layer in layer_items:
                            li = int(layer_idx)
                            if stacked.dim() < 1 or li >= int(stacked.size(0)):
                                continue
                            try:
                                t_sc0 = time.perf_counter()
                                _scatter_by_slots(kv_layer, sm_slice, stacked[li])
                                _prof_record("worker.start_load_kv.bundleT.scatter", time.perf_counter() - t_sc0)
                            except RuntimeError:
                                break
                    # Attribution: count one "loaded chunk" per chunk (not per layer)
                    self._stats["loaded_chunks"] += 1
                    self._stats["bytes_loaded"] += fr.memory_obj.get_size()
                    self._io_by_tier[tier]["bytes_loaded"] += fr.memory_obj.get_size()
                    self._io_by_tier[tier]["chunks_loaded"] += 1
                    dt = int(end - chunk_start)
                    loaded_tokens_for_req += dt
                    self._io_by_tier[tier]["tokens_loaded"] += dt
                    chunk_start = end
            else:
                # ---- legacy path: fetch per layer per chunk ----
                if isinstance(kv_caches, dict):
                    for layer_name, kv_layer in kv_caches.items():
                        layer_idx = _layer_index(str(layer_name), 0)
                        chunk_start = int(load_start)
                        for end in chunk_ends:
                            pref = ph_map.get(end)
                            if pref is None:
                                break
                            sm_slice = req.slot_mapping[chunk_start:end]
                            if int(sm_slice.numel()) != int(end - chunk_start):
                                break
                            aid = self._chunk_layer_id(pref, end, layer_idx)
                            try:
                                fr = self._engine.load(aid, promote=False)
                            except ArtifactNotFound:
                                break
                            gathered = self._decode_tensor_from_mo(fr.memory_obj, device=kv_layer.device)
                            try:
                                _scatter_by_slots(kv_layer, sm_slice, gathered)
                            except RuntimeError:
                                break
                            self._stats["loaded_chunks"] += 1
                            self._stats["bytes_loaded"] += fr.memory_obj.get_size()
                            tier = self._tier_names[fr.hit_tier]
                            self._io_by_tier[tier]["bytes_loaded"] += fr.memory_obj.get_size()
                            self._io_by_tier[tier]["chunks_loaded"] += 1
                            if layer_idx == 0:
                                dt = int(end - chunk_start)
                                loaded_tokens_for_req += dt
                                self._io_by_tier[tier]["tokens_loaded"] += dt
                            chunk_start = end
                else:
                    for layer_idx, kv_layer in enumerate(list(kv_caches)):
                        chunk_start = int(load_start)
                        for end in chunk_ends:
                            pref = ph_map.get(end)
                            if pref is None:
                                break
                            sm_slice = req.slot_mapping[chunk_start:end]
                            if int(sm_slice.numel()) != int(end - chunk_start):
                                break
                            aid = self._chunk_layer_id(pref, end, layer_idx)
                            try:
                                fr = self._engine.load(aid, promote=False)
                            except ArtifactNotFound:
                                break
                            gathered = self._decode_tensor_from_mo(fr.memory_obj, device=kv_layer.device)
                            try:
                                _scatter_by_slots(kv_layer, sm_slice, gathered)
                            except RuntimeError:
                                break
                            self._stats["loaded_chunks"] += 1
                            self._stats["bytes_loaded"] += fr.memory_obj.get_size()
                            tier = self._tier_names[fr.hit_tier]
                            self._io_by_tier[tier]["bytes_loaded"] += fr.memory_obj.get_size()
                            self._io_by_tier[tier]["chunks_loaded"] += 1
                            if layer_idx == 0:
                                dt = int(end - chunk_start)
                                loaded_tokens_for_req += dt
                                self._io_by_tier[tier]["tokens_loaded"] += dt
                            chunk_start = end

            self._stats["external_loaded_tokens_total"] += int(loaded_tokens_for_req)
            self._loaded_or_attempted.add(req.req_id)
        _prof_record("worker.start_load_kv", time.perf_counter() - _t0_prof)

        # Do not log per-step stats; we log once per request in request_finished.

    def wait_for_layer_load(self, layer_name: str) -> None:
        _ = layer_name
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs) -> None:
        _t0_prof = time.perf_counter()
        _ = (attn_metadata, kwargs)
        if self._debug and not self._dbg_logged_worker_pid:
            logger.info(
                "StrataCache(vLLM/worker) entry pid=%d impl_id=%s chain_id=%s",
                os.getpid(),
                hex(id(self)),
                hex(id(self._chain)),
            )
            self._dbg_logged_worker_pid = True
        connector_meta = None
        if hasattr(self._parent, "_get_connector_metadata"):
            connector_meta = self._parent._get_connector_metadata()
        if connector_meta is None:
            connector_meta = kwargs.get("connector_metadata") or kwargs.get("kv_connector_metadata")
        if not isinstance(connector_meta, StrataConnectorMetadata):
            _prof_record("worker.save_kv_layer", time.perf_counter() - _t0_prof)
            return

        idx = _layer_index(str(layer_name), fallback=0)

        # Normal logging is via periodic stats; debug may log per-request IO.

        for req in connector_meta.requests:
            n = int(req.tokens_to_save)
            if n <= 0:
                continue
            cs = max(1, int(self._chunk_size))
            # Fast path: in bundle mode, do minimal work here and defer heavy gather/encode/store
            # to wait_for_save (called once per step). This avoids blocking the attention forward.
            if self._bundle_layers:
                exp = int(self._expected_num_layers or 0)
                if exp <= 0:
                    # If layer count is unknown, keep legacy behavior by falling back to per-layer store.
                    exp = 0
                bundle_key = (req.req_id, -1)
                prev_bundle = int(self._saved_upto_by_req_layer.get(bundle_key, 0))
                if prev_bundle >= n:
                    continue
                # Only layer0 computes boundaries/hashes and creates pending entries.
                if int(idx) == 0:
                    chunk_ends = [
                        e
                        for e in (req.chunk_ends_to_save or [])
                        if e <= n and (e % cs == 0 or (self._save_partial_chunks and e == n))
                    ]
                    chunk_ends = [e for e in chunk_ends if e > prev_bundle]
                    if not chunk_ends:
                        continue
                    self._pending_ends_by_req[str(req.req_id)] = list(int(e) for e in chunk_ends)
                    ph_map = {e: req.ph_map[e] for e in chunk_ends if e in req.ph_map}
                    missing = [e for e in chunk_ends if e not in ph_map]
                    if missing:
                        ph_map.update(_prefix_hashes(req.token_ids, missing))
                    chunk_start = int(prev_bundle)
                    for end in chunk_ends:
                        pref = ph_map.get(end)
                        if pref is None:
                            break
                        pkey = (req.req_id, int(end))
                        ent = self._pending_save.get(pkey)
                        if ent is None:
                            ent = {
                                "pref": pref,
                                "end": int(end),
                                "chunk_start": int(chunk_start),
                                "slot_mapping": req.slot_mapping[chunk_start:end],
                                "layers": {},
                            }
                            self._pending_save[pkey] = ent
                            self._pending_layers[pkey] = set()
                            self._pending_keys_by_req.setdefault(str(req.req_id), set()).add(pkey)
                        ent["layers"][0] = kv_layer
                        self._pending_layers[pkey].add(0)
                        chunk_start = int(end)
                else:
                    ends = self._pending_ends_by_req.get(str(req.req_id), []) or []
                    if not ends:
                        continue
                    for end in ends:
                        pkey = (req.req_id, int(end))
                        ent = self._pending_save.get(pkey)
                        if ent is None:
                            continue
                        ent["layers"][int(idx)] = kv_layer
                        self._pending_layers[pkey].add(int(idx))
                continue

            # Non-bundle mode: keep legacy per-layer store behavior.
            layer_key = (req.req_id, idx)
            prev_layer = int(self._saved_upto_by_req_layer.get(layer_key, 0))
            if prev_layer >= n:
                continue
            chunk_ends = [
                e
                for e in (req.chunk_ends_to_save or [])
                if e <= n and (e % cs == 0 or (self._save_partial_chunks and e == n))
            ]
            chunk_ends = [e for e in chunk_ends if e > prev_layer]
            if not chunk_ends:
                continue
            ph_map = {e: req.ph_map[e] for e in chunk_ends if e in req.ph_map}
            missing = [e for e in chunk_ends if e not in ph_map]
            if missing:
                ph_map.update(_prefix_hashes(req.token_ids, missing))
            chunk_start = prev_layer
            for end in chunk_ends:
                pref = ph_map.get(end)
                if pref is None:
                    break
                gathered = _gather_by_slots(kv_layer, req.slot_mapping[chunk_start:end]).detach()
                base_attrs = {"chunk_start": chunk_start, "chunk_end": end}
                memory_obj, payload_size = self._make_kv_memory_obj(gathered, base_attrs)
                self._engine.store(
                    self._chunk_layer_id(pref, end, idx),
                    memory_obj,
                )
                self._stats["stored_chunks"] += 1
                self._stats["bytes_stored"] += payload_size
                remaining = payload_size
                for ti, tname in enumerate(self._tier_names):
                    if ti == 0:
                        self._io_by_tier[tname]["bytes_stored"] += remaining
                        self._io_by_tier[tname]["chunks_stored"] += 1
                        if idx == 0:
                            self._io_by_tier[tname]["tokens_stored"] += int(end - chunk_start)
                        continue
                    if ti - 1 < len(self._chain.links) and self._chain.links[ti - 1] == LinkPolicy.WRITE_THROUGH:
                        self._io_by_tier[tname]["bytes_stored"] += remaining
                        self._io_by_tier[tname]["chunks_stored"] += 1
                        if idx == 0:
                            self._io_by_tier[tname]["tokens_stored"] += int(end - chunk_start)
                    else:
                        break
                if idx == 0:
                    self._engine.store(
                        self._chunk_manifest_id(pref, end),
                        BytesMemoryObj(
                            b"",
                            ArtifactMeta(
                                artifact_type=ArtifactType.KV_BLOCKS,
                                attrs={"tokens": end},
                            ),
                        ),
                    )
                    # Internal write-back control: not part of StorageEngine public API.
                    self._engine.chain.flush(self._chunk_manifest_id(pref, end))
                chunk_start = end
            self._saved_upto_by_req_layer[layer_key] = max(prev_layer, max(chunk_ends))

            # No per-request save logging (too noisy); stats are emitted on request_finished.
        _prof_record("worker.save_kv_layer", time.perf_counter() - _t0_prof)

    def _maybe_log_stats(self) -> None:
        """
        Rate-limited logging of connector stats.

        Avoids the previous bug where total_io==0 would satisfy modulo checks and spam logs.
        """
        now = time.time()
        total_io = int(self._stats["stored_chunks"] + self._stats["loaded_chunks"])
        if total_io <= 0 and self._stats["sched_calls"] <= 0:
            return
        if now - self._last_log_t < self._log_min_interval_s:
            return

        # log when IO progressed, or every N scheduler calls
        sched_calls = int(self._stats["sched_calls"])
        if total_io == self._last_logged_io_total and (sched_calls - self._last_logged_sched_calls) < max(1, self._log_every):
            return

        prompt_total = int(self._stats["prompt_tokens_total"])
        gpu_hit_total = int(self._stats["gpu_hit_tokens_total"])
        matched_total = int(self._stats["external_matched_tokens_total"])
        loaded_total = int(self._stats["external_loaded_tokens_total"])

        external_match_rate = 0.0 if prompt_total == 0 else (matched_total / prompt_total)
        external_load_rate = 0.0 if prompt_total == 0 else (loaded_total / prompt_total)
        gpu_hit_rate = 0.0 if prompt_total == 0 else (gpu_hit_total / prompt_total)

        # Deprecated: we no longer emit periodic stats logs. We log once per
        # request in request_finished to align with vLLM's engine logger.
        return

        self._last_log_t = now
        self._last_logged_io_total = total_io
        self._last_logged_sched_calls = sched_calls

    def wait_for_save(self) -> None:
        # v0.13.0 calls this at forward_context exit; use it as the synchronization
        # point to do heavy gather/encode/store once per chunk.
        _t0 = time.perf_counter()
        if not self._bundle_layers:
            _prof_record("worker.wait_for_save", time.perf_counter() - _t0)
            return None

        exp = int(self._expected_num_layers or 0)
        if exp <= 0:
            _prof_record("worker.wait_for_save", time.perf_counter() - _t0)
            return None

        # Commit any fully-enqueued bundles.
        keys = list(self._pending_save.keys())
        for pkey in keys:
            ent = self._pending_save.get(pkey)
            if ent is None:
                continue
            layers: dict[int, Any] = dict(ent.get("layers", {}) or {})
            if len(layers) < exp:
                continue
            req_id, end = pkey
            pref = str(ent.get("pref", ""))
            chunk_start = int(ent.get("chunk_start", 0))
            sm_slice = ent.get("slot_mapping")
            if sm_slice is None:
                continue

            # Cross-request dedup: if this chunk is already cached, skip
            # expensive gather/encode/store and only advance watermark.
            # We always write bundleT in the default config, so a single
            # contains check is sufficient.
            if self._engine.contains(self._chunk_bundle_tensor_id(pref, int(end))).exists:
                bundle_key = (req_id, -1)
                self._saved_upto_by_req_layer[bundle_key] = max(
                    int(self._saved_upto_by_req_layer.get(bundle_key, 0)),
                    int(end),
                )
                self._pending_save.pop(pkey, None)
                self._pending_layers.pop(pkey, None)
                req_set = self._pending_keys_by_req.get(str(req_id))
                if req_set is not None:
                    req_set.discard(pkey)
                    if not req_set:
                        self._pending_keys_by_req.pop(str(req_id), None)
                continue

            base_attrs = {
                "chunk_start": int(chunk_start),
                "chunk_end": int(end),
                "bundle_layers": exp,
                "bundle_format": "tensor",
            }

            # A1+A2 fast path: pipelined gather straight into a slab-backed
            # host buffer, no per-layer Python stack.
            t_g0 = time.perf_counter()
            pipelined = self._make_kv_memory_obj_pipelined(layers, sm_slice, base_attrs, exp)
            if pipelined is not None:
                memory_obj, payload_size = pipelined
                _prof_record("worker.wait_for_save.bundleT.pipelined", time.perf_counter() - t_g0)
            else:
                # Legacy: per-layer gather + stack + encode.
                gathered_layers = []
                for li in range(exp):
                    kv_layer = layers.get(li)
                    if kv_layer is None:
                        break
                    gathered_layers.append(_gather_by_slots(kv_layer, sm_slice).detach())
                if len(gathered_layers) != exp:
                    continue
                stacked = torch.stack(gathered_layers, dim=0)
                _prof_record("worker.wait_for_save.bundleT.gather_stack", time.perf_counter() - t_g0)

                t_e0 = time.perf_counter()
                memory_obj, payload_size = self._make_kv_memory_obj(stacked, base_attrs)
                _prof_record("worker.wait_for_save.bundleT.encode_tensor", time.perf_counter() - t_e0)

            t_s0 = time.perf_counter()
            self._engine.store(
                self._chunk_bundle_tensor_id(pref, int(end)),
                memory_obj,
            )
            _prof_record("worker.wait_for_save.bundleT.store", time.perf_counter() - t_s0)

            self._stats["stored_chunks"] += 1
            self._stats["bytes_stored"] += payload_size
            remaining = payload_size
            for ti, tname in enumerate(self._tier_names):
                if ti == 0:
                    self._io_by_tier[tname]["bytes_stored"] += remaining
                    self._io_by_tier[tname]["chunks_stored"] += 1
                    self._io_by_tier[tname]["tokens_stored"] += int(end - chunk_start)
                    continue
                if ti - 1 < len(self._chain.links) and self._chain.links[ti - 1] == LinkPolicy.WRITE_THROUGH:
                    self._io_by_tier[tname]["bytes_stored"] += remaining
                    self._io_by_tier[tname]["chunks_stored"] += 1
                    self._io_by_tier[tname]["tokens_stored"] += int(end - chunk_start)
                else:
                    break

            # Advance per-request bundle watermark only after commit.
            bundle_key = (req_id, -1)
            self._saved_upto_by_req_layer[bundle_key] = max(int(self._saved_upto_by_req_layer.get(bundle_key, 0)), int(end))

            # Cleanup pending entry.
            self._pending_save.pop(pkey, None)
            self._pending_layers.pop(pkey, None)
            req_set = self._pending_keys_by_req.get(str(req_id))
            if req_set is not None:
                req_set.discard(pkey)
                if not req_set:
                    self._pending_keys_by_req.pop(str(req_id), None)

        _prof_record("worker.wait_for_save", time.perf_counter() - _t0)
        return None

    def get_finished(self, finished_req_ids: set[str]):
        _ = finished_req_ids
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def get_kv_events(self) -> list:
        """
        Drain the type-agnostic backend event stream, translated to
        CacheStoreEvent / CacheRemoveEvent. Empty list when the
        translator is disabled (default).
        """
        if self._kv_event_translator is None:
            return []
        return self._kv_event_translator.drain_events()

    def shutdown(self) -> None:
        _release_chain(getattr(self, "_chain_key", ""))


class StrataCacheConnectorV1(KVConnectorBase_V1):  # type: ignore[misc]
    """
    vLLM KVConnector v1 for StrataCache.

    Example kv-transfer-config:
      {"kv_connector":"StrataCacheConnectorV1",
       "kv_connector_module_path":"stratacache.adapters.vllm.connector_v1",
       "kv_role":"kv_both"}
    """

    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None):
        # vLLM v0.13 expects connectors to accept kv_cache_config and pass it to super.
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._impl = _StrataConnectorImpl(vllm_config, role, self, kv_cache_config)

    # Worker-side
    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        return self._impl.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return self._impl.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs) -> None:
        return self._impl.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        return self._impl.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]):
        return self._impl.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._impl.get_block_ids_with_load_errors()

    def get_kv_events(self) -> list:
        return self._impl.get_kv_events()

    def shutdown(self):
        return self._impl.shutdown()

    # Scheduler-side
    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[Optional[int], bool]:
        return self._impl.get_num_new_matched_tokens(request, num_computed_tokens), False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int):
        return self._impl.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        return self._impl.build_connector_meta(scheduler_output)

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        return self._impl.request_finished(request, block_ids)
