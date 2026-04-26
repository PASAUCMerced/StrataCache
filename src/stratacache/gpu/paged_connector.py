"""
MultiLayerPagedConnector: pure-Torch port of LMCache's
`VLLMPagedMemGPUConnectorV2.to_gpu / from_gpu` (Apache-2.0).

Minimum-viable version: no custom CUDA kernel, but the rest of the
optimizations are present:

- A pre-built per-layer pointer / view table is held once per process
  rather than rediscovered per call.
- Dedicated `load_stream` and `store_stream` so PCIe traffic can overlap
  with the compute stream's attention work.
- Non-blocking `copy_` between pinned-host slabs and device buffers.
- Stacked CPU buffer for "all layers, one chunk" so we issue exactly
  `num_layers` H2D / D2H copies per chunk instead of `num_layers *
  num_chunks` separate Python store/load calls.

When CUDA isn't available the connector still works (CPU device, no
streams, copies become memcpy). That keeps tests on machines without
GPUs runnable.

The connector deliberately stays type-agnostic above the call boundary:
it accepts a stacked `host_buffer` of shape `[num_layers, slots, ...]`
and a `slot_mapping` and does not know about chunk hashes or prefix keys.
KV-specific orchestration sits in `adapters/vllm/connector_v1.py`.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class PagedConnectorStreams:
    """Per-device pair of streams used for overlapping load and store."""

    device: Any  # torch.device
    load_stream: Any  # torch.cuda.Stream | None
    store_stream: Any  # torch.cuda.Stream | None


class MultiLayerPagedConnector:
    """
    Per-process singleton that owns the per-layer KV view table and the
    overlap streams. Construction is cheap; first call lazily discovers
    the layer layout from `kv_caches` (a list or dict matching vLLM's
    `forward_context.no_compile_layers[*].kv_cache[ve]` convention).

    Public entry points:
    - `gather_chunk(kv_caches, slot_mapping, host_buffer)`:
        D2H. For each layer, gathers `slot_mapping`-indexed rows out of
        the layer's paged KV tensor and writes them into
        `host_buffer[layer_idx]`. host_buffer must be on CPU; pinned for
        true async behaviour.
    - `scatter_chunk(kv_caches, slot_mapping, host_buffer)`:
        H2D. Inverse of gather_chunk.

    Both calls run on `store_stream` / `load_stream` respectively; they
    return immediately and the caller is expected to synchronize via
    `synchronize_load()` / `synchronize_store()` before reading the
    destination.
    """

    def __init__(self) -> None:
        if torch is None:
            raise RuntimeError("torch is required for MultiLayerPagedConnector")
        self._lock = threading.RLock()
        # Cached metadata per (kv_caches identity, device).
        self._cached_layer_views: list[Any] = []
        self._cached_kv_caches_id: Optional[int] = None
        self._cached_device: Any = None
        self._streams: Optional[PagedConnectorStreams] = None

    # ---- stream / device ----

    def _ensure_streams(self, device: Any) -> PagedConnectorStreams:
        if self._streams is not None and self._streams.device == device:
            return self._streams
        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            self._streams = PagedConnectorStreams(
                device=device,
                load_stream=torch.cuda.Stream(device=device),
                store_stream=torch.cuda.Stream(device=device),
            )
        else:
            # Non-CUDA fallback: streams are None and copies happen on the
            # default stream (synchronous semantics on CPU anyway).
            self._streams = PagedConnectorStreams(
                device=device, load_stream=None, store_stream=None
            )
        return self._streams

    def synchronize_load(self) -> None:
        if self._streams is None or self._streams.load_stream is None:
            return
        # Make the compute stream wait so the next attention forward sees
        # the scattered KV without us having to host-block here.
        compute_stream = torch.cuda.current_stream(self._streams.device)
        compute_stream.wait_stream(self._streams.load_stream)

    def synchronize_store(self) -> None:
        if self._streams is None or self._streams.store_stream is None:
            return
        self._streams.store_stream.synchronize()

    # ---- layer view table ----

    def _build_layer_views(self, kv_caches: Any) -> list[Any]:
        """
        Normalise vLLM's KV cache containers (dict[name, tensor] or
        list[tensor]) into an ordered list of per-layer tensors. Each
        tensor is the per-layer paged KV memory; we keep it as-is and
        flatten the slot dim only at gather/scatter time.
        """
        if isinstance(kv_caches, dict):
            items = []
            from re import findall

            for name, tens in kv_caches.items():
                # Use the trailing integer in the layer name as the index.
                m = findall(r"(\d+)", str(name))
                idx = int(m[-1]) if m else len(items)
                items.append((idx, tens))
            items.sort(key=lambda x: x[0])
            return [t for _, t in items]
        return list(kv_caches)

    def _maybe_refresh(self, kv_caches: Any, device: Any) -> None:
        cur_id = id(kv_caches)
        if cur_id == self._cached_kv_caches_id and device == self._cached_device:
            return
        with self._lock:
            self._cached_layer_views = self._build_layer_views(kv_caches)
            self._cached_kv_caches_id = cur_id
            self._cached_device = device

    @staticmethod
    def _flatten_slots(kv_layer: Any) -> Any:
        """
        Same convention as the inline helper in the legacy connector.
        Returns a view; no copy.
        """
        if kv_layer.dim() < 2:
            raise ValueError(f"Unsupported kv_layer ndim={kv_layer.dim()}")
        if kv_layer.dim() >= 3 and kv_layer.size(0) == 2:
            b = int(kv_layer.size(1))
            s = int(kv_layer.size(2))
            return kv_layer.reshape(2, b * s, *kv_layer.shape[3:])
        b = int(kv_layer.size(0))
        s = int(kv_layer.size(1))
        return kv_layer.reshape(b * s, *kv_layer.shape[2:])

    @staticmethod
    def _maybe_contig_range(slot_mapping: Any) -> Optional[tuple[int, int]]:
        """
        Return (start, length) when slot_mapping is a contiguous increasing
        run on CPU, no -1 sentinels. Lets us short-circuit to a slice.
        """
        if torch is None or not isinstance(slot_mapping, torch.Tensor):
            return None
        if slot_mapping.device.type != "cpu" or slot_mapping.numel() == 0:
            return None
        try:
            if int(slot_mapping.min().item()) < 0:
                return None
            if slot_mapping.numel() == 1 or bool(
                torch.all((slot_mapping[1:] - slot_mapping[:-1]) == 1).item()
            ):
                return int(slot_mapping[0].item()), int(slot_mapping.numel())
        except Exception:  # noqa: BLE001
            return None
        return None

    # ---- gather (D2H) ----

    def gather_chunk(
        self,
        kv_caches: Any,
        slot_mapping: Any,
        host_buffer: Any,
    ) -> None:
        """
        For each layer, copy `slot_mapping`-indexed rows out of the
        per-layer paged KV tensor into `host_buffer[layer_idx]`.

        Shapes:
            host_buffer: [num_layers, ...] matching the per-layer chunk
                         shape (i.e. the slot dim folded out).
        """
        self._maybe_refresh(kv_caches, host_buffer.device)
        layers = self._cached_layer_views
        if len(layers) == 0:
            return
        streams = self._ensure_streams(layers[0].device)

        rng = self._maybe_contig_range(slot_mapping)
        if streams.store_stream is not None:
            # Make store_stream wait for the compute stream so the KV
            # writes from attention are visible to the D2H copies.
            compute_stream = torch.cuda.current_stream(layers[0].device)
            streams.store_stream.wait_stream(compute_stream)
            ctx: Any = torch.cuda.stream(streams.store_stream)
        else:
            ctx = _NullCtx()
        with ctx:
            for li, kv_layer in enumerate(layers):
                flat = self._flatten_slots(kv_layer)
                if rng is not None:
                    s0, ln = rng
                    if flat.dim() >= 2 and flat.size(0) == 2:
                        view = flat[:, s0 : s0 + ln]
                    else:
                        view = flat[s0 : s0 + ln]
                else:
                    slots = slot_mapping.to(device=kv_layer.device)
                    mask = slots >= 0
                    slots = slots[mask].to(dtype=torch.long)
                    if flat.dim() >= 2 and flat.size(0) == 2:
                        view = flat.index_select(1, slots).contiguous()
                    else:
                        view = flat.index_select(0, slots).contiguous()
                # view is on device; copy into the host_buffer slice.
                host_buffer[li].copy_(view, non_blocking=True)

    # ---- scatter (H2D) ----

    def scatter_chunk(
        self,
        kv_caches: Any,
        slot_mapping: Any,
        host_buffer: Any,
    ) -> None:
        """
        For each layer, write rows from `host_buffer[layer_idx]` back into
        the per-layer paged KV tensor at `slot_mapping` positions.
        """
        self._maybe_refresh(kv_caches, host_buffer.device)
        layers = self._cached_layer_views
        if len(layers) == 0:
            return
        streams = self._ensure_streams(layers[0].device)

        rng = self._maybe_contig_range(slot_mapping)
        if streams.load_stream is not None:
            # Make load_stream wait for the compute stream so any prior
            # attention reads of these KV slots have completed before we
            # overwrite them.
            compute_stream = torch.cuda.current_stream(layers[0].device)
            streams.load_stream.wait_stream(compute_stream)
            ctx: Any = torch.cuda.stream(streams.load_stream)
        else:
            ctx = _NullCtx()
        with ctx:
            for li, kv_layer in enumerate(layers):
                flat = self._flatten_slots(kv_layer)
                hb = host_buffer[li]
                if rng is not None:
                    s0, ln = rng
                    if flat.dim() >= 2 and flat.size(0) == 2:
                        flat[:, s0 : s0 + ln].copy_(hb, non_blocking=True)
                    else:
                        flat[s0 : s0 + ln].copy_(hb, non_blocking=True)
                else:
                    slots = slot_mapping.to(device=kv_layer.device)
                    mask = slots >= 0
                    slots_pos = slots[mask].to(dtype=torch.long)
                    g = hb.to(device=kv_layer.device, non_blocking=True)
                    if flat.dim() >= 2 and flat.size(0) == 2:
                        if g.size(1) == mask.numel():
                            g = g[:, mask]
                        flat[:, slots_pos] = g
                    else:
                        if g.size(0) == mask.numel():
                            g = g[mask]
                        flat[slots_pos] = g


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
