# StrataCache vLLM Connector — Usage Guide

`StrataCacheConnectorV1` is StrataCache's KVConnector v1 implementation
for vLLM v0.13.0. vLLM calls into it on the scheduler and worker sides,
and it stores/loads prefix-keyed KV chunks on top of StrataCache's
tiered storage plane (CPU memory + optional CXL).

## Prerequisites

```bash
# From the stratacache/ directory
python -m pip install -U uv
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

`vllm==0.13.0` is already pinned in `pyproject.toml`. Optional features:

- **Pinned host slab + zero-copy fast path:** requires CUDA + `torch`.
  Auto-detected; falls back to pageable host memory when CUDA is absent.
- **CXL DAX tier:** requires a CXL DAX character device (`/dev/dax*.*`)
  and the bundled C library — see
  [Building the CXL backend](#building-the-cxl-backend) below.
- **NUMA binding (B6):** requires `libnuma`. Skipped silently when
  unavailable.
- **Lazy slab expansion (B5):** requires `sortedcontainers` (already a
  transitive dep).

## Building the CXL backend

The CXL DAX tier is implemented in C and lives at
[`csrc/cxl/`](../../../../csrc/cxl/). Build it once after install:

```bash
cd stratacache/csrc/cxl
make
```

That produces `libcxl_shm.so` next to the sources, which is exactly
where `binding.py` looks first; no further setup needed when
`use_cxl: true`. `make clean` removes the build artefacts.

Build prerequisites: a C compiler with `-mclflushopt -mclwb` support
(GCC ≥ 5 / Clang ≥ 7) and `pthread`. No external library dependencies.

## Quick Start

### Minimal launch (CPU tier only)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.8 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"StrataCacheConnectorV1","kv_connector_module_path":"stratacache.adapters.vllm.connector_v1","kv_role":"kv_both"}'
```

### Launch with a YAML config (CPU + CXL)

A ready-to-use CPU+CXL config is at
[`examples/vllm/config_cxl.yaml`](../../../../examples/vllm/config_cxl.yaml).

```bash
STRATACACHE_CONFIG_FILE=examples/vllm/config_cxl.yaml \
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.8 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"StrataCacheConnectorV1","kv_connector_module_path":"stratacache.adapters.vllm.connector_v1","kv_role":"kv_both"}' \
  --kv-events-config '{"enable_kv_cache_events":"True","publisher":"zmq","topic":"kv-events"}'
```

When `--kv-events-config` is set, also turn on `expose_kv_events: true`
in the YAML so the connector forwards admit/evict events into vLLM's
publisher.

### Where the YAML is read from

The connector searches in this order, first match wins:

1. `kv_connector_extra_config["stratacache.config_path"]` (passed inside
   `--kv-transfer-config`).
2. `STRATACACHE_CONFIG_FILE` env variable.
3. Repo defaults: `./config.yaml`, `~/.stratacache/config.yaml`,
   `<package>/config.yaml`.

## Configuration Reference (vLLM serve pipeline only)

All keys live under `stratacache.connector.*` in YAML. They can also be
overridden inline via `kv_connector_extra_config["stratacache.<key>"]`
in `--kv-transfer-config`.

### Tier shape

| Key | Type | Default | Meaning |
|---|---|---:|---|
| `use_cxl` | bool | `false` | Append a CXL DAX tier below the CPU tier. Requires `libcxl_shm.so`. |
| `writeback` | bool | `false` | Use write-back semantics on the CPU→CXL link. When `false`, writes go through synchronously. |
| `cpu_capacity_gb` | int | `60` | CPU tier capacity in GB. Effective size is clamped against `available - reserve_local_cpu_mb` (B9). |
| `cxl_dax_device` | str/null | `null` | Path to the CXL DAX character device, e.g. `/dev/dax1.0`. Can also be set via `STRATACACHE_CXL_DAX_DEVICE`. |
| `cxl_reset_metadata` | bool | `false` | Reset CXL metadata on connector init. |

### KV chunking and codec

| Key | Type | Default | Meaning |
|---|---|---:|---|
| `chunk_size` | int | `256` | Token-count per chunk; chunks are the unit of store/load. |
| `bundle_layers` | bool | `true` | Pack all layers of a chunk into one bundleT artifact (faster). |
| `save_partial_chunks` | bool | `true` | Save the trailing chunk that is shorter than `chunk_size`. |
| `tensor_codec` | str | `stable` | `stable` (raw bytes), `stable` + `tensor_header_in_payload`, or `torchsave` (slow legacy). |
| `tensor_header_in_payload` | bool | `false` | When false (default), dtype/shape live in `ArtifactMeta`, not the payload header. |

### Phase-3 features

| Key | Type | Default | Meaning |
|---|---|---:|---|
| `use_pinned_slab` | bool | `true` | Allocate a pinned host slab and store MemoryObjs as zero-copy tensor views into it. Auto-fallback to pageable if CUDA is unavailable. |
| `use_lazy_allocator` | bool | `false` | Grow the slab from a small initial chunk on a background thread instead of all-at-once. Avoids multi-second engine-boot stall on large pools. |
| `lazy_initial_mb` | int | `0` | Lazy mode initial slab size; `0` = auto-pick (≤ `min(cap, max(64, cap/16))` MiB). |
| `lazy_growth_step_mb` | int | `0` | Lazy mode per-step growth; `0` = auto. |
| `use_layerwise_pipeline` | bool | `false` | Use the GPU paged connector + dedicated CUDA streams for `wait_for_save` (gather) and `start_load_kv` (scatter). Overlaps PCIe traffic with attention compute. Off by default. |
| `use_token_database` | bool | `false` | Drive prefix hashing via `ChunkedTokenDatabase` (incremental, vLLM-block-hash compatible). Off by default during migration; aim is on. |
| `expose_kv_events` | bool | `false` | Translate backend admit/evict events into `CacheStoreEvent` / `CacheRemoveEvent` and surface them via `get_kv_events()`. Required when launching vLLM with `--kv-events-config`. |
| `numa_node` | int | `-1` | NUMA node to bind the pinned slab to (`-1` = no binding). Requires `libnuma`; silently no-ops otherwise. |
| `reserve_local_cpu_mb` | int | `1024` | Headroom (MiB) subtracted from system-available memory when clamping `cpu_capacity_gb` (B9). |

### Logging / debug

| Key | Type | Default | Meaning |
|---|---|---:|---|
| `log_stats` | bool | `true` | Emit one summary line per request: `req_done: req=... gpu=N cpu=N cxl=N | cum_hit_rate: ...`. |
| `log_every` | int | `50` | Cap log frequency by scheduler call count. |
| `log_min_interval_s` | float | `2.0` | Cap log frequency by wall time. |
| `debug` | bool | `false` | Verbose pid / impl_id traces. |

### Telemetry

| Key | Type | Default | Meaning |
|---|---|---:|---|
| `telemetry.export` | bool | `true` | Start a FastAPI `/metrics` endpoint on `0.0.0.0:6954`. |

### Environment variables

| Var | Effect |
|---|---|
| `STRATACACHE_CONFIG_FILE` | Path to a YAML config (overrides repo defaults). |
| `STRATACACHE_<KEY>` | Override any single config key (e.g. `STRATACACHE_USE_CXL=true`). |
| `STRATACACHE_PROFILE` | `1` enables per-method latency histograms; dumped at process exit / SIGTERM. |
| `STRATACACHE_CXL_DAX_DEVICE` | DAX device path (overrides `cxl_dax_device` in YAML). |
| `STRATACACHE_CXL_DAX_DEVICE_SIZE` | DAX device size in bytes (default 64 GiB). |
| `STRATACACHE_CXL_VERBOSE_NOT_FOUND` | Set to `1` to log every CXL "object not found" miss to stderr. |

## Inline overrides via `--kv-transfer-config`

Anything in `kv_connector_extra_config` under `stratacache.<key>` (or
just `<key>`) overrides the YAML. Useful for ablations:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --kv-transfer-config '{
    "kv_connector":"StrataCacheConnectorV1",
    "kv_connector_module_path":"stratacache.adapters.vllm.connector_v1",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "stratacache.config_path":"/path/to/cfg.yaml",
      "stratacache.use_layerwise_pipeline": true,
      "stratacache.use_pinned_slab": true,
      "stratacache.cpu_capacity_gb": 30
    }
  }'
```
