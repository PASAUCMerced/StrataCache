# StrataCache

StrataCache is a standalone tiered storage library for inference workloads. It provides a unified storage API over multiple memory/media layers (CPU memory, CXL, and future tiers) while keeping model/runtime-specific logic in adapters.

The project is designed for two concurrent goals:

- Reuse and manage KV cache for inference engines such as vLLM.
- Reuse the same storage plane for non-KV artifacts such as model parameter chunks used by offload/prefetch flows.

In short, StrataCache is not “KV-only cache code”; it is a generalized artifact storage substrate for LLM systems.

## Why StrataCache (Motivation)

Modern inference systems move data across heterogeneous memory/media (GPU, CPU, CXL, SSD). Most implementations solve this in a task-specific way (for example KV-only), which creates duplicated logic and inconsistent policies.

StrataCache addresses this by:

- Defining a generalized artifact abstraction (`ArtifactId`, metadata, bytes payload).
- Providing a single tier coordinator (`TierChain`) with write-through/write-back semantics.
- Exposing one public API (`StorageEngine`) used by both adapters and direct clients.

This allows KV caching and parameter offload/prefetch to share the same storage management model without key collisions or policy conflicts.

## Main Use Cases

1. vLLM KV cache reuse
- Use `StrataCacheConnectorV1` to store/load KV chunks by prefix and chunk boundaries.

2. Parameter offload and prefetch
- Use `ParameterStoreClient` on top of `StorageEngine` to store/load parameter chunks.

3. Mixed workloads in one storage plane
- Run KV connector and parameter offload together with unified tier policies.

## Installation

From the `stratacache/` directory:

```bash
python -m pip install -U uv
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

`vllm==0.13.0` is already included in `pyproject.toml` dependencies.

## Minimal Run (Qwen + CPU Memory)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"StrataCacheConnectorV1","kv_connector_module_path":"stratacache.adapters.vllm.connector_v1","kv_role":"kv_both"}'
```

## Configuration (YAML)

StrataCache connector reads YAML configuration.

- Default config file in this repo: `config.yaml`
- Packaged fallback config for installed module: `src/stratacache/config.yaml`
- Optional override path in vLLM config: `kv_connector_extra_config["stratacache.config_path"]`

If no YAML file is found, built-in defaults are used.

### Connector Config Keys

| Key | Type | Default | Meaning |
|---|---|---:|---|
| `use_cxl` | bool | `false` | Enable CXL tier |
| `writeback` | bool | `false` | Use write-back between CPU and CXL |
| `cpu_capacity_mb` | int | `61440` | CPU tier capacity in MB |
| `chunk_size` | int | `256` | Token chunk size for KV matching/store/load |
| `bundle_layers` | bool | `true` | Store/load layered KV in bundled format |
| `tensor_codec` | str | `stable` | Tensor payload codec |
| `tensor_header_in_payload` | bool | `false` | Keep dtype/shape in payload header |
| `save_partial_chunks` | bool | `true` | Save/load last partial chunk |
| `log_stats` | bool | `true` | Enable connector stats logging |
| `log_every` | int | `50` | Stats logging interval |
| `log_min_interval_s` | float | `2.0` | Min seconds between logs |
| `debug` | bool | `false` | Debug logging |
| `cxl_dax_device` | str/null | `null` | CXL DAX device path |
| `cxl_reset_metadata` | bool | `false` | Reset CXL metadata on init |
| `exporter.file.enabled` | bool | `false` | Enable telemetry export to file |
| `exporter.wandb.enabled` | bool | `false` | Enable telemetry export to WandB |

## Design Overview

StrataCache separates storage responsibilities from task semantics.

### Core Components

1. Artifact model (`core/`)
- Generic object identity + metadata + payload.

2. Memory layers (`backend/`)
- Per-medium implementation (`CpuMemoryLayer`, `CxlMemoryLayer`).

3. Tier coordinator (`tiering/TierChain`)
- Ordered tiers, read-through lookup, promotion, write-through/write-back propagation.

4. Public API (`engine/StorageEngine`)
- `store/load/contains/delete` with `chain/exact/prefer` access modes.

5. Task adapters (`adapters/`)
- vLLM KV connector for prefix/chunk/slot mapping logic.
- Torch parameter client for parameter chunk workflows.

### Core Logic (Read/Write Path)

Write path:

- Client calls `StorageEngine.store(...)`.
- `TierChain` writes to selected tier and applies link policy.
- In write-back mode, dirty tracking and background convergence are handled internally.

Read path:

- Client calls `StorageEngine.load(...)`.
- `TierChain` scans tiers top-down (`chain`) or uses directed tier strategy (`exact` / `prefer`).
- On lower-tier hit, optional promotion updates upper tiers.

KV-specific matching and token-boundary logic are adapter responsibilities, not storage-core responsibilities.

## Project Layout

```text
stratacache/
├── config.yaml
├── pyproject.toml
├── README.md
├── src/stratacache/
│   ├── core/          # artifact schema, codecs, errors, key builder
│   ├── backend/       # CPU/CXL memory layers
│   ├── tiering/       # TierChain and link policies
│   ├── writeback/     # dirty tracking and worker
│   ├── engine/        # StorageEngine public facade
│   └── adapters/
│       ├── vllm/      # KVConnector v1 integration
│       └── torch/     # parameter chunk client
└── tests/             # assert-based tests
```

## Testing

```bash
PYTHONPATH=$(pwd)/src python tests/run_all.py
```
