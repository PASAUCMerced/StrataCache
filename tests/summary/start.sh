#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/venv/bin/activate"

clear

(cd "$DIR/../.." && uv pip install -e .)

vllm serve Qwen/Qwen2-0.5B-Instruct \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"StrataCacheConnectorV1","kv_connector_module_path":"stratacache.adapters.vllm.connector_v1","kv_role":"kv_both"}'
