"""
GPU-side helpers for inference-engine adapters.

`paged_connector.py` provides the type-agnostic paged-KV gather/scatter
primitives. Higher-level KV chunk decisions (bundleT layout etc.) stay in
`artifacts/kv/`; here we only handle the device-side data movement.
"""
from stratacache.gpu.paged_connector import (
    MultiLayerPagedConnector,
    PagedConnectorStreams,
)

__all__ = ["MultiLayerPagedConnector", "PagedConnectorStreams"]
