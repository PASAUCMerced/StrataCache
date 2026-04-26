from __future__ import annotations

from pathlib import Path
import tempfile


def run() -> None:
    from stratacache.adapters.vllm.connector_v1 import _load_connector_config, yaml

    cfg = _load_connector_config({})
    assert int(cfg["cpu_capacity_gb"]) == 60

    if yaml is None:
        return

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.yaml"
        p.write_text(
            "stratacache:\n"
            "  connector:\n"
            "    cpu_capacity_gb: 12\n"
            "    chunk_size: 64\n"
        )
        cfg2 = _load_connector_config({"stratacache.config_path": str(p)})
        assert int(cfg2["cpu_capacity_gb"]) == 12
        assert int(cfg2["chunk_size"]) == 64
