from __future__ import annotations


def run() -> None:
    try:
        import torch
    except Exception:
        return

    from stratacache.gpu import MultiLayerPagedConnector

    # Two layers, each shaped [num_blocks=4, block_size=2, head_dim=4]. We
    # use CPU tensors so the test runs anywhere; the connector falls back
    # to default-stream copies when CUDA is unavailable.
    L = 2
    B = 4
    S = 2
    D = 4
    layers = [torch.zeros(B, S, D, dtype=torch.float32) for _ in range(L)]
    # Fill each layer with a per-slot watermark so we can verify scatter.
    for li, kv in enumerate(layers):
        flat = kv.reshape(B * S, D)
        for s in range(B * S):
            flat[s] = float(100 * li + s)

    conn = MultiLayerPagedConnector()

    # Gather a contiguous range [3, 6) into a stacked host buffer.
    sm = torch.tensor([3, 4, 5], dtype=torch.long)
    host = torch.empty(L, len(sm), D, dtype=torch.float32)
    conn.gather_chunk(layers, sm, host)
    conn.synchronize_store()
    for li in range(L):
        for i, s in enumerate(sm.tolist()):
            assert torch.equal(host[li, i], torch.full((D,), float(100 * li + s)))

    # Scatter the same buffer to a fresh pair of layers and verify only the
    # masked slots got written.
    layers2 = [torch.full((B, S, D), -1.0) for _ in range(L)]
    conn.scatter_chunk(layers2, sm, host)
    conn.synchronize_load()
    for li in range(L):
        flat = layers2[li].reshape(B * S, D)
        for s in range(B * S):
            if s in sm.tolist():
                expected = float(100 * li + s)
                assert torch.equal(flat[s], torch.full((D,), expected))
            else:
                assert torch.equal(flat[s], torch.full((D,), -1.0))
