from __future__ import annotations


def run() -> None:
    try:
        import torch  # noqa: F401
        import sortedcontainers  # noqa: F401
    except Exception:
        return

    from stratacache.backend.cpu import LazyCpuAllocator

    # 4KB target, 1KB initial, 1KB step. Background thread should grow.
    lz = LazyCpuAllocator(
        capacity_bytes=4096,
        initial_bytes=1024,
        growth_step_bytes=1024,
        growth_pause_s=0.001,
        pin_memory=False,
    )
    try:
        # Initially we can grab a 1KB slot.
        s = lz.try_allocate(1024)
        assert s is not None
        s.free()

        # Wait for full target.
        ok = lz.wait_until_full(timeout_s=2.0)
        assert ok, "lazy expander did not reach target capacity in time"
        assert lz.stats().capacity_bytes == 4096
    finally:
        lz.stop()
