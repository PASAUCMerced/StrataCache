import logging

logger = logging.getLogger(__name__)

cpu_telemetry = None
gpu_telemetry = None  # noqa: F841 (reserved for the GPU backend opt-in below)


def create_telemetry_backends(telemetry):
    """Create and register all telemetry backends.

    Each backend is wrapped in try/except so a missing optional dependency
    (cupti / pcm / etc.) only disables that backend instead of breaking
    telemetry init.
    """
    global cpu_telemetry
    if cpu_telemetry is None:
        try:
            from stratacache.telemetry.backend.cpu.cpu_telemetry import (
                StrataCPUTelemetry,
            )

            cpu_telemetry = StrataCPUTelemetry(telemetry)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to initialise CPU telemetry backend")

    # GPU backend is opt-in (requires CUPTI). Re-enable here when needed.
    # global gpu_telemetry
    # if gpu_telemetry is None:
    #     try:
    #         from stratacache.telemetry.backend.gpu.gpu_telemetry import (
    #             StrataGPUTelemetry,
    #         )
    #         gpu_telemetry = StrataGPUTelemetry(telemetry)
    #     except Exception:  # noqa: BLE001
    #         logger.exception("Failed to initialise GPU telemetry backend")

