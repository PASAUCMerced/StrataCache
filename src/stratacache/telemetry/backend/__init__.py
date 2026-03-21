from stratacache.telemetry.backend.gpu.gpu_telemetry import StrataGPUTelemetry
from stratacache.telemetry.backend.cpu.cpu_telemetry import StrataCPUTelemetry

gpu_telemetry = None
cpu_telemetry = None

def create_telemetry_backends(telemetry):
    """Create and register all telemetry backends.
    
    Args:
        telemetry: StrataTelemetry instance to register backends with
    """
        
    # global gpu_telemetry
    # if gpu_telemetry is None:
    #     gpu_telemetry = StrataGPUTelemetry(telemetry)

    global cpu_telemetry
    if cpu_telemetry is None:
        cpu_telemetry = StrataCPUTelemetry(telemetry)

