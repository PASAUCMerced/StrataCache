from stratacache.telemetry.backend.cpu_telemetry import StrataCPUTelemetry

cpu_telemetry = None

def create_telemetry_backends(telemetry):
    """Create and register all telemetry backends.
    
    Args:
        telemetry: StrataTelemetry instance to register backends with
    """
    global cpu_telemetry
    if cpu_telemetry is None:
        cpu_telemetry = StrataCPUTelemetry(telemetry)
