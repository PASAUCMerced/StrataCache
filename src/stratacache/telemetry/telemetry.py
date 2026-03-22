import stratacache.config as config
from stratacache.telemetry.utils import thread_safe, human_readable_size
from stratacache.telemetry.exporters import (
    ExporterManager, 
    FileExporter, 
    WandBExporter, 
    TelemetrySnapshot
)

from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import queue

import copy
import os
import threading
from datetime import datetime

import logging

logger = logging.getLogger(__name__)

# Check if WandB is available for optional telemetry export
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class StrataTierType(Enum):
    CPU = 0
    CXL = 1
    NIXL = 2
    DISK = 3
    
    UNKNOWN = 99

@dataclass
class StrataTierStats():
    # Operation counts
    total_ops: int = 0
    total_read_ops: int = 0
    total_write_ops: int = 0
    total_delete_ops: int = 0
    
    # Cache payload stats
    byte_read: int = 0
    byte_written: int = 0
    byte_delete: int = 0
    
    # Cache size
    size: int = 0
    
    # Read stats
    read_throughput: float = 0.0
    avg_read_throughput: float = 0.0
    max_read_throughput: float = 0.0
    read_latency_us: float = 0.0
    avg_read_latency_us: float = 0.0
    max_read_latency_us: float = 0.0
    
    # Write stats
    write_throughput: float = 0.0
    avg_write_throughput: float = 0.0
    max_write_throughput: float = 0.0
    write_latency_us: float = 0.0
    avg_write_latency_us: float = 0.0
    max_write_latency_us: float = 0.0
    
    # Overall stats (read/write)
    throughput: float = 0.0
    avg_throughput: float = 0.0
    max_throughput: float = 0.0
    latency_us: float = 0.0
    avg_latency_us: float = 0.0
    max_latency_us: float = 0.0

class StrataTierTelemetry(ABC):
    def __init__(self, tier: StrataTierType, telemetry: "StrataTelemetry"):
        self._lock = threading.RLock()
        self._tier = tier
        telemetry.register_tier(tier, self)
        
    def get_stats(self) -> StrataTierStats:
        with self._lock:
            return copy.deepcopy(self._stats)
    
    @abstractmethod
    def process_op(self, op_type: str, **kwargs):
        raise NotImplementedError("process_op must be implemented by subclasses")
    
    @abstractmethod
    def print_stats(self):
        raise NotImplementedError("print_stats must be implemented by subclasses")
    
    def dump_stats(self) -> dict:
        """Dump tier stats as dictionary for serialization.
        Can be overridden by subclasses for custom dump logic.
        """
        tier_stats = self.get_stats()
        return tier_stats.__dict__


@dataclass
class StrataSystemStats():
    # Operation counts
    total_ops: int = 0
    total_read_ops: int = 0
    total_write_ops: int = 0
    total_delete_ops: int = 0
    
    # Cache payload stats
    byte_read: int = 0
    byte_written: int = 0
    byte_delete: int = 0
    
    # Cache size
    size: int = 0
    
    # Read stats
    read_throughput: float = 0.0
    avg_read_throughput: float = 0.0
    max_read_throughput: float = 0.0
    read_latency_us: float = 0.0
    avg_read_latency_us: float = 0.0
    max_read_latency_us: float = 0.0
    
    # Write stats
    write_throughput: float = 0.0
    avg_write_throughput: float = 0.0
    max_write_throughput: float = 0.0
    write_latency_us: float = 0.0
    avg_write_latency_us: float = 0.0
    max_write_latency_us: float = 0.0
    
    # Overall stats (read/write)
    throughput: float = 0.0
    avg_throughput: float = 0.0
    max_throughput: float = 0.0
    latency_us: float = 0.0
    avg_latency_us: float = 0.0
    max_latency_us: float = 0.0
    
    def print_stats(self):
        logger.info(f"System Stats: total_ops={self.total_ops}, total_read_ops={self.total_read_ops}, total_write_ops={self.total_write_ops}, total_delete_ops={self.total_delete_ops}, "
                    f"byte_read={human_readable_size(self.byte_read)}, byte_written={human_readable_size(self.byte_written)}, byte_delete={human_readable_size(self.byte_delete)}, size={human_readable_size(self.size)}, "
                    f"read_throughput={human_readable_size(self.read_throughput)}/s, avg_read_throughput={human_readable_size(self.avg_read_throughput)}/s, max_read_throughput={human_readable_size(self.max_read_throughput)}/s, "
                    f"read_latency_us={self.read_latency_us:.2f}us, avg_read_latency_us={self.avg_read_latency_us:.2f}us, max_read_latency_us={self.max_read_latency_us:.2f}us, "
                    f"write_throughput={human_readable_size(self.write_throughput)}/s, avg_write_throughput={human_readable_size(self.avg_write_throughput)}/s, max_write_throughput={human_readable_size(self.max_write_throughput)}/s, "
                    f"write_latency_us={self.write_latency_us:.2f}us, avg_write_latency_us={self.avg_write_latency_us:.2f}us, max_write_latency_us={self.max_write_latency_us:.2f}us, "
                    f"throughput={human_readable_size(self.throughput)}/s, avg_throughput={human_readable_size(self.avg_throughput)}/s, max_throughput={human_readable_size(self.max_throughput)}/s, "
                    f"latency_us={self.latency_us:.2f}us, avg_latency_us={self.avg_latency_us:.2f}us, max_latency_us={self.max_latency_us:.2f}us")
    
class StrataTelemetry:
    """_summary_
    StrataTelemetry collects telemetry data for StrataCache.
    It provides telemetry from two perspectives: system-overall and by-tier.
    """
    
    def __init__(self):
        self._system_stats_lock = threading.RLock()
        self._system_stats = StrataSystemStats()
        self._per_tier_telemetry: dict[StrataTierType, StrataTierTelemetry] = {}
        self._message_queue = queue.Queue()
        
        self._initialize_exporters()
        self._initialize_backends()
        
        # Start background threads for processing messages and exporting stats
        self._msg_process_thread = threading.Thread(target=self._process_messages, daemon=True)
        self._msg_process_thread.start()
        self._exporter_thread = threading.Thread(target=self._export_stats, daemon=True)
        self._exporter_thread.start()
    
    _instance = None

    @staticmethod
    @thread_safe
    def get_or_create() -> "StrataTelemetry":
        if StrataTelemetry._instance is None:
            StrataTelemetry._instance = StrataTelemetry()
        return StrataTelemetry._instance
    
    def on_tier_op_async(self, tier: StrataTierType, op_type: str, **kwargs):
        self._message_queue.put_nowait((tier, op_type, kwargs))

    def get_system_stats(self) -> StrataSystemStats:
        with self._system_stats_lock:
            return copy.deepcopy(self._system_stats)
    
    def get_tier_stats(self, tier: StrataTierType) -> StrataTierStats:
        """Get telemetry stats snapshot for a specific tier.

        Args:
            tier (StrataTierType): Tier

        Raises:
            ValueError: If the tier type is invalid

        Returns:
            StrataTierStats: Telemetry stats for the specified tier (snapshot)
        """
        telemetry = self._per_tier_telemetry.get(tier, None)
        if telemetry is None:
            raise ValueError(f"Invalid tier type: {tier}")
        return telemetry.get_stats()
    
    def register_tier(self, tier: StrataTierType, telemetry: StrataTierTelemetry):
        """Register telemetry for a specific tier.

        Args:
            tier (StrataTierType): Tier type
            telemetry (StrataTierTelemetry): Telemetry instance for the tier
        """
        logging.info(f"Registering telemetry for tier: {tier}")
        self._per_tier_telemetry[tier] = telemetry
    
    def enable_wandb(self, entity: str, project: str = "stratacache", name: str = None, config: dict = None):
        if not WANDB_AVAILABLE:
            logger.error("WandB not installed. Install it with: pip install wandb")
            raise ImportError("WandB not installed. Install it with: pip install wandb")
        
        if entity is None or entity.strip() == "":
            logger.error("WandB entity is required to enable WandB integration")
            raise ValueError("WandB entity is required to enable WandB integration")
        
        try:
            if wandb.run is not None:
                logger.debug("WandB already initialized")
                return True
            
            init_config = {"module": "stratacache.telemetry"}
            if config:
                init_config.update(config)
            
            wandb.init(
                entity=entity,
                project=project,
                name=name or "stratacache-experiment",
                config=init_config
            )
            logger.info(f"WandB initialized: project={project}, name={wandb.run.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            return False
    
    def disable_wandb(self):
        """Finalize WandB run.
        
        This closes the WandB run and should be called before application exit.
        Also removes WandBExporter from the exporter manager.
        """
        # Remove WandB exporter
        self._exporter_manager.remove_exporter(WandBExporter)
        
        # Close WandB run
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                wandb.finish()
                logger.info("WandB run finalized")
            except Exception as e:
                logger.warning(f"Error closing WandB: {e}")
    
    def add_exporter(self, exporter):
        """Add a telemetry exporter.
        
        Args:
            exporter: TelemetryExporter instance (FileExporter, WandBExporter, etc.)
            
        Example:
            from stratacache.telemetry.exporters import PrometheusExporter
            telemetry = StrataTelemetry.get_or_create()
            telemetry.add_exporter(PrometheusExporter())
        """
        self._exporter_manager.add_exporter(exporter)
    
    def remove_exporter(self, exporter_class):
        """Remove all exporters of a given class.
        
        Args:
            exporter_class: Class to remove (e.g., WandBExporter, FileExporter)
            
        Example:
            from stratacache.telemetry.exporters import FileExporter
            telemetry = StrataTelemetry.get_or_create()
            telemetry.remove_exporter(FileExporter)
        """
        self._exporter_manager.remove_exporter(exporter_class)
    
    def _initialize_exporters(self):
        """Initialize telemetry exporters based on configuration.
        """
        self._exporter_manager = ExporterManager()
        if config.get_config().exporter_file_enabled == True:
            self._exporter_manager.add_exporter(FileExporter(config.get_config().exporter_file_folder))
        if config.get_config().exporter_wandb_enabled == True:
            self.enable_wandb(
                entity=config.get_config().exporter_wandb_entity,
                project=config.get_config().exporter_wandb_project,
                name=f"{config.get_config().exporter_wandb_run_name}-{os.getpid()}",
            )
            self._exporter_manager.add_exporter(WandBExporter())
    
    def _initialize_backends(self):
        """Initialize all telemetry backends.
        """
        try:
            from stratacache.telemetry.backend import create_telemetry_backends
            create_telemetry_backends(self)
        except Exception:
            logger.exception("Failed to initialize telemetry backends")
    
    def _process_messages(self):
        logger.info("Starting telemetry message processing thread.")
        
        while True:
            tier, op_type, kwargs = self._message_queue.get()
            
            logger.debug(f"Processing telemetry message: tier={tier}, op_type={op_type}, kwargs={kwargs}")
            
            # Generic tier metrics
            latency_us = kwargs.get("latency_us", 0.0)
            size = kwargs.get("size", 0)
            released_size = kwargs.get("released_size", 0)
            tier_bytes_used = kwargs.get("tier_bytes_used", 0)  # Get actual bytes_used from backend
            throughput = (size / latency_us * 1_000_000 if latency_us > 0 else 0)

            # Update tier stats
            telemetry = self._per_tier_telemetry.get(tier, None)
            if telemetry is not None:
                with telemetry._lock:
                    tier_stats = telemetry._stats
                    tier_stats.total_ops += 1
                    
                    tier_stats.latency_us = latency_us
                    tier_stats.avg_latency_us = ((tier_stats.avg_latency_us * (tier_stats.total_ops - 1)) + latency_us) / tier_stats.total_ops
                    tier_stats.max_latency_us = max(tier_stats.max_latency_us, latency_us)

                    tier_stats.throughput = throughput
                    tier_stats.avg_throughput = ((tier_stats.avg_throughput * (tier_stats.total_ops - 1)) + throughput) / tier_stats.total_ops
                    tier_stats.max_throughput = max(tier_stats.max_throughput, throughput)
                    
                    if op_type == "load":
                        tier_stats.total_read_ops += 1

                        tier_stats.byte_read += size
                        
                        tier_stats.read_latency_us = latency_us
                        tier_stats.avg_read_latency_us = ((tier_stats.avg_read_latency_us * (tier_stats.total_read_ops - 1)) + latency_us) / tier_stats.total_read_ops
                        tier_stats.max_read_latency_us = max(tier_stats.max_read_latency_us, latency_us)
                        
                        tier_stats.read_throughput = throughput
                        tier_stats.avg_read_throughput = ((tier_stats.avg_read_throughput * (tier_stats.total_read_ops - 1)) + throughput) / tier_stats.total_read_ops
                        tier_stats.max_read_throughput = max(tier_stats.max_read_throughput, throughput)

                    elif op_type == "store":
                        tier_stats.total_write_ops += 1
                        
                        tier_stats.byte_written += size
                        
                        tier_stats.write_latency_us = latency_us
                        tier_stats.avg_write_latency_us = ((tier_stats.avg_write_latency_us * (tier_stats.total_write_ops - 1)) + latency_us) / tier_stats.total_write_ops
                        tier_stats.max_write_latency_us = max(tier_stats.max_write_latency_us, latency_us)

                        tier_stats.write_throughput = throughput                
                        tier_stats.avg_write_throughput = ((tier_stats.avg_write_throughput * (tier_stats.total_write_ops - 1)) + throughput) / tier_stats.total_write_ops
                        tier_stats.max_write_throughput = max(tier_stats.max_write_throughput, throughput)
                        
                        # Use actual bytes_used from backend instead of accumulating
                        tier_stats.size = tier_bytes_used
                    elif op_type == "delete":
                        logger.debug(f"Delete operation: size={size}, latency_us={latency_us}")
                        tier_stats.total_delete_ops += 1
                        tier_stats.byte_delete += size
                        
                        # Use actual bytes_used from backend after deletion
                        tier_stats.size = tier_bytes_used
                    else:
                        logger.warning(f"Unknown operation type: {op_type}")
                    
                    # Call tier-specific processing
                    telemetry.process_op(op_type, **kwargs)
                    telemetry.print_stats()
            
            # Update system stats
            ## NOTICE: System stats should aggregate all tiers' current bytes_used
            with self._system_stats_lock:
                sys_stats = self._system_stats
                
                sys_stats.total_ops += 1
                
                sys_stats.latency_us = latency_us
                sys_stats.avg_latency_us = ((sys_stats.avg_latency_us * (sys_stats.total_ops - 1)) + latency_us) / sys_stats.total_ops
                sys_stats.max_latency_us = max(sys_stats.max_latency_us, latency_us)
                
                sys_stats.throughput = throughput
                sys_stats.avg_throughput = ((sys_stats.avg_throughput * (sys_stats.total_ops - 1)) + throughput) / sys_stats.total_ops
                sys_stats.max_throughput = max(sys_stats.max_throughput, throughput)
                
                if op_type == "load":
                    sys_stats.total_read_ops += 1
                    
                    sys_stats.byte_read += size
                    
                    sys_stats.read_latency_us = latency_us
                    sys_stats.avg_read_latency_us = ((sys_stats.avg_read_latency_us * (sys_stats.total_read_ops - 1)) + latency_us) / sys_stats.total_read_ops
                    sys_stats.max_read_latency_us = max(sys_stats.max_read_latency_us, latency_us)

                    sys_stats.read_throughput = throughput
                    sys_stats.avg_read_throughput = ((sys_stats.avg_read_throughput * (sys_stats.total_read_ops - 1)) + throughput) / sys_stats.total_read_ops
                    sys_stats.max_read_throughput = max(sys_stats.max_read_throughput, throughput)

                elif op_type == "store":
                    sys_stats.total_write_ops += 1
                    
                    sys_stats.byte_written += size
                    
                    sys_stats.write_latency_us = latency_us
                    sys_stats.avg_write_latency_us = ((sys_stats.avg_write_latency_us * (sys_stats.total_write_ops - 1)) + latency_us) / sys_stats.total_write_ops
                    sys_stats.max_write_latency_us = max(sys_stats.max_write_latency_us, latency_us)

                    sys_stats.write_throughput = throughput
                    sys_stats.avg_write_throughput = ((sys_stats.avg_write_throughput * (sys_stats.total_write_ops - 1)) + throughput) / sys_stats.total_write_ops
                    sys_stats.max_write_throughput = max(sys_stats.max_write_throughput, throughput)

                elif op_type == "delete":
                    logger.debug(f"Delete operation: size={size}, latency_us={latency_us}")
                    sys_stats.total_delete_ops += 1
                    sys_stats.byte_delete += size
                else:
                    logger.warning(f"Unknown operation type: {op_type}")
                
                # Update system size: sum of all tiers' current bytes_used
                total_bytes_used = 0
                for tier_telemetry in self._per_tier_telemetry.values():
                    with tier_telemetry._lock:
                        total_bytes_used += tier_telemetry._stats.size
                sys_stats.size = total_bytes_used
                sys_stats.print_stats()
    
    def _export_stats(self):
        """Export telemetry stats to all registered exporters.
        """
        export_interval = 1  # frequency of exporting stats (seconds)
        
        while True:
            try:
                threading.Event().wait(export_interval)
                
                timestamp = datetime.now().isoformat()
                
                # Build system stats dict
                system_stats_dict = self.get_system_stats().__dict__
                
                # Build per-tier stats dict
                tier_stats_dict = {}
                for tier, telemetry in self._per_tier_telemetry.items():
                    tier_stats_dict[tier.name] = telemetry.dump_stats()
                
                # Create telemetry snapshot
                snapshot = TelemetrySnapshot(
                    timestamp=timestamp,
                    system_stats=system_stats_dict,
                    tier_stats=tier_stats_dict
                )
                
                # Export via all registered exporters
                _ = self._exporter_manager.export(snapshot)
                
            except Exception as e:
                logger.error(f"Error in _export_stats: {e}", exc_info=True)
