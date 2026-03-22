"""
Telemetry data exporters for StrataCache.

This module provides pluggable exporters for telemetry data to different backends:
- FileExporter: Export to JSON files
- WandBExporter: Export to Weights & Biases
- PrometheusExporter: (Future) Export to Prometheus format
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class TelemetrySnapshot:
    """Snapshot of telemetry data at a point in time."""
    timestamp: str
    system_stats: Dict[str, Any]
    tier_stats: Dict[str, Dict[str, Any]]


class TelemetryExporter(ABC):
    """Abstract base class for telemetry exporters."""
    
    @abstractmethod
    def export(self, snapshot: TelemetrySnapshot) -> bool:
        """Export telemetry snapshot.
        
        Args:
            snapshot: TelemetrySnapshot containing system and tier stats
            
        Returns:
            True if export was successful, False otherwise
        """
        raise NotImplementedError("export must be implemented by subclasses")
    
    @abstractmethod
    def close(self):
        """Clean up exporter resources."""
        raise NotImplementedError("close must be implemented by subclasses")


class FileExporter(TelemetryExporter):
    """Export telemetry data to JSON files."""
    
    def __init__(self, output_dir: str = "/tmp/stratacache-exports"):
        """Initialize file exporter.
        
        Args:
            output_dir: Directory to save JSON files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"FileExporter initialized: output_dir={output_dir}")
    
    def export(self, snapshot: TelemetrySnapshot) -> bool:
        """Export snapshot to JSON file."""
        try:
            filename = os.path.join(
                self.output_dir,
                f"stats_{datetime.fromisoformat(snapshot.timestamp).strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            data = {
                "timestamp": snapshot.timestamp,
                "system_stats": snapshot.system_stats,
                "tier_stats": snapshot.tier_stats,
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Exported stats to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to file: {e}")
            return False
    
    def close(self):
        """No resources to clean up."""
        pass


class WandBExporter(TelemetryExporter):
    """Export telemetry data to Weights & Biases."""
    
    def __init__(self):
        """Initialize WandB exporter."""
        try:
            import wandb
            self.wandb = wandb
            if self.wandb.run is None:
                raise RuntimeError("WandB not initialized. Call enable_wandb() first.")
            logger.info("WandBExporter initialized")
        except ImportError:
            raise ImportError("wandb package not installed. Install with: pip install wandb")
        except RuntimeError as e:
            logger.warning(f"WandB not available: {e}")
            self.wandb = None
    
    def export(self, snapshot: TelemetrySnapshot) -> bool:
        """Export snapshot to WandB."""
        if self.wandb is None:
            return False
        
        try:
            wandb_data = {}
            
            # System metrics with System/ prefix
            for key, value in snapshot.system_stats.items():
                wandb_data[f"System/{key}"] = value
            
            # Per-tier metrics with tier name prefix
            for tier_name, tier_stats in snapshot.tier_stats.items():
                for key, value in tier_stats.items():
                    wandb_data[f"{tier_name}/{key}"] = value
            
            self.wandb.log(wandb_data)
            logger.debug(f"Logged {len(wandb_data)} metrics to WandB")
            return True
        except Exception as e:
            logger.warning(f"Failed to export to WandB: {e}")
            return False
    
    def close(self):
        """Close WandB run."""
        if self.wandb and self.wandb.run is not None:
            try:
                self.wandb.finish()
                logger.debug("WandB run finished")
            except Exception as e:
                logger.warning(f"Error closing WandB: {e}")


class PrometheusExporter(TelemetryExporter):
    """Export telemetry data to Prometheus format.
    
    Produces metrics in Prometheus text format suitable for scraping.
    """
    
    def __init__(self, output_file: str = "/tmp/stratacache_metrics.txt"):
        """Initialize Prometheus exporter.
        
        Args:
            output_file: File to write Prometheus metrics
        """
        pass
    
    def export(self, snapshot: TelemetrySnapshot) -> bool:
        """Export snapshot to Prometheus text format."""
        raise NotImplementedError("PrometheusExporter is not implemented yet")
    
    def close(self):
        """No resources to clean up."""
        pass


class ExporterManager:
    """Manages multiple telemetry exporters."""
    
    def __init__(self):
        """Initialize exporter manager."""
        self.exporters: List[TelemetryExporter] = []
    
    def add_exporter(self, exporter: TelemetryExporter):
        """Add an exporter.
        
        Args:
            exporter: TelemetryExporter instance
        """
        self.exporters.append(exporter)
        logger.info(f"Added exporter: {exporter.__class__.__name__}")
    
    def remove_exporter(self, exporter_class):
        """Remove all exporters of a given class.
        
        Args:
            exporter_class: Class to remove (e.g., WandBExporter)
        """
        self.exporters = [e for e in self.exporters if not isinstance(e, exporter_class)]
    
    def export(self, snapshot: TelemetrySnapshot) -> Dict[str, bool]:
        """Export snapshot to all registered exporters.
        
        Args:
            snapshot: TelemetrySnapshot to export
            
        Returns:
            Dictionary mapping exporter name to success status
        """
        results = {}
        for exporter in self.exporters:
            name = exporter.__class__.__name__
            try:
                results[name] = exporter.export(snapshot)
            except Exception as e:
                logger.error(f"Exporter {name} raised exception: {e}")
                results[name] = False
        return results
    
    def close_all(self):
        """Close all exporters."""
        for exporter in self.exporters:
            try:
                exporter.close()
            except Exception as e:
                logger.warning(f"Error closing {exporter.__class__.__name__}: {e}")
        self.exporters.clear()
