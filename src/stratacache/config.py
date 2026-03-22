# SPDX-License-Identifier: Apache-2.0
"""
StrataCache Configuration Management

Configuration system for StrataCache inspired by LMCache that:
- Loads configuration from YAML file, environment variables, or parameters
- Supports configuration aliases and deprecation handling
- Provides convenient access to configuration values
- Enables runtime validation and logging
"""

# Standard
from typing import Any, Dict, Optional, List
import json
import logging
import os
import threading
import uuid

# Third Party
import yaml

# Get logger
logger = logging.getLogger(__name__)


# Configuration parsing utilities
def _to_bool(value: Optional[Any]) -> bool:
    """Convert value to boolean."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1", "yes", "on"]


def _to_int(value: Optional[Any]) -> Optional[int]:
    """Convert value to integer."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _to_float(value: Optional[Any]) -> Optional[float]:
    """Convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _to_str_list(value: Optional[Any]) -> Optional[List[str]]:
    """Convert value to list of strings."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts if parts else None
    return None


def _to_int_list(value: Optional[Any]) -> Optional[List[int]]:
    """Convert value to list of integers."""
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [int(p) for p in parts] if parts else None
    return None


def _parse_quoted_string(value: str) -> str:
    """Parse a string that may be surrounded by quotes and handle escape characters."""
    if not value:
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


# Single configuration definition center
_CONFIG_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Connector basic configurations
    "connector_use_cxl": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable CXL (Compute Express Link) support",
    },
    "connector_writeback": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable writeback feature",
    },
    "connector_cpu_capacity_mb": {
        "type": int,
        "default": 61440,
        "env_converter": _to_int,
        "description": "CPU memory capacity in MB",
    },
    "connector_chunk_size": {
        "type": int,
        "default": 256,
        "env_converter": _to_int,
        "description": "Chunk size in tokens",
    },
    "connector_bundle_layers": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
        "description": "Bundle layers feature",
    },
    "connector_tensor_codec": {
        "type": str,
        "default": "stable",
        "env_converter": str,
        "description": "Tensor codec type",
    },
    "connector_tensor_header_in_payload": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Include tensor header in payload",
    },
    "connector_save_partial_chunks": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
        "description": "Save partial chunks to storage",
    },
    "connector_log_stats": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable statistics logging",
    },
    "connector_log_every": {
        "type": int,
        "default": 50,
        "env_converter": _to_int,
        "description": "Log statistics every N operations",
    },
    "connector_log_min_interval_s": {
        "type": float,
        "default": 2.0,
        "env_converter": _to_float,
        "description": "Minimum interval between log entries in seconds",
    },
    "connector_debug": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable debug mode",
    },
    "connector_cxl_dax_device": {
        "type": Optional[str],
        "default": None,
        "env_converter": lambda x: x if x else None,
        "description": "CXL DAX device path",
    },
    "connector_cxl_reset_metadata": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Reset CXL metadata on startup",
    },
    # Exporter configurations - File
    "exporter_file_enabled": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable file exporter",
    },
    "exporter_file_folder": {
        "type": str,
        "default": "/tmp/stratacache-exports",
        "env_converter": str,
        "description": "File exporter output folder",
    },
    # Exporter configurations - WandB
    "exporter_wandb_enabled": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable Weights and Biases exporter",
    },
    "exporter_wandb_entity": {
        "type": Optional[str],
        "default": None,
        "env_converter": lambda x: x if x else None,
        "description": "WandB entity name",
    },
    "exporter_wandb_project": {
        "type": Optional[str],
        "default": "stratacache",
        "env_converter": str,
        "description": "WandB project name",
    },
    "exporter_wandb_run_name": {
        "type": Optional[str],
        "default": "default-run",
        "env_converter": str,
        "description": "WandB run name",
    },
    # Cache policy
    "cache_policy": {
        "type": str,
        "default": "LRU",
        "env_converter": str,
        "description": "Cache eviction policy: LRU, FIFO, LFU"
    },
    # Extra configurations
    "extra_config": {
        "type": Optional[dict],
        "default": None,
        "env_converter": lambda x: x if isinstance(x, dict) else json.loads(x) if x else None,
        "description": "Extra configuration dictionary",
    },
}

# Configuration aliases for backward compatibility
_CONFIG_ALIASES = {
    "use_cxl": "connector_use_cxl",
    "chunk_size": "connector_chunk_size",
    "cpu_capacity_mb": "connector_cpu_capacity_mb",
}

# Deprecated configurations
_DEPRECATED_CONFIGS = {
    "nixl_backends": "nixl_backends is deprecated, use extra_config instead",
}


def _validate_config(self):
    """Validate configuration"""
    if self.connector_cpu_capacity_mb <= 0:
        raise ValueError(
            f"connector_cpu_capacity_mb must be positive, got {self.connector_cpu_capacity_mb}"
        )
    if self.connector_chunk_size <= 0:
        raise ValueError(
            f"connector_chunk_size must be positive, got {self.connector_chunk_size}"
        )
    if self.cache_policy not in ["LRU", "FIFO", "LFU"]:
        raise ValueError(
            f"cache_policy must be one of [LRU, FIFO, LFU], got {self.cache_policy}"
        )
    return self


def _log_config(self):
    """Log configuration"""
    config_dict = {}
    for name in _CONFIG_DEFINITIONS:
        value = getattr(self, name, None)
        config_dict[name] = value
    logger.info(f"StrataCache Configuration: {config_dict}")
    return self


def _get_extra_config_value(self, key, default_value=None):
    """Get value from extra_config dictionary"""
    if hasattr(self, "extra_config") and self.extra_config is not None:
        return self.extra_config.get(key, default_value)
    return default_value


def _flatten_config_dict(config_dict: Dict, prefix: str = "") -> Dict:
    """Recursively flatten nested configuration dictionary.
    
    Converts nested dicts like:
        {'stratacache': {'connector': {'use_cxl': false, 'exporter': {'wandb': {'enabled': true}}}}}
    To flat keys like:
        {'connector_use_cxl': false, 'exporter_wandb_enabled': true}
    """
    flat = {}
    for key, value in config_dict.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            # Skip 'stratacache' root key
            if key == "stratacache":
                flat.update(_flatten_config_dict(value, ""))
            else:
                flat.update(_flatten_config_dict(value, new_key))
        else:
            flat[new_key] = value
    return flat


def _update_config_from_env(self):
    """Update an existing config object with environment variable configurations."""
    def get_env_name(attr_name: str) -> str:
        return f"STRATACACHE_{attr_name.upper()}"

    # Collect environment variables
    env_config = {}
    for name in _CONFIG_DEFINITIONS:
        env_name = get_env_name(name)
        env_value = os.getenv(env_name)
        if env_value is not None:
            env_config[name] = env_value

    # Update config object with environment values
    for name, config in _CONFIG_DEFINITIONS.items():
        if name in env_config:
            try:
                raw_value = env_config[name]
                value = _parse_quoted_string(raw_value)
                converted_value = config["env_converter"](value)
                setattr(self, name, converted_value)
                # Mark as user-set
                self._user_set_keys.add(name)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Failed to parse {get_env_name(name)}={raw_value!r}: {e}"
                )
    self.validate()
    return self


def _find_config_file() -> Optional[str]:
    """Find config.yaml in default locations.
    
    Searches in the following order:
    1. Current working directory (./config.yaml)
    2. User home directory (~/.stratacache/config.yaml)
    3. Same directory as this module
    
    Returns:
        Path to the first config file found, or None if no config file exists
    """
    locations = [
        "./config.yaml",
        os.path.expanduser("~/.stratacache/config.yaml"),
        os.path.join(os.path.dirname(__file__), "config.yaml"),
    ]

    for loc in locations:
        if os.path.exists(loc):
            logger.info(f"Found config file: {loc}")
            return loc

    return None


# Create configuration class
class StrataCacheConfig:
    """StrataCache Configuration class"""
    
    def __init__(self, **kwargs):
        """Initialize configuration with given values"""
        # Initialize user-set keys tracking
        object.__setattr__(self, "_user_set_keys", set())
        
        # Set all configuration values
        for name, config in _CONFIG_DEFINITIONS.items():
            if name in kwargs:
                value = kwargs[name]
                self._user_set_keys.add(name)
            else:
                value = config["default"]
            setattr(self, name, value)
        
        # Generate instance ID
        object.__setattr__(self, "stratacache_instance_id", f"stratacache_{uuid.uuid4().hex[:8]}")
    
    @classmethod
    def from_defaults(cls, **kwargs):
        """Create configuration from defaults with optional overrides"""
        config_values = {}
        user_set_keys = set()
        for name, config in _CONFIG_DEFINITIONS.items():
            if name in kwargs:
                value = kwargs[name]
                user_set_keys.add(name)
            else:
                value = config["default"]
            config_values[name] = value
        
        instance = cls(**config_values)
        object.__setattr__(instance, "_user_set_keys", user_set_keys)
        return instance
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        def get_env_name(attr_name: str) -> str:
            return f"STRATACACHE_{attr_name.upper()}"
        
        config_values = {}
        user_set_keys = set()
        for name, config in _CONFIG_DEFINITIONS.items():
            env_name = get_env_name(name)
            env_value = os.getenv(env_name)
            if env_value is not None:
                try:
                    value = _parse_quoted_string(env_value)
                    config_values[name] = config["env_converter"](value)
                    user_set_keys.add(name)
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(
                        f"Failed to parse {env_name}={env_value!r}: {e}, using default"
                    )
                    config_values[name] = config["default"]
            else:
                config_values[name] = config["default"]
        
        instance = cls(**config_values)
        object.__setattr__(instance, "_user_set_keys", user_set_keys)
        return instance
    
    @classmethod
    def from_file(cls, file_path: str):
        """Load configuration from YAML file"""
        try:
            with open(file_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            file_config = {}
        
        # Flatten nested dict to match config keys
        flat_config = _flatten_config_dict(file_config)
        
        config_values = {}
        user_set_keys = set()
        for name, config in _CONFIG_DEFINITIONS.items():
            if name in flat_config:
                value = flat_config[name]
                user_set_keys.add(name)
            else:
                value = config["default"]
            config_values[name] = value
        
        instance = cls(**config_values)
        object.__setattr__(instance, "_user_set_keys", user_set_keys)
        logger.info(f"Loaded config from {file_path}")
        return instance
    
    def validate(self):
        """Validate configuration"""
        return _validate_config(self)
    
    def log_config(self):
        """Log configuration"""
        return _log_config(self)
    
    def get_extra_config_value(self, key, default_value=None):
        """Get value from extra_config"""
        return _get_extra_config_value(self, key, default_value)
    
    def update_config_from_env(self):
        """Update configuration from environment variables"""
        return _update_config_from_env(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {name: getattr(self, name) for name in _CONFIG_DEFINITIONS}
    
    def __repr__(self) -> str:
        return f"StrataCacheConfig(instance_id={self.stratacache_instance_id})"


# Thread-safe singleton management
_config_instance: Optional[StrataCacheConfig] = None
_config_lock = threading.Lock()


def get_config(config_path: Optional[str] = None) -> StrataCacheConfig:
    """Get or create the global config instance.
    
    Searches for config files in the following order:
    1. Explicitly provided config_path
    2. STRATACACHE_CONFIG_FILE environment variable
    3. Default locations: ./config.yaml, ~/.stratacache/config.yaml, <module_dir>/config.yaml
    4. Environment variables only
    
    Args:
        config_path: Optional path to config file (only used on first call)
        
    Returns:
        StrataCacheConfig instance
        
    Example:
        config = get_config()
        use_cxl = config.connector_use_cxl
    """
    global _config_instance
    
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                # Determine which config file to use (in priority order)
                config_file = None
                
                if config_path and os.path.exists(config_path):
                    config_file = config_path
                elif env_file := os.getenv("STRATACACHE_CONFIG_FILE"):
                    if os.path.exists(env_file):
                        config_file = env_file
                else:
                    # Try to find config file in default locations
                    config_file = _find_config_file()
                
                # Load configuration
                if config_file:
                    logger.info(f"Loading config from: {config_file}")
                    _config_instance = StrataCacheConfig.from_file(config_file)
                    # Allow environment variables to override file settings
                    _config_instance.update_config_from_env()
                else:
                    logger.info("No config file found, loading from environment variables")
                    _config_instance = StrataCacheConfig.from_env()
                
                _config_instance.validate()
                _config_instance.log_config()
    
    return _config_instance


def reset_config():
    """Reset the global config instance (mainly for testing)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


def load_config_with_overrides(
    config_file_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> StrataCacheConfig:
    """Load configuration with support for file, env vars, and overrides.
    
    Searches for config files in the following order:
    1. Explicitly provided config_file_path
    2. STRATACACHE_CONFIG_FILE environment variable
    3. Default locations: ./config.yaml, ../config.yaml, ~/.stratacache/config.yaml, <module_dir>/config.yaml
    4. Environment variables only
    
    Args:
        config_file_path: Optional direct path to config file
        overrides: Optional dictionary of configuration overrides
        
    Returns:
        Loaded and validated StrataCacheConfig instance
    """
    # Determine which config file to use (in priority order)
    config_file = None
    
    if config_file_path and os.path.exists(config_file_path):
        config_file = config_file_path
    elif env_file := os.getenv("STRATACACHE_CONFIG_FILE"):
        if os.path.exists(env_file):
            config_file = env_file
    else:
        # Try to find config file in default locations
        config_file = _find_config_file()
    
    # Load configuration
    if config_file:
        logger.info(f"Loading config file: {config_file}")
        config = StrataCacheConfig.from_file(config_file)
        # Allow environment variables to override file settings
        config.update_config_from_env()
    else:
        logger.info("No config file found, loading from environment variables")
        config = StrataCacheConfig.from_env()
    
    # Validate and log
    config.validate()
    config.log_config()
    return config
