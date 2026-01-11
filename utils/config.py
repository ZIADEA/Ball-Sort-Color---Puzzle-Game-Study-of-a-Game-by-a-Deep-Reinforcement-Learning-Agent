"""Configuration management utilities."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil


class ConfigManager:
    """Manage configuration loading, merging, and saving."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to config file (YAML or JSON)
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
        
        # Load default config
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with custom config if provided
        if self.config_path:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                        custom_config = yaml.safe_load(f)
                    elif config_path.suffix == '.json':
                        custom_config = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config format: {config_path.suffix}")
                
                # Deep merge
                config = self._deep_merge(config, custom_config)
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example: config.get('training.seed')
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def save(self, run_name: str, output_dir: str = "result/configs"):
        """
        Save configuration to file.
        
        Args:
            run_name: Name of the run
            output_dir: Directory to save config
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{run_name}_{timestamp}.yaml"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to: {filepath}")
        return str(filepath)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()


def setup_directories(config: ConfigManager):
    """Create all necessary directories based on config."""
    paths = config.get('paths', {})
    for key, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    print("✓ All directories created")


def get_run_name(algo: str, level: int = 1) -> str:
    """Generate a run name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{algo}_level{level}_{timestamp}"
