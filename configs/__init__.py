"""Config loading utilities."""
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config_cache = None


def load_config():
    """Load config.yaml and cache it."""
    global _config_cache
    if _config_cache is None:
        with open(_CONFIG_PATH, 'r') as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_dataset_paths(dataset_name: str) -> dict:
    """Get dataset paths from config.

    Args:
        dataset_name: Dataset name (e.g., 'kitti', 'ms2', 'tartanair')

    Returns:
        dict with path keys (e.g., {'rgb_path': ..., 'depth_path': ...})
    """
    config = load_config()
    if dataset_name not in config['datasets']:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(config['datasets'].keys())}")
    return config['datasets'][dataset_name]


def get_model_config(model_name: str) -> dict:
    """Get model config from config.yaml.

    Args:
        model_name: Model name (e.g., 'depth_anything_v2_rel')

    Returns:
        dict with model config
    """
    config = load_config()
    if model_name not in config['models']:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(config['models'].keys())}")
    return config['models'][model_name]
