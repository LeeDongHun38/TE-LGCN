"""Configuration loading and saving utilities."""

import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file.

    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save YAML file
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
