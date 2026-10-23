import os
import random
import json
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(verbose: bool = True) -> torch.device:
    """Detects available hardware and prints environment details."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        if verbose:
            print(f"Selected Device: {device} ({torch.cuda.get_device_name(0)})")
            print(f"CUDA Version: {torch.version.cuda}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Available VRAM: {vram_gb:.2f} GB")
    else:
        device = torch.device('cpu')
        if verbose:
            print("CUDA is not available. Using CPU fallback.")
    return device


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: dict, file_path: str) -> None:
    """Saves dictionary to JSON file with formatting."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str) -> dict:
    """Loads dictionary from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
