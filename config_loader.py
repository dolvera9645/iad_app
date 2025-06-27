# config_loader.py
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "g_value": 0.8,
    "reference_wavelength": 600,
    "fit_min": 600,
    "fit_max": 750,
    "use_dual_beam": False,
    "iad_exe": "iad.exe",
    "input_dir": "iad_inputs",
    "output_dir": "iad_outputs",
    "font_size": 18
}

def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with open(config_path, 'r') as file:
        user_config = yaml.safe_load(file) or {}
    return {**DEFAULT_CONFIG, **user_config}
