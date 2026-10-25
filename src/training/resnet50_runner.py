

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _validate_seed(seed: object) -> None:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise ValueError("experiment.seed must be an integer in [0, 2**32).")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read YAML and validate the settings used by experiment setup."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")

    for section in ("experiment", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping.")

    name = config["experiment"].get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("experiment.name must be a non-empty string.")

    _validate_seed(config["experiment"].get("seed"))

    if config["training"].get("device") not in ("cpu", "cuda"):
        raise ValueError("training.device must be 'cpu' or 'cuda'.")

    return config


def seed_everything(seed: int) -> None:
    """Seed random generators and configure deterministic cuDNN behavior."""
    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name: str) -> torch.device:
    """Select CPU or CUDA, requiring CUDA when it is requested."""
    if name not in ("cpu", "cuda"):
        raise ValueError("Device must be 'cpu' or 'cuda'.")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Run with the CUDA-enabled project environment."
        )

    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resnet50.yaml"),
        help="YAML configuration path; run from the repository root.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        device = select_device(config["training"]["device"])
        seed_everything(config["experiment"]["seed"])
    except (OSError, yaml.YAMLError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Configuration: {args.config.resolve()}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Seed: {config['experiment']['seed']}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("Setup complete. Data loading and training are not connected yet.")


if __name__ == "__main__":
    main()
