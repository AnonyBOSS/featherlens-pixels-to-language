#!/usr/bin/env python3
"""Standalone executable entry point for hyperparameter tuning."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import _cmd_tune


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grid Search & Random Search tuning")
    parser.add_argument("--task", default="classification", choices=["classification"])
    parser.add_argument("--config", default="configs/cls.yaml", help="Path to YAML config")
    args = parser.parse_args()
    _cmd_tune(args)


if __name__ == "__main__":
    main()
