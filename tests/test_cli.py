"""Offline unit tests for CLI argument parsing and dispatch."""

import subprocess
import sys

import pytest


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "train" in result.stdout
    assert "tune" in result.stdout
    assert "gradcam" in result.stdout
    assert "predict" in result.stdout


def test_cli_subcommand_help():
    for subcmd in ("train", "tune", "gradcam", "predict"):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", subcmd, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert f"cli.py {subcmd}" in result.stdout
