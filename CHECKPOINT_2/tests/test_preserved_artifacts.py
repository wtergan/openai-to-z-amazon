"""Offline smoke tests for preserved Checkpoint 2 artifacts.

These tests use only the Python standard library so the suite has at least one
useful check even before geospatial/API dependencies are installed.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_preserved_top5_outputs_are_valid_json():
    for relative_path in ["test-run_top5.json", "test-run_top5_llm.json"]:
        path = ROOT / relative_path
        assert path.exists(), f"missing preserved output: {relative_path}"
        with path.open() as f:
            data = json.load(f)
        assert data, f"preserved output is empty: {relative_path}"


def test_checkpoint_notebook_is_preserved_in_checkpoint_directory():
    assert (ROOT / "Checkpoint_2.ipynb").exists()
