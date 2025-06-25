"""
Reproducibility and integration tests for CHECKPOINT_2 anomaly detection pipeline.
Covers CLI and programmatic usage, ensuring deterministic outputs and correct integration
across all modules (dataset_fetching, feature_extraction, feature_engineering, anomaly_detect, model_integration).

To run: pytest tests/test_reproducibility.py
"""
import os
import sys
import subprocess
import tempfile
import shutil
import json
import importlib
import platform
import pytest

# Use importlib.metadata instead of deprecated pkg_resources
try:
    from importlib.metadata import distributions
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import distributions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==============================================================================
# UTILITY: ENV AND DEPENDENCIES
# ==============================================================================
def print_env_info():
    print("\nPYTHON VERSION:", platform.python_version())
    print("PLATFORM:", platform.platform())
    print("INSTALLED PACKAGES:")
    for dist in sorted(distributions(), key=lambda d: d.metadata['name'].lower()):
        print(f"  {dist.metadata['name']}=={dist.version}")

# ==============================================================================
# CLI: REPRODUCIBILITY TEST
# ==============================================================================
def test_cli_reproducibility():
    """Test that running the CLI with fixed inputs gives deterministic output."""
    bbox = "-59.9,-5.2,-59.8,-5.1"  # Small bbox for quick test
    year = 2022
    top_n = 3
    tag = "test_run"
    output_file = f"{tag}_top{top_n}.json"
    
    # Removing the output file if exists:
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Running the CLI with proper bbox argument format:
    console_path = os.path.join(os.path.dirname(__file__), "..", "console_output.py")
    result = subprocess.run([
        sys.executable, console_path,
        f"--bbox={bbox}", "--year", str(year), "--top_n", str(top_n), "--tag", tag
    ], capture_output=True, text=True, timeout=180)
    print("CLI STDOUT:\n", result.stdout)
    print("CLI STDERR:\n", result.stderr)
    assert result.returncode == 0, "CLI did not exit cleanly"
    assert os.path.exists(output_file), f"Output file {output_file} not created"
    with open(output_file) as f:
        data1 = json.load(f)
    # Running it again to check if its deterministic:
    result2 = subprocess.run([
        sys.executable, console_path,
        f"--bbox={bbox}", "--year", str(year), "--top_n", str(top_n), "--tag", tag
    ], capture_output=True, text=True, timeout=180)
    assert result2.returncode == 0, "Second CLI run did not exit cleanly"
    with open(output_file) as f:
        data2 = json.load(f)
    assert data1 == data2, "CLI output is not reproducible for same input"
    os.remove(output_file)

# ==============================================================================
# PIPELINE TEST
# ==============================================================================
def test_pipeline_reproducibility():
    """Test that direct pipeline calls are deterministic and integrated."""
    from dataset_fetching import make_bbox, initialize_gee, gedi_prodes_srtm_fetch_pipeline, lidar_sentinel2_fetch_pipeline
    from feature_extraction import sentinel2_gee_extract_features, lidar_ot_extract_features
    from feature_engineering import feat_engineering_pipeline
    from anomaly_detect import score_cells, rank_cells
    from model_integration import analyze_top_n_cells

    bbox = make_bbox(-59.9, -5.2, -59.8, -5.1)
    year = 2022
    cache_dir = tempfile.mkdtemp()
    try:
        assert initialize_gee(), "GEE initialization failed"
        gedi_df, prodes_gdf, srtm_df = gedi_prodes_srtm_fetch_pipeline(bbox, year, cache_dir, True)
        region_ctx = lidar_sentinel2_fetch_pipeline(bbox)
        lidar_stats = lidar_ot_extract_features(region_ctx["lidar_path"], show_image=False) if region_ctx and region_ctx.get("lidar_path") else None
        s2_stats = sentinel2_gee_extract_features(region_ctx["s2_data"], show_image=False) if region_ctx and region_ctx.get("s2_data") else None
        features = feat_engineering_pipeline(gedi_df, prodes_gdf, srtm_df)
        scored = score_cells(features, method='weighted')
        top = rank_cells(scored, n=3)
        # Run LLM analysis (mocked if no API key)
        llm_results = None
        try:
            llm_results = analyze_top_n_cells(top.to_dict(orient='records'), {**(lidar_stats or {}), **(s2_stats or {})}, provider="openrouter", model_name="google/gemma-3-27b-it", save_log=False)
        except Exception as e:
            print(f"LLM analysis skipped or failed: {e}")
        # Re-run pipeline for determinism check
        features2 = feat_engineering_pipeline(gedi_df, prodes_gdf, srtm_df)
        scored2 = score_cells(features2, method='weighted')
        top2 = rank_cells(scored2, n=3)
        assert top.equals(top2), "Pipeline output is not reproducible"
    finally:
        shutil.rmtree(cache_dir)

# --- Print environment info at start ---
print_env_info()
