# Checkpoint 2: Archaeological Anomaly Detection Pipeline

This checkpoint expands the project into a multi-source anomaly-detection workflow for the Amazon rainforest using GEDI canopy structure, PRODES deforestation context, SRTM terrain data, H3 spatial aggregation, and LLM-assisted assessment.

The notebook is the historical challenge artifact; the Python modules preserve the same workflow in a more reusable form.

## What this checkpoint does

### 1. Feature engineering (`feature_engineering.py`)

- Snaps GEDI shots to H3 cells at resolution 9.
- Joins canopy, terrain, and disturbance context into per-cell feature vectors.
- Computes derived features for canopy variability, shot density, terrain complexity, and deforestation impact.
- Normalizes scoring features while preserving raw values for interpretation.

### 2. Anomaly scoring (`anomaly_detect.py`)

- Provides a weighted heuristic score based on archaeological search assumptions.
- Provides an Isolation Forest option for unsupervised anomaly ranking.
- Produces ranked candidate cells for follow-up review.

### 3. LLM interpretation (`model_integration.py`)

- Supports OpenAI and OpenRouter-style model calls.
- Builds multimodal prompts using per-cell metrics plus regional LiDAR/Sentinel-2 context.
- Includes a batch path so the regional context can be sent once while assessing multiple top cells.

### 4. Tests and preserved artifacts

- Offline artifact tests validate preserved top-N and LLM output files.
- Integration tests are intentionally gated behind environment variables because full reruns require live credentials and external services.

## Preserved result snapshot

Preserved project artifacts show a run that produced:

- thousands of GEDI shots and SRTM terrain samples processed into H3 cells;
- ranked candidate cells in `test-run_top5.json`;
- regional and per-cell LLM assessments in `test-run_top5_llm.json`.

The results should be treated as candidate leads and technical evidence of the workflow, not as confirmed archaeological discoveries.

## Usage

From this directory or the repository root, install the project dependencies and run the offline-safe tests first:

```bash
PYTHONPATH=CHECKPOINT_2 pytest CHECKPOINT_2/tests -q
```

Live reruns require Google Earth Engine, OpenTopography, and model-provider credentials. See the root-level docs for the fuller setup and reproducibility notes:

- [`../docs/setup.md`](../docs/setup.md)
- [`../docs/reproducibility.md`](../docs/reproducibility.md)
- [`../docs/results.md`](../docs/results.md)

## Direct module usage

```python
from feature_engineering import feat_engineering_pipeline
from anomaly_detect import score_cells, rank_cells
from model_integration import analyze_top_n_cells_batch

features_df = feat_engineering_pipeline(gedi_df, prodes_gdf, srtm_df)
scored_df = score_cells(features_df, method="weighted")
top_cells = rank_cells(scored_df, n=5)

assessment = analyze_top_n_cells_batch(
    top_cells.to_dict("records"),
    lidar_s2_context,
    provider="openrouter",
)
```

## File structure

```text
CHECKPOINT_2/
├── Checkpoint_2.ipynb
├── README.md
├── anomaly_detect.py
├── console_output.py
├── dataset_fetching.py
├── feature_engineering.py
├── feature_extraction.py
├── model_integration.py
├── test-run_top5.json
├── test-run_top5_llm.json
├── requirements.txt
└── tests/
```

## Reproducibility notes

The safest path today is to inspect the preserved notebooks, JSON artifacts, and offline tests first. Full end-to-end reruns depend on service availability, current API credentials, Google Earth Engine access, and geospatial package setup.
