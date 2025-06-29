# Setup

This document explains the practical setup required to inspect or rerun the OpenAI to Z Challenge code in this repository.

The project was built as a challenge prototype, not a packaged Python library. The checkpoint notebooks are preserved historical artifacts, while the `.py` files are the modularized version of the same workflow.

## Recommended environment

Use Python 3.10 or newer. Python 3.12 is the current local audit target.

From the repo root:

```bash
cd /path/to/openai-to-z-challenge
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

For the broadest local setup, install both checkpoint requirement sets:

```bash
pip install -r CHECKPOINT_1/requirements.txt -r CHECKPOINT_2/requirements.txt pytest
```

The Checkpoint 2 dependency story still needs cleanup. In practice, Checkpoint 2 uses some packages also listed in Checkpoint 1, including model/API and raster tooling.

## Required environment variables

Create a `.env` file in the repo root or checkpoint directory with the credentials needed for the paths you plan to run:

```bash
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
OT_API_KEY=your_opentopography_key_here
GEE_PROJECT_ID=your_ee_project_id_here
```

Not every workflow needs every key:

| Variable | Used for |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI model interpretation |
| `OPENROUTER_API_KEY` | OpenRouter model interpretation |
| `OT_API_KEY` | OpenTopography LiDAR / DEM access |
| `GEE_PROJECT_ID` | Google Earth Engine project initialization |

## Google Earth Engine authentication

Several paths depend on Google Earth Engine:

- Sentinel-2 imagery;
- GEDI data;
- SRTM terrain data;
- parts of the full Checkpoint 2 pipeline.

Authenticate before running those paths:

```bash
earthengine authenticate
```

or use the equivalent Earth Engine Python authentication flow.

If Earth Engine auth is missing or expired, GEE-dependent reruns should be expected to fail even if the Python environment is otherwise correct.

## Checkpoint 1 modular workflow

Checkpoint 1 is the easiest path to inspect and partially rerun.

```bash
cd CHECKPOINT_1
python console_output.py
```

To use the Sentinel-2 path:

```bash
cd CHECKPOINT_1
export DATASET_TYPE=sentinel2
python console_output.py
```

To use OpenRouter instead of OpenAI:

```bash
export API_TYPE=openrouter
python console_output.py
```

## Checkpoint 2 modular workflow

Checkpoint 2 is more demanding because it combines multiple data sources and more external services.

Example command:

```bash
cd CHECKPOINT_2
python console_output.py --bbox=-59.9,-5.2,-59.8,-5.1 --year 2022 --top_n 5
```

With LLM assessment:

```bash
python console_output.py \
  --bbox=-59.9,-5.2,-59.8,-5.1 \
  --year 2022 \
  --top_n 5 \
  --llm \
  --llm_provider openrouter \
  --llm_model google/gemma-3-27b-it
```

Expect this path to require working Earth Engine authentication and live external services.

## Notebook execution

The notebooks are preserved as challenge artifacts:

- `CHECKPOINT_1/Checkpoint_1.ipynb`
- `CHECKPOINT_2/Checkpoint_2.ipynb`

They define substantial logic directly inside notebook cells. They are not currently thin wrappers around the modularized `.py` files.

To rerun them, use Colab, Jupyter, or Kaggle with the required dependencies and credentials. Treat any full rerun as historical workflow reproduction rather than a guaranteed deterministic build.

## Known setup caveats

- Full reruns depend on external service availability and credentials.
- Integration tests are opt-in through `RUN_GEE_INTEGRATION_TESTS=1` or `RUN_API_INTEGRATION_TESTS=1`.
- Preserved JSON outputs and notebooks are the safest artifacts for review without rerunning the whole pipeline.
