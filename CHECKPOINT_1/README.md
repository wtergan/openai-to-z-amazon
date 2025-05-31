# Remote Sensing + OpenAI Natural Language Analysis

This project fetches a sample remote-sensing dataset (LiDAR or Sentinel-2), extracts minimal features, and generates a plain-English summary using an OpenAI model.

## Quickstart

### Option 1: Using conda (recommended for easier dependency management)
1. **Environment setup**
   - Install [conda](https://docs.conda.io/en/latest/)
   - Create and activate the environment:
     ```sh
     conda create -n amazon-env python=3.11
     conda activate amazon-env
     conda install -c conda-forge rasterio laspy pdal
     pip install openai boto3 python-dotenv
     ```

### Option 2: Using venv
1. **Environment setup**
   - Create and activate a virtual environment:
     ```sh
     python -m venv amazon-env
     source amazon-env/bin/activate  # On Windows: .\amazon-env\Scripts\activate
     pip install rasterio laspy pdal openai boto3 python-dotenv
     ```
   - Note: On some systems, you may need to install system dependencies first:
     ```sh
     # Ubuntu/Debian
     sudo apt-get install -y python3-dev libgdal-dev
     
     # macOS (using Homebrew)
     brew install gdal
     ```
2. **API key**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=sk-...
     ```
3. **Run the workflow**
   - By default, runs LiDAR. To run Sentinel-2, set `DATASET_TYPE=sentinel2` in your environment.
   - Run:
     ```sh
     python console_output.py
     ```

## Files
- `dataset_fetching.py` – Download LiDAR or Sentinel-2 data
- `feature_extraction.py` – Compute minimal stats for each dataset
- `openai_integration.py` – Send stats to OpenAI and get summary
- `console_output.py` – Main script: fetch, extract, call model, print

## Customization
- Change dataset IDs or model in the respective Python files or via environment variables.

## Reproducibility
- All dependencies are listed in the environment setup.
- API keys are kept out of code via `.env`.

## License
MIT
