# Remote Sensing + OpenAI Natural Language Analysis

This project fetches a sample remote-sensing dataset (LiDAR or Sentinel-2), extracts minimal features, and generates a plain-English summary using an OpenAI model.

## Environment Setup

### Option 1: Using conda (recommended)
1. **Install conda**
   - Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

2. **Create and activate environment**
   ```bash
   # Navigate to the CHECKPOINT_1 directory
   cd CHECKPOINT_1
   
   # Create environment from environment.yml
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate amazon-env-checkpoint1
   ```

### Option 2: Using pip/venv
1. **Create and activate virtual environment**
   ```bash
   # Navigate to the CHECKPOINT_1 directory
   cd CHECKPOINT_1
   
   # Create virtual environment
   python -m venv venv
   
   # Activate the environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # .\venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **System Dependencies (if needed)**
   Some packages might require system-level dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y python3-dev libgdal-dev
   
   # macOS (using Homebrew)
   brew install gdal
   ```

## Configuration

1. **API Keys**
   Create a `.env` file in the CHECKPOINT_1 directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OT_API_KEY=your_ot_api_key_here
   GEE_PROJECT_ID=your_ee_project_id_here
   ..........
   ```

## Usage

1. **Run the workflow**
   ```bash
   # Default: Runs LiDAR analysis using OpenAI
   python console_output.py
   
   # For Sentinel-2 analysis
   export DATASET_TYPE=sentinel2
   python console_output.py
   
   # To use OpenRouter instead of OpenAI
   export API_TYPE=openrouter
   python console_output.py
   
   # Combine options (e.g., Sentinel-2 with OpenRouter)
   export DATASET_TYPE=sentinel2
   export API_TYPE=openrouter
   python console_output.py
   ```

### Environment Variables
- `DATASET_TYPE`: Set to `lidar` (default) or `sentinel2`
- `API_TYPE`: Set to `openai` (default) or `openrouter`

## Project Structure

- `dataset_fetching.py` – Download LiDAR or Sentinel-2 data
- `feature_extraction.py` – Compute minimal stats for each dataset
- `openai_integration.py` – Send stats to OpenAI and get summary
- `console_output.py` – Main script: fetch, extract, call model, print
- `requirements.txt` – Python package dependencies (for pip)
- `environment.yml` – Conda environment specification

## Customization
- Change dataset IDs or model in the respective Python files or via environment variables.

## Reproducibility
- All dependencies are listed in the environment setup.
- API keys are kept out of code via `.env`.

## License
MIT
