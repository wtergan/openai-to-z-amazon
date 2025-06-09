import os
from pprint import pprint
from dataset_fetching import fetch_dataset, initialize_gee
from feature_extraction import lidar_ot_extract_features, sentinel2_gee_extract_features
from openai_integration import call_model_responses, call_openai_responses, OPENROUTER_PROVIDER, OPENROUTER_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL

# ===============================================================================
# ENVIRONMENT SETUP
# ===============================================================================
DATASET_TYPE = os.environ.get("DATASET_TYPE", "lidar")  # 'lidar' or 'sentinel2'.
API_TYPE = os.environ.get("API_TYPE", "openai")  # 'openai' or 'openrouter'.
if DATASET_TYPE not in ["lidar", "sentinel2"]:
    raise ValueError("DATASET_TYPE must be 'lidar' or 'sentinel2'")
if API_TYPE not in ["openai", "openrouter"]:
    raise ValueError("API_TYPE must be 'openai' or 'openrouter'")

# Optional: overrides default dataset types in dataset_fetching.py
# DATASET_ID = os.environ.get("DATASET_ID")
# if DATASET_ID is None:
#     print("No Optional DATASET_ID provided, using default from dataset_fetching.py")

GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID")

# ===============================================================================
# MAIN FUNCTION
# ===============================================================================
def main():
    # Safety check for GEE initialization:
    if not initialize_gee():
        if DATASET_TYPE == "sentinel2":
            print("CRITICAL: GEE initialization failed. Cannot proceed with Sentinel-2 dataset type.")
            return # Exit if GEE needed but not available
        else:
            print("Warning: GEE initialization failed, but proceeding as dataset type is not Sentinel-2.")

    # Fetching specified dataset data:
    print(f"Fetching {DATASET_TYPE} dataset for data retrieval...")
    data = fetch_dataset(DATASET_TYPE)
    if data:
        print(f"Successfully fetched {DATASET_TYPE} dataset.")
    else:
        print(f"Error: Failed to fetch {DATASET_TYPE} dataset.")
        return

    # Analysis of fetched data; if data not valid in any case, exit:
    print(f"Analyzing fetched {DATASET_TYPE} dataset...")
    analysis_results = None
    if DATASET_TYPE == "lidar":
        if data and isinstance(data, str):
            print(f"Extracting features and statistics from the LiDAR data from {data}\n\n")
            analysis_results = lidar_ot_extract_features(data)
            print(f"Successfully extracted features and statistics from the LiDAR data from {data}")
            stats_to_display = {k: v for k, v in analysis_results.items() if k not in ['image', 'ndvi_image', 'false_color_image']}
            pprint(stats_to_display)
        else:
            print(f"Error: LiDAR data file path is invalid or not fetched: {data}\n")
            return
    elif DATASET_TYPE == "sentinel2":
        if data and isinstance(data, dict) and data.get("image"):
            print(f"Extracting features and statistics from the Sentinel-2 data from {data}")
            analysis_results = sentinel2_gee_extract_features(data)
            print(f"Successfully extracted features and statistics from the Sentinel-2 data from {data}\n")
            stats_to_display = {k: v for k, v in analysis_results.items() if k not in ['image', 'ndvi_image', 'false_color_image']}
            pprint(stats_to_display)
        elif data and isinstance(data, dict) and data.get("error"):
            print(f"Error: {data['error']}")
            return
        else:
            print(f"Error: Sentinel-2 data is invalid or not fetched: {data}")
            return
    else:
        raise ValueError("Unknown DATASET_TYPE")

    # If analysis_results is not valid, exit as well:
    if not analysis_results or not analysis_results.get("statistics"):
        print("Error: Analysis results are invalid or not fetched.")
        return
    
    # Call the model (defaults to OpenRouter):
    print("\nCalling model for prompting...")
    if analysis_results:
        if API_TYPE == "openai":
            llm_response = call_openai_responses(analysis_results, dataset_type=DATASET_TYPE, model=OPENAI_DEFAULT_MODEL)
        elif API_TYPE == "openrouter":
            llm_response = call_openrouter_responses(analysis_results, dataset_type=DATASET_TYPE, model=OPENROUTER_DEFAULT_MODEL)
        print(llm_response)
    
    #print(f"Provider: {OPENROUTER_PROVIDER}")
    #print(f"Model: {OPENROUTER_DEFAULT_MODEL}")
    #print(f"Dataset type: {DATASET_TYPE}")
    #print(f"Stats: {stats}")
    #print("Model output:\n", summary)

if __name__ == "__main__":
    main()
