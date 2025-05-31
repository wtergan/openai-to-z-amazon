import os
from dataset_fetching import fetch_dataset
from feature_extraction import laspy_stats, sentinel2_stats
from openai_integration import call_openai_responses, MODEL_NAME

# ===============================================================================
# ENVIRONMENT SETUP
# ===============================================================================
DATASET_TYPE = os.environ.get("DATASET_TYPE", "lidar")  # 'lidar' or 'sentinel2'
if DATASET_TYPE not in ["lidar", "sentinel2"]:
    raise ValueError("DATASET_TYPE must be 'lidar' or 'sentinel2'")

DATASET_ID = os.environ.get("DATASET_ID")  # Optional: override default in dataset_fetching.py
if DATASET_ID is None:
    print("No DATASET_ID provided, using default from dataset_fetching.py")

# ===============================================================================
# MAIN FUNCTION
# ===============================================================================
def main():
    # Fetch dataset
    data = fetch_dataset(DATASET_TYPE)
    if DATASET_TYPE == "lidar":
        stats = laspy_stats(data)
    elif DATASET_TYPE == "sentinel2":
        stats = sentinel2_stats(data)
    else:
        raise ValueError("Unknown DATASET_TYPE")

    # Call OpenAI
    summary = call_openai_responses(stats)

    # Print outputs
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset type: {DATASET_TYPE}")
    print(f"Stats: {stats}")
    print("Model output:\n", summary)

if __name__ == "__main__":
    main()
