import os
from dataset_fetching import fetch_dataset
from feature_extraction import ot_lidar_plot, sentinel2_stats
from openai_integration import call_model_responses, OPENROUTER_PROVIDER, OPENROUTER_DEFAULT_MODEL

# ===============================================================================
# ENVIRONMENT SETUP
# ===============================================================================
DATASET_TYPE = os.environ.get("DATASET_TYPE", "lidar")  # 'lidar' or 'sentinel2'
if DATASET_TYPE not in ["lidar", "sentinel2"]:
    raise ValueError("DATASET_TYPE must be 'lidar' or 'sentinel2'")

# Optional: overrides default dataset types in dataset_fetching.py
# DATASET_ID = os.environ.get("DATASET_ID")
# if DATASET_ID is None:
#     print("No Optional DATASET_ID provided, using default from dataset_fetching.py")

# ===============================================================================
# MAIN FUNCTION
# ===============================================================================
def main():
    # Fetch dataset
    data = fetch_dataset(DATASET_TYPE)
    if DATASET_TYPE == "lidar":
        plot_stats = ot_lidar_plot(data)
    elif DATASET_TYPE == "sentinel2":
        stats = sentinel2_stats(data)
    else:
        raise ValueError("Unknown DATASET_TYPE")

    # Printing plot stats first:
    print({k: v for k, v in plot_stats.items() if k != 'plot'})
    
    # Call the model (defaults to OpenRouter)
    summary = call_model_responses(plot_stats, provider=OPENROUTER_PROVIDER, model=OPENROUTER_DEFAULT_MODEL)
    print(summary)
    
    #print(f"Provider: {OPENROUTER_PROVIDER}")
    #print(f"Model: {OPENROUTER_DEFAULT_MODEL}")
    #print(f"Dataset type: {DATASET_TYPE}")
    #print(f"Stats: {stats}")
    #print("Model output:\n", summary)

if __name__ == "__main__":
    main()
