import os
import tempfile
import requests
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from .env if present

# ===============================================================================
# LiDAR PARAMETERS
# ===============================================================================
LIDAR_DATASET_ID = "OT.072016.32611.1"  # Example OpenTopography dataset
LIDAR_URL = (
    "https://portal.opentopography.org"
    f"/getOTDataset?datasetID={LIDAR_DATASET_ID}&fileFormat=LAZ"
)

def fetch_lidar(url: str = LIDAR_URL) -> str:
    """Download a small LiDAR .laz file from OpenTopography."""
    tf = tempfile.NamedTemporaryFile(suffix=".laz", delete=False)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    tf.write(resp.content)
    tf.close()
    return tf.name

# ===============================================================================
# Sentinel-2 PARAMETERS
# ===============================================================================
S2_TILE = "22/MU/L"
S2_DATE_PATH = "2024/5/26/0"
S2_BANDS = ["2", "3", "4"]  # B02 (blue), B03 (green), B04 (red)

def fetch_sentinel2_bands(tile=S2_TILE, date_path=S2_DATE_PATH, bands=S2_BANDS):
    """Download three Sentinel-2 RGB bands as small .jp2 files from AWS."""
    base_url = (
        "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/"
        f"{tile}/{date_path}/B0{{band}}.jp2"
    )
    band_files = {}
    for band in bands:
        url = base_url.format(band=band)
        tf = tempfile.NamedTemporaryFile(suffix=f"_B0{band}.jp2", delete=False)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tf.write(resp.content)
        tf.close()
        band_files[band] = tf.name
    return band_files

# ===============================================================================
# DATASET SELECTION
# ===============================================================================
def fetch_dataset(dataset_type: str = "lidar") -> str:
    """
    Download a dataset based on type.
    :param dataset_type: 'lidar' or 'sentinel2'
    :return: path(s) to downloaded file(s)
    """
    if dataset_type == "lidar":
        return fetch_lidar()
    elif dataset_type == "sentinel2":
        return fetch_sentinel2_bands()
    else:
        raise ValueError("dataset_type must be 'lidar' or 'sentinel2'")

# Example usage (do not run if just producing code):
# lidar_file = fetch_dataset("lidar")
# s2_files = fetch_dataset("sentinel2")
