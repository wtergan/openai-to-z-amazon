import os
import tempfile
import requests
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from .env if present
OT_API_KEY = os.getenv("OT_API_KEY")

# ===============================================================================
# LiDAR PARAMETERS: OpenTopography API
# ===============================================================================

def get_ot_lidar(demtype: str, south: int, north: int, west: int, east: int, 
    api_key=OT_API_KEY) -> str:
    """
    Download a small LiDAR .tif file from OpenTopography API.
    Takes in the available global raster dataset type (demtype), the bbox coordinates,
    and the name to save the file as.
    """
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }
    tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    tf.write(resp.content)
    print(f"Downloaded {tf.name}")
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
        print(f"Downloaded {tf.name}")
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
        return get_ot_lidar(demtype="COP30", south=-5.253821, north=-3.983349, west=-59.813892, east=-58.332325)
    elif dataset_type == "sentinel2":
        return fetch_sentinel2_bands()
    else:
        raise ValueError("dataset_type must be 'lidar' or 'sentinel2'")

# Example usage (do not run if just producing code):
# lidar_file = fetch_dataset("lidar")
# s2_files = fetch_dataset("sentinel2")
