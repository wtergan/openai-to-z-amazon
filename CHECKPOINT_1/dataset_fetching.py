import os
import tempfile
import requests
import base64
import ee
import urllib.request
import io
from PIL import Image
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# ===============================================================================
# ENVIRONMENT SETUP
# ===============================================================================
load_dotenv()  
OT_API_KEY = os.getenv("OT_API_KEY")

# GEE Initialization state and Project ID:
gee_initialized_successfully = False
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID")

def initialize_gee():
    global gee_initialized_successfully
    if gee_initialized_successfully:
        return True
    try:
        print("Attempting GEE initialization...")
        if GEE_PROJECT_ID:
            ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine.googleapis.com')
        else:
            ee.Initialize(opt_url='https://earthengine.googleapis.com')
        print("GEE initialized successfully (using existing credentials or default project).")
        gee_initialized_successfully = True
    except ee.EEException as e_init:
        print(f"GEE auto-initialization failed: {e_init}. Attempting authentication flow.")
        try:
            ee.Authenticate() # This will open a browser tab for auth code.
            if GEE_PROJECT_ID:
                ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine.googleapis.com')
            else:
                ee.Initialize(opt_url='https://earthengine.googleapis.com')
            print("GEE authenticated and initialized successfully.")
            gee_initialized_successfully = True
        except Exception as e_auth:
            print(f"CRITICAL: GEE authentication and initialization failed: {e_auth}")
            gee_initialized_successfully = False
    return gee_initialized_successfully

# ===============================================================================
# DEFAULT GLOBAL BBOX COORDINATES (Amazonas, Brazil), used for now
# ===============================================================================
S2_DEFAULT_SOUTH = -5.253821
S2_DEFAULT_NORTH = -3.983349
S2_DEFAULT_WEST = -59.813892
S2_DEFAULT_EAST = -58.332325
S2_DEFAULT_START_DATE = "2023-01-01"
S2_DEFAULT_END_DATE = "2023-12-31"

# ===============================================================================
# LiDAR PARAMETERS: OpenTopography API
# ===============================================================================
def fetch_lidar_ot_data(demtype: str, south: float, north: float, west: float, east: float, 
    api_key=OT_API_KEY) -> Optional[str]:
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
    try:
        tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        tf.write(resp.content)
        tf.close()
        print(f"LiDAR data successfully downloaded to temporary file: {tf.name}")
        return tf.name
    except Exception as e:
        print(f"Error downloading LiDAR data: {e}")
        return None

# ===============================================================================
# Sentinel-2 PARAMETERS: Google Earth Engine
# ===============================================================================
def cloud_mask_s2_sr(image: ee.Image) -> ee.Image:
    """Cloud mask creation for Sentinel-2 Surface Reflectance using SCL band."""
    scl = image.select('SCL')
    # Mask out cloud shadow, medium/high probability cloud, and cirrus, saturated/defective pixels:
    clear_mask = scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    valid_data_mask = image.select('B2').gt(0)
    return image.updateMask(clear_mask.And(valid_data_mask)).divide(10000) # Scale data to 0-1 range.

def fetch_sentinel2_gee_data(
    south: float, north: float, west: float, east: float, start_date: str, end_date: str,
    max_cloud_percentage: float = 20.0
) -> Optional[Dict[str, Any]]:
    """
    Fetching Sentinel-2 L2A median composite from GEE for a given bbox and date range.
    Returns a dictionary containing the GEE image object and ROI.
    """
    if not gee_initialized_successfully:
        if not initialize_gee():
            raise RuntimeError("GEE could not be initialized. Cannot fetch Sentinel-2 data.")
    roi = ee.Geometry.Rectangle([west, south, east, north])

    # Filtering ImageCollection by ROI, date range, as well as cloud percentage:
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(ee.Date(start_date), ee.Date(end_date))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percentage))
    )
    count = s2_collection.size().getInfo()
    if count == 0:
        print(f"Warning: No Sentinel-2 images found for the given criteria in GEE.")
        print(f"ROI: {west},{south},{east},{north}")
        print(f"Date: {start_date} to {end_date}, Cloud %: < {max_cloud_percentage}")
        return {"image": None, "roi": roi, "count": 0, "error": "No images found"}
    print(f"Found {count} Sentinel-2 images. Creating median composite...")
    
    # Creating median composite via applying cloud/valid mask, selecting defined bands, and clipping to ROI:
    composite_image = (
        s2_collection.map(cloud_mask_s2_sr)
        .select(['B2', 'B3', 'B4', 'B8'])
        .median()
        .clip(roi)
    )
    # Lets check if the median composite image is valid: if no band names, all pixels masked, thus empty. Not good.
    try:
        composite_image.bandNames().getInfo()
    except ee.EEException as e:
        print(f"Error creating valid GEE composite (likely all pixels masked): {e}")
        return {"image": None, "roi": roi, "count": count, "error": "Composite image is empty/invalid."}
    print(f"Sentinel-2 GEE median composite created for period {start_date} to {end_date}.")
    return {
        "image": composite_image,
        "roi": roi,
        "count": count,
        "start_date": start_date,
        "end_date": end_date,
        "roi_bounds": [west, south, east, north]
    }

# ===============================================================================
# DATASET SELECTION
# ===============================================================================
def fetch_dataset(dataset_type: str = "lidar") -> dict:
    """
    Download/fetch dataset based on type.
    :param dataset_type: 'lidar' or 'sentinel2'
    :return: Path to downloaded LiDAR file (str) or dict with GEE image and ROI for Sentinel-2.
    """
    if dataset_type == "lidar":
        return fetch_lidar_ot_data(demtype="COP30", south=S2_DEFAULT_SOUTH, north=S2_DEFAULT_NORTH, west=S2_DEFAULT_WEST, east=S2_DEFAULT_EAST)
    elif dataset_type == "sentinel2":
        return fetch_sentinel2_gee_data(
            south=S2_DEFAULT_SOUTH, north=S2_DEFAULT_NORTH,
            west=S2_DEFAULT_WEST, east=S2_DEFAULT_EAST,
            start_date=S2_DEFAULT_START_DATE, end_date=S2_DEFAULT_END_DATE
        )
    else:
        raise ValueError("dataset_type must be 'lidar' or 'sentinel2'")

# Example usage (do not run if just producing code):
# lidar_file = fetch_dataset("lidar")
# s2_files = fetch_dataset("sentinel2")
