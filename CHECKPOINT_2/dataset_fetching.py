"""
Dataset fetching module for GEDI L2A, PRODES, SRTM, LiDAR, and Sentinel-2 data.

This module provides robust functions to fetch and locally cache:
- GEDI L2A lidar shots 
- TerraBrasilis PRODES deforestation polygons
- SRTM elevation and slope data
- OpenTopography LiDAR data
- Sentinel-2 imagery via Google Earth Engine

Includes quality filtering, error handling, and reproducible caching for downstream 
feature engineering and image generation.
"""
import os
import json
import time
import hashlib
import logging
import tempfile
import base64
import urllib.request
import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urljoin
from functools import lru_cache

from datetime import datetime

import ee
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from dotenv import load_dotenv
from owslib.wfs import WebFeatureService

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables for OpenTopography and GEE:
load_dotenv()  
OT_API_KEY = os.getenv("OT_API_KEY")

# GEE Initialization state and Project ID:
gee_initialized_successfully = False
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID")

# Default coordinates for current pipeline (Amazonas, Brazil), subject to change of course.
S2_DEFAULT_SOUTH = -5.253821
S2_DEFAULT_NORTH = -3.983349
S2_DEFAULT_WEST = -59.813892
S2_DEFAULT_EAST = -58.332325
S2_DEFAULT_START_DATE = "2022-01-01"
S2_DEFAULT_END_DATE = "2022-12-31"

# ==============================================================================
# BOUNDING BOX UTILITIES 
# ==============================================================================
def make_bbox(min_lon, min_lat, max_lon, max_lat):
    """Construct a bounding box as a dict."""
    return {
        'min_lon': min_lon,
        'min_lat': min_lat,
        'max_lon': max_lon,
        'max_lat': max_lat
    }

def bbox_to_polygon(bbox):
    """Convert bbox dict to Shapely polygon."""
    return box(bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat'])

def bbox_to_ee_geometry(bbox):
    """Convert bbox dict to Earth Engine geometry."""
    return ee.Geometry.Rectangle([
        bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat']
    ])

# ==============================================================================
# SESSION AND CACHE HELPERS
# ==============================================================================
def get_requests_session():
    """Create a reusable requests session with retry strategy."""
    session = requests.Session()  
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # Set a default User-Agent
    session.headers.update({"User-Agent": "openai-to-z/0.1 (Dataset fetcher)"})
    return session

def ensure_cache_dir(cache_dir):
    """Ensure the cache directory exists, creating it if necessary."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path

def generate_cache_key(bbox, year, dataset):
    """
    Generate a cache key based on bbox, year, and dataset.
    Converts key data to a sorted json string, then returns a md5 hash of the string.
    """
    key_data = {
        "bbox": [bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat']],
        "year": year,
        "dataset": dataset,
        "version": "1.0"
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cache_path(cache_dir, cache_key, dataset, ext=".csv"):
    """Get the full cache path for a given dataset and cache key."""
    return Path(cache_dir) / f"{dataset}_{cache_key}{ext}"

# ==============================================================================
# GEE INITIALIZATION CONFIGURATION
# ==============================================================================
def initialize_gee():
    """Initialize Google Earth Engine with proper authentication handling."""
    global gee_initialized_successfully
    if gee_initialized_successfully:
        return True
    try:
        logger.info("Attempting GEE initialization...")
        if GEE_PROJECT_ID:
            ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine.googleapis.com')
        else:
            ee.Initialize(opt_url='https://earthengine.googleapis.com')
        logger.info("GEE initialized successfully (using existing credentials or default project).")
        gee_initialized_successfully = True
    except ee.EEException as e_init:
        logger.warning(f"GEE auto-initialization failed: {e_init}. Attempting authentication flow.")
        try:
            ee.Authenticate() # This will open a browser tab for auth code.
            if GEE_PROJECT_ID:
                ee.Initialize(project=GEE_PROJECT_ID, opt_url='https://earthengine.googleapis.com')
            else:
                ee.Initialize(opt_url='https://earthengine.googleapis.com')
            logger.info("GEE authenticated and initialized successfully.")
            gee_initialized_successfully = True
        except Exception as e_auth:
            logger.error(f"CRITICAL: GEE authentication and initialization failed: {e_auth}")
            gee_initialized_successfully = False
    return gee_initialized_successfully

# ==============================================================================
# LIDAR DATA FETCHING (OpenTopography API)
# ==============================================================================
def fetch_lidar_ot_data(demtype: str, south: float, north: float, west: float, east: float, 
    api_key=OT_API_KEY) -> Optional[str]:
    """
    Download a LiDAR .tif file from OpenTopography API.
    Takes in the available global raster dataset type (demtype), the bbox coordinates,
    and returns the path to the downloaded temporary file.
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
        logger.info(f"LiDAR data successfully downloaded to temporary file: {tf.name}")
        return tf.name
    except Exception as e:
        logger.error(f"Error downloading LiDAR data: {e}")
        return None

# ==============================================================================
# SENTINEL-2 DATA FETCHING AND PROCESSING
# ==============================================================================
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
    Fetching Sentinel-2 L2A median composite from GEE (Copernicus/S2_SR_HARMONIZED) for a given bbox and date range.
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
        logger.warning(f"No Sentinel-2 images found for the given criteria in GEE.")
        logger.warning(f"ROI: {west},{south},{east},{north}")
        logger.warning(f"Date: {start_date} to {end_date}, Cloud %: < {max_cloud_percentage}")
        return {"image": None, "roi": roi, "count": 0, "error": "No images found"}
    logger.info(f"Found {count} Sentinel-2 images. Creating median composite...")
    
    # All 12 multispectral bands in S2_SR_HARMONIZED... B10, which is used to detech cirrus clouds and used 
    # for atomospheric correction purposes in L1C data, is not included in L2A, since its already atmos corrected.
    all_spectral_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    
    # Creating median composite via applying cloud/valid mask, selecting defined bands, and clipping to ROI:
    composite_image = (
        s2_collection.map(cloud_mask_s2_sr)
        .select(all_spectral_bands)
        .median()
        .clip(roi)
    )
    # Lets check if the median composite image is valid: if no band names, all pixels masked, thus empty. Not good.
    try:
        composite_image.bandNames().getInfo()
    except ee.EEException as e:
        logger.error(f"Error creating valid GEE composite (likely all pixels masked): {e}")
        return {"image": None, "roi": roi, "count": count, "error": "Composite image is empty/invalid."}
    logger.info(f"Sentinel-2 GEE median composite created for period {start_date} to {end_date}.")
    return {
        "image": composite_image,
        "roi": roi,
        "count": count,
        "start_date": start_date,
        "end_date": end_date,
        "roi_bounds": [west, south, east, north],
        "bands_included": all_spectral_bands
    }

# ==============================================================================
# SRTM DATA FETCHING AND PROCESSING (Google Earth Engine)
# ==============================================================================
def fetch_srtm(bbox, cache_dir="data/raw", force_refresh=False, sample_scale=30):
    """
    Fetch SRTM elevation data within the bounding box and compute slope metrics.
    Returns a DataFrame with columns: lon, lat, elevation, slope_degrees, aspect_degrees.
    Uses Google Earth Engine for data access. 
    """
    ensure_cache_dir(cache_dir)
    cache_year = 2000 # SRTM is static, thus using a fixed 2000 year for caching purposes.
    cache_key = generate_cache_key(bbox, cache_year, "srtm")
    cache_path = get_cache_path(cache_dir, cache_key, "srtm")
    if not force_refresh and cache_path.exists():
        logger.info(f"Loading SRTM data from cache: {cache_path}")
        try:
            return pd.read_csv(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
    logger.info(f"Fetching SRTM elevation and slope data via Google Earth Engine")
    try:
        # Initialize Earth Engine if needed:
        if not gee_initialized_successfully:
            if not initialize_gee():
                logger.error("Failed to initialize Google Earth Engine for SRTM data")
                return pd.DataFrame(columns=['lon', 'lat', 'elevation', 'slope_degrees', 'aspect_degrees'])
        # Loading SRTM elevation data (30m), selecting elevation band, then computing slope and aspect in degrees:
        srtm = ee.Image('USGS/SRTMGL1_003')
        elevation = srtm.select('elevation')
        slope = ee.Terrain.slope(elevation)
        aspect = ee.Terrain.aspect(elevation)
        
        # Combining elevation, slope, and aspect into a single image:
        terrain_image = elevation.addBands([slope, aspect]).rename(['elevation', 'slope_degrees', 'aspect_degrees'])

        # Computing the appropriate bbox area to determine the optimal number of pixels to sample:
        bbox_geometry = bbox_to_ee_geometry(bbox)
        area_km2 = bbox_geometry.area().divide(1000000).getInfo()  # Convert to km²
        
        # Conservative sampling to avoid GEE limits - use lower density for larger areas:
        if area_km2 > 1000:  # Large area
            target_density = 100
        elif area_km2 > 100:  # Medium area
            target_density = 300
        else:  # Small area
            target_density = 500
            
        optimal_pixels = int(area_km2 * target_density)
        # Strict bounds to avoid GEE collection limits:
        num_pixels = max(100, min(optimal_pixels, 3000))  # Much lower max limit
        logger.info(f"Sampling {num_pixels} pixels for the bbox area of {area_km2:.2f} km²")
        
        samples = terrain_image.sample(
            region=bbox_geometry,
            scale=sample_scale,
            numPixels=num_pixels,
            seed=42,
            geometries=True
        )
        # Checking if there are any raster samples found:
        raster_data = samples.getInfo()
        if not raster_data['features']:
            logger.warning("No SRTM elevation data found for the specified bbox")
            return pd.DataFrame(columns=['lon', 'lat', 'elevation', 'slope_degrees', 'aspect_degrees'])
        
        # Constructing a records list from the raster_data features:
        records = []
        for feature in raster_data['features']:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            record = {
                'lon': coords[0],
                'lat': coords[1],
                'elevation': props.get('elevation'),
                'slope_degrees': props.get('slope_degrees'),
                'aspect_degrees': props.get('aspect_degrees')
            }
            records.append(record)
        
        # Constructing a pandas DataFrame from the records list, filtering out invalid coords, rasters outside bbox:
        df = pd.DataFrame(records)
        df = df.dropna(subset=['lon', 'lat', 'elevation'])
        df = df[(df['lon'] >= bbox['min_lon']) & (df['lon'] <= bbox['max_lon'])]
        df = df[(df['lat'] >= bbox['min_lat']) & (df['lat'] <= bbox['max_lat'])]
        df = df.round(4) # Rounding values for consistency.
        logger.info(f"Fetched {len(df)} SRTM elevation points with terrain metrics")
        logger.info(f"Elevation range: {df['elevation'].min():.1f} - {df['elevation'].max():.1f} m")
        logger.info(f"Slope range: {df['slope_degrees'].min():.1f} - {df['slope_degrees'].max():.1f} degrees")
        
        # Caching the DataFrame to a CSV file for further usage:
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached SRTM data to: {cache_path}")
        return df        
    except Exception as e:
        logger.error(f"Failed to fetch SRTM data: {e}")
        return pd.DataFrame(columns=['lon', 'lat', 'elevation', 'slope_degrees', 'aspect_degrees'])

# ==============================================================================
# GEDI DATA FETCHING AND PROCESSING (Google Earth Engine)
# ==============================================================================
def fetch_gedi(bbox, year, cache_dir="data/raw", force_refresh=False):
    """
    Fetch GEDI L2A vector shots within the bounding box for the specified year.
    Uses Google Earth Engine for data access. Filters for quality_flag == 1
    and removes shots with invalid coordinates or missing canopy height.
    Returns DataFrame with columns: lon, lat, rh98, canopy_height, quality_flag, shot_number
    Note: canopy_height is mapped from rh100, lat/lon from lat_highestreturn/lon_highestreturn
    """
    ensure_cache_dir(cache_dir)
    cache_key = generate_cache_key(bbox, year, "gedi")
    cache_path = get_cache_path(cache_dir, cache_key, "gedi")
    
    if not force_refresh and cache_path.exists():
        logger.info(f"Loading GEDI data from cache: {cache_path}")
        try:
            return pd.read_csv(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
    
    logger.info(f"Fetching GEDI L2A vector data for year {year} via Google Earth Engine")
    
    try:
        # Initialize Earth Engine if needed:
        if not gee_initialized_successfully:
            if not initialize_gee():
                logger.error("Failed to initialize Google Earth Engine for GEDI data")
                return pd.DataFrame(columns=['lon', 'lat', 'rh98', 'canopy_height', 'quality_flag', 'shot_number'])
        
        # Usage of GEDI L2A table index to find available data tables; creating date filter for the specified year:
        gedi_index = ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002_INDEX")
        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"
        
        # Filtering the index by date and bounds to find relevant tables, then getting the list of table IDs:
        bbox_geometry = bbox_to_ee_geometry(bbox)
        relevant_tables = (gedi_index
                          .filter(ee.Filter.And(
                              ee.Filter.gte('time_start', start_date),
                              ee.Filter.lte('time_end', end_date)
                          ))
                          .filterBounds(bbox_geometry))
        table_info = relevant_tables.getInfo()
        if not table_info['features']:
            logger.warning("No GEDI tables found for the specified bbox and year")
            return pd.DataFrame(columns=['lon', 'lat', 'rh98', 'canopy_height', 'quality_flag', 'shot_number'])
        logger.info(f"Found {len(table_info['features'])} GEDI tables for the specified period")
        
        # Computing the optimal sampling based on bbox area:
        area_km2 = bbox_geometry.area().divide(1000000).getInfo()
        target_density = 1000
        optimal_samples = int(area_km2 * target_density)
        max_samples = min(optimal_samples, 50000)  # GEE limit consideration:
        samples_per_table = max(100, max_samples // len(table_info['features']))
        logger.info(f"Sampling up to {samples_per_table} shots per table for bbox area of {area_km2:.2f} km²")
        
        # Collecting data from all relevant tables:
        all_records = []
        for table_feature in table_info['features']:
            table_id = table_feature['properties']['table_id']
            try:
                # Loading the specific GEDI table (table_id already includes full path):
                gedi_table = ee.FeatureCollection(table_id)
                # Filtering by bounds and quality:
                gedi_filtered = (gedi_table
                               .filterBounds(bbox_geometry)
                               .filter(ee.Filter.eq('quality_flag', 1)))
                
                # Selecting only the fields we need:
                gedi_selected = gedi_filtered.select([
                    'lat_highestreturn', 'lon_highestreturn',  # Correct coordinate fields.
                    'rh98', 'rh100', 'quality_flag', 'shot_number'
                ])
                
                # Sampling the table if it's too large:
                table_size = gedi_selected.size().getInfo()
                if table_size > samples_per_table:
                    gedi_sampled = gedi_selected.randomColumn('random').sort('random').limit(samples_per_table)
                else:
                    gedi_sampled = gedi_selected
                
                # Getting the data:
                table_data = gedi_sampled.getInfo()
                if table_data['features']:
                    all_records.extend(table_data['features'])
                    logger.info(f"Collected {len(table_data['features'])} shots from table {table_id}")
                
            except Exception as e:
                logger.warning(f"Failed to process table {table_id}: {e}")
                continue
        
        if not all_records:
            logger.warning("No GEDI shots found after processing all tables")
            return pd.DataFrame(columns=['lon', 'lat', 'rh98', 'canopy_height', 'quality_flag', 'shot_number'])
        logger.info(f"Total collected: {len(all_records)} GEDI shots from {len(table_info['features'])} tables")
        
        # Extracting records from features:
        records = []
        for feature in all_records:
            props = feature['properties']
            
            # Skipping if any critical values are null:
            if (props.get('rh98') is None or 
                props.get('rh100') is None or 
                props.get('lat_highestreturn') is None or 
                props.get('lon_highestreturn') is None):
                continue
                
            record = {
                'lon': props.get('lon_highestreturn'),  # Map from lon_highestreturn
                'lat': props.get('lat_highestreturn'),  # Map from lat_highestreturn
                'rh98': props.get('rh98'),
                'canopy_height': props.get('rh100'),  # Map rh100 to canopy_height for compatibility
                'quality_flag': props.get('quality_flag', 1),
                'shot_number': props.get('shot_number')
            }
            records.append(record)
        
        # Creating a DataFrame and filtering:
        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No valid GEDI records after filtering")
            return df
            
        # Additional spatial filtering and data cleaning:
        df = df.dropna(subset=['lon', 'lat', 'rh98', 'canopy_height'])
        df = df[(df['lon'] >= bbox['min_lon']) & (df['lon'] <= bbox['max_lon'])]
        df = df[(df['lat'] >= bbox['min_lat']) & (df['lat'] <= bbox['max_lat'])]
        
        logger.info(f"Fetched {len(df)} quality GEDI shots")
        
        # Caching the results:
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached GEDI data to: {cache_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch GEDI data: {e}")
        return pd.DataFrame(columns=['lon', 'lat', 'rh98', 'canopy_height', 'quality_flag', 'shot_number'])

# ==============================================================================
# PRODES DATA FETCHING AND PROCESSING (TerraBrasilis API)
# ==============================================================================
def fetch_prodes(bbox, year, cache_dir="data/raw", force_refresh=False):
    """
    Load PRODES deforestation polygons from local GeoPackage file for the specified year. 
    Returns a GeoDataFrame with the deforestation polygons and attributes.
    """
    ensure_cache_dir(cache_dir)
    cache_key = generate_cache_key(bbox, year, "prodes")
    cache_path = get_cache_path(cache_dir, cache_key, "prodes", ".geojson")
    
    if not force_refresh and cache_path.exists():
        logger.info(f"Loading PRODES data from cache: {cache_path}")
        try:
            return gpd.read_file(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
    
    logger.info(f"Loading PRODES deforestation data for year {year} from local GeoPackage")
    
    # Path to the local GeoPackage file
    geopackage_path = Path(cache_dir) / "prodes_amazonia_nb.gpkg"
    
    if not geopackage_path.exists():
        logger.error(f"PRODES GeoPackage not found at: {geopackage_path}")
        logger.error("Please ensure prodes_amazonia_nb.gpkg is available in the data/raw directory")
        return gpd.GeoDataFrame(columns=['geometry', 'year', 'area_ha'])
    
    try:
        logger.info(f"Loading GeoPackage from: {geopackage_path}")
        # Load the yearly deforestation layer specifically (better for annual data)
        layer_name = 'yearly_deforestation_biome'
        gdf = gpd.read_file(geopackage_path, layer=layer_name)
        logger.info(f"Loaded GeoPackage layer '{layer_name}' with {len(gdf)} total polygons")
        
        # Check available columns to handle different naming conventions
        logger.info(f"Available columns in GeoPackage: {list(gdf.columns)}")
        
        # Find the year column (common variations)
        year_col = None
        for col in ['year', 'Year', 'YEAR', 'ano', 'Ano', 'ANO']:
            if col in gdf.columns:
                year_col = col
                break
        
        if year_col is None:
            logger.warning("Could not find year column in GeoPackage. Available columns: " + str(list(gdf.columns)))
            # If no year column found, assume all data is for the requested year
            gdf['year'] = year
            year_col = 'year'
        
        # Filter by year if specified
        if year:
            gdf_year = gdf[gdf[year_col] == year]
            logger.info(f"Filtered to {len(gdf_year)} polygons for year {year}")
        else:
            gdf_year = gdf
            logger.info(f"No year filter applied, using all {len(gdf_year)} polygons")
        
        if gdf_year.empty:
            logger.warning(f"No PRODES polygons found for year {year}")
            return gpd.GeoDataFrame(columns=['geometry', 'year', 'area_ha'])
        
        # Create bbox polygon for spatial filtering
        bbox_polygon = bbox_to_polygon(bbox)
        
        # Clip to bounding box using spatial intersection
        logger.info(f"Filtering polygons by bbox: {bbox}")
        gdf_clipped = gdf_year[gdf_year.geometry.intersects(bbox_polygon)]
        logger.info(f"After bbox filtering: {len(gdf_clipped)} polygons")
        
        if gdf_clipped.empty:
            logger.warning(f"No PRODES polygons found within bbox {bbox} for year {year}")
            return gpd.GeoDataFrame(columns=['geometry', 'year', 'area_ha'])
        
        # Ensure we have the required columns - handle area_km to area_ha conversion
        gdf_clipped = gdf_clipped.copy()  # Avoid SettingWithCopyWarning
        if 'area_ha' not in gdf_clipped.columns:
            if 'area_km' in gdf_clipped.columns:
                logger.info("Converting area_km to area_ha")
                gdf_clipped['area_ha'] = gdf_clipped['area_km'] * 100  # Convert km² to hectares
            else:
                logger.info("Calculating area_ha from geometry")
                gdf_clipped['area_ha'] = gdf_clipped.geometry.to_crs('EPSG:3857').area / 10000
        
        # Ensure we have year column with consistent name
        if year_col != 'year':
            gdf_clipped = gdf_clipped.copy()
            gdf_clipped['year'] = gdf_clipped[year_col]
        
        # Select only the required columns, preserving any additional useful columns
        required_cols = ['geometry', 'year', 'area_ha']
        available_cols = [col for col in required_cols if col in gdf_clipped.columns]
        
        # Add any additional useful columns that might be present
        extra_cols = []
        for col in gdf_clipped.columns:
            if col not in required_cols and col != year_col:
                # Include columns that might contain useful attributes
                if any(keyword in col.lower() for keyword in ['class', 'tipo', 'type', 'bioma', 'biome', 'estado', 'state']):
                    extra_cols.append(col)
        
        final_cols = available_cols + extra_cols
        gdf_result = gdf_clipped[final_cols]
        
        # Cache the results
        gdf_result.to_file(cache_path, driver='GeoJSON', index=False)
        logger.info(f"Fetched {len(gdf_result)} PRODES polygons covering {gdf_result['area_ha'].sum():.1f} hectares")
        logger.info(f"Cached PRODES data to: {cache_path}")
        
        return gdf_result
        
    except Exception as e:
        logger.error(f"Failed to load PRODES data from GeoPackage: {e}")
        return gpd.GeoDataFrame(columns=['geometry', 'year', 'area_ha'])

# ==============================================================================
# FETCH PIPELINE FOR GEDI, PRODES, and SRTM
# ==============================================================================
def gedi_prodes_srtm_fetch_pipeline(bbox: dict = None, year: int = 2022, cache_dir="data/raw", force_refresh: bool = False):
    """
    Fetching pipeline for the GEDI, PRODES, and SRTM datasets for the specified bbox and year.
    If bbox and/or year is None, uses the default coordinates and/or year.
    Returns a tuple of the GEDI and SRTM DataFrames, as well as the PRODES GeoDataFrame. 
    force_refresh determines whether to force a refresh of the cached data. Defaults to 'False'.
    GEDI data comes from 'GEDI L2A Raster Canopy Top Height (Version 2, Monthly)', via Google Earth Engine (GEE).
        - https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_A_002_MONTHLY#description
    PRODES data comes from 'TerraBrasilis Native Vegetation Suppression Map (PRODES)', loaded from local GeoPackage file.
        - Source: https://terrabrasilis.dpi.inpe.br/app/map/deforestation?hl=en
    SRTM data comes from 'NASA SRTM Digital Elevation V3', via Google Earth Engine (GEE).
        - https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
    """
    # Use default coordinates, created via make_box function for easy plugin; useful for test runs:
    if bbox is None:
        bbox = make_bbox(S2_DEFAULT_WEST, S2_DEFAULT_SOUTH, S2_DEFAULT_EAST, S2_DEFAULT_NORTH)
    # If bbox is already a proper dict with the required keys, use it as-is
    # Otherwise, assume it needs to be properly formatted
    
    # Retrieving the GEDI data; if there is no data for the specified bbox/year, function will return empty DataFrame:
    logger.info(f"FETCHING GEDI L2A RASTER CANOPY TOP HEIGHT V2 DATA FROM GEE FOR THE FOLLOWING BBOX:\n{bbox}\nYEAR:{year}\n")
    gedi_df = fetch_gedi(bbox, year, cache_dir, force_refresh)
    if gedi_df.empty is True:
        logger.warning("FETCH DID NOT RETURN ANY DATA. THERE WAS AN ERROR IN THE FETCHING PROCESS OR NO DATA AVAILABLE \
            FOR THE SPECIFIED BBOX AND YEAR.\n")

    # Retrieving the PRODES data; if there is no data for the specified bbox/year, function will return empty DataFrame:
    logger.info(f"FETCHING PRODES DATA FROM LOCAL GEOPACKAGE FOR THE FOLLOWING BBOX:\n{bbox}\nYEAR:{year}\n")
    prodes_gdf = fetch_prodes(bbox, year, cache_dir, force_refresh)
    if prodes_gdf.empty is True:
        logger.warning("FETCH DID NOT RETURN ANY DATA. THERE WAS AN ERROR IN THE FETCHING PROCESS OR NO DATA AVAILABLE \
            FOR THE SPECIFIED BBOX AND YEAR.\n")

    # Retrieving the SRTM data; if there is no data for the specified bbox/year, function will return empty DataFrame:
    logger.info(f"FETCHING SRTM DATA FROM GEE FOR THE FOLLOWING BBOX:\n{bbox}\n")
    srtm_df = fetch_srtm(bbox, cache_dir, force_refresh)
    if srtm_df.empty is True:
        logger.warning("FETCH DID NOT RETURN ANY DATA. THERE WAS AN ERROR IN THE FETCHING PROCESS OR NO DATA AVAILABLE \
            FOR THE SPECIFIED BBOX AND YEAR.\n")

    # Returning some preliminary data for debugging, informational purposes:
    logger.info(f"GEDI_PRODES_SRTM_FETCH_PIPELINE COMPLETED. RETURNING THE FOLLOWING DATA:\n\
        GEDI Shots: {len(gedi_df)}\n\
        PRODES Polygons: {len(prodes_gdf)}\n\
        SRTM Points: {len(srtm_df)}")
    
    # Finally, returning the LiDAR and Sentinel-2 data for further processing and usage:
    return gedi_df, prodes_gdf, srtm_df

# ==============================================================================
# FETCHING PIPELINE FOR LIDAR AND SENTINEL-2
# ==============================================================================
def lidar_sentinel2_fetch_pipeline(bbox: dict = None) -> dict:
    """
    Fetching pipeline for the LiDAR and Sentinel-2 datasets, given a specified bounding box (bbox).
    If bbox is None, uses the default coordinates.
    Returns a dict with the LiDAR file path (GeoTIFF data) and the Sentinel-2 GEE median composite image and ROI.
    LiDAR data comes from 'Copernicus Global Digital Elevation Models' (COP30), via OpenTopography API.
        - https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.032021.4326.1  
    Sentinel-2 data comes from 'Copernicus Harmonnized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A (SR)', via Google Earth Engine (GEE).
        - https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    """
    logger.info(f"FETCHING LIDAR DATA FROM COP30 (OpenTopography API) FOR THE FOLLOWING BBOX:\n{bbox}\n")
    # Use default coordinates; useful for test runs:
    if bbox is None:
        south, north = S2_DEFAULT_SOUTH, S2_DEFAULT_NORTH
        west, east = S2_DEFAULT_WEST, S2_DEFAULT_EAST
    # Use the specified bbox coordinates:    
    else:
        south, north = bbox['min_lat'], bbox['max_lat']
        west, east = bbox['min_lon'], bbox['max_lon']
        
    # Retrieving the LiDAR data; if there is no data for the specified bbox, set to None but continue.
    lidar_path = fetch_lidar_ot_data(demtype="COP30", south=south, north=north, west=west, east=east)
    
    # Retrieving the Sentinel-2 data; if there is no data for the specified bbox, set to None but continue:
    print(f"FETCHING SENTINEL-2 COMPOSITE DATA FROM COPERNICUS S2 SR HARMONIZED (GEE) FOR THE FOLLOWING BBOX:\n{bbox}\n")
    s2_data = fetch_sentinel2_gee_data(
            south=south, north=north, west=west, east=east,
            start_date=S2_DEFAULT_START_DATE, end_date=S2_DEFAULT_END_DATE
        )
    
    # Only return None if both data sources failed
    if lidar_path is None and s2_data is None:
        logger.warning("Both LiDAR and Sentinel-2 data fetching failed")
        return None

    # Returning some preliminary data for debugging, informational purposes: 
    logger.info(f"LIDAR and SENTINEL-2 FETCHING COMPLETED. RETURNING THE FOLLOWING DATA:\n\
        LiDAR Path: {lidar_path}\n\
        Sentinel-2 Data: {s2_data}")
    
    # Finally, returning the LiDAR and Sentinel-2 data for further processing and usage:
    return {
        'lidar_path': lidar_path,
        's2_data': s2_data
    }

# ============================================================================== 
#   MAIN
# ============================================================================== 
if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch GEDI, PRODES, and SRTM datasets")
    parser.add_argument("--bbox", required=True, help="Bounding box as 'min_lon,min_lat,max_lon,max_lat'")
    parser.add_argument("--year", type=int, required=True, help="Year to fetch data for")
    parser.add_argument("--cache-dir", default="data/raw", help="Cache directory")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh cached data")
    
    args = parser.parse_args()
    
    # Parsing the bounding box:
    bbox_coords = [float(x) for x in args.bbox.split(',')]
    bbox = make_bbox(*bbox_coords)
    
    # Fetching the GEDI, PRODES and SRTM data:
    gedi_df, prodes_gdf, srtm_df = gedi_prodes_srtm_fetch_pipeline(bbox, args.year, args.cache_dir, args.force_refresh)
    print(f"Fetched {len(gedi_df)} GEDI shots, {len(prodes_gdf)} PRODES polygons, and {len(srtm_df)} SRTM points")
    
    # Fetching the regional LiDAR and Sentinel-2 data:
    regional_ctx = lidar_sentinel2_fetch_pipeline(bbox)
    if regional_ctx:
        print(f"Fetched lidar data via its path: {regional_ctx['lidar_path']}, and Sentinel 2 data: {regional_ctx['s2_data']}")
    else:
        print("No regional LiDAR/Sentinel-2 data could be fetched")