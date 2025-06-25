#!/usr/bin/env python3
"""
Simple test script for SRTM elevation and slope data fetching.
Tests the fetch_srtm function with a small bounding box using the current API.
"""

import os
import sys
import logging
import pandas as pd

# ===============================================================================
# DIRECTORY SETUP AND LOGGING CONFIGURATION
# ===============================================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================================================
# IMPORTING ALL PERTINENT FUNCTIONS AND VALUES FOR TESTING
# ===============================================================================
from dataset_fetching import (
    fetch_srtm, make_bbox, initialize_gee,
    S2_DEFAULT_SOUTH, S2_DEFAULT_NORTH, S2_DEFAULT_WEST, S2_DEFAULT_EAST
)

# ===============================================================================
# TESTING: SRTM DATA FETCHING
# ===============================================================================
def test_srtm_fetching():
    """Test SRTM data fetching with a small Amazon region using current API."""
    
    # Initialize Google Earth Engine first
    logger.info("Initializing Google Earth Engine...")
    if not initialize_gee():
        logger.error("Failed to initialize Google Earth Engine!")
        return False
    
    # Use the default coordinates from dataset_fetching.py testing (Amazonas, Brazil):
    bbox = make_bbox(S2_DEFAULT_WEST, S2_DEFAULT_SOUTH, S2_DEFAULT_EAST, S2_DEFAULT_NORTH)
    logger.info("Testing SRTM data fetching...")
    logger.info(f"Bounding box: {bbox}")
    
    try:
        # Fetch SRTM data using current API (no year parameter needed for SRTM):
        srtm_df = fetch_srtm(bbox, cache_dir="data/test_cache", force_refresh=True)
        if srtm_df.empty:
            logger.error("No SRTM data returned!")
            return False
        
        logger.info(f"Successfully fetched {len(srtm_df)} SRTM points")
        logger.info(f"Columns: {list(srtm_df.columns)}")
        
        # Checking expected columns based on current API:
        expected_columns = ['lon', 'lat', 'elevation', 'slope_degrees', 'aspect_degrees']
        missing_columns = [col for col in expected_columns if col not in srtm_df.columns]
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            return False
        
        # Displaying basic statistics:
        logger.info("\nSRTM Data Statistics:")
        logger.info(f"Elevation range: {srtm_df['elevation'].min():.1f} - {srtm_df['elevation'].max():.1f} m")
        logger.info(f"Slope range: {srtm_df['slope_degrees'].min():.1f} - {srtm_df['slope_degrees'].max():.1f} degrees")
        logger.info(f"Aspect range: {srtm_df['aspect_degrees'].min():.1f} - {srtm_df['aspect_degrees'].max():.1f} degrees")
        logger.info(f"Mean elevation: {srtm_df['elevation'].mean():.1f} m")
        logger.info(f"Mean slope: {srtm_df['slope_degrees'].mean():.1f} degrees")
        
        # Check for reasonable values:
        if srtm_df['elevation'].min() < -500 or srtm_df['elevation'].max() > 9000:
            logger.warning("Elevation values seem unreasonable for Amazon region")
        
        if srtm_df['slope_degrees'].min() < 0 or srtm_df['slope_degrees'].max() > 90:
            logger.warning("Slope values outside expected range [0, 90] degrees")
        
        if srtm_df['aspect_degrees'].min() < 0 or srtm_df['aspect_degrees'].max() > 360:
            logger.warning("Aspect values outside expected range [0, 360] degrees")
        
        # Testing data quality:
        null_counts = srtm_df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts.to_dict()}")
        
        # Testing coordinate bounds:
        lat_range = (srtm_df['lat'].min(), srtm_df['lat'].max())
        lon_range = (srtm_df['lon'].min(), srtm_df['lon'].max())
        logger.info(f"Latitude range: {lat_range[0]:.4f} - {lat_range[1]:.4f}")
        logger.info(f"Longitude range: {lon_range[0]:.4f} - {lon_range[1]:.4f}")
        
        # Verifying coordinates are within bbox:
        if (lat_range[0] < bbox['min_lat'] or lat_range[1] > bbox['max_lat'] or
            lon_range[0] < bbox['min_lon'] or lon_range[1] > bbox['max_lon']):
            logger.warning("Some coordinates fall outside the specified bounding box")
        
        logger.info("SRTM test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"SRTM test failed: {e}")
        return False

# ===============================================================================
# TESTING: SRTM CACHING FUNCTIONALITY
# ===============================================================================
def test_srtm_caching():
    """Test SRTM data caching functionality."""
    logger.info("Testing SRTM caching functionality...")
    
    bbox = make_bbox(S2_DEFAULT_WEST, S2_DEFAULT_SOUTH, S2_DEFAULT_EAST, S2_DEFAULT_NORTH)
    cache_dir = "data/test_cache"
    
    try:
        # First fetch (should create cache):
        logger.info("First fetch (creating cache)...")
        start_time = pd.Timestamp.now()
        srtm_df1 = fetch_srtm(bbox, cache_dir=cache_dir, force_refresh=True)
        first_duration = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Second fetch (should use cache):
        logger.info("Second fetch (using cache)...")
        start_time = pd.Timestamp.now()
        srtm_df2 = fetch_srtm(bbox, cache_dir=cache_dir, force_refresh=False)
        second_duration = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Comparing results:
        if srtm_df1.equals(srtm_df2):
            logger.info("Cached data matches original data")
        else:
            logger.warning("Cached data differs from original data")
        
        logger.info(f"First fetch: {first_duration:.2f}s, Second fetch: {second_duration:.2f}s")
        if second_duration < first_duration * 0.5:  # Cache should be significantly faster.
            logger.info("Cache provides performance improvement")
        else:
            logger.warning("Cache may not be working properly")
            
        return True
        
    except Exception as e:
        logger.error(f"SRTM caching test failed: {e}")
        return False

# ===============================================================================
# MAIN TEST SCRIPT
# ===============================================================================
if __name__ == "__main__":
    success1 = test_srtm_fetching()
    success2 = test_srtm_caching()
    
    if success1 and success2:
        print("\nAll SRTM tests passed!")
    else:
        print("\nSome SRTM tests failed!")
        exit(1) 