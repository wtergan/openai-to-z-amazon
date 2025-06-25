#!/usr/bin/env python3
"""
Test script exclusively for the GEDI-PRODES-SRTM anomaly detection pipeline.

This script demonstrates the complete workflow from data fetching to feature engineering
using the current pipeline APIs.
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path

# ===============================================================================
# DIRECTORY SETUP AND LOGGING CONFIGURATION
# ===============================================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===============================================================================
# IMPORTING ALL PERTINENT FUNCTIONS AND VALUES FOR TESTING
# ===============================================================================
try:    
    from dataset_fetching import (
        initialize_gee, make_bbox, gedi_prodes_srtm_fetch_pipeline,
        fetch_gedi, fetch_prodes, fetch_srtm,
        S2_DEFAULT_SOUTH, S2_DEFAULT_NORTH, S2_DEFAULT_WEST, S2_DEFAULT_EAST
    )
    from feature_engineering import (
        feat_engineering_pipeline, grid_snap, build_percell_vectors,
        spatial_join_with_srtm, spatial_join_with_prodes
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# ===============================================================================
# TESTING: COMPLETE PIPELINE
# ===============================================================================
def test_small_amazon_region():
    """Test the complete pipeline with a small Amazon region using the current API."""
    logger.info("Testing GEDI-PRODES-SRTM pipeline with default Amazon region")
    
    # Use the default coordinates from dataset_fetching.py for testing (Amazonas, Brazil):
    bbox = make_bbox(S2_DEFAULT_WEST, S2_DEFAULT_SOUTH, S2_DEFAULT_EAST, S2_DEFAULT_NORTH)
    year = 2022
    cache_dir = "data/test_cache"
    logger.info(f"Test region: {bbox}")
    logger.info(f"Test year: {year}")
    
    try:
        # Initialize Google Earth Engine:
        logger.info("Initializing Google Earth Engine...")
        if not initialize_gee():
            logger.error("Failed to initialize Google Earth Engine!")
            return False
        
        # Step 1:Fetching the pertinent datasets using the gedi_prodes_srtm_fetch_pipeline function:
        logger.info("=" * 50)
        logger.info("STEP 1: FETCHING DATASETS USING THE PIPELINE:")
        logger.info("=" * 50)

        gedi_df, prodes_gdf, srtm_df = gedi_prodes_srtm_fetch_pipeline(
            bbox, year, cache_dir, force_refresh=False
        )
        logger.info(f"Pipeline fetched {len(gedi_df)} GEDI shots")
        logger.info(f"Pipeline fetched {len(prodes_gdf)} PRODES polygons")
        logger.info(f"Pipeline fetched {len(srtm_df)} SRTM points")
        
        # Displaying the data information:
        if len(gedi_df) > 0:
            logger.info(f"GEDI columns: {list(gedi_df.columns)}")
            logger.info(f"GEDI sample stats:")
            if 'canopy_height' in gedi_df.columns:
                logger.info(f"  - Canopy height range: {gedi_df['canopy_height'].min():.2f} - {gedi_df['canopy_height'].max():.2f} m")
            if 'rh98' in gedi_df.columns:
                logger.info(f"  - RH98 range: {gedi_df['rh98'].min():.2f} - {gedi_df['rh98'].max():.2f} m")
        else:
            logger.warning("No GEDI data available for this region/year.")
            
        if len(prodes_gdf) > 0:
            logger.info(f"PRODES columns: {list(prodes_gdf.columns)}")
            if 'area_ha' in prodes_gdf.columns:
                logger.info(f"PRODES total area: {prodes_gdf['area_ha'].sum():.1f} hectares")
        else:
            logger.warning("No PRODES data available for this region/year")
            
        if len(srtm_df) > 0:
            logger.info(f"SRTM columns: {list(srtm_df.columns)}")
            if 'elevation' in srtm_df.columns:
                logger.info(f"SRTM elevation range: {srtm_df['elevation'].min():.1f} - {srtm_df['elevation'].max():.1f} m")
        else:
            logger.warning("No SRTM data available for this region")

        # Step 2: Feature engineering using the current pipeline:
        logger.info("=" * 50)
        logger.info("STEP 2: FEATURE ENGINEERING USING THE CURRENT PIPELINE:")
        logger.info("=" * 50)
        
        features = feat_engineering_pipeline(gedi_df, prodes_gdf, srtm_df)
        logger.info(f"Generated {len(features)} feature vectors")
        
        if len(features) > 0:
            logger.info(f"Feature columns: {list(features.columns)}")
            
            # Show sample statistics:
            numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
            logger.info(f"Sample feature statistics:")
            for col in numeric_cols[:8]:  # Show first 8 numeric columns.
                values = features[col].dropna()
                if len(values) > 0:
                    logger.info(f"  - {col}: mean={values.mean():.3f}, std={values.std():.3f}")
            
            # Checking for normalized features, if available:
            norm_cols = [col for col in features.columns if col.endswith('_norm')]
            logger.info(f"Generated {len(norm_cols)} normalized features: {norm_cols[:5]}...")
            
            # Checking for core feature categories:
            core_features = ['shot_count', 'mean_canopy_height', 'height_variability']
            available_core = [col for col in core_features if col in features.columns]
            logger.info(f"Core features available: {available_core}")
            
            # Checking for terrain features:
            terrain_features = ['mean_elevation', 'mean_slope', 'terrain_complexity']
            available_terrain = [col for col in terrain_features if col in features.columns]
            logger.info(f"Terrain features available: {available_terrain}")
            
            # Checking for deforestation features:
            deforest_features = ['deforested', 'deforestation_year', 'deforest_impact']
            available_deforest = [col for col in deforest_features if col in features.columns]
            logger.info(f"Deforestation features available: {available_deforest}")
            
            # Saving the test features to a CSV file:
            output_path = Path(cache_dir) / "test_features.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(output_path, index=False)
            logger.info(f"Saved test features to: {output_path}")
        else:
            logger.warning("No features generated - this may be expected for regions with sparse data")
            
        logger.info("=" * 50)
        logger.info("PIPELINE TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        logger.error("Check your Earth Engine authentication and network connection")
        return False

# ===============================================================================
# TESTING: INDIVIDUAL FUNCTIONS
# ===============================================================================
def test_individual_functions():
    """Test individual functions in the pipeline for robustness."""
    logger.info("Testing individual pipeline functions")
    
    # Use the default coordinates from dataset_fetching.py for testing (Amazonas, Brazil):
    bbox = make_bbox(S2_DEFAULT_WEST, S2_DEFAULT_SOUTH, S2_DEFAULT_EAST, S2_DEFAULT_NORTH)
    year = 2022
    cache_dir = "data/test_cache"
    
    try:
        # Testing individual fetch functions:
        logger.info("Testing individual fetch functions...")
        
        # Testing GEDI fetching:
        logger.info("Testing GEDI fetch...")
        gedi_df = fetch_gedi(bbox, year, cache_dir)
        logger.info(f"GEDI fetch: {len(gedi_df)} shots")
        
        # Testing PRODES fetching:
        logger.info("Testing PRODES fetch...")
        prodes_gdf = fetch_prodes(bbox, year, cache_dir)
        logger.info(f"PRODES fetch: {len(prodes_gdf)} polygons")
        
        # Testing SRTM fetching:
        logger.info("Testing SRTM fetch...")
        srtm_df = fetch_srtm(bbox, cache_dir)
        logger.info(f"SRTM fetch: {len(srtm_df)} points")
        
        # Testing feature engineering components:
        logger.info("Testing feature engineering components...")
        
        if len(gedi_df) > 0:
            # Testing grid snapping procedure:
            logger.info("Testing grid snapping procedure...")
            cell_stats = grid_snap(gedi_df, resolution=9, min_shots_per_cell=2)
            logger.info(f"Grid snapping: {len(cell_stats)} H3 cells")
            
            if len(cell_stats) > 0:
                # Testing individual spatial joins:
                if len(srtm_df) > 0:
                    logger.info("Testing SRTM spatial join...")
                    srtm_joined = spatial_join_with_srtm(cell_stats, srtm_df)
                    logger.info(f"SRTM join: {len(srtm_joined)} cells with terrain data")
                
                if len(prodes_gdf) > 0:
                    logger.info("Testing PRODES spatial join...")
                    prodes_joined = spatial_join_with_prodes(cell_stats, prodes_gdf)
                    logger.info(f"PRODES join: {len(prodes_joined)} cells with deforestation data")
                
                # Testing complete feature building procedure:
                logger.info("Testing complete feature building procedure...")
                features = build_percell_vectors(cell_stats, prodes_gdf, srtm_df)
                logger.info(f"Feature building: {len(features)} feature vectors with {len(features.columns)} columns")
        else:
            logger.warning("No GEDI data available for individual function testing")
            
        logger.info("Individual function tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Individual function test failed: {e}")
        return False

# ===============================================================================
# TESTING: MOCK DATA
# ===============================================================================
def test_mock_data():
    """Test feature engineering with mock data (no network required)."""
    logger.info("Testing feature engineering with mock data")
    
    try:
        import pandas as pd
        import numpy as np
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
        
        # Create mock GEDI data with current expected columns using default bbox:
        np.random.seed(42)
        n_shots = 100
        mock_gedi = pd.DataFrame({
            'lon': np.random.uniform(S2_DEFAULT_WEST, S2_DEFAULT_EAST, n_shots),
            'lat': np.random.uniform(S2_DEFAULT_SOUTH, S2_DEFAULT_NORTH, n_shots),
            'rh98': np.random.uniform(5, 35, n_shots),
            'canopy_height': np.random.uniform(3, 30, n_shots),
            'quality_flag': 1,
            'shot_number': range(1, n_shots + 1)
        })
        logger.info(f"Created mock GEDI data: {len(mock_gedi)} shots")
        
        # Create mock SRTM data using default bbox:
        n_srtm = 50
        mock_srtm = pd.DataFrame({
            'lon': np.random.uniform(S2_DEFAULT_WEST, S2_DEFAULT_EAST, n_srtm),
            'lat': np.random.uniform(S2_DEFAULT_SOUTH, S2_DEFAULT_NORTH, n_srtm),
            'elevation': np.random.uniform(100, 300, n_srtm),
            'slope_degrees': np.random.uniform(0, 15, n_srtm),
            'aspect_degrees': np.random.uniform(0, 360, n_srtm)
        })
        logger.info(f"Created mock SRTM data: {len(mock_srtm)} points")
        
        # Create mock PRODES data:
        n_prodes = 5
        mock_prodes_data = []
        for i in range(n_prodes):
            # Create small polygons using default bbox:
            center_lon = np.random.uniform(S2_DEFAULT_WEST, S2_DEFAULT_EAST)
            center_lat = np.random.uniform(S2_DEFAULT_SOUTH, S2_DEFAULT_NORTH)
            size = 0.01  # Small polygon
            polygon = Polygon([
                (center_lon - size, center_lat - size),
                (center_lon + size, center_lat - size),
                (center_lon + size, center_lat + size),
                (center_lon - size, center_lat + size)
            ])
            mock_prodes_data.append({
                'geometry': polygon,
                'year': 2022,
                'area_ha': np.random.uniform(10, 100)
            })
        
        mock_prodes = gpd.GeoDataFrame(mock_prodes_data, crs='EPSG:4326')
        logger.info(f"Created mock PRODES data: {len(mock_prodes)} polygons")
        
        # Test grid snapping procedure:
        cell_stats = grid_snap(mock_gedi, resolution=9, min_shots_per_cell=2)
        logger.info(f"Grid snapping: {len(cell_stats)} H3 cells")
        
        # Test complete feature pipeline with mock data:
        features = feat_engineering_pipeline(mock_gedi, mock_prodes, mock_srtm)
        logger.info(f"Feature pipeline: {len(features)} feature vectors")
        
        if len(features) > 0:
            logger.info(f"Mock test features: {list(features.columns)}")
            
            # Check for key feature categories:
            basic_features = ['h3_cell', 'lat', 'lon', 'shot_count']
            norm_features = [col for col in features.columns if col.endswith('_norm')]
            logger.info(f"Basic features: {[f for f in basic_features if f in features.columns]}")
            logger.info(f"Normalized features: {len(norm_features)}")
            
        logger.info("Mock data test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Mock data test failed: {e}")
        return False

# ===============================================================================
# MAIN TEST SCRIPT
# ===============================================================================
if __name__ == "__main__":
    
    logger.info("Starting GEDI-PRODES-SRTM pipeline tests")
    
    # Test 1: Mock data (no network required):
    success1 = test_mock_data()
    
    # Test 2: Individual functions:
    success2 = test_individual_functions()
    
    # Test 3: Complete pipeline (requires Earth Engine):
    success3 = test_small_amazon_region()
    
    # Summary:
    passed_tests = sum([success1, success2, success3])
    total_tests = 3
    
    logger.info(f"\nTest Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("All tests passed!")
        sys.exit(0)
    elif passed_tests >= 2:
        logger.warning("Most tests passed, but some issues detected")
        sys.exit(1)
    else:
        logger.error("Multiple tests failed")
        sys.exit(1) 