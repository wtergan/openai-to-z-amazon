"""
Feature engineering module for GEDI-PRODES-SRTM-Sentinel2 anomaly detection.

This module processes GEDI points by snapping them to a deterministic H3 grid,
computes per-cell statistics, and joins with PRODES attributes. Outputs are
normalized and ready for anomaly scoring.
In-line comments heavily annotated to help explain heavy DataFrame transformations.
Note that LiDAR data is not used in this module, as it is only used for general plotting and 
feeding of simple plots and stats to the LLM.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any

from datetime import datetime

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_GRID_CONFIG = {
    'resolution': 9,               # H3 resolution (9 = ~174m edge length)
    'min_shots_per_cell': 3,       # Minimum GEDI shots required per cell
    'canopy_height_threshold': 2.0 # Minimum canopy height (meters)
}

# ===============================================================================
# GRID SNAPPING GEDI POINTS TO H3 GRID
# ===============================================================================
def grid_snap(
    gedi_df: pd.DataFrame,
    resolution: int = DEFAULT_GRID_CONFIG['resolution'],
    min_shots_per_cell: int = DEFAULT_GRID_CONFIG['min_shots_per_cell'],
    canopy_height_threshold: float = DEFAULT_GRID_CONFIG['canopy_height_threshold']
) -> pd.DataFrame:
    """
    Snap GEDI points to H3 grid and compute per-cell statistics.
    Returns DataFrame with H3 cells and aggregated canopy statistics.
    canopy_height_threshold: the minimum canopy height (meters) to consider a GEDI shot as valid.
    """
    logger.info(f"Snapping {len(gedi_df)} GEDI shots to H3 grid (resolution {resolution})")
    if gedi_df.empty:
        logger.warning("Empty GEDI DataFrame provided")
        return pd.DataFrame(columns=['h3_cell', 'lat', 'lon', 'shot_count', 'mean_canopy_height',
                                   'std_canopy_height', 'min_canopy_height', 'max_canopy_height', 'mean_rh98',
                                   'canopy_cover_ratio', 'height_variability'])
    # Filtering out invalid shots, as well as shots with the canopy height below the given threshold:
    valid_shots = gedi_df.dropna(subset=['lon', 'lat', 'rh98'])
    valid_shots = valid_shots[valid_shots['canopy_height'] >= canopy_height_threshold]
    if valid_shots.empty:
        logger.warning("No valid GEDI shots after filtering")
        return pd.DataFrame(columns=['h3_cell', 'lat', 'lon', 'shot_count', 'mean_canopy_height',
                                   'std_canopy_height', 'min_canopy_height', 'max_canopy_height', 'mean_rh98',
                                   'canopy_cover_ratio', 'height_variability'])

    # Converting each valid shot into H3 cells: resolution 9 ~174m edge length, ~0.11 km² (small neighborhood/block scale).
    h3_cells = []
    for _, row in valid_shots.iterrows():
        try:
            h3_cell = h3.latlng_to_cell(row['lat'], row['lon'], resolution)
            h3_cells.append(h3_cell)
        except Exception as e:
            logger.warning(f"Failed to convert coordinates to H3: {e}")
            h3_cells.append(None)
    valid_shots = valid_shots.copy()
    valid_shots['h3_cell'] = h3_cells
    valid_shots = valid_shots.dropna(subset=['h3_cell'])
    if valid_shots.empty:
        logger.warning("No valid H3 cells generated")
        return pd.DataFrame(columns=['h3_cell', 'lat', 'lon', 'shot_count', 'mean_canopy_height',
                                   'std_canopy_height', 'min_canopy_height', 'max_canopy_height', 'mean_rh98',
                                   'canopy_cover_ratio', 'height_variability'])
    
    # Grouping valid_shots by h3_cell, then aggregating stats for canopy_height, rh98, lat, lon assigned to each cell:
    cell_stats = valid_shots.groupby('h3_cell').agg({
        'canopy_height': ['count', 'mean', 'std', 'min', 'max'],
        'rh98': ['mean', 'std'],
        'lat': 'mean',
        'lon': 'mean'
    }).round(4)
    # Renaming columns for clarity:
    cell_stats.columns = ['shot_count', 'mean_canopy_height', 'std_canopy_height', 'min_canopy_height', 'max_canopy_height',
                         'mean_rh98', 'std_rh98', 'lat', 'lon']

    # Filtering out cells with minimum shot count per cell:
    cell_stats = cell_stats[cell_stats['shot_count'] >= min_shots_per_cell]
    if cell_stats.empty:
        logger.warning(f"No cells meet minimum shot count requirement ({min_shots_per_cell})")
        return pd.DataFrame(columns=['h3_cell', 'lat', 'lon', 'shot_count', 'mean_canopy_height',
                                   'std_canopy_height', 'min_canopy_height', 'max_canopy_height', 'mean_rh98',
                                   'canopy_cover_ratio', 'height_variability'])
    # Computing canopy cover ratio; normalizing the shot count per cell relative to the max shot count.
    cell_stats['canopy_cover_ratio'] = (
        cell_stats['shot_count'] / cell_stats['shot_count'].max()
    ).round(4)
    # Computing height variability, measuring how much the canopy height varies within each cell.
    cell_stats['height_variability'] = (
        cell_stats['std_canopy_height'] / (cell_stats['mean_canopy_height'] + 1e-6)
    ).round(4)
    # Filling in missing values for std_canopy_height and std_rh_98 with 0, then resetting the index for the DataFrame.
    cell_stats['std_canopy_height'] = cell_stats['std_canopy_height'].fillna(0)
    cell_stats['std_rh98'] = cell_stats['std_rh98'].fillna(0)
    cell_stats = cell_stats.reset_index()

    # Computing centroids for each H3 cell, which is essentially the lat and lon of the center of the H3 cell::
    centroids = []
    for h3_cell in cell_stats['h3_cell']:
        try:
            lat, lon = h3.cell_to_latlng(h3_cell)
            centroids.append((lat, lon))
        except Exception as e:
            logger.warning(f"Failed to get centroid for H3 cell {h3_cell}: {e}")
            centroids.append((cell_stats[cell_stats['h3_cell'] == h3_cell]['lat'].iloc[0],
                            cell_stats[cell_stats['h3_cell'] == h3_cell]['lon'].iloc[0]))
            
    # Converting the centroids into DataFrame, joining it with cell_stats:
    centroid_df = pd.DataFrame(centroids, columns=['h3_lat', 'h3_lon'])
    cell_stats = pd.concat([cell_stats.reset_index(drop=True), centroid_df], axis=1)

    # Copying h3_lat, h3_lon derived from centroids into lat, lon, then dropping h3_lat, h3_lon:
    cell_stats['lat'] = cell_stats['h3_lat']
    cell_stats['lon'] = cell_stats['h3_lon']
    cell_stats = cell_stats.drop(columns=['h3_lat', 'h3_lon'])
    logger.info(f"Generated {len(cell_stats)} H3 cells with sufficient GEDI coverage")

    # Finally, returning the cell_stats DataFrame:
    return cell_stats

# ===============================================================================
# SPATIAL JOIN H3 CELLS WITH SRTM ELEVATION AND SLOPE DATA
# ===============================================================================
def spatial_join_with_srtm(cell_stats: pd.DataFrame, srtm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform spatial join between H3 cells and SRTM elevation and slope data. Takes the cell_stats DataFrame
    as well as the srtm_df DataFrame and returns an amalgamated DataFrame with the SRTM attributes joined to 
    the H3 cells. For each H3 cell, finds nearby SRTM points via a buffer, and computes terrain statistics.
    """
    if cell_stats.empty or srtm_df.empty:
        # Add SRTM columns with default values:
        cell_stats['mean_elevation'] = 0.0
        cell_stats['std_elevation'] = 0.0
        cell_stats['mean_slope'] = 0.0
        cell_stats['max_slope'] = 0.0
        cell_stats['terrain_roughness'] = 0.0
        cell_stats['elevation_norm'] = 0.0
        return cell_stats
    logger.info(f"Joining {len(cell_stats)} H3 cells with {len(srtm_df)} SRTM points")
    
    # Creating GeoDataFrames from H3 cells and SRTM data points respectively:
    cell_points = [Point(lon, lat) for lon, lat in zip(cell_stats['lon'], cell_stats['lat'])]
    cells_gdf = gpd.GeoDataFrame(cell_stats, geometry=cell_points, crs='EPSG:4326')
    srtm_points = [Point(lon, lat) for lon, lat in zip(srtm_df['lon'], srtm_df['lat'])]
    srtm_gdf = gpd.GeoDataFrame(srtm_df, geometry=srtm_points, crs='EPSG:4326')
    
    # For each H3 cell, find SRTM points within a reasonable distance.
    # H3 resolution 9 has ~174m edge length, so use ~300m buffer to capture nearby points:
    buffer_distance = 0.003  # ~300m in degrees (approximate).
    
    # Computing the terrain stats for each H3 cell, then joining the terrain stats to the cell_stats DataFrame:
    terrain_stats = []
    for idx, cell in cells_gdf.iterrows():
        # Creating a buffer around the cell centroid, then finding SRTM points within that buffer:
        cell_buffer = cell.geometry.buffer(buffer_distance)
        nearby_srtm = srtm_gdf[srtm_gdf.geometry.within(cell_buffer)]
        if len(nearby_srtm) > 0:
            # Computing the elevation stats for each H3 cell, given the nearby SRTM points in the buffer:
            elevation_stats = {
                'mean_elevation': nearby_srtm['elevation'].mean(),
                'std_elevation': nearby_srtm['elevation'].std() if len(nearby_srtm) > 1 else 0.0,
                'mean_slope': nearby_srtm['slope_degrees'].mean(),
                'max_slope': nearby_srtm['slope_degrees'].max(),
                # Computing the terrain roughness; the standard deviation of the slope degrees:
                'terrain_roughness': nearby_srtm['slope_degrees'].std() if len(nearby_srtm) > 1 else 0.0
            }
        else:
            # No nearby SRTM points, use default values:
            elevation_stats = {
                'mean_elevation': 0.0,
                'std_elevation': 0.0,
                'mean_slope': 0.0,
                'max_slope': 0.0,
                'terrain_roughness': 0.0
            }
        terrain_stats.append(elevation_stats)
    
    # Converting the terrain stats to a DataFrame, then joining it with the cell_stats DataFrame:
    terrain_df = pd.DataFrame(terrain_stats)
    result = pd.concat([cell_stats.reset_index(drop=True), terrain_df], axis=1)
    
    # Computing normalized elevation for better canopy-elevation relationships:
    # Normalize elevation within the local context to remove macro-relief effects:
    if 'mean_elevation' in result.columns and result['mean_elevation'].std() > 0:
        # Use z-score normalization to center around local mean:
        elevation_mean = result['mean_elevation'].mean()
        elevation_std = result['mean_elevation'].std()
        result['elevation_norm'] = ((result['mean_elevation'] - elevation_mean) / elevation_std).round(4)
    else:
        # If no elevation variation, set normalized elevation to 0:
        result['elevation_norm'] = 0.0
    
    # Rounding values for consistency:
    terrain_columns = ['mean_elevation', 'std_elevation', 'mean_slope', 'max_slope', 'terrain_roughness']
    for col in terrain_columns:
        if col in result.columns:
            result[col] = result[col].round(4)
    logger.info(f"Added terrain features to {len(result)} H3 cells.")

    # Finally, returning the resultant amalgamated DataFrame:
    return result

# ===============================================================================
# SPATIAL JOIN H3 CELLS WITH PRODES DEFORESTATION POLYGONS
# ===============================================================================
def spatial_join_with_prodes(cell_stats: pd.DataFrame, prodes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Perform spatial join between H3 cells and PRODES polygons. For each point in the cell_stats DataFrame,
    check if it lies inside any of the polygons in the PRODES GeoDataFrame; if yes, copy the polygons' attrs
    into the cell_stats DataFrame, then return the resultant amalgamated DataFrame.
    """
    if cell_stats.empty or prodes_gdf.empty:
        # Add PRODES columns with default values: 
        cell_stats['deforested'] = False
        cell_stats['deforestation_year'] = None
        cell_stats['deforestation_area_ha'] = 0.0
        cell_stats['yrs_since_deforest'] = None
        return cell_stats
    logger.info(f"Joining {len(cell_stats)} H3 cells with {len(prodes_gdf)} PRODES polygons")
    
    # Create GeoDataFrame from H3 cells; 'cell_points': list of shapely Point objs, representing centroid of an H3 cell.
    # Setting this to the 'geometry' column of the GeoDataFrame to enable spatial operations like spatial joins:
    cell_points = [Point(lon, lat) for lon, lat in zip(cell_stats['lon'], cell_stats['lat'])]
    cells_gdf = gpd.GeoDataFrame(cell_stats, geometry=cell_points, crs='EPSG:4326')
        
    # Ensure PRODES has consistent CRS:
    if prodes_gdf.crs != 'EPSG:4326':
        prodes_gdf = prodes_gdf.to_crs('EPSG:4326')
        
    # Spatial join: find which cells from 'cells_gdf' intersect with deforestation polygons in 'prodes_gdf'.
    joined = gpd.sjoin(cells_gdf, prodes_gdf, how='left', predicate='intersects')
        
    # Handle multiple intersections per cell by retaining only the largest polygon in that cell.
    if 'area_ha' in joined.columns and not joined['area_ha'].isna().all():
        valid_areas = joined.dropna(subset=['area_ha'])
        if not valid_areas.empty:
            max_indices = valid_areas.groupby('h3_cell')['area_ha'].idxmax()
            # Keeping only the row with max area_ha and those that did not have area_ha to begin with:
            joined = pd.concat([
                joined.loc[max_indices],
                joined[~joined.index.isin(valid_areas.index)]
            ]).drop_duplicates()
        
    # Computing years since deforestation for each cell, then creating a deforestation indicator:
    current_year = datetime.now().year
    joined['yrs_since_deforest'] = None
    if 'year' in joined.columns:
        mask = joined['year'].notna()
        joined.loc[mask, 'yrs_since_deforest'] = current_year - joined.loc[mask, 'year']
    joined['deforested'] = joined['year'].notna()
        
    # Renaming and selecting relevant columns:
    column_mapping = {
            'year': 'deforestation_year',
            'area_ha': 'deforestation_area_ha'
        }
    for old_col, new_col in column_mapping.items():
        if old_col in joined.columns:
            joined[new_col] = joined[old_col]
    joined['deforestation_area_ha'] = joined.get('deforestation_area_ha', 0.0).fillna(0.0)
        
    # Removing geometry and extra columns for return:
    columns_to_keep = [
            'h3_cell', 'lat', 'lon', 'shot_count', 'mean_canopy_height', 'std_canopy_height',
            'min_canopy_height', 'max_canopy_height', 'mean_rh98', 'std_rh98', 'canopy_cover_ratio', 'height_variability',
            'deforested', 'deforestation_year', 'deforestation_area_ha', 'yrs_since_deforest'
        ]
    result_columns = [col for col in columns_to_keep if col in joined.columns]
    result = joined[result_columns].copy() 

    # Finally, returning the resultant amalgamated DataFrame:
    return result

# ===============================================================================
# ADDING DERIVED FEATURES TO PER-CELL DATAFRAME
# ===============================================================================
def add_derived_percell_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the DataFrame and return the modified DataFrame."""
    # Computing the shot density; number of shots per unit area, where the unit area is the area of an H3 cell.
    # For H3 resolution 9, the area of an H3 cell is ~0.105 km²:
    h3_area_km2 = 0.105
    df['shot_density'] = (df['shot_count'] / h3_area_km2).astype(float).round(4)
    
    # Computing the canopy height range; the true range (max - min) gives better spread information:
    df['canopy_height_range'] = (df['max_canopy_height'] - df['min_canopy_height']).astype(float).round(4)  
    
    # Computing the rh98_height_difference; difference captures subtle variations between these similar metrics:
    # RH98 and canopy height measure similar things, so difference reveals measurement inconsistencies or forest structure anomalies:
    df['rh98_height_difference'] = (
        df['mean_rh98'] - df['mean_canopy_height']    
        ).astype(float).round(4)
        
    # Computing the deforestation impact proxy; decaying impact over time, where the impact is 1.0 for the first year,
    # and decays by 1/2 every year thereafter:
    if 'yrs_since_deforest' in df.columns:
        # Ensure the column is numeric and handle NaN values properly
        yrs_since = pd.to_numeric(df['yrs_since_deforest'], errors='coerce')
        df['deforest_impact'] = np.where(
            yrs_since.notna(),
            1.0 / (yrs_since + 1),
            0.0
        )
        df['deforest_impact'] = df['deforest_impact'].round(4)
    else:
        df['deforest_impact'] = 0.0
    
    # Computing terrain-based derived features if elevation data is available:
    if 'mean_elevation' in df.columns and 'mean_slope' in df.columns:
        # Computing the canopy-elevation relationship using normalized elevation to remove macro-relief effects:
        # This gives a more meaningful relationship between local canopy and local terrain context:
        if 'elevation_norm' in df.columns:
            df['canopy_elevation_difference'] = (
                df['mean_canopy_height'] - df['elevation_norm']
            ).round(4)
        else:
            # Fallback to simple difference if normalization not available:
            df['canopy_elevation_difference'] = (
                df['mean_canopy_height'] - df['mean_elevation']
            ).round(4)
        
        # Computing the slope-canopy interaction: how canopy varies with terrain slope:
        df['slope_canopy_interaction'] = (
            df['mean_slope'] * df['height_variability']
        ).round(4)
        
        # Computing the terrain complexity index: combines slope and elevation variability:
        df['terrain_complexity'] = (
            (df['mean_slope'] * 0.6) + (df['terrain_roughness'] * 0.4)
        ).round(4)
    else:
        # Default values if terrain data not available:
        df['canopy_elevation_difference'] = 0.0
        df['slope_canopy_interaction'] = 0.0
        df['terrain_complexity'] = 0.0
    
    # Computing the anomaly potential score (pre-normalization); updated to include terrain features:
    terrain_weight = 0.1 if 'mean_elevation' in df.columns else 0.0
    base_weights_sum = 1.0 - terrain_weight # Ensuring the sum of weights is 1.0 even if terrain data is not available.
    
    df['anomaly_potential'] = (
        df['height_variability'] * (0.3 * base_weights_sum) +
        df['canopy_height_range'] * (0.2 * base_weights_sum) +
        df['shot_density'] * (0.2 * base_weights_sum) +
        df['deforest_impact'] * (0.3 * base_weights_sum) +
        df['terrain_complexity'] * terrain_weight
    ).round(4)
    
    # Finally, returning the DataFrame with the derived features:
    return df
    
# ===============================================================================
# NORMALIZING PER-CELL FEATURES FOR SCORING
# ===============================================================================
def normalize_percell_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize only per-cell features used in anomaly scoring pipeline.
    Excludes regional features which should be handled separately.
    """
    # Per-cell features that MUST be normalized for scoring:
    percell_scoring_features = [
        'height_variability', 'canopy_height_range', 'shot_density', 
        'rh98_height_difference', 'deforest_impact', 'terrain_complexity',
        'canopy_elevation_difference', 'slope_canopy_interaction', 
        'mean_slope', 'anomaly_potential'
    ]
    
    # Per-cell features to keep raw for LLM interpretability:
    percell_context_features = [
        'shot_count', 'mean_canopy_height', 'std_canopy_height', 'min_canopy_height', 'max_canopy_height',
        'mean_rh98', 'std_rh98', 'canopy_cover_ratio',
        'mean_elevation', 'std_elevation', 'elevation_norm', 'max_slope', 'terrain_roughness',
        'deforested', 'deforestation_year', 'deforestation_area_ha', 'yrs_since_deforest'
    ]
    
    # Filter scoring features to only include those present in the DataFrame:
    available_scoring_features = [col for col in percell_scoring_features if col in df.columns]
    if not available_scoring_features:
        logger.warning("No per-cell scoring features available for normalization.")
        return df
        
    logger.info(f"Normalizing {len(available_scoring_features)} per-cell scoring features, "
               f"keeping {len([col for col in percell_context_features if col in df.columns])} context features raw")

    # Normalization process for per-cell scoring features only:
    normalized_df = df.copy()
    scaler = MinMaxScaler()
    try:
        # Handle potential issues with constant features or NaN values:
        feature_data = df[available_scoring_features].fillna(0)
        normalized_values = scaler.fit_transform(feature_data)
        
        # Ensure values are strictly in [0, 1] range:
        normalized_values = np.clip(normalized_values, 0, 1)
        
        # Create normalized feature columns:
        normalized_feature_df = pd.DataFrame(
            normalized_values,
            columns=[f"{col}_norm" for col in available_scoring_features],
            index=df.index
        )
        normalized_df = pd.concat([normalized_df, normalized_feature_df], axis=1)
        logger.info(f"Successfully normalized {len(available_scoring_features)} per-cell features for scoring")
    except Exception as e:
        logger.error(f"Per-cell feature normalization failed: {e}")
        # Add dummy normalized columns:
        for col in available_scoring_features:
            normalized_df[f"{col}_norm"] = df[col]
    
    # Finally, returning the normalized per-cell feature DataFrame:
    return normalized_df

# ===============================================================================
# PER-CELL FEATURE VECTOR BUILDING PIPELINE (spatially joined features only)
# ===============================================================================
def build_percell_vectors(cell_stats: pd.DataFrame, prodes_gdf: gpd.GeoDataFrame = None, 
                         srtm_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Per-cell feature vector building pipeline for only the spatially-joined features (SRTM, PRODES).
    These features vary meaningfully between H3 cells based on local spatial characteristics.
    """
    logger.info(f"Building per-cell feature vectors for {len(cell_stats)} H3 cells")
    
    if cell_stats.empty:
        logger.warning("Empty cell statistics provided")
        return pd.DataFrame()
    
    # Start with GEDI-derived cell stats (already per-cell):
    feature_df = cell_stats.copy()
    
    # Join with SRTM data if available (per-cell spatial join):
    if srtm_df is not None and not srtm_df.empty:
        logger.info(f"Spatially joining with {len(srtm_df)} SRTM elevation points")
        feature_df = spatial_join_with_srtm(feature_df, srtm_df)
    else:
        logger.warning("No SRTM data provided, adding default terrain features")
        terrain_columns = ['mean_elevation', 'std_elevation', 'mean_slope', 'max_slope', 'terrain_roughness', 'elevation_norm']
        for col in terrain_columns:
            feature_df[col] = 0.0
        
    # Join with PRODES data if available (per-cell spatial join):
    if prodes_gdf is not None and not prodes_gdf.empty:
        logger.info(f"Spatially joining with {len(prodes_gdf)} PRODES polygons")
        feature_df = spatial_join_with_prodes(feature_df, prodes_gdf)
    else:
        logger.warning("No PRODES data provided, adding default deforestation features")
        feature_df['deforested'] = False
        feature_df['deforestation_year'] = None
        feature_df['deforestation_area_ha'] = 0.0
        feature_df['yrs_since_deforest'] = None
        
    # Add derived features based on per-cell data, and then do per-cell normalization of pertinent features:
    derived_df = add_derived_percell_features(feature_df)
    normalized_df = normalize_percell_features(derived_df)
    
    # Finally, returning the normalized per-cell feature DataFrame:
    logger.info(f"Generated per-cell feature vectors with {len(normalized_df.columns)} features")
    return normalized_df

# ================================================================================================
# COMPLETE FEATURE VECTOR BUILDING PIPELINE (gridsnap, per-cell feature vector building)
# ================================================================================================
def feat_engineering_pipeline(gedi_df: pd.DataFrame, prodes_gdf: gpd.GeoDataFrame = None, srtm_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    The complete feature engineering pipeline for the spatially joined, per-cell feature vectors.
    Takes the GEDI DataFrame, PRODES GeoDataFrame, and SRTM DataFrame as inputs, snaps the GEDI 
    shots to an H3 grid, perform the per-cell feature engineering, then finally returning a 
    DataFrame with the spatially joined, per-cell feature vectors.
    """
    logger.info("Starting per-cell feature engineering pipeline")
    
    # Step 1: Grid snapping; snapping the GEDI shots to an H3 grid:
    cell_stats = grid_snap(gedi_df)
    if cell_stats.empty:
        logger.warning("No H3 cells generated from GEDI data.")
        return pd.DataFrame()
        
    # Step 2: Build per-cell feature vectors (spatially-joined features only):
    percell_features = build_percell_vectors(cell_stats, prodes_gdf, srtm_df)
    
    logger.info(f"Feature engineering complete: {len(percell_features)} cells × {len(percell_features.columns)} features")
    
    # Finally, returning the per-cell feature vector DataFrame:
    return percell_features


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Example usage...
    import argparse
    import sys
    from pathlib import Path
    
    # Adding the parent directory to path for imports:
    sys.path.append(str(Path(__file__).parent))
    
    from dataset_fetching import gedi_prodes_srtm_fetch_pipeline, make_bbox
    
    parser = argparse.ArgumentParser(description="Process GEDI, PRODES, and SRTM data for feature engineering")
    parser.add_argument("--bbox", required=True, help="Bounding box as 'min_lon,min_lat,max_lon,max_lat'")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    parser.add_argument("--resolution", type=int, default=9, help="H3 resolution (default: 9)")
    parser.add_argument("--min-shots", type=int, default=3, help="Minimum shots per cell (default: 3)")
    parser.add_argument("--include-srtm", action="store_true", help="Include SRTM elevation/slope data")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Parsing the bounding box:
    bbox_coords = [float(x) for x in args.bbox.split(',')]
    bbox = make_bbox(*bbox_coords)
    
    # Fetching the datasets:
    if args.include_srtm:
        gedi_df, prodes_gdf, srtm_df = gedi_prodes_srtm_fetch_pipeline(bbox, args.year)
        print(f"Fetched {len(gedi_df)} GEDI shots, {len(prodes_gdf)} PRODES polygons, {len(srtm_df)} SRTM points")
    else:
        gedi_df, prodes_gdf, srtm_df = gedi_prodes_srtm_fetch_pipeline(bbox, args.year)
        srtm_df = None  # Ignore SRTM data if not requested
        print(f"Fetched {len(gedi_df)} GEDI shots, {len(prodes_gdf)} PRODES polygons")
    
    # Processing the features:
    features = feat_engineering_pipeline(gedi_df, prodes_gdf, srtm_df)
    
    print(f"Generated {len(features)} feature vectors")
    print(f"Feature columns: {list(features.columns)}")
    
    if args.output:
        features.to_csv(args.output, index=False)
        print(f"Saved features to: {args.output}") 