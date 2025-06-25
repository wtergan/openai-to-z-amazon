"""
Comprehensive test script for the anomaly detection pipeline.
Tests feature engineering, anomaly scoring (weighted + isolation forest), 
and LLM assessment (individual + batch approaches).
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Adding current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dataset_fetching import gedi_prodes_srtm_fetch_pipeline, make_bbox
from feature_engineering import feat_engineering_pipeline  
from anomaly_detect import score_cells, rank_cells, PERCELL_FEATURES
from model_integration import analyze_top_n_cells, analyze_top_n_cells_batch

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===============================================================================
# MOCK REGIONAL LIDAR AND SENTINEL-2 CONTEXT FOR TESTING
# ===============================================================================
def create_mock_lidar_s2_context() -> Dict[str, Any]:
    """
    Create mock regional LiDAR and Sentinel-2 context for testing.
    In real usage, this would come from actual regional data processing.
    """
    return {
        "statistics": {
            "lidar_stats": {
                "mean_elevation": 145.2,
                "std_elevation": 12.8,
                "min_elevation": 120.5,
                "max_elevation": 180.3,
                "slope_stats": {
                    "mean_slope": 2.1,
                    "max_slope": 15.6,
                    "terrain_roughness": 3.2
                }
            },
            "sentinel2_stats": {
                "ndvi_stats": {
                    "mean": 0.76,
                    "std": 0.12,
                    "min": 0.45,
                    "max": 0.95
                },
                "spectral_bands": {
                    "red_mean": 0.08,
                    "nir_mean": 0.35,
                    "swir_mean": 0.12
                },
                "cloud_coverage": 5.2,
                "acquisition_date": "2024-01-15"
            },
            "regional_summary": {
                "total_area_km2": 25.6,
                "forest_coverage_pct": 78.4,
                "deforestation_pct": 3.2,
                "water_bodies_pct": 1.1
            }
        },
        # In real usage, these would be actual base64-encoded images
        "image": None,  # Base64 encoded LiDAR elevation visualization
        "ndvi_image": None,  # Base64 encoded NDVI composite
        "false_color_image": None  # Base64 encoded false color composite
    }

# ===============================================================================
# FEATURE ENGINEERING TESTING
# ===============================================================================
def test_feature_engineering(bbox: Dict[str, float], year: int) -> Optional[pd.DataFrame]:
    """Test the complete feature engineering pipeline."""
    logger.info("=" * 60)
    logger.info("TESTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Fetch datasets
        logger.info(f"Fetching datasets for bbox {bbox} and year {year}")
        gedi_df, prodes_gdf, srtm_df = gedi_prodes_srtm_fetch_pipeline(bbox, year)
        
        logger.info(f"Fetched {len(gedi_df)} GEDI shots, {len(prodes_gdf)} PRODES polygons, {len(srtm_df)} SRTM points")
        
        if gedi_df.empty:
            logger.warning("No GEDI data available for testing")
            return None
            
        # Run feature engineering pipeline
        logger.info("Running feature engineering pipeline...")
        features_df = feat_engineering_pipeline(gedi_df, prodes_gdf, srtm_df)
        
        if features_df.empty:
            logger.warning("Feature engineering produced no results")
            return None
            
        logger.info(f"Generated {len(features_df)} feature vectors with {len(features_df.columns)} features")
        logger.info(f"Feature columns: {list(features_df.columns)}")
        
        # Display sample statistics
        logger.info("\nSample feature statistics:")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        logger.info(f"Numeric features: {len(numeric_cols)}")
        
        if len(numeric_cols) > 0:
            logger.info("\nFeature value ranges:")
            for col in numeric_cols[:10]:  # Show first 10 features
                if features_df[col].notna().any():
                    logger.info(f"  {col}: {features_df[col].min():.4f} to {features_df[col].max():.4f}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
        return None

# ===============================================================================
# ANOMALY SCORING TESTING
# ===============================================================================
def test_anomaly_scoring(features_df: pd.DataFrame, n_top: int = 5) -> Dict[str, pd.DataFrame]:
    """Test both weighted and isolation forest scoring methods."""
    logger.info("=" * 60)
    logger.info("TESTING ANOMALY SCORING METHODS")
    logger.info("=" * 60)
    
    results = {}
    
    # Test weighted scoring
    logger.info("Testing weighted scoring...")
    try:
        weighted_df = score_cells(features_df, method='weighted', weights=PERCELL_FEATURES)
        weighted_top_n = rank_cells(weighted_df, n=n_top)
        results['weighted'] = weighted_top_n
        
        logger.info(f"Weighted scoring: Top {n_top} cells identified")
        logger.info("Top weighted scores:")
        for i, (_, row) in enumerate(weighted_top_n.iterrows(), 1):
            cell_id = row.get('h3_cell', f'cell_{i}')
            score = row.get('score', 0)
            logger.info(f"  {i}. Cell {cell_id}: score={score:.4f}")
            
    except Exception as e:
        logger.error(f"Weighted scoring test failed: {e}")
        
    # Test isolation forest scoring
    logger.info("\nTesting isolation forest scoring...")
    try:
        # Get available normalized features for isolation forest
        # PERCELL_FEATURES keys already have '_norm' suffix, so use them directly
        available_features = [col for col in PERCELL_FEATURES.keys() 
                            if col in features_df.columns]
        
        if not available_features:
            logger.warning("No normalized features available for isolation forest")
        else:
            iforest_df = score_cells(
                features_df, 
                method='iforest', 
                feature_cols=available_features,
                contamination=0.05,
                random_state=424242
            )
            iforest_top_n = rank_cells(iforest_df, n=n_top)
            results['iforest'] = iforest_top_n
            
            logger.info(f"Isolation forest scoring: Top {n_top} cells identified")
            logger.info("Top isolation forest scores:")
            for i, (_, row) in enumerate(iforest_top_n.iterrows(), 1):
                cell_id = row.get('h3_cell', f'cell_{i}')
                score = row.get('score', 0)
                logger.info(f"  {i}. Cell {cell_id}: score={score:.4f}")
                
    except Exception as e:
        logger.error(f"Isolation forest scoring test failed: {e}")
    
    return results

# ===============================================================================
# LLM ASSESSMENT TESTING
# ===============================================================================
def test_llm_assessment(top_n_results: Dict[str, pd.DataFrame], 
                       lidar_s2_ctx: Dict[str, Any],
                       provider: str = "openrouter",
                       model_name: Optional[str] = None,
                       test_batch: bool = True) -> Dict[str, Any]:
    """Test LLM assessment using both individual and batch approaches."""
    logger.info("=" * 60)
    logger.info("TESTING LLM ASSESSMENT")
    logger.info("=" * 60)
    
    assessment_results = {}
    
    for scoring_method, top_n_df in top_n_results.items():
        logger.info(f"\nTesting LLM assessment for {scoring_method} scoring results...")
        
        # Convert DataFrame to list of dictionaries for LLM assessment
        top_n_cells = top_n_df.to_dict('records')
        
        # Test individual assessment approach
        logger.info(f"Testing individual assessment approach ({len(top_n_cells)} cells)...")
        try:
            individual_results = analyze_top_n_cells(
                top_n_cells,
                lidar_s2_ctx,
                provider=provider,
                model_name=model_name,
                temperature=0.7,
                save_log=True
            )
            assessment_results[f"{scoring_method}_individual"] = individual_results
            logger.info(f"Individual assessment completed: {len(individual_results)} responses")
            
            # Show sample response
            if individual_results:
                sample_response = individual_results[0]['llm_response']
                logger.info(f"Sample individual response: {sample_response[:200]}...")
                
        except Exception as e:
            logger.error(f"Individual assessment failed for {scoring_method}: {e}")
        
        # Test batch assessment approach
        if test_batch:
            logger.info(f"Testing batch assessment approach ({len(top_n_cells)} cells)...")
            try:
                batch_result = analyze_top_n_cells_batch(
                    top_n_cells,
                    lidar_s2_ctx,
                    provider=provider,
                    model_name=model_name,
                    temperature=0.7,
                    save_log=True
                )
                assessment_results[f"{scoring_method}_batch"] = batch_result
                logger.info(f"Batch assessment completed for {batch_result['num_cells']} cells")
                
                # Show sample response
                batch_response = batch_result['llm_response']
                logger.info(f"Sample batch response: {batch_response[:200]}...")
                
            except Exception as e:
                logger.error(f"Batch assessment failed for {scoring_method}: {e}")
    
    return assessment_results

# ===============================================================================
# MAIN TESTING PIPELINE
# ===============================================================================
def run_full_pipeline_test(
    bbox: Dict[str, float],
    year: int,
    n_top: int = 3,
    provider: str = "openrouter",
    model_name: Optional[str] = None,
    test_llm: bool = True,
    test_batch: bool = True
) -> Dict[str, Any]:
    """Run the complete anomaly detection pipeline test."""
    logger.info("STARTING COMPREHENSIVE ANOMALY DETECTION PIPELINE TEST")
    logger.info(f"Bbox: {bbox}")
    logger.info(f"Year: {year}")
    logger.info(f"Top N cells: {n_top}")
    logger.info(f"LLM Provider: {provider}")
    logger.info(f"Test LLM: {test_llm}")
    
    results = {}
    
    # Step 1: Test feature engineering
    features_df = test_feature_engineering(bbox, year)
    if features_df is None:
        logger.error("Feature engineering failed, stopping pipeline test")
        return {"error": "Feature engineering failed"}
    
    results['features'] = {
        'num_cells': len(features_df),
        'num_features': len(features_df.columns),
        'feature_columns': list(features_df.columns)
    }
    
    # Step 2: Test anomaly scoring
    scoring_results = test_anomaly_scoring(features_df, n_top)
    if not scoring_results:
        logger.error("Anomaly scoring failed, stopping pipeline test")
        return {"error": "Anomaly scoring failed"}
    
    results['scoring'] = scoring_results
    
    # Step 3: Test LLM assessment (if enabled)
    if test_llm:
        # Create mock regional context
        lidar_s2_ctx = create_mock_lidar_s2_context()
        
        assessment_results = test_llm_assessment(
            scoring_results, 
            lidar_s2_ctx, 
            provider, 
            model_name, 
            test_batch
        )
        results['llm_assessment'] = assessment_results
    else:
        logger.info("Skipping LLM assessment tests")
    
    logger.info("=" * 60)
    logger.info("PIPELINE TEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return results

# ===============================================================================
# COMMAND LINE INTERFACE
# ===============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test the complete anomaly detection pipeline")
    parser.add_argument("--bbox", required=True, 
                       help="Bounding box as 'min_lon,min_lat,max_lon,max_lat'")
    parser.add_argument("--year", type=int, required=True, 
                       help="Year to process")
    parser.add_argument("--n-top", type=int, default=3, 
                       help="Number of top cells to analyze (default: 3)")
    parser.add_argument("--provider", default="openrouter", 
                       choices=["openai", "openrouter"],
                       help="LLM provider to use (default: openrouter)")
    parser.add_argument("--model", 
                       help="Specific model name (optional)")
    parser.add_argument("--skip-llm", action="store_true", 
                       help="Skip LLM assessment tests")
    parser.add_argument("--skip-batch", action="store_true", 
                       help="Skip batch LLM assessment tests")
    parser.add_argument("--output", 
                       help="Output file for test results (optional)")
    
    args = parser.parse_args()
    
    # Parse bounding box
    bbox_coords = [float(x) for x in args.bbox.split(',')]
    if len(bbox_coords) != 4:
        parser.error("Bounding box must have exactly 4 coordinates")
    
    bbox = make_bbox(*bbox_coords)
    
    # Run tests
    results = run_full_pipeline_test(
        bbox=bbox,
        year=args.year,
        n_top=args.n_top,
        provider=args.provider,
        model_name=args.model,
        test_llm=not args.skip_llm,
        test_batch=not args.skip_batch
    )
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with output_path.open('w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Test results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if 'features' in results:
        print(f"Features: {results['features']['num_cells']} cells, {results['features']['num_features']} features")
    
    if 'scoring' in results:
        for method in results['scoring']:
            print(f"Scoring ({method}): ✓ Completed")
    
    if 'llm_assessment' in results:
        for assessment in results['llm_assessment']:
            print(f"LLM Assessment ({assessment}): ✓ Completed")
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")

if __name__ == "__main__":
    main() 