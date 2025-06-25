"""
CLI entry point for anomaly detection pipeline (CHECKPOINT_2):
- Deeply follows and integrates the logic of dataset_fetching.py, feature_extraction.py, feature_engineering.py, anomaly_detect.py,
  model_integration.py.
- Exposes all advanced data sources (GEDI, PRODES, SRTM, LiDAR, Sentinel-2).
- Supports LLM-based cell analysis with rich regional context.
- Robust progress, error, and output handling.
"""
import os
import json
import logging
import argparse
from dataset_fetching import initialize_gee, make_bbox, gedi_prodes_srtm_fetch_pipeline, lidar_sentinel2_fetch_pipeline
from feature_extraction import sentinel2_gee_extract_features, lidar_ot_extract_features
from feature_engineering import feat_engineering_pipeline
from anomaly_detect import score_cells, rank_cells
from model_integration import analyze_top_n_cells, analyze_top_n_cells_batch

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run full anomaly detection pipeline with LLM integration")
    parser.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--year", required=True, type=int, help="Year to fetch data for")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top anomalies to return")
    parser.add_argument("--tag", type=str, help="Optional tag to save results")
    parser.add_argument("--cache_dir", type=str, default="data/raw", help="Cache directory path")
    parser.add_argument("--force_refresh", action="store_true", help="Force refresh of cached data")
    parser.add_argument("--llm", action="store_true", help="Run LLM-based cell analysis (requires API key)")
    parser.add_argument("--llm_provider", type=str, default="openrouter", help="LLM provider: openai or openrouter")
    parser.add_argument("--llm_model", type=str, default="google/gemma-3-27b-it", help="LLM model name (optional override)")
    parser.add_argument("--show_images", action="store_true", help="Display images during processing")
    args = parser.parse_args()

    # Parsing the provided bounding box (bbox) coordinates for usage:
    try:
        coords = [float(x) for x in args.bbox.split(',')]
        bbox = make_bbox(*coords)
    except Exception as e:
        logging.error(f"Invalid bbox format: {e}")
        return

    # Initializing Google Earth Engine:
    logging.info("Initializing Google Earth Engine...")
    if not initialize_gee():
        logging.error("GEE initialization failed.")
        return

    # Dataset fetching pipeline for pertinent GEDI, PRODES, and SRTM data:
    logging.info("Fetching pertinent GEDI, PRODES, and SRTM datasets...")
    gedi_df, prodes_gdf, srtm_df = gedi_prodes_srtm_fetch_pipeline(
        bbox, args.year, args.cache_dir, args.force_refresh
    )
    logging.info(f"Fetched {len(gedi_df)} GEDI shots, {len(prodes_gdf)} PRODES polygons, {len(srtm_df)} SRTM points")

    # Dataset fetching and feature extraction pipeline for pertinent regional LiDAR and Sentinel-2 data:
    logging.info("Fetching pertinent regional LiDAR and Sentinel-2 data...")
    region_ctx = lidar_sentinel2_fetch_pipeline(bbox)
    if region_ctx and region_ctx.get("lidar_path"):
        lidar_stats = lidar_ot_extract_features(region_ctx["lidar_path"], show_image=args.show_images)
    else:
        lidar_stats = None
    if region_ctx and region_ctx.get("s2_data"):
        s2_stats = sentinel2_gee_extract_features(region_ctx["s2_data"], show_image=args.show_images)
    else:
        s2_stats = None
    # Merging LiDAR and S2 data for LLM analysis:
    region_features = {}
    if lidar_stats:
        region_features.update(lidar_stats)
    if s2_stats:
        region_features.update(s2_stats)

    # Feature engineering pipeline for GEDI, PRODES, and SRTM data:
    logging.info("Engineering per-cell features...")
    features = feat_engineering_pipeline(gedi_df, prodes_gdf, srtm_df)
    logging.info(f"Generated {len(features)} per-cell feature vectors")

    # Anomaly scoring and ranking pipeline:
    logging.info("Scoring anomalies and top-N ranking...")
    scored = score_cells(features, method='weighted')
    top = rank_cells(scored, n=args.top_n)
    logging.info(f"Top {args.top_n} anomalies:")
    for idx, row in top.iterrows():
        logging.info(f"{idx+1}. Cell {row['h3_cell']}: score={row['score']:.4f}, location=({row['lat']:.4f},{row['lon']:.4f})")

    # LLM-based per-cell assessment pipeline:
    llm_results = None
    if args.llm:
        logging.info("\nRunning LLM-based assessment for top cells...")
        llm_results = analyze_top_n_cells_batch(
            top.to_dict(orient='records'),
            region_features,
            provider=args.llm_provider,
            model_name=args.llm_model,
            save_log=True               # Saves to persistent, cumulative log for audit history.
        )
        # llm_results is a batch result dict with cell_assessments containing the individual cell results
        if isinstance(llm_results, dict) and "cell_assessments" in llm_results:
            logging.info(f"\nRegional Assessment:")
            logging.info(llm_results.get('regional_assessment', 'No regional assessment'))
            
            cell_assessments = llm_results.get("cell_assessments", [])
            for i, entry in enumerate(cell_assessments, 1):
                cell_id = entry.get('cell_id', f'Cell {i}')
                logging.info(f"\nLLM Assessment for {cell_id}:")
                logging.info(entry.get('llm_response', 'No response'))
        else:
            logging.error(f"Unexpected llm_results format: {type(llm_results)} - {llm_results}")

    # Save per-run results if requested; both the top-N cells and their respective LLM assessments: 
    if args.tag:
        out_file = f"{args.tag}_top{args.top_n}.json"
        with open(out_file, 'w') as f:
            json.dump(top.to_dict(orient='records'), f, indent=2)
        logging.info(f"Saved results to {out_file}")
        if llm_results:
            llm_file = f"{args.tag}_top{args.top_n}_llm.json"
            with open(llm_file, 'w') as f:
                json.dump(llm_results, f, indent=2)
            logging.info(f"Saved LLM assessments to {llm_file}")

if __name__ == '__main__':
    main()
