import numpy as np
import laspy
import rasterio
import matplotlib.pyplot as plt
import io
import base64
import ee
import urllib.request
from PIL import Image
from matplotlib.colors import LightSource
import os

# ===============================================================================
# LiDAR FEATURE EXTRACTION: OpenTopography API
# ===============================================================================
def lidar_ot_extract_features(lidar_path: str, show_image: bool = True) -> dict:
    """
    Generate and display a plot of LiDAR data along with some stats.
    Returns a dict containing the plot as a BytesIO obj, and the stats.
    Cleans up the temporary lidar_path file upon completion or error.
    """
    try:
        with rasterio.open(lidar_path) as src:
            print("LiDAR data read.")
            lidar_arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                lidar_arr = np.where(lidar_arr == src.nodata, np.nan, lidar_arr)
            
            # Usage of 2-98 percentiles for avoiding outliers that may affect color mapping:
            vmin, vmax = np.nanpercentile(lidar_arr, [2, 98])

            # Figure with subplots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Main elevation plot:
            print("Generating LiDAR elevation plot...")
            im1 = ax1.imshow(lidar_arr, cmap='terrain', vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=ax1, label='Elevation (m)')
            ax1.set_title("LiDAR Elevation Data (2-98% range)")

            # Hillshade for better terrain visualization, using LightSource for azimuth and altitude:
            print("Generating LiDAR hillshade...")
            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(lidar_arr, vert_exag=1, dx=1, dy=1, fraction=1.0)
            ax2.imshow(hillshade, cmap='gray', alpha=0.8)
            ax2.set_title("Hillshade Visualization")
            
            # Show the plot if show_image is True:
            print("Displaying the plot(s)...")
            if show_image:
                plt.show()

            # Save plot to BytesIO buffer as JPEG, tight bbox, 150 DPI:
            print("Saving plot to BytesIO buffer...")
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="JPEG", bbox_inches="tight", dpi=150)
            plt.close()
            buf.seek(0)
            
            # Binary to base64 conversion:
            print("Converting plot to base64 and computation of stats...")
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            # Calculate basic stats:
            ot_stats = {
                "image": image_base64,
                "statistics": {
                    "mean": float(np.nanmean(lidar_arr)),
                    "median": float(np.nanmedian(lidar_arr)),
                    "std": float(np.nanstd(lidar_arr)),
                    "min": float(np.nanmin(lidar_arr)),
                    "max": float(np.nanmax(lidar_arr)),
                    "percentile_2": float(vmin),
                    "percentile_98": float(vmax),
                    "shape": lidar_arr.shape
                },
                # Some spatial metadata: coordinate reference system, and bbox info:
                "crs": str(src.crs) if src.crs else None,
                "bounds": list(src.bounds)
            }
            buf.close()
            return ot_stats
    finally:
        # Clean up the temporary file
        if os.path.exists(lidar_path):
            os.unlink(lidar_path)
            print(f"Temporary file {lidar_path} deleted.")    

# ===============================================================================
# Sentinel-2 FEATURE EXTRACTION: Google Earth Engine
# ===============================================================================
def sentinel2_gee_extract_features(
    gee_data: dict,
    scale: int = 30, # Resolution for reduceRegion, 10/20/30/60m for S2 bands.
    thumb_dimensions: str = '768x768', # Thumbnail dimensions.
    show_image: bool = True
) -> dict:
    """
    Computing stats and generating a thumbnail for a Sentinel-2 GEE Image.
    Expects a dictionary from fetch_sentinel2_gee_data.
    Returns a dict containing the plot as base64, as well as pertinent stats.
    """
    if not gee_data or not gee_data.get("image"):
        error_msg = gee_data.get("error", "GEE image not provided or invalid.")
        print(f"Error: {error_msg}")
        return {
            "image": None,
            "statistics": {"error": error_msg},
            "roi_bounds": gee_data.get("roi_bounds"),
            "gee_image_details": "No image processed."
        }

    image = gee_data["image"]
    roi = gee_data["roi"]
    roi_bounds = gee_data["roi_bounds"]

    # Creating a composite reducer that computes multiple stats all in one operation:
    reducers = (
        ee.Reducer.mean().unweighted()
        .combine(ee.Reducer.minMax().unweighted(), sharedInputs=True)
        .combine(ee.Reducer.stdDev().unweighted(), sharedInputs=True)
        .combine(ee.Reducer.percentile([2, 98]).unweighted(), sharedInputs=True)
        .combine(ee.Reducer.count().unweighted(), sharedInputs=True)
    )
    selected_bands = ['B2', 'B3', 'B4', 'B8']
    try:
        # Computation of the stats for each selected band, given image, ROI, scale, and reducers:
        print(f"Calculating stats for bands: {selected_bands} at {scale}m scale...")
        band_stats = image.select(selected_bands).reduceRegion(
            reducer=reducers,
            geometry=roi,
            scale=scale,
            maxPixels=1e10,
            tileScale=4
        ).getInfo()
        print("Band stats received.")
    except ee.EEException as e:
        print(f"GEE Error calculating band stats: {e}")
        return {
            "image": None, 
            "statistics": {"error": f"GEE band stats error: {e}"},
            "roi_bounds": roi_bounds,
            "gee_image_details": "Failed during band statistics."
        }
    except Exception as e_gen:
        print(f"General Error calculating band stats: {e_gen}")
        return {
            "image": None, 
            "statistics": {"error": f"General band stats error: {e_gen}"},
            "roi_bounds": roi_bounds,
            "gee_image_details": "Failed during band statistics."
        }

    # Calculating NDVI and its stats:
    ndvi_stats = {}
    try:
        print("Computing Normalized Difference Vegetation Index (NDVI)...")
        ndvi_image = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndvi_stats_raw = ndvi_image.reduceRegion(
            reducer=reducers,
            geometry=roi,
            scale=scale,
            maxPixels=1e10,
            tileScale=4
        ).getInfo()
        print("NDVI stats received.")
        for key, value in ndvi_stats_raw.items():
            ndvi_stats[f"NDVI_{key}"] = value
    except ee.EEException as e:
        print(f"GEE Error calculating NDVI stats: {e}")
    except Exception as e_gen:
        print(f"General Error calculating NDVI stats: {e_gen}")

    # Merging the stats from all bands and NDVI:
    all_stats = {}
    for band in selected_bands:
        for stat_key, reducer_key_part in {
            "mean": "mean", "min": "min", "max": "max", "std": "stdDev", 
            "p2": "p2", "p98": "p98", "count": "count" }.items():
            gee_key = f"{band}_{reducer_key_part}"
            all_stats[f"{band}_{stat_key}"] = band_stats.get(gee_key)
    all_stats.update(ndvi_stats)

    # Generating the RGB thumbnail for LLM processing:
    image_base64 = None
    pil_image = None
    try:
        print("Generating RGB thumbnail...")
        # Vis params for 0-1 scaled reflectance. Common S2 vis is 0-0.3 range.
        rgb_vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3, 'gamma': 1.4}
        region_payload = roi.getInfo()['coordinates'] if hasattr(roi, 'getInfo') else roi
        rgb_thumbnail_url = image.visualize(**rgb_vis_params).getThumbURL({
            'region': region_payload,
            'dimensions': thumb_dimensions,
            'format': 'jpg'
        })

        # Downloading and processing the thumbnail; binary to base64 conversion:
        print(f"RGB thumbnail URL (first 100 chars): {rgb_thumbnail_url[:100]}...")
        with urllib.request.urlopen(rgb_thumbnail_url, timeout=60) as response:
            img_data = response.read()
        pil_image = Image.open(io.BytesIO(img_data))
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        print("RGB thumbnail generated and encoded.")

        # Displaying the thumbnail image if show_image is True:
        if show_image and pil_image:
            print("Displaying S2 thumbnail...")
            try:
                pil_image.show() # Usage of local default image viewer
            except Exception as e_show:
                print(f"Could not display the thumbnail image using pil_image.show(): {e_show}")
                print("Consider saving it to a file instead, if you need to view it.")


    except ee.EEException as e:
        print(f"GEE Error generating/downloading thumbnail: {e}")
    except urllib.error.URLError as e_url:
        print(f"URL Error downloading thumbnail: {e_url}")
    except Exception as e_thumb:
        print(f"General error generating/downloading thumbnail: {e_thumb}")

    return {
        "image": image_base64,
        "statistics": all_stats,
        "roi_bounds": roi_bounds,
        "gee_image_details": (
            f"Median composite from COPERNICUS/S2_SR_HARMONIZED "
            f"({gee_data.get('start_date')} to {gee_data.get('end_date')}), "
            f"{gee_data.get('count')} images processed. "
            f"Stats scale: {scale}m. Thumb: {thumb_dimensions}."
        )
    }

# ===============================================================================
# Sentinel-2 FEATURE EXTRACTION: AWS (Legacy, for reference)
# ===============================================================================
def sentinel2_stats(band_files: dict) -> dict:
    """
    Compute mean NDVI or mean RGB band stats from Sentinel-2 band files.
    Expects: dict {band: filepath}
    Returns: dict of mean values (and NDVI if possible).
    """
    bands = {}
    for band, path in band_files.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            bands[band] = arr

    stats = {}
    # Mean of each band
    for band, arr in bands.items():
        stats[f"mean_B0{band}"] = float(np.nanmean(arr))
    # NDVI if red (B04) and NIR (B08) are present
    if "4" in bands and "8" in bands:
        red = bands["4"]
        nir = bands["8"]
        ndvi = (nir - red) / (nir + red + 1e-6)
        stats["mean_NDVI"] = float(np.nanmean(ndvi))
    return stats

# Example usage (do not run if just producing code):
# lidar_file = fetch_dataset("lidar")
# lidar_stats = lidar_ot_extract_features(lidar_file)
# s2_files = fetch_dataset("sentinel2")
# s2_gee_stats = sentinel2_gee_extract_features(s2_files)