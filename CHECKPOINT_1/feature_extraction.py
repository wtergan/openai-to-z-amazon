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
from typing import Optional, Dict, Any

# LiDAR FEATURE EXTRACTION: OpenTopography API
def lidar_ot_extract_features(lidar_path: str, show_image: bool = True) -> Optional[Dict[str, Any]]:
    """
    Generate and display a plots of LiDAR GeoTIFF data along with some stats.
    Returns a dict containing the plot as a BytesIO obj, and the stats.
    Cleans up the temporary lidar_path file upon completion or error.
    """
    try:
        with rasterio.open(lidar_path) as src:
            print("LiDAR data read.")
            lidar_arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                lidar_arr = np.where(lidar_arr == src.nodata, np.nan, lidar_arr)
            if np.all(np.isnan(lidar_arr)):
                print("Error: LiDAR data is empty or all NoData values.")
                return None

            # Usage of 2-98 percentiles for avoiding outliers that may affect color mapping:
            vmin, vmax = np.nanpercentile(lidar_arr, [2, 98])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle("LiDAR Data Analysis", fontsize=16)

            print("Generating LiDAR elevation plot...")
            im1 = ax1.imshow(lidar_arr, cmap='terrain', vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=ax1, label='Elevation (m)')
            ax1.set_title("LiDAR Elevation Data (2-98% range)")

            # Hillshade for better terrain visualization, using LightSource for azimuth and altitude:
            print("Generating LiDAR hillshade...")
            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(lidar_arr, vert_exag=1, dx=src.res[0], dy=src.res[1], fraction=1.0)
            ax2.imshow(hillshade, cmap='gray', alpha=0.8)
            ax2.set_title("Hillshade Visualization")

            print("Displaying the plot(s)...")
            if show_image:
                plt.show()

            print("Saving plot to BytesIO buffer...")
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="JPEG", bbox_inches="tight", dpi=150)
            plt.close()
            buf.seek(0)

            print("Converting plot to base64 and computation of stats...")
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

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
                "crs": str(src.crs) if src.crs else "N/A",
                "bounds": list(src.bounds)
            }
            buf.close()
            return ot_stats
    except Exception as e:
        print(f"Error processing LiDAR data: {e}")
        return None
    finally:
        if os.path.exists(lidar_path):
            os.unlink(lidar_path)
            print(f"Temporary file {lidar_path} deleted.")

# Sentinel-2 FEATURE EXTRACTION: Google Earth Engine
def sentinel2_gee_extract_features(
    gee_data: dict,
    scale: int = 20, # Compromise resolution for comprehensive band statistics.
    thumb_dimensions: str = '768x768', # Thumbnail dimensions.
    show_image: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Computing stats and generating multiple visualizations for a Sentinel-2 GEE Image.
    Returns a dict containing RGB composite, NDVI heatmap, false-color composite as base64,
    along with comprehensive statistics and metadata for AI analysis.
    """
    if not gee_data or not gee_data.get("image"):
        error_msg = gee_data.get("error", "GEE image not provided or invalid.")
        print(f"Error: {error_msg}")
        return {
            "rgb_image": None,
            "statistics": {"error": error_msg},
            "roi_bounds": gee_data.get("roi_bounds"),
            "gee_image_details": "No image processed."
        }
    image = gee_data["image"]
    roi = gee_data["roi"]
    roi_bounds = gee_data["roi_bounds"]
    all_spectral_bands = gee_data["bands_included"]

    # Creating a composite reducer that computes multiple stats all in one operation:
    reducers = (
        ee.Reducer.mean().unweighted()
        .combine(ee.Reducer.minMax().unweighted(), sharedInputs=True)
        .combine(ee.Reducer.stdDev().unweighted(), sharedInputs=True)
        .combine(ee.Reducer.percentile([2, 98]).unweighted(), sharedInputs=True)
        .combine(ee.Reducer.count().unweighted(), sharedInputs=True)
    )

    try:
        # Computation of the stats for each selected band, given image, ROI, scale, and reducers:
        print(f"Calculating stats for bands: {all_spectral_bands} at {scale}m scale...")
        band_stats = image.select(all_spectral_bands).reduceRegion(
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
            "rgb_image": None,
            "statistics": {"error": f"GEE band stats error: {e}"},
            "roi_bounds": roi_bounds,
            "gee_image_details": "Failed during band statistics."
        }
    except Exception as e_gen:
        print(f"General Error calculating band stats: {e_gen}")
        return {
            "rgb_image": None,
            "statistics": {"error": f"General band stats error: {e_gen}"},
            "roi_bounds": roi_bounds,
            "gee_image_details": "Failed during band statistics."
        }

    # Computing the NDVI (Normalized Difference Vegetation Index) and its stats:
    # NDVI is computed as (NIR - Red) / (NIR + Red), where NIR=near-infrared, band 8, and Red is band 4.
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

    # Merging the stats from all bands and NDVI into a single dict for usage:
    all_stats = {}
    for band in all_spectral_bands:
        for stat_key, reducer_key_part in {
            "mean": "mean", "min": "min", "max": "max", "std": "stdDev",
            "p2": "p2", "p98": "p98", "count": "count" }.items():
            gee_key = f"{band}_{reducer_key_part}"
            all_stats[f"{band}_{stat_key}"] = band_stats.get(gee_key)
    all_stats.update(ndvi_stats)

    # Additional regional analysis stats for LLM consumption:
    regional_analysis = {}
    if all_stats:
        # Core NDVI regional statistics:
        print("Computing additional regional stats for analysis...")
        regional_analysis['region_ndvi_mean'] = all_stats.get('NDVI_mean', 0.0)
        regional_analysis['region_ndvi_std'] = all_stats.get('NDVI_stdDev', 0.0)
        regional_analysis['region_ndvi_range'] = (all_stats.get('NDVI_max', 0.0) -
                                                 all_stats.get('NDVI_min', 0.0))

        # Band means for vegetation health calculation:
        b4_mean = all_stats.get('B4_mean', 0.0)
        b5_mean = all_stats.get('B5_mean', 0.0)  # Red edge (if available)
        b8_mean = all_stats.get('B8_mean', 0.0)

        # Regional vegetation health composite (NDVI weighted 70% + red edge NDVI 30%):
        if (b8_mean + b5_mean) > 0:
            red_edge_ndvi = (b8_mean - b5_mean) / (b8_mean + b5_mean)
            regional_analysis['region_vegetation_health'] = (0.7 * regional_analysis['region_ndvi_mean'] +
                                                           0.3 * red_edge_ndvi)
        else:
            regional_analysis['region_vegetation_health'] = regional_analysis['region_ndvi_mean']

        # Regional spectral baseline (NIR/SWIR vs visible contrast):
        visible_bands = [all_stats.get('B2_mean', 0.0), all_stats.get('B3_mean', 0.0), b4_mean]
        nir_swir_bands = [b8_mean, all_stats.get('B11_mean', 0.0), all_stats.get('B12_mean', 0.0)]

        if any(b > 0 for b in visible_bands + nir_swir_bands):
            visible_avg = np.mean([b for b in visible_bands if b > 0])
            nir_swir_avg = np.mean([b for b in nir_swir_bands if b > 0])
            if visible_avg > 0:
                regional_analysis['region_spectral_baseline'] = nir_swir_avg / visible_avg
            else:
                regional_analysis['region_spectral_baseline'] = 0.0
        else:
            regional_analysis['region_spectral_baseline'] = 0.0

    # Adding the regional analysis to the statistics:
    all_stats.update(regional_analysis)

    rgb_image_base64 = None
    ndvi_image_base64 = None
    false_color_image_base64 = None
    rgb_pil_image = None
    ndvi_pil_image = None
    false_color_pil_image = None

    # Generating the RGB composite thumbnail for LLM processing:
    try:
        print("Generating RGB composite thumbnail...")
        # Vis params for 0-1 scaled reflectance. Common S2 vis is 0-0.3 range.
        rgb_vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3, 'gamma': 1.4}
        region_payload = roi.getInfo()['coordinates'] if hasattr(roi, 'getInfo') else roi
        rgb_thumbnail_url = image.visualize(**rgb_vis_params).getThumbURL({
            'region': region_payload,
            'dimensions': thumb_dimensions,
            'format': 'jpg'
        })

        print(f"RGB composite thumbnail URL (first 100 chars): {rgb_thumbnail_url[:100]}...")
        with urllib.request.urlopen(rgb_thumbnail_url, timeout=60) as response:
            img_data = response.read()
        rgb_pil_image = Image.open(io.BytesIO(img_data))
        buf = io.BytesIO()
        rgb_pil_image.save(buf, format="JPEG")
        buf.seek(0)
        rgb_image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        print("RGB composite thumbnail generated and encoded.")

    except ee.EEException as e:
        print(f"GEE Error generating RGB composite thumbnail: {e}")
    except urllib.error.URLError as e_url:
        print(f"URL Error downloading RGB composite thumbnail: {e_url}")
    except Exception as e_thumb:
        print(f"General error generating RGB composite thumbnail: {e_thumb}")

    # Generating NDVI heatmap visualization:
    try:
        print("Generating NDVI heatmap visualization...")
        # NDVI visualization parameters: red-yellow-green scale for vegetation health
        # Green/yellow for healthy vegetation (0.3-1.0), red/brown for bare soil/stress (-1.0-0.3)
        ndvi_vis_params = {
            'bands': ['NDVI'],
            'min': -0.2,
            'max': 0.8,
            'palette': ['8B4513', 'CD853F', 'DEB887', 'F0E68C', 'ADFF2F', '32CD32', '228B22']  # Brown to green
        }

        ndvi_thumbnail_url = ndvi_image.visualize(**ndvi_vis_params).getThumbURL({
            'region': region_payload,
            'dimensions': thumb_dimensions,
            'format': 'jpg'
        })

        print(f"NDVI thumbnail URL (first 100 chars): {ndvi_thumbnail_url[:100]}...")
        with urllib.request.urlopen(ndvi_thumbnail_url, timeout=60) as response:
            ndvi_img_data = response.read()
        ndvi_pil_image = Image.open(io.BytesIO(ndvi_img_data))
        ndvi_buf = io.BytesIO()
        ndvi_pil_image.save(ndvi_buf, format="JPEG")
        ndvi_buf.seek(0)
        ndvi_image_base64 = base64.b64encode(ndvi_buf.getvalue()).decode('utf-8')
        ndvi_buf.close()
        print("NDVI heatmap generated and encoded.")

    except ee.EEException as e:
        print(f"GEE Error generating NDVI heatmap: {e}")
        ndvi_image_base64 = None
    except urllib.error.URLError as e_url:
        print(f"URL Error downloading NDVI heatmap: {e_url}")
        ndvi_image_base64 = None
    except Exception as e_ndvi:
        print(f"General error generating NDVI heatmap: {e_ndvi}")
        ndvi_image_base64 = None

    # Generating false-color composite (NIR-Red-Green):
    try:
        print("Generating false-color composite (NIR-Red-Green)...")
        # False-color visualization: NIR->Red, Red->Green, Green->Blue
        # Vegetation appears red/pink, water appears blue/black, urban appears cyan/blue
        false_color_vis_params = {
            'bands': ['B8', 'B4', 'B3'],  # NIR, Red, Green mapped to RGB
            'min': 0.0,
            'max': 0.3,
            'gamma': 1.2
        }

        false_color_thumbnail_url = image.visualize(**false_color_vis_params).getThumbURL({
            'region': region_payload,
            'dimensions': thumb_dimensions,
            'format': 'jpg'
        })

        print(f"False-color thumbnail URL (first 100 chars): {false_color_thumbnail_url[:100]}...")
        with urllib.request.urlopen(false_color_thumbnail_url, timeout=60) as response:
            false_color_img_data = response.read()
        false_color_pil_image = Image.open(io.BytesIO(false_color_img_data))
        false_color_buf = io.BytesIO()
        false_color_pil_image.save(false_color_buf, format="JPEG")
        false_color_buf.seek(0)
        false_color_image_base64 = base64.b64encode(false_color_buf.getvalue()).decode('utf-8')
        false_color_buf.close()
        print("False-color composite generated and encoded.")

    except ee.EEException as e:
        print(f"GEE Error generating false-color composite: {e}")
        false_color_image_base64 = None
    except urllib.error.URLError as e_url:
        print(f"URL Error downloading false-color composite: {e_url}")
        false_color_image_base64 = None
    except Exception as e_fc:
        print(f"General error generating false-color composite: {e_fc}")
        false_color_image_base64 = None

    if show_image:
        if rgb_pil_image:
            print("Displaying RGB thumbnail...")
            try:
                rgb_pil_image.show()
            except Exception as e_show:
                print(f"Could not display RGB thumbnail: {e_show}")

        if ndvi_pil_image:
            print("Displaying NDVI heatmap...")
            try:
                ndvi_pil_image.show()
            except Exception as e_show:
                print(f"Could not display NDVI heatmap: {e_show}")

        if false_color_pil_image:
            print("Displaying false-color composite...")
            try:
                false_color_pil_image.show()
            except Exception as e_show:
                print(f"Could not display false-color composite: {e_show}")

    print("Compiling results with an enhanced data structure for AI analysis and further usage...")
    return {
        "image": rgb_image_base64,
        "ndvi_image": ndvi_image_base64,
        "false_color_image": false_color_image_base64, # For backward compatibility.

        # Enhanced metadata for AI processing:
        "image_metadata": {
            "rgb_composite": {
                "description": "Natural color composite using Red, Green, Blue bands",
                "bands": "B4 (Red), B3 (Green), B2 (Blue)",
                "purpose": "General landscape features and natural color interpretation"
            },
            "ndvi_heatmap": {
                "description": "NDVI vegetation health heatmap with color mapping",
                "color_mapping": "Brown/Red = bare soil/stressed vegetation (-0.2 to 0.3), Yellow/Green = healthy vegetation (0.3 to 0.8)",
                "purpose": "Vegetation health assessment and stress identification",
                "available": ndvi_image_base64 is not None
            },
            "false_color_composite": {
                "description": "False-color composite highlighting vegetation patterns",
                "bands": "B8 (NIR->Red), B4 (Red->Green), B3 (Green->Blue)",
                "interpretation": "Red/Pink = vegetation, Blue/Black = water, Cyan/Blue = urban/bare soil",
                "purpose": "Vegetation boundary detection and pattern analysis",
                "available": false_color_image_base64 is not None
            }
        },

        "statistics": all_stats,
        "roi_bounds": roi_bounds,
        "gee_image_details": (
            f"Median composite from COPERNICUS/S2_SR_HARMONIZED "
            f"({gee_data.get('start_date')} to {gee_data.get('end_date')}), "
            f"{gee_data.get('count')} images processed. "
            f"Stats scale: {scale}m. Thumb: {thumb_dimensions}. "
            f"Visualizations: RGB={'Present' if rgb_image_base64 else 'Absent'}, "
            f"NDVI={'Present' if ndvi_image_base64 else 'Absent'}, "
            f"False-color={'Present' if false_color_image_base64 else 'Absent'}."
        )
    }

