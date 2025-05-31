import numpy as np
import laspy
import rasterio

# ===============================================================================
# LiDAR FEATURE EXTRACTION
# ===============================================================================
def laspy_stats(laz_path: str) -> dict:
    """
    Compute basic stats from a LiDAR .laz file.
    Returns: dict with mean, min, max elevation and point count.
    """
    las = laspy.read(laz_path)
    z = las.z                           # Retrieves the z-coords as a np array.
    return {
        "mean_elev": float(np.mean(z)), 
        "min_elev": float(np.min(z)),
        "max_elev": float(np.max(z)),
        "pt_count": int(len(z))
    }

# ===============================================================================
# Sentinel-2 FEATURE EXTRACTION
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
# lidar_stats = laspy_stats(lidar_file)
# s2_files = fetch_dataset("sentinel2")
# s2_stats = sentinel2_stats(s2_files)