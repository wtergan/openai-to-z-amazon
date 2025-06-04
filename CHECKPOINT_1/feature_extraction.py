import numpy as np
import laspy
import rasterio
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import LightSource

# ===============================================================================
# LiDAR FEATURE EXTRACTION: OpenTopography API
# ===============================================================================
def ot_lidar_plot(lidar_path: str, show_plot: bool = True) -> dict:
    """
    Generate and display a plot of LiDAR data along with some stats.
    Returns a dict containing the plot as a BytesIO obj, and the stats.
    """
    with rasterio.open(lidar_path) as src:
        lidar_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            lidar_arr = np.where(lidar_arr == src.nodata, np.nan, lidar_arr)
        
        # Usage of 2-98 percentiles for avoiding outliers that may affect color mapping
        vmin, vmax = np.nanpercentile(lidar_arr, [2, 98])

        # Figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Main elevation plot
        im1 = ax1.imshow(lidar_arr, cmap='terrain', vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=ax1, label='Elevation (m)')
        ax1.set_title("LiDAR Elevation Data (2-98% range)")

        # Hillshade for better terrain visualization, using LightSource for azimuth and altitude
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(lidar_arr, vert_exag=1, dx=1, dy=1, fraction=1.0)
        ax2.imshow(hillshade, cmap='gray', alpha=0.8)
        ax2.set_title("Hillshade Visualization")
        
        # Save plot to BytesIO buffer as JPEG, tight bbox, 150 DPI
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="JPEG", bbox_inches="tight", dpi=150)

        if show_plot:
            plt.show()
        plt.close()
        buf.seek(0)
        
        # Binary to base64 conversion
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        # Calculate basic stats
        plot_stats = {
            "plot": plot_base64,
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
            # Some spatial metadata: coordinate reference system, and bbox info.
            "crs": str(src.crs) if src.crs else None,
            "bounds": list(src.bounds)
        }
        buf.close()
        return plot_stats    

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
# lidar_plot = ot_lidar_plot(lidar_file)
# s2_files = fetch_dataset("sentinel2")
# s2_stats = sentinel2_stats(s2_files)