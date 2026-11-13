#!/usr/bin/env python3
"""
build_processed_cube.py
=======================
Harmonizes and rasterizes multi-sensor remote sensing observations into a unified
spatial Data Cube for Martian ground-ice inversion.

Ingests:
1. MGS MOLA MEGDR Topography (64 pix/deg or 16 pix/deg)
2. 2001 Mars Odyssey GRS Averaged Neutron Data (AND) Epithermal Count Rates (CEPI)

Computes:
1. Surface elevation (meters relative to areoid datum)
2. Spherical metric terrain slope (degrees) via latitudinally corrected gradients
3. Continuous epithermal neutron flux (counts per second) via bicubic spline interpolation

Outputs:
data/processed/cubes/arcadia_planitia_multimodal_cube.npz

Usage:
    python3 scripts/02_preprocessing/build_processed_cube.py
"""

import argparse
import os
import sys
import numpy as np

# Physical constants for Mars
MARS_RADIUS_M = 3396190.0

REGIONS = {
    "arcadia": {
        "name": "Arcadia Planitia",
        "lat_min": 37.0,
        "lat_max": 50.0,
        "lon_min": 180.0,
        "lon_max": 205.0,
        "out_file": "data/processed/cubes/arcadia_planitia_multimodal_cube.npz"
    },
    "utopia": {
        "name": "Utopia Planitia",
        "lat_min": 40.0,
        "lat_max": 55.0,
        "lon_min": 80.0,
        "lon_max": 115.0,
        "out_file": "data/processed/cubes/utopia_planitia_multimodal_cube.npz"
    },
    "acidalia": {
        "name": "Acidalia Planitia",
        "lat_min": 35.0,
        "lat_max": 52.0,
        "lon_min": 310.0,
        "lon_max": 350.0,
        "out_file": "data/processed/cubes/acidalia_planitia_multimodal_cube.npz"
    }
}

def process_cube(region_key="arcadia"):
    cfg = REGIONS[region_key]
    print(f"\nProcessing Multi-Modal Data Cube for: {cfg['name']}")
    print(f"  Lat bounds: [{cfg['lat_min']}°N, {cfg['lat_max']}°N]")
    print(f"  Lon bounds: [{cfg['lon_min']}°E, {cfg['lon_max']}°E]")

    os.makedirs(os.path.dirname(cfg["out_file"]), exist_ok=True)

    # 1. Load MOLA Topography (use global grid to cover all ground truth sites)
    # Ingest the 64 pix/deg grid (megt90n180gb.img) for high-resolution regional cubes.
    # The 16 pix/deg global grid serves as fallback for regions outside 64 pix/deg coverage.
    mola_64_path = "data/raw/mola/megt90n180gb.img"
    mola_16_path = "data/raw/mola/megt90n000eb.img"

    use_64 = (
        os.path.exists(mola_64_path)
        and os.path.getsize(mola_64_path) == 5760 * 11520 * 2
        and 0.0 <= cfg["lat_min"] and cfg["lat_max"] <= 90.0
        and cfg["lon_min"] >= 180.0 and cfg["lon_max"] <= 360.0
    )

    if use_64:
        print(f"  Ingesting high-resolution MOLA 64 pix/deg grid: {mola_64_path}")
        # Grid: 90N to 0N (5760 rows), 180E to 360E (11520 cols)
        mola_full = np.fromfile(mola_64_path, dtype=">i2").reshape((5760, 11520))
        scale = 64.0
        row_start = int((90.0 - cfg["lat_max"]) * scale)
        row_end = int((90.0 - cfg["lat_min"]) * scale)
        col_start = int((cfg["lon_min"] - 180.0) * scale)
        col_end = int((cfg["lon_max"] - 180.0) * scale)
        mola_crop = mola_full[row_start:row_end, col_start:col_end].astype(np.float32)
        pix_res_m = MARS_RADIUS_M * (1.0 / scale) * (np.pi / 180.0)  # ~925 m/pix
    elif os.path.exists(mola_16_path) and os.path.getsize(mola_16_path) == 2880 * 5760 * 2:
        print(f"  Ingesting global MOLA 16 pix/deg grid: {mola_16_path}")
        # Grid: 90N to -90S (2880 rows), 0E to 360E (5760 cols)
        mola_full = np.fromfile(mola_16_path, dtype=">i2").reshape((2880, 5760))
        scale = 16.0
        row_start = int((90.0 - cfg["lat_max"]) * scale)
        row_end = int((90.0 - cfg["lat_min"]) * scale)
        col_start = int(cfg["lon_min"] * scale)
        col_end = int(cfg["lon_max"] * scale)
        mola_crop = mola_full[row_start:row_end, col_start:col_end].astype(np.float32)
        pix_res_m = MARS_RADIUS_M * (1.0 / scale) * (np.pi / 180.0)  # ~3700 m/pix
    else:
        sys.exit(f"ERROR: No usable MOLA MEGDR grid found for region '{region_key}'.\n"
                 f"  Checked: {mola_64_path} (64 pix/deg, 90N-0N / 180-360E)\n"
                 f"           {mola_16_path} (16 pix/deg, global)")

    print(f"  Cropped elevation grid: {mola_crop.shape} | Min: {mola_crop.min():.1f}m | Max: {mola_crop.max():.1f}m")

    # 2. Compute Terrain Slope with Spherical Metric Correction
    # In cylindrical planetocentric coordinates, longitude spacing contracts by cos(latitude).
    n_rows, n_cols = mola_crop.shape
    crop_lats = np.linspace(cfg["lat_max"], cfg["lat_min"], n_rows)
    crop_lons = np.linspace(cfg["lon_min"], cfg["lon_max"], n_cols)

    dy_m = pix_res_m
    dx_m_per_row = pix_res_m * np.cos(np.radians(crop_lats))  # Latitude convergence correction

    dy = np.gradient(mola_crop, dy_m, axis=0)
    dx = np.gradient(mola_crop, axis=1) / dx_m_per_row[:, None]
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    print(f"  Computed slope grid (spherical metric corrected): {slope_deg.shape} | Mean: {slope_deg.mean():.2f}° | Max: {slope_deg.max():.2f}°")

    # 3. Ingest and Interpolate Odyssey GRS Epithermal Neutrons (Smooth Bicubic Interpolation)
    grs_path = "data/raw/odyssey_grs/and_04_000_015.dat"
    if not os.path.exists(grs_path):
        sys.exit(f"ERROR: Missing Odyssey GRS dataset at {grs_path}.\n"
                 "Run scripts/01_data_acquisition/download_planetary_data.sh to download real PDS data.")

    from scipy.interpolate import RegularGridInterpolator
    grs_raw = np.fromfile(grs_path, dtype=">f4").reshape((2592, 11))
    lats_unique = np.unique(grs_raw[:, 0])[::-1]  # 87.5 down to -87.5 (36 bands)
    lons_unique = np.unique(grs_raw[:, 1])        # 2.5 to 357.5 (72 bands)
    cepi_grid = grs_raw[:, 5].reshape((36, 72))

    # Construct bicubic regular grid interpolator across Mars sphere
    rgi_cubic = RegularGridInterpolator(
        (lats_unique[::-1], lons_unique),
        cepi_grid[::-1, :],
        method='cubic',
        bounds_error=False,
        fill_value=None
    )

    lon_grid, lat_grid = np.meshgrid(crop_lons, crop_lats)
    query_pts = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
    grs_interpolated = rgi_cubic(query_pts).reshape(n_rows, n_cols)
    print(f"  Interpolated GRS Epithermal flux (smooth bicubic): {grs_interpolated.shape} | Mean: {grs_interpolated.mean():.2f} cps | Min: {grs_interpolated.min():.2f} cps | Max: {grs_interpolated.max():.2f} cps")

    # 4. Save Multi-Modal Data Cube (.npz)
    np.savez_compressed(
        cfg["out_file"],
        elevation=mola_crop,
        slope=slope_deg,
        epithermal_neutrons=grs_interpolated,
        lat_bounds=[cfg["lat_min"], cfg["lat_max"]],
        lon_bounds=[cfg["lon_min"], cfg["lon_max"]],
        pixel_scale_deg=1.0 / scale,
        resolution_m=pix_res_m
    )
    print(f"  Wrote: {cfg['out_file']} ({os.path.getsize(cfg['out_file']) / 1024:.1f} KB)")
    print(f"Data cube generation for {cfg['name']} complete!\n")

def main():
    parser = argparse.ArgumentParser(description="Build harmonized planetary multi-modal data cube.")
    parser.add_argument("--region", choices=["arcadia", "utopia", "acidalia", "all"], default="arcadia", help="Target study sector")
    args = parser.parse_args()

    if args.region == "all":
        for r in ["arcadia", "utopia", "acidalia"]:
            process_cube(r)
    else:
        process_cube(args.region)

if __name__ == "__main__":
    main()
