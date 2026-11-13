#!/usr/bin/env python3
"""
fig1_study_area.py
==================
Generates Figure 1: Physiographic setting of the Arcadia Planitia study area.

Renders MGS MOLA MEGDR elevation (64 pix/deg) referenced to the IAU Mars 2000 datum
with hillshaded shaded-relief terrain, the official IAU Arcadia Planitia boundary
(USGS Feature #348), catalogued benchmark impact craters, a zoomed planetary context inset,
a cartographic scale bar above the lower-right legend, a true North arrow at the top right,
and an updated publication-ready legend.

Usage:
    python3 scripts/04_visualization/fig1_study_area.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon as MplPolygon
import shapefile
from scipy.interpolate import splprep, splev

IAU_BOUNDARY_SHP = "data/shapefiles/iau_arcadia_planitia_boundary.shp"
GLOBAL_MOLA_16 = "data/raw/mola/megt90n000eb.img"
CUBE_NPZ = "data/processed/cubes/arcadia_planitia_multimodal_cube.npz"
OUT_FIG = "figures/fig1_study_area_and_shapefile_boundary.png"


def load_iau_boundary():
    """Return points for Part 0 and Part 1 of the official IAU Arcadia Planitia boundary."""
    reader = shapefile.Reader(IAU_BOUNDARY_SHP)
    shape = reader.shape(0)
    parts = list(shape.parts) + [len(shape.points)]
    pts_p0 = shape.points[parts[0]:parts[1]]
    pts_p1 = shape.points[parts[1]:parts[2]]
    return pts_p0, pts_p1


def generate_figure():
    print("Loading data for Figure 1...")
    cube = np.load(CUBE_NPZ)
    elevation = cube["elevation"]
    lat_min, lat_max = 37.0, 50.0
    lon_min, lon_max = 180.0, 205.0

    pts_p0, pts_p1 = load_iau_boundary()

    os.makedirs("figures", exist_ok=True)

    # 1. Main map shaded relief using planetary MOLA elevation palette
    ls = LightSource(azdeg=315, altdeg=45)
    elev_cmap = LinearSegmentedColormap.from_list(
        'mola_palette',
        ['#000077', '#002699', '#0055d4', '#0099e6', '#00bfa5', '#66bb6a',
         '#d4e157', '#ffee58', '#ffa726', '#e65100', '#8d6e63', '#4e342e']
    )
    shaded = ls.shade(
        elevation, cmap=elev_cmap, blend_mode='soft',
        vmin=-5000, vmax=-2800, dx=927, dy=927, vert_exag=14
    )

    print("Rendering Figure 1...")
    fig, ax = plt.subplots(figsize=(12, 6.4))
    im = ax.imshow(shaded, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max])

    # 2. Authentic IAU Arcadia Planitia boundary clipped dynamically to corridor margins
    # Intersects corridor at southern latitude 37.0°N and eastern longitude 205.0°E
    reader = shapefile.Reader(IAU_BOUNDARY_SHP)
    shp_pts = reader.shape(0).points
    corridor_pts_full = [pt for pt in shp_pts if (184.0 <= pt[0] <= 208.0 and 36.0 <= pt[1] <= 50.0)]
    corridor_pts_full.sort(key=lambda p: p[1])

    p_below, p_above_bot = corridor_pts_full[0], corridor_pts_full[1]
    t_bot = (lat_min - p_below[1]) / (p_above_bot[1] - p_below[1])
    lon_bot = p_below[0] + t_bot * (p_above_bot[0] - p_below[0])

    p_below_top, p_above_top = corridor_pts_full[-2], corridor_pts_full[-1]
    t_right = (lon_max - p_below_top[0]) / (p_above_top[0] - p_below_top[0])
    lat_right = p_below_top[1] + t_right * (p_above_top[1] - p_below_top[1])

    boundary_pts = [(lon_bot, lat_min)] + corridor_pts_full[1:-1] + [(lon_max, lat_right)]
    b_lons = [p[0] for p in boundary_pts]
    b_lats = [p[1] for p in boundary_pts]

    tck, u = splprep([b_lons, b_lats], s=0, k=2)
    u_new = np.linspace(0, 1, 100)
    smooth_lons, smooth_lats = splev(u_new, tck)

    ax.plot(smooth_lons, smooth_lats, color='#ffd700', lw=2.4, ls='--',
            label='Official IAU Arcadia Planitia boundary (USGS Feature #348)',
            path_effects=[pe.withStroke(linewidth=4.0, foreground='#111111')], zorder=4)

    # 3. Ground truth benchmark sites with collision-free label positioning
    corridor_ice = [
        ("Dundas_05", 190.25, 39.11, (9, -9), "top", "left"),
        ("HiRISE_Ice_02", 188.51, 46.14, (9, -7), "top", "left"),
        ("HiRISE_Ice_07", 194.85, 46.57, (9, 5), "bottom", "left"),
    ]
    corridor_dry = [
        ("Dry_03", 185.50, 40.33, (9, 3), "bottom", "left"),
        ("Dry_04", 203.32, 43.75, (-9, 0), "center", "right"),
        ("Dry_05", 190.05, 39.55, (-9, 6), "bottom", "right"),
        ("Dry_10", 182.20, 37.32, (9, 5), "bottom", "left"),
    ]

    for name, lon, lat, xytext, va, ha in corridor_ice:
        ax.plot(lon, lat, 'o', color='#0066cc', markeredgecolor='white',
                markeredgewidth=1.5, markersize=8.5, zorder=5)
        ax.annotate(name, (lon, lat), fontsize=8.5, fontweight='bold', color='#003399',
                    xytext=xytext, textcoords='offset points', va=va, ha=ha,
                    path_effects=[pe.withStroke(linewidth=2.8, foreground='white')])

    for name, lon, lat, xytext, va, ha in corridor_dry:
        ax.plot(lon, lat, 's', color='#d32f2f', markeredgecolor='white',
                markeredgewidth=1.5, markersize=7.8, zorder=5)
        ax.annotate(name, (lon, lat), fontsize=8.5, fontweight='bold', color='#990000',
                    xytext=xytext, textcoords='offset points', va=va, ha=ha,
                    path_effects=[pe.withStroke(linewidth=2.8, foreground='white')])

    # HiRISE_Ice_01 with callout arrow pointing to the marker, right-aligned to stay safely inside 205°E
    ax.plot(204.35, 43.88, 'o', color='#0066cc', markeredgecolor='white',
            markeredgewidth=1.5, markersize=8.5, zorder=5)
    ax.annotate('HiRISE_Ice_01', (204.35, 43.88), fontsize=8.5, fontweight='bold', color='#003399',
                xytext=(6, 14), textcoords='offset points', va='bottom', ha='right',
                arrowprops=dict(arrowstyle='->', color='#003399', lw=1.2, shrinkB=4),
                path_effects=[pe.withStroke(linewidth=2.8, foreground='white')], zorder=6)

    # Comprehensive legend entries
    ax.plot([], [], '-', color='#d32f2f', lw=2.2, label='Study area corridor (37°–50°N, 180°–205°E)')
    ax.plot([], [], 'o', color='#0066cc', markeredgecolor='white', markeredgewidth=1.2,
            markersize=8, linestyle='none', label='Ice-exposing impact crater (HiRISE)')
    ax.plot([], [], 's', color='#d32f2f', markeredgecolor='white', markeredgewidth=1.2,
            markersize=7.5, linestyle='none', label='Non-icy impact crater / Dry control (HiRISE)')

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title('MOLA Topography — Arcadia Planitia Study Corridor (37°–50°N, 180°–205°E)\nNASA Mars Global Surveyor MEGDR (64 pix/deg)',
                 fontsize=12.5, fontweight='bold', pad=12)
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)

    leg = ax.legend(loc='lower right', fontsize=8.6, framealpha=0.92, facecolor='white', edgecolor='#aaaaaa',
                    labelspacing=0.45, borderpad=0.55, handletextpad=0.6)
    leg.set_zorder(8)

    plt.tight_layout()
    fig.canvas.draw()
    p = ax.get_position()

    # 4. Zoomed Planetary Context Inset Map (positioned comfortably at p.x0 + 0.028)
    if os.path.exists(GLOBAL_MOLA_16):
        w_inset = 0.215
        h_inset = 0.255
        # Moved slightly left from 0.042 to 0.028 to maintain balanced margins
        ax_inset = fig.add_axes([p.x0 + 0.028, p.y1 - h_inset - 0.040, w_inset, h_inset], zorder=8)

        # Read zoomed regional MOLA (30°N to 68°N, 160°E to 215°E)
        mola_16 = np.fromfile(GLOBAL_MOLA_16, dtype=">i2").reshape((2880, 5760))
        r_start, r_end = int((90 - 68) * 16), int((90 - 30) * 16)
        c_start, c_end = int(160 * 16), int(215 * 16)
        sub_mola = mola_16[r_start:r_end, c_start:c_end]

        ls_inset = LightSource(azdeg=315, altdeg=45)
        shaded_inset = ls_inset.shade(
            sub_mola, cmap=elev_cmap, blend_mode='soft', vmin=-5500, vmax=0, vert_exag=4
        )
        ax_inset.imshow(shaded_inset, origin='upper', extent=[160, 215, 30, 68])

        # Seamless outer IAU Arcadia Planitia boundary
        poly_pts = pts_p0[:18] + pts_p1[2:] + [pts_p1[1], pts_p0[18], pts_p0[0]]
        poly_lons = [pt[0] for pt in poly_pts]
        poly_lats = [pt[1] for pt in poly_pts]

        iau_patch = MplPolygon(poly_pts, closed=True, facecolor='#ffd700', alpha=0.18, edgecolor='none')
        ax_inset.add_patch(iau_patch)
        ax_inset.plot(poly_lons, poly_lats, color='#ffd700', lw=1.8,
                      path_effects=[pe.withStroke(linewidth=2.8, foreground='#111111')])

        # Study corridor bounding box in vibrant red
        corridor_box = [180, 205, 205, 180, 180]
        corridor_lats = [37, 37, 50, 50, 37]
        ax_inset.plot(corridor_box, corridor_lats, color='#d32f2f', lw=2.0, ls='-',
                      path_effects=[pe.withStroke(linewidth=3.0, foreground='white')])

        ax_inset.text(187, 55, 'Arcadia\nPlanitia', fontsize=6.6, color='#ffffcc', fontweight='bold', ha='center',
                      path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])
        ax_inset.text(187, 43.5, 'Study Area', fontsize=6.0, color='#ffffff', fontweight='bold', ha='center',
                      path_effects=[pe.withStroke(linewidth=2.0, foreground='#990000')])

        # Title positioned clearly ABOVE the inset box with stroke for visibility
        title_txt = ax_inset.set_title('Planetary Context', fontsize=8.5, fontweight='bold', pad=5, color='#111111')
        title_txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground='white')])

        # Precise latitude and longitude coordinate grid
        ax_inset.set_xlim(160, 215)
        ax_inset.set_ylim(30, 68)
        ax_inset.set_xticks([170, 190, 210])
        ax_inset.set_xticklabels(['170°E', '190°E', '210°E'], fontsize=6.8, fontweight='bold')
        ax_inset.set_yticks([35, 50, 65])
        ax_inset.set_yticklabels(['35°N', '50°N', '65°N'], fontsize=6.8, fontweight='bold')
        ax_inset.tick_params(direction='out', length=2.5, width=0.8, colors='#111111', pad=2)

        for lbl in ax_inset.get_xticklabels() + ax_inset.get_yticklabels():
            lbl.set_path_effects([pe.withStroke(linewidth=2.4, foreground='white')])

        for spine in ax_inset.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

    # 5. North Arrow at Right Top Side
    na_x = 203.6
    na_y_bot = 48.65
    na_y_top = 49.30
    na_w = 0.28

    left_wing = MplPolygon([[na_x, na_y_top], [na_x - na_w, na_y_bot], [na_x, na_y_bot + 0.14]],
                           closed=True, facecolor='black', edgecolor='black', lw=0.8, zorder=7)
    right_wing = MplPolygon([[na_x, na_y_top], [na_x, na_y_bot + 0.14], [na_x + na_w, na_y_bot]],
                            closed=True, facecolor='white', edgecolor='black', lw=0.8, zorder=7)
    ax.add_patch(left_wing)
    ax.add_patch(right_wing)
    left_wing.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])
    right_wing.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])

    n_txt = ax.text(na_x, na_y_top + 0.08, 'N', fontsize=9.5, fontweight='bold', ha='center', va='bottom', color='black', zorder=7)
    n_txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground='white')])

    # 6. Cartographic Measurement Scale Bar (Top of legend, right side)
    # Calibrated to IAU 2000 Mars sphere (R = 3396.19 km) at latitude 43.5°N
    # 1° lon = 42.996 km -> 100 km = 2.3258° lon, 200 km = 4.6516° lon
    sb_x0 = 197.50
    sb_y0 = 39.75
    sb_len_100 = 2.3258
    sb_h = 0.14

    rect1 = Rectangle((sb_x0, sb_y0), sb_len_100, sb_h, facecolor='#222222', edgecolor='white', linewidth=1.0, zorder=7)
    rect2 = Rectangle((sb_x0 + sb_len_100, sb_y0), sb_len_100, sb_h, facecolor='#ffffff', edgecolor='black', linewidth=1.0, zorder=7)
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    t0 = ax.text(sb_x0, sb_y0 + sb_h + 0.06, '0', fontsize=8.0, fontweight='bold', ha='center', va='bottom', color='#111111', zorder=8)
    t1 = ax.text(sb_x0 + sb_len_100, sb_y0 + sb_h + 0.06, '100', fontsize=8.0, fontweight='bold', ha='center', va='bottom', color='#111111', zorder=8)
    t2 = ax.text(sb_x0 + 2 * sb_len_100, sb_y0 + sb_h + 0.06, '200 km', fontsize=8.0, fontweight='bold', ha='center', va='bottom', color='#111111', zorder=8)
    ts = ax.text(sb_x0 + sb_len_100, sb_y0 - 0.08, 'Scale (at 43.5°N datum)', fontsize=7.2, fontstyle='italic', fontweight='bold', color='#222222', ha='center', va='top', zorder=8)

    for t in [t0, t1, t2, ts]:
        t.set_path_effects([pe.withStroke(linewidth=2.8, foreground='white')])

    # 7. Colorbar matched to aspect-corrected map box
    aspect = (lat_max - lat_min) / (lon_max - lon_min)
    h_map = min(p.height, p.width * fig.get_figwidth() * aspect / fig.get_figheight())
    y_mid = p.y0 + p.height / 2.0
    cax = fig.add_axes([p.x1 + 0.015, y_mid - h_map / 2.0, 0.016, h_map])
    norm = matplotlib.colors.Normalize(vmin=-5000, vmax=-2800)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=elev_cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks(np.arange(-5000, -2800, 500))
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Elevation (m)', fontsize=11, labelpad=6)
    cbar.outline.set_linewidth(0.8)

    plt.savefig(OUT_FIG, dpi=600, bbox_inches='tight')
    out_pdf = OUT_FIG.replace('.png', '.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Figure 1 successfully generated: {OUT_FIG} and {out_pdf}")


if __name__ == "__main__":
    generate_figure()
