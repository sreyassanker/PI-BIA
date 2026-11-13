#!/usr/bin/env python3
"""
fig4_arcadia_inversion_composite.py
====================================
Generates Figure 4: Regional Bayesian inversion composite across Arcadia Planitia.

Two-tier architecture:
- Top Tier (Panels a–c): 2D Spatial Inversion Maps
    - Panel a: Physics-constrained ground ice probability P(ice) with stability contours
               and empirical ground-truth impact crater markers.
    - Panel b: Inverted shallow ice-table burial depth D(x) benchmarked against
               HiRISE and Dundas empirical excavation depths.
    - Panel c: Laplace epistemic uncertainty sigma_prob(x) quantifying predictive parameter variance.
- Bottom Tier (Panels d–f): 1D Quantitative Validation Transects & Distributions
    - Panel d: Latitudinal ice probability transition with +/- 1-sigma regional spread.
    - Panel e: Permafrost table shallowing transect from southern plains to high latitudes.
    - Panel f: Epistemic uncertainty probability density function (PDF).

Usage:
    python3 scripts/04_visualization/fig4_arcadia_inversion_composite.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from scipy.ndimage import gaussian_filter


def generate_figure():
    print("Loading data for Figure 4...")
    inv = np.load("data/processed/arcadia_inversion_results.npz", allow_pickle=True)
    grid_prob = inv["ice_probability"]
    grid_depth = inv["ice_table_depth_m"]
    grid_unc = inv["ice_probability_unc"]
    lats = inv["lats"]
    lons = inv["lons"]

    cube = np.load("data/processed/cubes/arcadia_planitia_multimodal_cube.npz")
    sub_elev = cube["elevation"][::4, ::4]

    os.makedirs("figures", exist_ok=True)

    # Shaded relief topography drape
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(sub_elev, vert_exag=6, dx=927 * 4, dy=927 * 4)

    def apply_hillshade(data, cmap, vmin, vmax, hs, blend=0.20):
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgb = cmap(norm(data))[:, :, :3]
        intensity = (1.0 - blend) + blend * hs[:, :, None]
        return np.clip(rgb * intensity, 0, 1)

    ice_cmap = LinearSegmentedColormap.from_list('ice', ['#f7fbff', '#9ecae1', '#2171b5', '#08306b'])
    depth_cmap = plt.cm.YlGnBu_r
    unc_cmap = plt.cm.inferno

    rgb_prob = apply_hillshade(grid_prob, ice_cmap, 0.0, 1.0, hs)
    rgb_depth = apply_hillshade(grid_depth, depth_cmap, 0.05, 0.75, hs)
    rgb_unc = apply_hillshade(grid_unc, unc_cmap, 0.03, 0.35, hs)

    extent = [180.0, 205.0, 37.0, 50.0]           # Arcadia Planitia, north-up
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])

    print("Rendering 6-panel composite layout...")
    fig = plt.figure(figsize=(18.5, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.72], hspace=0.28, wspace=0.30,
                          left=0.045, right=0.955, top=0.93, bottom=0.09)

    ax_map0 = fig.add_subplot(gs[0, 0])
    ax_map1 = fig.add_subplot(gs[0, 1])
    ax_map2 = fig.add_subplot(gs[0, 2])

    ax_prof0 = fig.add_subplot(gs[1, 0])
    ax_prof1 = fig.add_subplot(gs[1, 1])
    ax_prof2 = fig.add_subplot(gs[1, 2])

    # Verified ground-truth coordinates
    corridor_ice = [
        ('Dundas_05', 190.25, 39.11, 0.90),
        ('HiRISE_Ice_01', 204.35, 43.88, 0.65),
        ('HiRISE_Ice_02', 188.51, 46.14, 0.48),
        ('HiRISE_Ice_07', 194.85, 46.57, 0.55)
    ]
    corridor_dry = [
        ('Dry_03', 185.50, 40.33),
        ('Dry_04', 203.32, 43.75),
        ('Dry_05', 190.05, 39.55),
        ('Dry_10', 182.20, 37.32)
    ]

    # =========================================================================
    # Panel a: Ground Ice Occurrence Probability P(ice)
    # =========================================================================
    ax_map0.imshow(rgb_prob, origin='upper', extent=extent)
    smooth_prob = gaussian_filter(grid_prob, sigma=3.2)
    CS0 = ax_map0.contour(lons, lats, smooth_prob, levels=[0.50, 0.80],
                          colors=['#ffffff', '#fff59d'], linewidths=[2.0, 1.6], linestyles=['-', '--'])
    fmt_map0 = {0.50: 'P = 0.50', 0.80: 'P = 0.80'}
    ax_map0.clabel(CS0, inline=True, fmt=fmt_map0, fontsize=8.0, manual=[(202.0, 40.2), (195.0, 45.4)])

    for name, lon, lat, _ in corridor_ice:
        ax_map0.plot(lon, lat, 'o', color='#00e5ff', markeredgecolor='black', markersize=6.8, markeredgewidth=1.1, zorder=6)
    for name, lon, lat in corridor_dry:
        ax_map0.plot(lon, lat, 's', color='#ff3d00', markeredgecolor='black', markersize=6.0, markeredgewidth=1.1, zorder=6)

    ax_map0.set_title('a.  Physics-Constrained Ice Probability P(ice)', fontweight='bold', fontsize=11.0, pad=5)
    ax_map0.set_xlabel('Longitude (°E)', fontsize=10.0)
    ax_map0.set_ylabel('Latitude (°N)', fontsize=10.0)

    # =========================================================================
    # Panel b: Inverted Ice-Table Burial Depth D(x)
    # =========================================================================
    ax_map1.imshow(rgb_depth, origin='upper', extent=extent)
    smooth_depth = gaussian_filter(grid_depth, sigma=2.5)
    CS1 = ax_map1.contour(lons, lats, smooth_depth, levels=[0.40, 0.60],
                          colors=['#ffffff', '#ffff99'], linewidths=[1.6, 1.6])
    ax_map1.clabel(CS1, inline=True, fmt=lambda x: f'{x:.2f} m', fontsize=8.0, manual=[(184.5, 44.1)])

    depth_calls = [
        ('Dundas_05 (0.90 m)', 190.25, 39.11, (10, 0), 'center', 'left'),
        ('HiRISE_Ice_01 (0.65 m)', 204.35, 43.88, (-10, -10), 'top', 'right'),
        ('HiRISE_Ice_02 (0.48 m)', 188.51, 46.14, (-8, 6), 'bottom', 'right'),
        ('HiRISE_Ice_07 (0.55 m)', 194.85, 46.57, (8, 6), 'bottom', 'left')
    ]
    for lbl, lon, lat, xytext, va, ha in depth_calls:
        ax_map1.plot(lon, lat, 'o', color='#ffff00', markeredgecolor='black', markersize=6.8, markeredgewidth=1.1, zorder=6)
        # Clean casual normal black font without strokes
        ax_map1.annotate(lbl, (lon, lat), fontsize=8.0, fontweight='normal', color='black',
                         xytext=xytext, textcoords='offset points', va=va, ha=ha, zorder=7)

    ax_map1.set_title('b.  Inverted Ice-Table Burial Depth D(x)', fontweight='bold', fontsize=11.0, pad=5)
    ax_map1.set_xlabel('Longitude (°E)', fontsize=10.0)
    ax_map1.set_ylabel('')
    ax_map1.set_yticklabels([])

    # =========================================================================
    # Panel c: Bayesian Epistemic Uncertainty sigma_prob(x)
    # =========================================================================
    ax_map2.imshow(rgb_unc, origin='upper', extent=extent)
    smooth_unc = gaussian_filter(grid_unc, sigma=2.5)
    CS2 = ax_map2.contour(lons, lats, smooth_unc, levels=[0.10, 0.15, 0.20],
                          colors=['#ffffff', '#ffcc80', '#ff8080'], linewidths=[1.6, 1.6, 1.6])
    ax_map2.clabel(CS2, inline=True, fmt=lambda x: f'σ = {x:.2f}', fontsize=8.0,
                   manual=[(190.0, 47.0), (195.0, 42.8), (200.5, 39.5)])

    ax_map2.set_title('c.  Bayesian Epistemic Uncertainty σ_prob(x)', fontweight='bold', fontsize=11.0, pad=5)
    ax_map2.set_xlabel('Longitude (°E)', fontsize=10.0)
    ax_map2.set_ylabel('')
    ax_map2.set_yticklabels([])

    # =========================================================================
    # Row 1 Colorbars (Matched to Aspect-Corrected Map Extents)
    # =========================================================================
    fig.canvas.draw()
    sm0 = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=ice_cmap)
    sm1 = plt.cm.ScalarMappable(norm=Normalize(vmin=0.05, vmax=0.75), cmap=depth_cmap)
    sm2 = plt.cm.ScalarMappable(norm=Normalize(vmin=0.03, vmax=0.35), cmap=unc_cmap)

    for ax, sm, ticks, label in [
            (ax_map0, sm0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 'P(ice)'),
            (ax_map1, sm1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 'Burial Depth (m)'),
            (ax_map2, sm2, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35], 'σ_prob')]:
        p = ax.get_position()
        h_map = min(p.height, p.width * fig.get_figwidth() * aspect / fig.get_figheight())
        cax = fig.add_axes([p.x1 + 0.007, p.y0 + p.height / 2 - h_map / 2, 0.010, h_map])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks(ticks)
        cbar.set_label(label, fontsize=9.2, labelpad=4)
        cbar.ax.tick_params(labelsize=8.2)
        cbar.outline.set_linewidth(0.8)

    # =========================================================================
    # Panel d: Latitudinal Ice Probability Transition Profile
    # =========================================================================
    p_mean = grid_prob.mean(axis=1)
    p_std = grid_prob.std(axis=1)
    ax_prof0.plot(lats, p_mean, color='#08306b', lw=2.2, label='Longitudinal Mean P(ice)')
    ax_prof0.fill_between(lats, np.clip(p_mean - p_std, 0, 1), np.clip(p_mean + p_std, 0, 1),
                          color='#9ecae1', alpha=0.45, label=r'±1σ Regional Spread')
    ax_prof0.axhline(0.50, color='gray', ls=':', lw=1.2, label='Stability Boundary (P = 0.50)')

    for name, lon, lat, _ in corridor_ice:
        ax_prof0.plot(lat, 1.00, 'o', color='#00e5ff', markeredgecolor='black', markersize=5.8, zorder=5)
    for name, lon, lat in corridor_dry:
        ax_prof0.plot(lat, 0.00, 's', color='#ff3d00', markeredgecolor='black', markersize=5.2, zorder=5)

    ax_prof0.plot([], [], 'o', color='#00e5ff', markeredgecolor='black', markersize=5.8, label='Excavated Ice (HiRISE)')
    ax_prof0.plot([], [], 's', color='#ff3d00', markeredgecolor='black', markersize=5.2, label='Dry Regolith Ground-Truth')

    ax_prof0.set_xlim(37, 50)
    ax_prof0.set_ylim(-0.08, 1.15)
    ax_prof0.set_xlabel('Latitude (°N)', fontsize=9.8)
    ax_prof0.set_ylabel('Ice Probability P(ice)', fontsize=9.8)
    ax_prof0.set_title('d.  Latitudinal Ice Probability Transition', fontweight='bold', fontsize=10.8, pad=5)
    ax_prof0.legend(loc='upper left', fontsize=7.2, framealpha=0.92)
    ax_prof0.grid(True, alpha=0.25)

    # =========================================================================
    # Panel e: Permafrost Table Shallowing Transect
    # =========================================================================
    d_mean = grid_depth.mean(axis=1)
    d_std = grid_depth.std(axis=1)
    ax_prof1.plot(lats, d_mean, color='#005a32', lw=2.2, label='Longitudinal Mean Depth')
    ax_prof1.fill_between(lats, np.clip(d_mean - d_std, 0.05, 1.0), np.clip(d_mean + d_std, 0.05, 1.0),
                          color='#a1d99b', alpha=0.45, label=r'±1σ Regional Spread')

    prof_depth_calls = [
        ('Dundas_05 (0.90 m)', 39.11, 0.90, (0, -12), 'top', 'center'),
        ('HiRISE_Ice_01 (0.65 m)', 43.88, 0.65, (0, 6), 'bottom', 'center'),
        ('HiRISE_Ice_02 (0.48 m)', 46.14, 0.48, (-10, 7), 'bottom', 'right'),
        ('HiRISE_Ice_07 (0.55 m)', 46.57, 0.55, (10, -11), 'top', 'left')
    ]
    for lbl, lat_val, depth_val, xytext, va, ha in prof_depth_calls:
        ax_prof1.plot(lat_val, depth_val, 'o', color='#ffff00', markeredgecolor='black', markersize=6.5, zorder=6)
        ax_prof1.annotate(lbl, (lat_val, depth_val), fontsize=7.2, fontweight='normal', color='black',
                          xytext=xytext, textcoords='offset points', va=va, ha=ha, zorder=7)

    ax_prof1.set_xlim(37, 50)
    ax_prof1.set_ylim(0.20, 1.02)
    ax_prof1.set_xlabel('Latitude (°N)', fontsize=9.8)
    ax_prof1.set_ylabel('Ice-Table Burial Depth (m)', fontsize=9.8)
    ax_prof1.set_title('e.  Permafrost Table Shallowing Transect', fontweight='bold', fontsize=10.8, pad=5)
    ax_prof1.legend(loc='lower left', fontsize=7.5, framealpha=0.92)
    ax_prof1.grid(True, alpha=0.25)

    # =========================================================================
    # Panel f: Regional Epistemic Uncertainty Distribution (PDF)
    # =========================================================================
    ax_prof2.hist(grid_unc.ravel(), bins=35, range=(0.03, 0.35),
                  color='#b10026', edgecolor='none', alpha=0.75, density=True)
    ax_prof2.axvline(grid_unc.mean(), color='#4a148c', lw=1.8, ls='--',
                     label=f'Mean σ = {grid_unc.mean():.3f} ± {grid_unc.std():.3f}')
    ax_prof2.axvspan(0.03, 0.10, color='#e8f5e9', alpha=0.6, label='Interior High Confidence (σ ≤ 0.10)')
    ax_prof2.set_xlim(0.03, 0.35)
    ax_prof2.set_xlabel('Epistemic Uncertainty σ_prob', fontsize=9.8)
    ax_prof2.set_ylabel('Probability Density', fontsize=9.8)
    ax_prof2.set_title('f.  Regional Epistemic Uncertainty Distribution', fontweight='bold', fontsize=10.8, pad=5)
    ax_prof2.legend(loc='upper right', fontsize=7.5, framealpha=0.92)
    ax_prof2.grid(True, alpha=0.25)

    out_file = "figures/fig4_arcadia_inversion_composite.png"
    print(f"Saving high-resolution figure to {out_file} at 600 DPI...")
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    out_pdf = out_file.replace('.png', '.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Figure 4 successfully generated: {out_file} and {out_pdf}")


if __name__ == "__main__":
    generate_figure()
