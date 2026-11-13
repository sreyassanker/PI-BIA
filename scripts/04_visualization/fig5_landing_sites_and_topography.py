#!/usr/bin/env python3
"""
fig5_landing_sites_and_topography.py
=====================================
Generates Figure 5: Resource Accessibility Index (RAI) and Candidate Landing Site Viability.

Panel a: Regional Resource Accessibility Index (RAI) continuous score map (0.0 to 0.60) integrating ice probability,
         burial depth, slope safety, EDL atmospheric pressure margin, and Laplace epistemic uncertainty.
         Dashed yellow contour marks the operational Prime Exploration Corridor (RAI >= 0.35).
Panel b: Multi-criteria operational engineering feasibility zones (Prime Operational Corridor, Moderate Resource Zone,
         and Engineering Exclusion / Low Resource) across southern and central Arcadia Planitia.
Panel c: Multi-Criteria Decision Analysis (MCDA) scorecard evaluating Candidate Sites #1 to #5 across five normalized
         engineering dimensions (Ice Probability, Burial Accessibility, Slope Safety, EDL Margin, and Normalized RAI).

Usage:
    python3 scripts/04_visualization/fig5_landing_sites_and_topography.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter


def generate_figure():
    print("Loading data for Figure 5...")
    cube = np.load("data/processed/cubes/arcadia_planitia_multimodal_cube.npz")
    sub_elev = cube["elevation"][::4, ::4]
    sub_slope = cube["slope"][::4, ::4]

    inv = np.load("data/processed/arcadia_inversion_results.npz", allow_pickle=True)
    rai_grid = inv["RAI"]
    grid_prob = inv["ice_probability"]
    grid_depth = inv["ice_table_depth_m"]
    top_sites = inv["top_sites"].tolist() if hasattr(inv["top_sites"], "tolist") else list(inv["top_sites"])

    os.makedirs("figures", exist_ok=True)

    rai_cmap = LinearSegmentedColormap.from_list(
        "rai", ["#0d2b16", "#1e6833", "#49a85a", "#f5d742", "#f39c12", "#d35400"]
    )

    # Multi-criteria operational engineering zones:
    # 0: Excluded (low ice P < 0.50, steep slope > 3.5 deg, or elevated terrain > -3000 m)
    # 1: Moderate Ice Resource Zone (0.50 <= P < 0.75, safe slope and EDL elevation)
    # 2: Prime Operational Corridor (P >= 0.75, accessible depth D <= 0.65 m, safe slope and EDL elevation)
    feas = np.zeros_like(grid_prob, dtype=int)
    safe_eng = (sub_elev <= -3000.0) & (sub_slope <= 3.5)
    feas[(grid_prob >= 0.50) & (grid_prob < 0.75) & safe_eng] = 1
    feas[(grid_prob >= 0.75) & (grid_depth <= 0.65) & safe_eng] = 2

    feas_colors = ["#d1d5db", "#7eb5d6", "#2d8659"]
    feas_cmap = ListedColormap(feas_colors)

    extent = [180.0, 205.0, 37.0, 50.0]

    print("Rendering Figure 5 at 600 DPI...")
    # Standard compact publication layout with tight inter-panel gaps
    fig = plt.figure(figsize=(16.8, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.20,
                          left=0.045, right=0.98, top=0.90, bottom=0.22)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # ----------------------------------------------------
    # Panel a: Continuous RAI Map + Operational Prime Corridor Contour (RAI >= 0.35)
    # ----------------------------------------------------
    im_rai = ax0.imshow(rai_grid, cmap=rai_cmap, origin="upper",
                        extent=extent, vmin=0.0, vmax=0.60)

    # Operational Prime Corridor contour (RAI >= 0.35)
    smooth_rai = gaussian_filter(rai_grid, sigma=1.8)
    ax0.contour(
        np.linspace(180, 205, rai_grid.shape[1]),
        np.linspace(50, 37, rai_grid.shape[0]),
        smooth_rai, levels=[0.35],
        colors=["#ffff00"], linewidths=[1.8], linestyles=["--"]
    )

    ax0.set_title("a.  Resource Accessibility Index (RAI)", fontweight="bold", fontsize=11.0, pad=6)
    ax0.set_xlabel("Longitude (°E)", fontsize=10.0)
    ax0.set_ylabel("Latitude (°N)", fontsize=10.0)

    for site in top_sites:
        ax0.plot(site["lon"], site["lat"], marker="*", markersize=13,
                 color="#d50000", markeredgecolor="white", markeredgewidth=1.1, zorder=5)
        ax0.annotate(f"#{site['rank']}", (site["lon"], site["lat"]),
                     fontsize=8.0, ha="center", va="bottom", color="black", fontweight="bold",
                     xytext=(0, 6), textcoords="offset points", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.88, edgecolor="none"))

    star_handle = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#d50000",
                             markeredgecolor="white", markersize=11, linestyle="none",
                             label="Top-5 candidate site (RAI-ranked)")
    contour_handle = plt.Line2D([0], [0], color="#ffff00", lw=1.8, ls="--",
                                label="Prime Corridor limit (RAI ≥ 0.35)")
    ax0.legend(handles=[star_handle, contour_handle], loc="lower right", fontsize=7.8, framealpha=0.92)

    # ----------------------------------------------------
    # Panel b: Engineering Feasibility & Candidate Exploration Corridors
    # ----------------------------------------------------
    im_feas = ax1.imshow(feas, cmap=feas_cmap, origin="upper",
                         extent=extent, vmin=-0.5, vmax=2.5)
    ax1.set_title("b.  Engineering Feasibility & Exploration Corridors",
                  fontweight="bold", fontsize=11.0, pad=6)
    ax1.set_xlabel("Longitude (°E)", fontsize=10.0)
    ax1.set_ylabel("")
    ax1.set_yticklabels([])

    for site in top_sites:
        ax1.plot(site["lon"], site["lat"], marker="*", markersize=13,
                 color="#d50000", markeredgecolor="white", markeredgewidth=1.1, zorder=5)
        ax1.annotate(f"#{site['rank']}", (site["lon"], site["lat"]),
                     fontsize=8.0, ha="center", va="bottom", color="black", fontweight="bold",
                     xytext=(0, 6), textcoords="offset points", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.88, edgecolor="none"))

    legend_patches = [
        mpatches.Patch(color="#2d8659", label="Prime Operational Corridor (P ≥ 0.75, D ≤ 0.65 m)"),
        mpatches.Patch(color="#7eb5d6", label="Moderate Resource Zone (0.50 ≤ P < 0.75)"),
        mpatches.Patch(color="#d1d5db", label="Engineering Exclusion / Desiccated Permafrost"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#d50000", markeredgecolor="white",
                   markersize=10, label="Candidate Sites (#1–#5)")
    ]
    ax1.legend(handles=legend_patches, loc="lower right", fontsize=7.0, framealpha=0.92)

    # Perfect baseline alignment across all panels
    fig.canvas.draw()
    p0 = ax0.get_position()
    p2 = ax2.get_position()
    ax2.set_position([p2.x0, p0.y0, p2.width, p0.height])

    # Matched colorbar for Panel a
    cax = fig.add_axes([p0.x1 + 0.006, p0.y0, 0.010, p0.height])
    cbar = fig.colorbar(im_rai, cax=cax)
    cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cbar.set_label("RAI Score", fontsize=9.2, labelpad=3)
    cbar.ax.tick_params(labelsize=8.0)
    cbar.outline.set_linewidth(0.8)

    # ----------------------------------------------------
    # Panel c: Candidate Site Multi-Criteria Scorecard
    # ----------------------------------------------------
    site_labels = [f"Site #{s['rank']}" for s in top_sites]
    y_pos = np.arange(len(top_sites))[::-1]

    ice_probs = [float(np.clip(s["predicted_ice_prob"], 0.0, 1.0)) for s in top_sites]
    depth_access = [float(np.clip(np.exp(-s["predicted_ice_depth_m"] / 1.25), 0.0, 1.0)) for s in top_sites]
    slope_safety = [float(np.clip(1.0 - (s["slope_deg"] / 10.0), 0.0, 1.0)) for s in top_sites]
    edl_safety = [float(np.clip((-s["elevation_m"] - 1500.0) / 2500.0, 0.0, 1.0)) for s in top_sites]
    rai_scores = [float(np.clip(s["RAI_score"] / 0.60, 0.0, 1.0)) for s in top_sites]

    bar_h = 0.15
    ax2.barh(y_pos + 2.0*bar_h, ice_probs, height=bar_h, color="#1565c0", label="Ice Probability P(ice)")
    ax2.barh(y_pos + 1.0*bar_h, depth_access, height=bar_h, color="#2e7d32", label=r"Burial Access ($\exp(-D/1.25\,\mathrm{m})$)")
    ax2.barh(y_pos + 0.0*bar_h, slope_safety, height=bar_h, color="#fbc02d", label="Slope Safety (1 - θ/10°)")
    ax2.barh(y_pos - 1.0*bar_h, edl_safety, height=bar_h, color="#f57c00", label="EDL Atmospheric Braking Margin")
    ax2.barh(y_pos - 2.0*bar_h, rai_scores, height=bar_h, color="#c2185b", label="Normalized RAI (Score/0.60)")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(site_labels, fontsize=8.8, fontweight="medium")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlabel("Normalized Performance Score", fontsize=9.5)
    ax2.set_title("c.  Candidate Site Multi-Criteria Scorecard", fontweight="bold", fontsize=11.0, pad=6)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.8, framealpha=0.92)
    ax2.grid(True, axis="x", alpha=0.25)

    out_file = "figures/fig5_landing_sites_and_topography.png"
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    out_pdf = out_file.replace(".png", ".pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Figure 5 successfully generated: {out_file} and {out_pdf}")


if __name__ == "__main__":
    generate_figure()
