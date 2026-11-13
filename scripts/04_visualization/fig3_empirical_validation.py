#!/usr/bin/env python3
"""
fig3_empirical_validation.py
============================
Generates Figure 3: Empirical validation scorecard and predictive calibration.

Panel a: Receiver Operating Characteristic (ROC) curve from physics-constrained model predictions
         with 5-fold CV +/- 1sigma envelope and optimal decision threshold operating point.
Panel b: Predicted ice occurrence probabilities across all 26 catalogued benchmark sites (16 ice excavations,
         10 dry controls), with error bars indicating Bayesian epistemic uncertainty (Laplace approximation)
         and shaded physical confidence zones.
Panel c: Stage II Ice-Table Burial Depth Parity Plot comparing observed excavation/trench depth
         against Bayesian Ridge model predictions, Leave-One-Out Cross-Validation (LOOCV),
         +/-10 cm operational tolerance envelope, and residual distribution inset.

Usage:
    python3 scripts/04_visualization/fig3_empirical_validation.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import BayesianRidge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03_modeling"))
import train_model as tm  # type: ignore


def generate_figure():
    print("Loading data for Figure 3...")
    metrics_file = "data/processed/evaluation_metrics.json"
    assert os.path.exists(metrics_file), "Run train_model.py first."
    metrics = json.load(open(metrics_file))
    depth_rmse = metrics["metrics_PI_BIA"]["Depth_RMSE_m"]
    depth_loocv_rmse = metrics["metrics_PI_BIA"]["Depth_LOOCV_RMSE_m"]

    # Re-evaluate live model on ground truth
    mola_64, mola_16, grs_raw = tm.load_global_datasets()
    samples = tm.load_ground_truth()
    X_raw, y_prob, y_depth = tm.extract_true_global_features(samples, mola_64, mola_16, grs_raw)

    mu = X_raw.mean(axis=0)
    std = X_raw.std(axis=0) + 1e-6
    X_norm = (X_raw - mu) / std

    w_fit, b_fit, cov_theta, _ = tm.train_physics_constrained_classifier(X_norm, y_prob)

    # Compute predicted probability and epistemic uncertainty
    logits = X_norm @ w_fit + b_fit
    p_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    X_tilde = np.hstack([X_norm, np.ones((len(X_norm), 1))])
    latent_var = np.sum((X_tilde @ cov_theta) * X_tilde, axis=1)
    p_unc = p_pred * (1.0 - p_pred) * np.sqrt(latent_var)

    fpr, tpr, thresholds = roc_curve(y_prob, p_pred)
    roc_auc = roc_auc_score(y_prob, p_pred)

    # 5-fold CV ROC distribution (fold-internal feature standardization, zero leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    cv_aucs = []
    for train_idx, test_idx in cv.split(X_raw, y_prob):
        mu_cv = X_raw[train_idx].mean(axis=0)
        std_cv = X_raw[train_idx].std(axis=0) + 1e-6
        X_tr = (X_raw[train_idx] - mu_cv) / std_cv
        X_te = (X_raw[test_idx] - mu_cv) / std_cv
        w_cv, b_cv, _, _ = tm.train_physics_constrained_classifier(X_tr, y_prob[train_idx])
        preds_cv = 1.0 / (1.0 + np.exp(-np.clip(X_te @ w_cv + b_cv, -30.0, 30.0)))
        cv_aucs.append(roc_auc_score(y_prob[test_idx], preds_cv))
        fpr_cv, tpr_cv, _ = roc_curve(y_prob[test_idx], preds_cv)
        interp_tpr = np.interp(mean_fpr, fpr_cv, tpr_cv)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)

    # Stage II: Ice-table depth modeling on confirmed ice excavations (n=16)
    pos = y_prob > 0.5
    X_pos_raw = X_raw[pos]
    y_pos = y_depth[pos]
    samples_pos = [s for s in samples if s["ice"] > 0.5]

    # In-sample Bayesian Ridge depth fit (standardized matching train_model.py)
    X_pos_norm = X_norm[pos]

    bridge = BayesianRidge(alpha_1=1e-3, alpha_2=1e-3, lambda_1=1e-3, lambda_2=1e-3)
    bridge.fit(X_pos_norm, y_pos)
    d_fit, d_std = bridge.predict(X_pos_norm, return_std=True)
    d_fit = np.clip(d_fit, 0.05, None)

    # Leave-One-Out Cross-Validation (LOOCV) for depth
    loo = LeaveOneOut()
    loo_preds = []
    for train_idx, test_idx in loo.split(X_pos_raw):
        mu_loo = X_pos_raw[train_idx].mean(axis=0)
        std_loo = X_pos_raw[train_idx].std(axis=0) + 1e-6
        X_tr = (X_pos_raw[train_idx] - mu_loo) / std_loo
        X_te = (X_pos_raw[test_idx] - mu_loo) / std_loo
        m_loo = BayesianRidge(alpha_1=1e-3, alpha_2=1e-3, lambda_1=1e-3, lambda_2=1e-3)
        m_loo.fit(X_tr, y_pos[train_idx])
        loo_preds.append(float(np.clip(m_loo.predict(X_te), 0.05, None)[0]))
    loo_preds = np.array(loo_preds)

    os.makedirs("figures", exist_ok=True)

    print("Rendering Figure 3 at 600 DPI...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    # ----------------------------------------------------
    # Panel a: ROC Curve with Confidence Envelope
    # ----------------------------------------------------
    axes[0].fill_between(fpr, 0, tpr, color="#0055d4", alpha=0.10, label=f"In-Sample AUC = {roc_auc:.3f}")
    axes[0].plot(fpr, tpr, color="#0055d4", lw=2.6, label=f"PI-BIA In-Sample (AUC = {roc_auc:.3f})")
    axes[0].plot(mean_fpr, mean_tpr, color="#d95f02", lw=1.8, ls="--", label=f"5-Fold CV Mean (AUC = {np.mean(cv_aucs):.3f})")
    axes[0].fill_between(mean_fpr, tpr_lower, tpr_upper, color="#6baed6", alpha=0.25, label=r"Empirical 5-Fold CV $\pm 1\sigma$ Spread")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1.2, label="Random Chance (AUC = 0.500)")

    # Dynamic evaluation of operating point at decision threshold P = 0.50
    pred_bin_05 = (p_pred >= 0.50).astype(int)
    tpr_05 = float(np.sum((pred_bin_05 == 1) & (y_prob == 1)) / np.sum(y_prob == 1))
    fpr_05 = float(np.sum((pred_bin_05 == 1) & (y_prob == 0)) / np.sum(y_prob == 0))

    axes[0].plot(fpr_05, tpr_05, marker="D", color="#ff6f00", markeredgecolor="black", markersize=8.5, zorder=6)
    axes[0].annotate(
        f"Decision Threshold\nP = 0.50 (TPR={tpr_05*100:.0f}%, FPR={fpr_05*100:.0f}%)", (fpr_05, tpr_05),
        fontsize=8.2, fontweight="normal", color="black",
        xytext=(14, -18), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0)
    )

    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].set_title("a.  Receiver Operating Characteristic (ROC)", fontweight="bold", fontsize=12, pad=6)
    axes[0].legend(loc="lower right", fontsize=8.8, framealpha=0.92)
    axes[0].grid(True, alpha=0.25)

    # ----------------------------------------------------
    # Panel b: Site-by-Site Epistemic Uncertainty & Confidence Zones
    # ----------------------------------------------------
    order = [i for i, s in enumerate(samples) if y_prob[i] == 1.0] + [i for i, s in enumerate(samples) if y_prob[i] == 0.0]
    ordered_samples = [samples[i] for i in order]
    ordered_pred = p_pred[order]
    ordered_unc = p_unc[order]
    ordered_y = y_prob[order]

    indices = np.arange(len(samples))
    is_ice = ordered_y == 1.0
    is_dry = ordered_y == 0.0
    n_ice = int(np.sum(is_ice))
    n_dry = int(np.sum(is_dry))

    axes[1].axhspan(0.70, 1.15, color="#e3f2fd", alpha=0.45, label="High-Confidence Ice (P ≥ 0.70)")
    axes[1].axhspan(0.40, 0.70, color="#fff9c4", alpha=0.45, label="Transition Zone (0.40 ≤ P < 0.70)")
    axes[1].axhspan(-0.02, 0.40, color="#ffebee", alpha=0.45, label="Dry Regolith (P < 0.40)")

    axes[1].axhline(0.50, color="gray", lw=1.2, ls=":", label="Decision Threshold (P = 0.50)")

    # Epistemic uncertainty whiskers physically bounded within [0, 1] probability domain
    yerr_ice = np.array([
        np.minimum(ordered_unc[is_ice], ordered_pred[is_ice]),
        np.minimum(ordered_unc[is_ice], 1.0 - ordered_pred[is_ice])
    ])
    yerr_dry = np.array([
        np.minimum(ordered_unc[is_dry], ordered_pred[is_dry]),
        np.minimum(ordered_unc[is_dry], 1.0 - ordered_pred[is_dry])
    ])

    axes[1].errorbar(
        indices[is_ice], ordered_pred[is_ice], yerr=yerr_ice,
        fmt="o", color="#0055d4", ecolor="#0055d4", elinewidth=1.5, capsize=3.5,
        markersize=6.8, label=f"Confirmed Ice Excavation (n={n_ice})"
    )
    axes[1].errorbar(
        indices[is_dry], ordered_pred[is_dry], yerr=yerr_dry,
        fmt="s", color="#d32f2f", ecolor="#d32f2f", elinewidth=1.5, capsize=3.5,
        markersize=6.4, label=f"Dry Control Crater (n={n_dry})"
    )

    divider = n_ice - 0.5
    axes[1].axvline(divider, color="#757575", lw=1.2, ls="--")
    axes[1].text((n_ice - 1) / 2.0, 1.08, f"Ice Excavations (n={n_ice})", ha="center", fontsize=8.8, color="#002b80", fontweight="bold")
    axes[1].text(n_ice + (n_dry - 1) / 2.0, 1.08, f"Dry Controls (n={n_dry})", ha="center", fontsize=8.8, color="#800000", fontweight="bold")

    axes[1].set_xticks(indices)
    axes[1].set_xticklabels([s["site_id"] for s in ordered_samples], rotation=48, ha="right", fontsize=7.8)
    axes[1].set_xlabel("Ground-Truth Benchmark Site", fontsize=11)
    axes[1].set_ylabel("Predicted Ice Probability P(ice)", fontsize=11)
    axes[1].set_title("b.  Benchmark Detection & Epistemic Uncertainty", fontweight="bold", fontsize=12, pad=6)
    axes[1].set_ylim(-0.02, 1.15)
    axes[1].legend(loc="lower left", fontsize=7.8, framealpha=0.92)
    axes[1].grid(True, alpha=0.25)

    # ----------------------------------------------------
    # Panel c: Depth Parity & Cross-Validation
    # ----------------------------------------------------
    axes[2].fill_between([0.0, 1.0], [-0.10, 0.90], [0.10, 1.10],
                         color="#2e7d32", alpha=0.08, label="±10 cm Operational Tolerance")
    axes[2].plot([0.0, 1.0], [0.0, 1.0], "k--", lw=1.4, label="1:1 Parity (Perfect Agreement)")

    axes[2].errorbar(
        y_pos, d_fit, yerr=d_std,
        fmt="o", color="#0066cc", ecolor="#6baed6", elinewidth=1.6, capsize=3.5,
        markersize=6.8, label=f"Bayesian Ridge In-sample (RMSE = {depth_rmse*100:.1f} cm)"
    )

    axes[2].plot(
        y_pos, loo_preds, "d", color="#d95f02", markeredgecolor="#d95f02",
        markerfacecolor="none", markersize=7.5, mew=1.5,
        label=f"LOOCV Out-of-fold (RMSE = {depth_loocv_rmse*100:.1f} cm)"
    )

    # Residuals Inset
    ax_ins = axes[2].inset_axes([0.06, 0.64, 0.30, 0.30])
    res = (d_fit - y_pos) * 100  # cm
    ax_ins.hist(res, bins=6, color="#90caf9", edgecolor="#1565c0", alpha=0.85, density=True)
    x_norm = np.linspace(-18, 18, 100)
    ax_ins.plot(x_norm, norm.pdf(x_norm, 0, depth_rmse * 100), color="#0d47a1", lw=1.5)
    ax_ins.axvline(0, color="gray", ls="--", lw=0.9)
    ax_ins.set_title(f"Residuals (cm)\nRMSE = {depth_rmse*100:.1f} cm", fontsize=7.2, pad=2)
    ax_ins.set_yticks([])
    ax_ins.tick_params(labelsize=6.8)
    ax_ins.set_xlim(-20, 20)

    phx_idx = samples_pos.index(next(s for s in samples_pos if s["site_id"] == "Phoenix"))
    d05_idx = samples_pos.index(next(s for s in samples_pos if s["site_id"] == "Dundas_05"))
    h09_idx = samples_pos.index(next(s for s in samples_pos if s["site_id"] == "HiRISE_Ice_09"))

    axes[2].annotate("Phoenix (0.05 m)", (0.05, float(d_fit[phx_idx])),
                     fontsize=8.0, fontweight="normal", color="black",
                     xytext=(10, -10), textcoords="offset points", va="top", ha="left",
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

    axes[2].annotate("HiRISE_Ice_09", (0.25, float(d_fit[h09_idx])),
                     fontsize=8.0, fontweight="normal", color="black",
                     xytext=(-10, 8), textcoords="offset points", va="bottom", ha="right")

    # Clean placement centered directly below point at x=0.90 m
    axes[2].annotate("Dundas_05\n(0.90 m)", (0.90, float(loo_preds[d05_idx])),
                     fontsize=8.0, fontweight="normal", color="black",
                     xytext=(0, -14), textcoords="offset points", va="top", ha="center")

    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xlabel("Measured Excavation / Trench Depth (m)", fontsize=11)
    axes[2].set_ylabel("Predicted Ice-Table Depth D(x) (m)", fontsize=11)
    axes[2].set_title("c.  Ice-Table Depth Parity & Cross-Validation", fontweight="bold", fontsize=12, pad=6)
    axes[2].legend(loc="lower right", fontsize=8.2, framealpha=0.92)
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()

    out_fig = "figures/fig3_empirical_validation_curves.png"
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    out_pdf = out_fig.replace(".png", ".pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 successfully generated: {out_fig} and {out_pdf}")


if __name__ == "__main__":
    generate_figure()
