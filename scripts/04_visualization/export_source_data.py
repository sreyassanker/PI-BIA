#!/usr/bin/env python3
"""
export_source_data.py
=====================
Exports discrete Source Data files for publication and open-science compliance.

Generates publication-ready CSV tables and a consolidated master Excel workbook
containing all numerical underlying data for display items:
- Figure 3 (Empirical Validation & Calibration):
    - Ground-truth benchmark sites (n=26): coordinates, class, measured depth, predicted ice
      probability, Laplace epistemic uncertainty (sigma), in-sample depth, and LOOCV depth.
    - Receiver Operating Characteristic (ROC): false positive rate (FPR), in-sample TPR, 5-fold CV
      mean TPR, standard deviation, and +/- 1-sigma confidence envelope.
- Figure 4 (Regional Inversion Composite & Transects):
    - 1D latitudinal transects: mean ice probability P(ice) and burial depth D(x) with +/- 1-sigma
      longitudinal spread from 37.0°N to 50.0°N across Arcadia Planitia.
    - Epistemic uncertainty distribution: probability density function (PDF) histogram bins matching
      Figure 4 panel f (35 bins over sigma in [0.03, 0.35]).
- Figure 5 (Candidate Landing Sites & Multi-Criteria Decision Scorecard):
    - Multi-Criteria Decision Analysis (MCDA) evaluation metrics for Candidate Sites #1 to #5
      across five normalized engineering dimensions (Ice Probability, Burial Accessibility, Slope Safety,
      EDL Margin, and Normalized RAI).
- Table 1 (Top-5 Landing Site Characteristics):
    - Physical coordinates, MOLA elevation, regional slope, predicted ice probability, burial depth with
      posterior predictive uncertainty (+/- 1-sigma), RAI composite score, and primary site features.

Usage:
    python3 scripts/04_visualization/export_source_data.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import BayesianRidge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03_modeling"))
import train_model as tm  # type: ignore

def export_all():
    out_dir = "data/source_data"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Exporting Source Data files to {out_dir}/...")

    # =========================================================================
    # 1. Figure 3 Source Data
    # =========================================================================
    mola_64, mola_16, grs_raw = tm.load_global_datasets()
    samples = tm.load_ground_truth()
    X_raw, y_prob, y_depth = tm.extract_true_global_features(samples, mola_64, mola_16, grs_raw)

    mu = X_raw.mean(axis=0)
    std = X_raw.std(axis=0) + 1e-6
    X_norm = (X_raw - mu) / std

    w_fit, b_fit, cov_theta, _ = tm.train_physics_constrained_classifier(X_norm, y_prob)

    logits = X_norm @ w_fit + b_fit
    p_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    X_tilde = np.hstack([X_norm, np.ones((len(X_norm), 1))])
    latent_var = np.sum((X_tilde @ cov_theta) * X_tilde, axis=1)
    p_unc = p_pred * (1.0 - p_pred) * np.sqrt(latent_var)

    # Stage II: In-sample & LOOCV depth estimation on confirmed ice excavations (n=16)
    pos = y_prob > 0.5
    X_pos_norm = X_norm[pos]
    X_pos_raw = X_raw[pos]
    y_pos = y_depth[pos]
    samples_pos = [s for s in samples if s["ice"] > 0.5]

    # In-sample Bayesian Ridge depth fit (standardized matching train_model.py & fig3)
    bridge = BayesianRidge(alpha_1=1e-3, alpha_2=1e-3, lambda_1=1e-3, lambda_2=1e-3)
    bridge.fit(X_pos_norm, y_pos)
    d_fit = np.clip(bridge.predict(X_pos_norm), 0.05, None)

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

    depth_insample_map = {samples_pos[j]["site_id"]: d_fit[j] for j in range(len(samples_pos))}
    depth_loocv_map = {samples_pos[j]["site_id"]: loo_preds[j] for j in range(len(samples_pos))}

    # Compile Fig 3 Benchmark table
    fig3_benchmarks = []
    for i, s in enumerate(samples):
        site_id = s["site_id"]
        is_ice = (s["ice"] == 1.0)
        row = {
            "Site_ID": site_id,
            "Latitude_deg_N": s["lat"],
            "Longitude_deg_E": s["lon"],
            "Class": "Ice Excavation" if is_ice else "Dry Control",
            "Ground_Truth_Ice": int(s["ice"]),
            "Measured_Depth_m": round(float(s["depth"]), 4) if is_ice else np.nan,
            "Predicted_Ice_Probability": round(float(p_pred[i]), 4),
            "Epistemic_Uncertainty_sigma": round(float(p_unc[i]), 4),
            "Predicted_Depth_InSample_m": round(float(depth_insample_map[site_id]), 4) if is_ice else np.nan,
            "Predicted_Depth_LOOCV_m": round(float(depth_loocv_map[site_id]), 4) if is_ice else np.nan
        }
        fig3_benchmarks.append(row)
    df_fig3_bench = pd.DataFrame(fig3_benchmarks)
    df_fig3_bench.to_csv(os.path.join(out_dir, "SourceData_Fig3_Benchmarks.csv"), index=False)

    # Fig 3 ROC curves (in-sample and 5-fold CV)
    fpr, tpr, thresholds = roc_curve(y_prob, p_pred)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    for train_idx, test_idx in cv.split(X_raw, y_prob):
        mu_cv = X_raw[train_idx].mean(axis=0)
        std_cv = X_raw[train_idx].std(axis=0) + 1e-6
        X_tr = (X_raw[train_idx] - mu_cv) / std_cv
        X_te = (X_raw[test_idx] - mu_cv) / std_cv
        w_cv, b_cv, _, _ = tm.train_physics_constrained_classifier(X_tr, y_prob[train_idx])
        preds_cv = 1.0 / (1.0 + np.exp(-np.clip(X_te @ w_cv + b_cv, -30.0, 30.0)))
        fpr_cv, tpr_cv, _ = roc_curve(y_prob[test_idx], preds_cv)
        interp_tpr = np.interp(mean_fpr, fpr_cv, tpr_cv)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    in_sample_tpr = np.interp(mean_fpr, fpr, tpr)
    in_sample_tpr[0] = 0.0
    in_sample_tpr[-1] = 1.0

    df_fig3_roc = pd.DataFrame({
        "FPR": np.round(mean_fpr, 4),
        "In_Sample_TPR": np.round(in_sample_tpr, 4),
        "Mean_TPR_5Fold_CV": np.round(mean_tpr, 4),
        "Std_TPR_5Fold_CV": np.round(std_tpr, 4),
        "TPR_Upper_1sigma": np.round(np.minimum(mean_tpr + std_tpr, 1.0), 4),
        "TPR_Lower_1sigma": np.round(np.maximum(mean_tpr - std_tpr, 0.0), 4)
    })
    df_fig3_roc.to_csv(os.path.join(out_dir, "SourceData_Fig3_ROC.csv"), index=False)

    # =========================================================================
    # 2. Figure 4 Source Data: 1D Latitudinal Transects & Uncertainty Distribution
    # =========================================================================
    inv = np.load("data/processed/arcadia_inversion_results.npz")
    lats = inv["lats"]
    p_grid = inv["ice_probability"]
    d_grid = inv["ice_table_depth_m"]
    u_grid = inv["ice_probability_unc"]

    p_mean = np.mean(p_grid, axis=1)
    p_std = np.std(p_grid, axis=1)
    d_mean = np.mean(d_grid, axis=1)
    d_std = np.std(d_grid, axis=1)

    df_fig4_transects = pd.DataFrame({
        "Latitude_deg_N": np.round(lats, 4),
        "Mean_Ice_Probability": np.round(p_mean, 4),
        "Std_Ice_Probability": np.round(p_std, 4),
        "Mean_Ice_Depth_m": np.round(d_mean, 4),
        "Std_Ice_Depth_m": np.round(d_std, 4)
    })
    df_fig4_transects.to_csv(os.path.join(out_dir, "SourceData_Fig4_Transects.csv"), index=False)

    # Histogram of regional epistemic uncertainty matching Figure 4 panel f
    counts, bin_edges = np.histogram(u_grid.flatten(), bins=35, range=(0.03, 0.35), density=True)
    df_fig4_hist = pd.DataFrame({
        "Bin_Left": np.round(bin_edges[:-1], 4),
        "Bin_Right": np.round(bin_edges[1:], 4),
        "Probability_Density": np.round(counts, 4)
    })
    df_fig4_hist.to_csv(os.path.join(out_dir, "SourceData_Fig4_Uncertainty_Dist.csv"), index=False)

    # =========================================================================
    # 3. Figure 5 Source Data: Candidate Landing Sites & Multi-Criteria Scorecard
    # =========================================================================
    with open("data/processed/evaluation_metrics.json") as f:
        metrics = json.load(f)
    top_sites = metrics["top_landing_sites_arcadia"]

    scorecard_rows = []
    for s in top_sites:
        scorecard_rows.append({
            "Rank": s["rank"],
            "Site": f"Site #{s['rank']}",
            "Latitude_deg_N": s["lat"],
            "Longitude_deg_E": s["lon"],
            "Elevation_m": s["elevation_m"],
            "Slope_deg": s["slope_deg"],
            "Predicted_Ice_Prob": s["predicted_ice_prob"],
            "Burial_Access_Score": round(float(np.exp(-s["predicted_ice_depth_m"] / 1.25)), 4),
            "Slope_Safety_Score": round(1.0 - s["slope_deg"] / 10.0, 4),
            "EDL_Margin_Score": round(float(np.clip((-s["elevation_m"] - 1500) / 2500.0, 0.0, 1.0)), 4),
            "Composite_RAI": s["RAI_score"],
            "Normalized_RAI_Score": round(s["RAI_score"] / 0.60, 4)
        })
    df_fig5 = pd.DataFrame(scorecard_rows)
    df_fig5.to_csv(os.path.join(out_dir, "SourceData_Fig5_Scorecard.csv"), index=False)

    # =========================================================================
    # 4. Table 1 Source Data
    # =========================================================================
    features = [
        "Highest ice probability",
        "Smooth regional plain",
        "Central basin plain",
        "Negligible slope hazard",
        "Lower latitude corridor"
    ]
    tab1_rows = []
    for s in top_sites:
        tab1_rows.append({
            "Site": f"#{s['rank']}",
            "Latitude_deg_N": s["lat"],
            "Longitude_deg_E": s["lon"],
            "Elevation_m": s["elevation_m"],
            "Slope_deg": s["slope_deg"],
            "P_ice": s["predicted_ice_prob"],
            "Predicted_Depth_m": s["predicted_ice_depth_m"],
            "Predictive_Uncertainty_m": s["depth_uncertainty_m"],
            "RAI_Score": s["RAI_score"],
            "Primary_Feature": features[s["rank"]-1]
        })
    df_tab1 = pd.DataFrame(tab1_rows)
    df_tab1.to_csv(os.path.join(out_dir, "SourceData_Table1_LandingSites.csv"), index=False)

    # =========================================================================
    # 5. Combined Master Excel Workbook
    # =========================================================================
    excel_path = os.path.join(out_dir, "SourceData_Master.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_fig3_bench.to_excel(writer, sheet_name="Fig3_Benchmarks", index=False)
        df_fig3_roc.to_excel(writer, sheet_name="Fig3_ROC_Curves", index=False)
        df_fig4_transects.to_excel(writer, sheet_name="Fig4_Transects", index=False)
        df_fig4_hist.to_excel(writer, sheet_name="Fig4_Uncertainty_Hist", index=False)
        df_fig5.to_excel(writer, sheet_name="Fig5_MCDA_Scorecard", index=False)
        df_tab1.to_excel(writer, sheet_name="Table1_LandingSites", index=False)

    print(f"Successfully generated all Source Data files in {out_dir}/ and workbook {excel_path}!")

if __name__ == "__main__":
    export_all()
