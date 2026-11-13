# Physics-Informed Bayesian Inversion of Mars Shallow Ground Ice (PI-BIA)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Target: Icarus (Elsevier)](https://img.shields.io/badge/Target-Icarus%20(Elsevier)-orange.svg)](https://www.sciencedirect.com/journal/icarus)
[![Reproducibility: End--to--End](https://img.shields.io/badge/Reproducibility-End--to--End-brightgreen.svg)](run_pipeline.sh)
[![Data: NASA PDS](https://img.shields.io/badge/Data-NASA%20PDS-orange.svg)](https://pds-geosciences.wustl.edu/)

<div align="justify">

This repository provides the official implementation, verification suite, and reproducibility pipeline for the research manuscript:

> **"Physics-informed Bayesian inversion of shallow ground ice in Arcadia Planitia on Mars"**  
> **Author:** Sreyas S. S.  
> **Affiliation:** Amity Institute of Geoinformatics and Remote Sensing (AIGIRS), Amity University, Sector 125, Noida, 201313, Uttar Pradesh, India  
> **Email:** `sreyassanker001@gmail.com`  
> **ORCID:** [0009-0008-2893-5146](https://orcid.org/0009-0008-2893-5146)  
> **Submission:** Prepared for submission to *Icarus* (Elsevier / AAS DPS)

</div>

---

## Table of Contents

- [Scientific Overview](#scientific-overview)
- [Key Numerical Results](#key-numerical-results)
- [Candidate Landing Sites](#candidate-landing-sites)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [End-to-End Pipeline & Reproducibility](#end-to-end-pipeline--reproducibility)
- [Publication Figures & Deliverables](#publication-figures--deliverables)
- [Source Data & Reproducibility Tables](#source-data--reproducibility-tables)
- [Repository Architecture](#repository-architecture)
- [Data Availability & Ingestion](#data-availability--ingestion)
- [Citation](#citation)
- [License](#license)

---

## Scientific Overview

<div align="justify">

Characterizing shallow subsurface ground ice on Mars is critical for planetary paleoclimate reconstruction and human in-situ resource utilization (ISRU). However, quantitative mapping of shallow permafrost burial depth is hindered by a fundamental spatial resolution disconnect between orbital remote sensing and in-situ ground truth:
1. **Orbital Epithermal Neutron Spectroscopy** (*2001 Mars Odyssey* GRS/HEND) detects water-equivalent hydrogen strictly within the uppermost ~1 m of regolith but averages signals over massive spatial footprints (~300–600 km FWHM).
2. **Orbital Sounding Radar** (*MRO* SHARAD, *MEX* MARSIS) resolves deep excess ice reflectors (>15–20 m) across Arcadia Planitia, but surface clutter and bandwidth obscure the uppermost 10–15 m of the regolith column.
3. **Orbital Thermal Models** (*MCS*, *TES*) provide decimeter-scale estimates but rely on idealized assumptions of uniform regolith thermophysical properties and vapor equilibrium.
4. **Opportunistic Fresh Impact Excavations** (*HiRISE*) and in-situ landers (*Phoenix*) offer direct physical proof of shallow ice ($0.05\text{–}0.90\text{ m}$), but only at sparse, isolated coordinates.

**PI-BIA** (*Physics-Informed Bayesian Inversion Architecture*) bridges this observational gap by coupling multi-frequency orbital remote sensing with thermodynamic boundary constraints:
- **Stage I: Physics-Constrained Logistic Classification** — Solves for the regional ground-ice occurrence probability $P_{\text{ice}}(x)$ by enforcing thermodynamic monotonicity inequalities ($\partial P / \partial \phi_{\text{CEPI}} \le 0$ for epithermal neutron flux suppression, and $\partial P / \partial \text{lat} \ge 0$ for thermal cold-trapping). Bayesian parameter uncertainty is analytically quantified via the **Laplace approximation**, yielding a closed-form posterior covariance matrix $\Sigma_\theta$ and continuous epistemic uncertainty field $\sigma_{\text{prob}}(x)$.
- **Stage II: Bayesian Ridge Ice-Table Depth Modeling** — Calibrates shallow burial depth $D(x)$ against high-resolution empirical impact excavations (HiRISE, Dundas et al., Byrne et al.) and the *Phoenix* Mars Lander trench anchor with regularized posterior variance.
- **Stage III: Regional Inversion Raster** — Discretizes Arcadia Planitia ($37.0^\circ\text{–}50.0^\circ\text{N}$, $180.0^\circ\text{–}205.0^\circ\text{E}$) onto a $208 \times 400$ regional inversion grid ($0.0625^\circ/\text{pixel} \approx 3.7\text{ km}$ resolution, subsampled $4\times$ from the $832 \times 1600$ base multi-modal cube at $64\text{ pix/deg}$). This spatial decimation ensures numerical tractability for full parameter covariance projection while aligning with the physical resolution footprint of orbital epithermal neutron moderation (~300–600 km FWHM).
- **Stage IV: Multi-Criteria Decision Analysis (MCDA)** — Synthesizes ice probability, burial depth, regional slope, atmospheric Entry, Descent, and Landing (EDL) margin, and epistemic uncertainty into the continuous **Resource Accessibility Index (RAI)** to delineate prime exploration corridors.

</div>

```
       [Orbital Datasets: MOLA Topography + Odyssey GRS Epithermal Neutrons]
                                      │
                                      ▼
                      [Multi-Modal Regional Data Cube]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   [Stage I: Physics-Constrained             [Stage II: Bayesian Ridge Depth]
    Monotonic Classifier]                     (Trained on positive ice excavations)
                 │                                         │
        P(ice) + Laplace σ_prob                     Burial Depth D(x)
                 └────────────────────┬────────────────────┘
                                      ▼
             [Stage III: 2D Spatial Inversion across Arcadia Planitia]
                                      │
                                      ▼
     [Stage IV: Multi-Criteria Decision Analysis -> Resource Accessibility Index (RAI)]
                                      │
                                      ▼
         [Operational Candidate Landing Sites #1 to #5 & Source Data]
```

---

## Key Numerical Results

<div align="justify">

All results reported in the manuscript are computed strictly from calibrated planetary observations and verified ground-truth benchmarks ($n = 26$: 16 confirmed ice excavations, 10 dry controls):

</div>

| Pipeline Metric | Value | Verification Source | Operational Significance |
| :--- | :---: | :--- | :--- |
| **In-sample ROC-AUC** | **0.931** | `evaluation_metrics.json` | Excellent discrimination between ice-bearing and dry regolith |
| **Classification F1-Score** | **0.867** | `evaluation_metrics.json` | High precision/recall balance on sparse planetary samples |
| **5-Fold Stratified CV F1** | **0.786 ± 0.072** | `evaluation_metrics.json` | Robust generalization with fold-internal standardization |
| **Brier Score Loss** | **0.121** | `evaluation_metrics.json` | Strong probabilistic accuracy and predictive confidence |
| **Expected Calibration Error (ECE)** | **0.123** | `evaluation_metrics.json` | Well-calibrated posterior probabilities across probability bins |
| **In-sample Depth RMSE** | **0.0656 m** (6.56 cm) | `evaluation_metrics.json` | Sub-decimetre agreement with excavation ground truth |
| **In-sample Depth MAE** | **0.0449 m** (4.49 cm) | `evaluation_metrics.json` | Low average absolute deviation across benchmark sites |
| **LOOCV Depth RMSE** | **0.0955 m** (9.55 cm) | `evaluation_metrics.json` | Rigorous out-of-sample depth error within $\pm 10\text{ cm}$ operational limit |
| **Mean Regional Uncertainty $\sigma_{\text{prob}}$** | **0.146 ± 0.047** | `arcadia_inversion_results.npz` | High interior confidence ($\sigma \le 0.10$) across central basin |

---

## Candidate Landing Sites

<div align="justify">

Multi-Criteria Decision Analysis identifies five optimal candidate landing sites within the smooth plains of southern and central Arcadia Planitia meeting strict robotic and crewed exploration constraints:

</div>

$$\text{RAI}(x) = P_{\text{ice}}(x) \times \exp\left(-\frac{D(x)}{1.25}\right) \times \left(1 - \frac{\theta(x)}{10}\right) \times \eta_{\text{EDL}}(z(x)) \times (1 - 1.5\,\sigma_{\text{prob}}(x))$$

| Rank | Latitude (°N) | Longitude (°E) | Elevation (m) | Slope (°) | $P_{\text{ice}}$ | Burial Depth (m) | RAI Score | Primary Operational Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | **48.49** | **190.03** | **−4071** | **0.00** | **0.917** | **$0.52 \pm 0.08$** | **0.533** | Highest ice probability; ultra-flat regional plain |
| **#2** | **47.43** | **198.67** | **−4001** | **0.03** | **0.898** | **$0.55 \pm 0.08$** | **0.497** | Smooth regional plain; high atmospheric density margin |
| **#3** | **46.42** | **193.72** | **−3994** | **0.00** | **0.861** | **$0.57 \pm 0.08$** | **0.458** | Central basin plain; minimal topographic roughness |
| **#4** | **45.42** | **190.34** | **−4030** | **0.03** | **0.817** | **$0.58 \pm 0.08$** | **0.419** | Negligible slope hazard; balanced EDL and resource depth |
| **#5** | **44.41** | **187.02** | **−4055** | **0.00** | **0.759** | **$0.60 \pm 0.08$** | **0.378** | Lower-latitude corridor; enhanced solar power availability |

<div align="justify">

*Note: All five candidate sites reside well below the $-2.0\text{ km}$ MOLA datum threshold (offering $> 2\text{ km}$ excess EDL atmospheric braking column) with surface slopes $< 0.1^\circ$ (hazard-free).*

</div>

---

## System Requirements

<div align="justify">

- **Operating System:** Linux (Ubuntu 20.04+, Debian 11+), macOS (macOS 12 Monterey or later), or Windows 10/11 (WSL2 recommended).
- **Python Version:** Python 3.10, 3.11, or 3.12.
- **Hardware Requirements:** Standard desktop or laptop computer (minimum 4 GB RAM; **no GPU or specialized accelerators required**).
- **Disk Storage:** ~250 MB for planetary data cubes, GIS shapefiles, rendered figures, and Source Data files.
- **Execution Time:** ~45 seconds for full end-to-end pipeline execution on standard desktop hardware.

</div>

---

## Installation

<div align="justify">

Clone the repository and install dependencies in an isolated virtual environment (installation takes **< 1 minute**):

</div>

```bash
# 1. Clone the repository
git clone https://github.com/sreyassanker/PI-BIA.git
cd PI-BIA

# 2. Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Upgrade pip and install pinned scientific dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## End-to-End Pipeline & Reproducibility

### One-Command Full Pipeline

<div align="justify">

The complete end-to-end scientific workflow executes via the master driver script:

</div>

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

<div align="justify">

The script automatically carries out six sequential stages:
1. **Environment Check:** Validates Python runtime and confirms all scientific packages (`numpy`, `scipy`, `sklearn`, `matplotlib`, `shapefile`, `pandas`, `openpyxl`).
2. **Dataset Verification:** Runs the 6-stage validation suite confirming raw planetary raster geometries, GRS neutron ranges, benchmark coordinates, and physical consistency.
3. **Preprocessing & Cube Construction:** Compiles the 26 ground-truth benchmark sites into IAU Mars 2000 GIS shapefiles and rasterizes the $832 \times 1600$ multi-modal regional cube.
4. **Model Training & Inversion:** Executes physics-constrained optimization, Laplace epistemic uncertainty estimation, Bayesian Ridge depth regression, and RAI site ranking.
5. **Figure Generation:** Renders all publication figures (Figures 1–5) as 600 DPI publication PNGs and resolution-independent vector PDFs.
6. **Source Data Export:** Generates discrete CSV tables and the consolidated master Excel workbook for open-science reproducibility.

</div>

### Individual Modular Stages

<div align="justify">

Each pipeline stage can also be executed individually:

</div>

```bash
# Stage 1: Validate dataset integrity and physical consistency
python3 scripts/01_data_acquisition/verify_datasets.py

# Stage 2: Compile benchmark ground-truth catalogs and generate GIS shapefiles
python3 scripts/02_preprocessing/compile_ground_truth.py

# Stage 3: Assemble multi-modal regional data cube across Arcadia Planitia
python3 scripts/02_preprocessing/build_processed_cube.py

# Stage 4: Run physics-constrained Bayesian inversion & MCDA site selection
python3 scripts/03_modeling/train_model.py

# Stage 5: Render publication figures (saved to figures/ as PNG and vector PDF)
python3 scripts/04_visualization/fig1_study_area.py
python3 scripts/04_visualization/fig3_empirical_validation.py
python3 scripts/04_visualization/fig4_arcadia_inversion_composite.py
python3 scripts/04_visualization/fig5_landing_sites_and_topography.py

# Stage 6: Export publication source data tables and master Excel workbook
python3 scripts/04_visualization/export_source_data.py
```

---

## Publication Figures & Deliverables

<div align="justify">

All figures comply with planetary science journal specifications (double-column width $\le 180\text{ mm}$, single-column width $\le 89\text{ mm}$, legible typography in Helvetica/Arial, 600 DPI raster PNGs alongside vector PDFs):

</div>

| Figure | Deliverable Files | Description |
| :---: | :--- | :--- |
| **Figure 1** | `figures/fig1_study_area_and_shapefile_boundary.png`<br>`figures/fig1_study_area_and_shapefile_boundary.pdf` | **Physiographic setting of Arcadia Planitia:** MGS MOLA MEGDR topography with shaded-relief hillshading, official IAU boundary (USGS Feature #348), catalogued benchmark impact craters, global context inset, cartographic scale bar, and true North arrow. |
| **Figure 2** | `figures/fig2_methodology_architecture.png`<br>`figures/fig2_methodology_architecture.pdf` | **Methodology architecture:** Two-stage inversion framework flowchart detailing data ingestion, thermodynamic monotonicity constraints, Laplace uncertainty quantification, Bayesian Ridge depth regression, and MCDA site viability. |
| **Figure 3** | `figures/fig3_empirical_validation_curves.png`<br>`figures/fig3_empirical_validation_curves.pdf` | **Empirical validation scorecard:** (a) ROC curve with 5-fold cross-validation $\pm 1\sigma$ envelope and optimal threshold; (b) Ground-truth benchmark ice probabilities with Bayesian Laplace epistemic error bars; (c) Ice-table burial depth parity plot with LOOCV and $\pm 10\text{ cm}$ operational tolerance. |
| **Figure 4** | `figures/fig4_arcadia_inversion_composite.png`<br>`figures/fig4_arcadia_inversion_composite.pdf` | **Regional Bayesian inversion composite:** Top tier: 2D maps of $P_{\text{ice}}(x)$, burial depth $D(x)$, and Laplace uncertainty $\sigma_{\text{prob}}(x)$. Bottom tier: 1D latitudinal transects ($37^\circ\text{–}50^\circ\text{N}$) and regional epistemic uncertainty probability density function (PDF). |
| **Figure 5** | `figures/fig5_landing_sites_and_topography.png`<br>`figures/fig5_landing_sites_and_topography.pdf` | **Landing site viability & MCDA scorecard:** (a) Continuous Resource Accessibility Index (RAI) map with Prime Exploration Corridor ($\text{RAI} \ge 0.35$); (b) Operational engineering feasibility zoning; (c) Multi-criteria scorecard across Candidate Sites #1–#5. |

---

## Source Data & Reproducibility Tables

<div align="justify">

In accordance with open-science reproducibility standards, numerical source data for all display items are archived in `data/source_data/`:

</div>

| Source Data File | Target Display Item | Content Description |
| :--- | :---: | :--- |
| **`SourceData_Fig3_Benchmarks.csv`** | Figure 3b, 3c | Coordinates, class, measured depth, predicted ice probability, Laplace epistemic uncertainty ($\sigma_{\text{prob}}$), in-sample depth, and LOOCV depth for all 26 benchmark sites. |
| **`SourceData_Fig3_ROC.csv`** | Figure 3a | FPR continuum, in-sample TPR (AUC = 0.931), 5-fold CV mean TPR, standard deviation, and upper/lower $\pm 1\sigma$ envelopes. |
| **`SourceData_Fig4_Transects.csv`** | Figure 4d, 4e | 1D latitudinal profiles ($37.0^\circ\text{–}50.0^\circ\text{N}$) of mean ice probability and burial depth with regional longitudinal standard deviation spread. |
| **`SourceData_Fig4_Uncertainty_Dist.csv`** | Figure 4f | Discrete histogram bin edges and empirical probability density values for regional epistemic uncertainty (35 bins over $\sigma \in [0.03, 0.35]$). |
| **`SourceData_Fig5_Scorecard.csv`** | Figure 5c | MCDA scorecard evaluation metrics for Candidate Sites #1–#5 across five normalized engineering feasibility dimensions. |
| **`SourceData_Table1_LandingSites.csv`** | Table 1 | Geographic coordinates, elevation, slope, predicted $P_{\text{ice}}$, burial depth with $\pm 1\sigma$ posterior uncertainty, RAI score, and primary site features. |
| **`SourceData_Master.xlsx`** | Figures 3–5, Table 1 | **Consolidated multi-sheet Excel workbook** containing all above source datasets formatted for archival and reproducibility. |

---

## Repository Architecture

```
PI-BIA/
├── LICENSE                                     # MIT License
├── README.md                                   # Comprehensive software and reproduction guide
├── requirements.txt                            # Pinned Python package dependencies
├── run_pipeline.sh                             # Master execution script (Stages 1–6)
├── data/
│   ├── ground_truth/
│   │   ├── fresh_ice_impacts/
│   │   │   ├── dundas_ice_exposing_impacts.csv # 15 fresh ice impact craters (scaling depths)
│   │   │   └── dundas_ice_exposing_impacts.geojson
│   │   ├── dry_impact_controls/
│   │   │   └── control_dry_impacts.csv         # 10 dry control craters (non-detection down to cavity)
│   │   └── phoenix_insitu/
│   │       └── phoenix_trench_ground_truth.json# Phoenix Lander in-situ trench physical anchor
│   ├── processed/
│   │   ├── arcadia_inversion_results.npz       # Regional 2D inversion grids (P_ice, depth, unc, RAI)
│   │   ├── evaluation_metrics.json             # Quantitative model performance and site metrics
│   │   └── cubes/
│   │       └── arcadia_planitia_multimodal_cube.npz  # Calibrated MOLA/GRS multimodal cube (832x1600)
│   ├── raw/
│   │   ├── mola/
│   │   │   ├── megt90n180gb.img / .lbl         # MGS MOLA MEGDR elevation (64 pix/deg northern quadrant)
│   │   │   └── megt90n000eb.img / .lbl         # MGS MOLA MEGDR elevation (16 pix/deg global)
│   │   └── odyssey_grs/
│   │       └── and_04_000_015.dat / .lbl       # 2001 Mars Odyssey GRS epithermal neutron flux
│   ├── shapefiles/                             # IAU Mars 2000 GIS vector layers (.shp, .dbf, .shx, .prj)
│   │   ├── iau_arcadia_planitia_boundary.*     # Official USGS/IAU Arcadia Planitia boundary
│   │   ├── arcadia_planitia_boundary.*         # Regional study corridor boundary polygon
│   │   ├── dundas_fresh_ice_impacts.*          # Benchmark fresh ice impact crater points
│   │   └── dry_impact_controls.*               # Benchmark dry control crater points
│   └── source_data/                            # Open Science Source Data & Calibration Tables
│       ├── SourceData_Fig3_Benchmarks.csv
│       ├── SourceData_Fig3_ROC.csv
│       ├── SourceData_Fig4_Transects.csv
│       ├── SourceData_Fig4_Uncertainty_Dist.csv
│       ├── SourceData_Fig5_Scorecard.csv
│       ├── SourceData_Table1_LandingSites.csv
│       └── SourceData_Master.xlsx              # Consolidated master Excel workbook
├── figures/
│   ├── fig1_study_area_and_shapefile_boundary.png / .pdf
│   ├── fig2_methodology_architecture.drawio / .png / .pdf
│   ├── fig3_empirical_validation_curves.png / .pdf
│   ├── fig4_arcadia_inversion_composite.png / .pdf
│   └── fig5_landing_sites_and_topography.png / .pdf
└── scripts/
    ├── 01_data_acquisition/
    │   ├── download_pipeline.py                # Automated NASA PDS data retrieval & verification
    │   ├── download_planetary_data.sh          # Shell wrapper for remote sensing datasets
    │   └── verify_datasets.py                  # 6-stage validation suite for raw/processed data
    ├── 02_preprocessing/
    │   ├── compile_ground_truth.py             # Ground-truth ingestion and GIS shapefile creation
    │   └── build_processed_cube.py             # Spatial resampling and multimodal cube assembly
    ├── 03_modeling/
    │   └── train_model.py                      # Bayesian physics-constrained inversion & MCDA
    └── 04_visualization/
        ├── fig1_study_area.py                  # Figure 1 renderer
        ├── fig2_methodology_architecture.drawio # Figure 2 diagram source
        ├── fig3_empirical_validation.py        # Figure 3 renderer
        ├── fig4_arcadia_inversion_composite.py # Figure 4 renderer
        ├── fig5_landing_sites_and_topography.py# Figure 5 renderer
        └── export_source_data.py               # Section 10 Source Data generator
```

---

## Data Availability & Ingestion

<div align="justify">

All primary planetary remote sensing datasets are publicly archived by NASA's Planetary Data System (PDS) Geosciences Node:
- **MGS MOLA MEGDR Topography:** Product ID `MGS-M-MOLA-5-MEGDR-L3-V1.0`, available at [PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html).
- **2001 Mars Odyssey GRS/HEND Epithermal Neutrons:** Product ID `ODY-M-GRS-5-AND-V2.0` (Averaged Neutron Data tables), available at [PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/odyssey/grs.html).
- **Empirical Ground-Truth Catalogs:** Compiled from peer-reviewed literature:
  - Dundas et al., *J. Geophys. Res. Planets* **119**, 109–127 (2014); DOI: [10.1002/2013JE004482](https://doi.org/10.1002/2013JE004482)
  - Byrne et al., *Science* **325**, 1674–1676 (2009); DOI: [10.1126/science.1175307](https://doi.org/10.1126/science.1175307)
  - Bramson et al., *Geophys. Res. Lett.* **42**, 6566–6574 (2015); DOI: [10.1002/2015GL064844](https://doi.org/10.1002/2015GL064844)
  - Smith et al., *Science* **325**, 58–61 (2009); DOI: [10.1126/science.1172339](https://doi.org/10.1126/science.1172339)

To download and re-ingest raw PDS products automatically:

</div>

```bash
python3 scripts/01_data_acquisition/download_pipeline.py
```

---

## Citation

<div align="justify">

If you find this work, data, or codebase helpful in your research, please cite:

</div>

```bibtex
@misc{Sreyas2026PIBIA,
  author       = {Sreyas, S. S.},
  title        = {Physics-informed Bayesian inversion of shallow ground ice in Arcadia Planitia on Mars},
  year         = {2026},
  howpublished = {\url{https://github.com/sreyassanker/PI-BIA}},
  note         = {Manuscript submitted for publication}
}
```

---

## License

<div align="justify">

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details. Free for academic, scientific, and commercial use with attribution.

</div>
