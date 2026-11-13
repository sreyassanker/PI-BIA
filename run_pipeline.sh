#!/usr/bin/env bash
# ==============================================================================
# Master Pipeline Execution Script
# "Physics-Informed Bayesian Inversion of Mars Shallow Ground Ice (PI-BIA)"
# Author: Sreyas S. S. (Amity Institute of Geoinformatics and Remote Sensing, Amity University, Noida; sreyassanker001@gmail.com)
# Target: Icarus (Elsevier) Submission
# ==============================================================================

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Step 1: Verify Python environment
echo -e "${CYAN}[Stage 1/6] Checking Python runtime and dependencies...${NC}"
python3 -c "
import sys
required = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'shapefile', 'pandas', 'openpyxl']
missing = []
for pkg in required:
    try: __import__(pkg)
    except ImportError: missing.append(pkg)
if missing:
    print(f'Error: Missing packages {missing}. Please install dependencies via: pip install -r requirements.txt', file=sys.stderr); sys.exit(1)
print(f'Python {sys.version.split()[0]} and all scientific libraries confirmed.')
"

# Step 2: Verify datasets
echo -e "\n${YELLOW}[Stage 2/6] Verifying planetary remote sensing datasets...${NC}"
python3 scripts/01_data_acquisition/verify_datasets.py

# Step 3: Compile ground truth and build data cube
echo -e "\n${YELLOW}[Stage 3/6] Compiling ground truth catalogs and spatial cubes...${NC}"
python3 scripts/02_preprocessing/compile_ground_truth.py
python3 scripts/02_preprocessing/build_processed_cube.py

# Step 4: Train models and compute inversion
echo -e "\n${YELLOW}[Stage 4/6] Training models and computing regional inversion...${NC}"
python3 scripts/03_modeling/train_model.py

# Step 5: Generate figures
echo -e "\n${YELLOW}[Stage 5/6] Rendering publication figures (600 DPI PNG & vector PDF)...${NC}"
for fig in fig1_study_area fig3_empirical_validation fig4_arcadia_inversion_composite fig5_landing_sites_and_topography; do
    python3 "scripts/04_visualization/${fig}.py"
done
# Figure 2 is a hand-maintained draw.io diagram; re-export it only if the CLI is present.
if command -v drawio >/dev/null 2>&1; then
    drawio -x --width 4800 -o figures/fig2_methodology_architecture.png \
        scripts/04_visualization/fig2_methodology_architecture.drawio
    drawio -x -o figures/fig2_methodology_architecture.pdf \
        scripts/04_visualization/fig2_methodology_architecture.drawio
    python3 -c "from PIL import Image; p='figures/fig2_methodology_architecture.png'; img=Image.open(p); img.save(p, dpi=(600, 600))"
else
    echo "  Note: drawio CLI not found — skipping fig2 re-export (edit the .drawio manually)."
fi

# Step 6: Export Open-Science Source Data
echo -e "\n${YELLOW}[Stage 6/6] Exporting publication Source Data tables and master workbook...${NC}"
python3 scripts/04_visualization/export_source_data.py

echo -e "\n${GREEN}================================================================================${NC}"
echo -e "${GREEN}  PIPELINE COMPLETE — ALL DELIVERABLES GENERATED SUCCESSFULLY!        ${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo "Primary Deliverables:"
echo "  • Publication Maps:  ${CYAN}figures/fig1${NC} to ${CYAN}figures/fig5${NC} (600 DPI PNG + vector PDF)"
echo "  • Source Data:       ${CYAN}data/source_data/${NC} (CSVs and master Excel workbook)"
echo "  • Ground Truth GIS:  ${CYAN}data/shapefiles/${NC}"
echo "  • Processed Data:    ${CYAN}data/processed/arcadia_inversion_results.npz${NC}"
echo -e "${GREEN}================================================================================${NC}\n"
