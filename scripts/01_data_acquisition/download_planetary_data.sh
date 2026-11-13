#!/usr/bin/env bash
# ==============================================================================
# Planetary Remote Sensing Data Download Script
# Ingests MGS MOLA MEGDR elevation rasters and Mars Odyssey GRS neutron tables
# directly from the NASA Planetary Data System (PDS) Geosciences Node archives.
# ==============================================================================

set -euo pipefail

mkdir -p data/raw/mola data/raw/odyssey_grs

echo "Downloading NASA PDS remote sensing datasets..."
echo ""

# MOLA Regional 64 pix/deg (Arcadia Planitia)
echo "Downloading MOLA regional grid (megt90n180gb.img, 132.7 MB)..."
curl -L -o data/raw/mola/megt90n180gb.img \
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg064/megt90n180gb.img"
echo "Downloading MOLA regional label (megt90n180gb.lbl)..."
curl -L -o data/raw/mola/megt90n180gb.lbl \
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg064/megt90n180gb.lbl"

# MOLA Global 16 pix/deg
echo "Downloading MOLA global grid (megt90n000eb.img, 33.2 MB)..."
curl -L -o data/raw/mola/megt90n000eb.img \
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/megt90n000eb.img"
echo "Downloading MOLA global label (megt90n000eb.lbl)..."
curl -L -o data/raw/mola/megt90n000eb.lbl \
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/megt90n000eb.lbl"

# Odyssey GRS AND (Year 04, Ls 0-15 deg)
echo "Downloading Odyssey GRS epithermal neutrons (and_04_000_015.dat, 114 kB)..."
curl -L -o data/raw/odyssey_grs/and_04_000_015.dat \
    "https://pds-geosciences.wustl.edu/ody/ody-m-grs-5-and-v2/odgn2_xxxx/yr04/yr04_ls000_015/and_04_000_015.dat"
echo "Downloading Odyssey GRS label (and_04_000_015.lbl)..."
curl -L -o data/raw/odyssey_grs/and_04_000_015.lbl \
    "https://pds-geosciences.wustl.edu/ody/ody-m-grs-5-and-v2/odgn2_xxxx/yr04/yr04_ls000_015/and_04_000_015.lbl"

echo ""
echo "All verified NASA PDS datasets downloaded."
echo "Run verify_datasets.py to confirm file integrity."
