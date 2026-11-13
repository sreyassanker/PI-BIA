#!/usr/bin/env python3
"""
download_pipeline.py
====================
NASA Planetary Data System (PDS) orbital observation ingestion pipeline.

Automates downloading calibrated planetary remote sensing data products:
- Mars Global Surveyor (MGS) MOLA MEGDR topography (64 pix/deg & 16 pix/deg)
- 2001 Mars Odyssey GRS/HEND Averaged Neutron Data (AND) epithermal flux tables

Usage:
    python3 scripts/01_data_acquisition/download_pipeline.py
"""

import argparse, os, sys, time, urllib.request, urllib.error, ssl
from typing import Dict, List, Optional

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

DATASET_REGISTRY = {
    "mola_regional_64": {
        "url": "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg064/megt90n180gb.img",
        "dest": "data/raw/mola/megt90n180gb.img",
        "expected_bytes": 132710400,
        "description": "MGS MOLA Regional Topography (64 pix/deg, ~925 m/pix, Arcadia Planitia)"
    },
    "mola_regional_64_lbl": {
        "url": "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg064/megt90n180gb.lbl",
        "dest": "data/raw/mola/megt90n180gb.lbl",
        "expected_bytes": 4820,
        "description": "MGS MOLA Regional Topography PDS3 Label"
    },
    "mola_global_16": {
        "url": "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/megt90n000eb.img",
        "dest": "data/raw/mola/megt90n000eb.img",
        "expected_bytes": 33177600,
        "description": "MGS MOLA Global Topography Grid (16 pix/deg, ~3.7 km/pix)"
    },
    "mola_global_16_lbl": {
        "url": "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/megt90n000eb.lbl",
        "dest": "data/raw/mola/megt90n000eb.lbl",
        "expected_bytes": 4813,
        "description": "MGS MOLA Global Topography PDS3 Label"
    },
    "grs_and_04_000_015": {
        "url": "https://pds-geosciences.wustl.edu/ody/ody-m-grs-5-and-v2/odgn2_xxxx/yr04/yr04_ls000_015/and_04_000_015.dat",
        "dest": "data/raw/odyssey_grs/and_04_000_015.dat",
        "expected_bytes": 114048,
        "description": "Odyssey GRS Epithermal Neutrons (Year 04, Ls 0-15 deg)"
    },
    "grs_and_04_000_015_lbl": {
        "url": "https://pds-geosciences.wustl.edu/ody/ody-m-grs-5-and-v2/odgn2_xxxx/yr04/yr04_ls000_015/and_04_000_015.lbl",
        "dest": "data/raw/odyssey_grs/and_04_000_015.lbl",
        "expected_bytes": 2544,
        "description": "Odyssey GRS PDS3 Label for Ls 0-15 deg"
    }
}

def download_file(url, dest_path, expected_bytes=None, max_retries=5, chunk_size=65536):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        current_size = os.path.getsize(dest_path)
        if expected_bytes and current_size == expected_bytes:
            print(f"[EXISTS] {dest_path} ({current_size:,} bytes). Skipping.")
            return True

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DOWNLOAD] Attempt {attempt}/{max_retries}: {url}")
            req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=30, context=ssl_context) as response:
                total_size = int(response.getheader("Content-Length") or expected_bytes)
                downloaded = 0
                with open(dest_path, "wb") as f_out:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk: break
                        f_out.write(chunk)
                        downloaded += len(chunk)
                final_size = os.path.getsize(dest_path)
                if expected_bytes and final_size != expected_bytes:
                    print(f"[WARNING] Size mismatch: {final_size} vs {expected_bytes}")
                    continue
                print(f"[SUCCESS] {dest_path} ({final_size:,} bytes)")
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"[ERROR] Attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return False

def verify_all_datasets():
    print("\nDATASET VERIFICATION")
    all_ok = True
    for key, info in DATASET_REGISTRY.items():
        path = info["dest"]
        exp = info["expected_bytes"]
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            all_ok = False
        else:
            actual = os.path.getsize(path)
            if actual == exp:
                print(f"  Confirmed: {path} ({actual:,} bytes)")
            else:
                print(f"  Size mismatch: {path}: {actual:,} vs {exp:,}")
                all_ok = False
    return all_ok

def main():
    parser = argparse.ArgumentParser(description="Download NASA PDS planetary datasets.")
    parser.add_argument("--target", choices=["all", "mola", "grs"], default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        all_ok = verify_all_datasets()
        sys.exit(0 if all_ok else 1)

    targets = list(DATASET_REGISTRY.keys()) if args.target == "all" else \
              [k for k in DATASET_REGISTRY if k.startswith(args.target)]
    for key in targets:
        item = DATASET_REGISTRY[key]
        download_file(item["url"], item["dest"], item.get("expected_bytes"))
    all_ok = verify_all_datasets()
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
