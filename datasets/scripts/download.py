#!/usr/bin/env python3
"""
download_midv.py ─ Fetch MIDV-500, MIDV-2019 et MIDV-2020
and convert it into COCO with fcakyon/midv500.

Dossier résultat :
    datasets/
        ├── raw/   <- raw datasets
        └── coco/  <- annotations COCO + images
"""

from pathlib import Path
import midv500 

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
COCO_DIR = Path(__file__).resolve().parent.parent / "coco"
RAW_DIR.mkdir(parents=True, exist_ok=True)
COCO_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["midv500", "midv-2019", "midv-2020"]

for name in DATASETS:
    print(f"\n=== {name.upper()} ===")
    midv500.download_dataset(str(RAW_DIR), name)
    midv500.convert_to_coco(str(RAW_DIR), str(COCO_DIR), name)
    print(f"{name} Done !")

print(f"\n  MIDV* downloaded under {RAW_DIR}\n"
    f"Coco annotations under {COCO_DIR}\n")
