# datasets/scripts/split_coco_to_yolo_seg.py
"""
Split a COCO **instance-segmentation** dataset into **train/val** and convert the
annotations to **YOLOv8 segmentation** format (Ultralytics *segment* task).

Output structure
────────────────
<out-dir>/
├── images/
│   ├── train/  (linked or moved images)
│   └── val/
├── labels/
│   ├── train/  (*.txt – YOLOv8 segmentation)
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json

Each line of a label file:
<class_id> x_center y_center width height x1 y1 x2 y2 … xn yn
(all values **normalisées 0-1** par rapport à la taille de l’image).

Usage
─────
```bash
python split_coco_to_yolo_seg.py \
    --coco-json datasets/coco/annotations/instances_all.json \
    --images-dir datasets/coco/images/all \
    --out-dir datasets/coco \
    --val-ratio 0.2
```

Requires
────────
* **pycocotools**
* Python ≥ 3.9
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Final, List

from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser("Split COCO dataset and convert to YOLOv8 segmentation format")
    p.add_argument("--coco-json", default="datasets/coco/midv500_coco.json", help="Path to the COCO annotation file")
    p.add_argument("--images-dir", default="datasets/raw/", help="Directory containing all images")
    p.add_argument("--out-dir", default="datasets/clean_seg", help="Root directory of the output dataset")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of images to use for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--move", action="store_true", help="Move files instead of hard-linking (slower)")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _link(src: Path, dst: Path, move: bool = False) -> None:
    """Hard-link (or move/copy) *src* → *dst*, creating parent directories."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if move:
        shutil.move(src, dst)
    else:
        try:
            dst.hardlink_to(src)
        except Exception:
            shutil.copy(src, dst)


def _flatten(poly: List) -> List[float]:
    """Flatte la liste potentiellement imbriquée des points de polygone."""
    out: List[float] = []
    for el in poly:
        if isinstance(el, list):
            out.extend(_flatten(el))
        else:
            out.append(float(el))
    return out


def convert_annotations_to_yolo_seg(coco_data: dict, out_root: Path, split: str) -> None:
    """Écrit les fichiers labels/<split>/*.txt au format YOLOv8 segmentation."""
    labels_dir = out_root / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Map image id → (file_name, width, height)
    img_meta = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco_data["images"]}

    # Grouper les annotations par image
    anns_by_image: dict[int, list] = {}
    for ann in coco_data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, anns in anns_by_image.items():
        file_name, w, h = img_meta[img_id]
        label_path = labels_dir / (Path(file_name).stem + ".txt")
        lines: List[str] = []
        for ann in anns:
            # Skip annotations sans segmentation ou en mode RLE
            seg = ann.get("segmentation")
            if not seg or isinstance(seg, dict):
                continue  # RLE (crowd) non supporté ici

            # Utilise bbox pour centroid/width/height
            bx, by, bw, bh = ann["bbox"]
            xc = (bx + bw / 2) / w
            yc = (by + bh / 2) / h
            bw_n = bw / w
            bh_n = bh / h

            # Concaténer tous les polygones de l'objet
            coords: List[float] = []
            for poly in seg:  # seg est List[List[float]]
                flat = _flatten(poly)
                # Normalisation des points
                for i, val in enumerate(flat):
                    flat[i] = val / w if i % 2 == 0 else val / h  # x / w  ou  y / h
                coords.extend(flat)

            if not coords:
                continue  # safety

            line = f"{ann['category_id']-1} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f} " + " ".join(f"{c:.6f}" for c in coords)
            lines.append(line)

        if lines:
            label_path.write_text("\n".join(lines))

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901
    args = parse_args()
    random.seed(args.seed)

    coco_path = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    out_root = Path(args.out_dir)
    ann_dir = out_root / "annotations"
    img_train_dir = out_root / "images" / "train"
    img_val_dir = out_root / "images" / "val"

    # 1. Load full COCO JSON
    with open(coco_path, "r", encoding="utf-8") as fp:
        coco = json.load(fp)

    image_ids = [img["id"] for img in coco["images"]]
    random.shuffle(image_ids)
    val_count = int(len(image_ids) * args.val_ratio)
    val_ids = set(image_ids[:val_count])

    # 2. Split images & annotations dicts
    coco_train = {k: [] if isinstance(v, list) else v for k, v in coco.items()}
    coco_val = {k: [] if isinstance(v, list) else v for k, v in coco.items()}

    def _copy_img_dict(img: dict, new_id: int):
        out = img.copy(); out["id"] = new_id; return out

    img_id_map_train: Final[dict[int, int]] = {}
    img_id_map_val: Final[dict[int, int]] = {}
    next_train_id = next_val_id = 1

    for img in coco["images"]:
        if img["id"] in val_ids:
            img_id_map_val[img["id"]] = next_val_id
            coco_val["images"].append(_copy_img_dict(img, next_val_id))
            next_val_id += 1
        else:
            img_id_map_train[img["id"]] = next_train_id
            coco_train["images"].append(_copy_img_dict(img, next_train_id))
            next_train_id += 1

    next_ann_train = next_ann_val = 1
    for ann in coco["annotations"]:
        if ann["image_id"] in val_ids:
            ann_val = ann.copy()
            ann_val["image_id"] = img_id_map_val[ann["image_id"]]
            ann_val["id"] = next_ann_val
            coco_val["annotations"].append(ann_val)
            next_ann_val += 1
        else:
            ann_tr = ann.copy()
            ann_tr["image_id"] = img_id_map_train[ann["image_id"]]
            ann_tr["id"] = next_ann_train
            coco_train["annotations"].append(ann_tr)
            next_ann_train += 1

    # 3. Save new JSONs
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "instances_train.json").write_text(json.dumps(coco_train))
    (ann_dir / "instances_val.json").write_text(json.dumps(coco_val))

    # 4. Link or move images
    print("Linking images …")
    for img in tqdm(coco["images"]):
        src = images_dir / img["file_name"]
        dst_root = img_val_dir if img["id"] in val_ids else img_train_dir
        _link(src, dst_root / src.name, move=args.move)

    # 5. Generate YOLOv8 segmentation label files
    convert_annotations_to_yolo_seg(coco_train, out_root, "train")
    convert_annotations_to_yolo_seg(coco_val, out_root, "val")

    print("✅ Split & YOLOv8 segmentation conversion done!")
    print(f"Train imgs: {len(coco_train['images'])} – Val imgs: {len(coco_val['images'])}")
    print("Labels stored in:", out_root / "labels")


if __name__ == "__main__":
    main()