# train_yolo_seg.py
"""
Train a YOLOv8/YOLOv9 instance-segmentation model on ID-card detection
with full MLflow tracking, without requiring a YAML config file.

Requirements:
------------
* ultralytics >= 8.3
* mlflow >= 2.12
* GPU with >= 8GB VRAM recommended
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("YOLO-Seg training with MLflow logging")
    p.add_argument("--model", default="yolov8s-seg.pt", help="Checkpoint or model type")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1024, help="Square img size")
    p.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--experiment", default="id-card-yolo-seg", help="MLflow experiment name")
    p.add_argument("--run-name", default=None, help="Optional MLflow run name")
    p.add_argument("--save-dir", default="runs/seg/idcard", help="Ultralytics output root")
    p.add_argument("--export", action="store_true", help="Export best.pt to ONNX")
    p.add_argument("--dynamic", action="store_true", help="Dynamic ONNX batch dims")
    return p.parse_args()


def log_ultralytics_metrics(metrics: dict[str, Any]) -> None:
    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = v.item()
        mlflow.log_metric(k, v)


def log_model_artifact(pt_path: Path, alias: str) -> None:
    mlflow.log_artifact(str(pt_path), artifact_path=f"models/{alias}")


def main() -> None:
    args = parse_args()
    
    mlflow.set_tracking_uri("/Users/rudy/personnals_projects/id-card-reader2/mlflow")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(args.experiment)
    run_name = args.run_name or f"{Path(args.model).stem}-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "dataset": "inline_coco_dict"
        })

        data_dict = {
            "train": "/Users/rudy/personnals_projects/id-card-reader2/datasets/splited/images/train",
            "val": "/Users/rudy/personnals_projects/id-card-reader2/datasets/splited/images/val",
            "nc": 1,
            "names": ["id_card"]
        }

        model = YOLO(args.model)
        mlflow.set_tag("yolo_version", model._version)

        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as temp_yaml:
            yaml.dump(data_dict, temp_yaml)
            temp_yaml.flush()
            temp_yaml_path = temp_yaml.name

        model.train(
            data=temp_yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(save_dir),
            exist_ok=True,
            verbose=True,
        )

        os.remove(temp_yaml_path)

        metrics = model.val()
        log_ultralytics_metrics(metrics)

        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            log_model_artifact(best_pt, alias="best_pt")
        else:
            print("-> best.pt not found; skipping weight upload")

        if args.export and best_pt.exists():
            onnx_path = best_pt.with_suffix(".onnx")
            export_model = YOLO(best_pt)
            export_model.export(
                format="onnx",
                dynamic=args.dynamic,
                simplify=True,
                imgsz=args.imgsz,
                opset=13,
            )
            if onnx_path.exists():
                log_model_artifact(onnx_path, alias="best_onnx")

        results_json = save_dir / "results.json"
        if results_json.exists():
            mlflow.log_artifact(str(results_json), artifact_path="results")
        else:
            results_yaml = save_dir / "results.yaml"
            if results_yaml.exists():
                with open(results_yaml, "r", encoding="utf-8") as yfp:
                    res_dict = yaml.safe_load(yfp)
                with open(results_json, "w", encoding="utf-8") as jfp:
                    json.dump(res_dict, jfp, indent=2)
                mlflow.log_artifact(str(results_json), artifact_path="results")

        if os.getenv("ARCHIVE_RUN_DIR", "0") == "1":
            archive_dest = Path("mlruns_artifacts") / run_name
            shutil.make_archive(str(archive_dest), "zip", save_dir)
            mlflow.log_artifact(f"{archive_dest}.zip", artifact_path="run_zip")

        print("\u2713 Training complete & MLflow run logged")


if __name__ == "__main__":
    main()
