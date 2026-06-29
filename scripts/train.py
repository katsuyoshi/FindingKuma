#!/usr/bin/env python3
"""
FindingKuma - Fine-tune YOLOv8 for bear detection.

Usage:
  # Download datasets and train
  python3 scripts/train.py --download --train

  # Download only
  python3 scripts/train.py --download

  # Train with existing data
  python3 scripts/train.py --train --data datasets/bear/data.yaml

  # Train with custom epochs
  python3 scripts/train.py --train --epochs 200

  # Evaluate model
  python3 scripts/train.py --eval --model runs/detect/train/weights/best.pt
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_DIR = Path("datasets/bear")


def download_datasets():
    """Download bear datasets from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: Set ROBOFLOW_API_KEY environment variable.")
        print("  Get your free API key at: https://app.roboflow.com/settings/api")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)

    # Download bear datasets from Roboflow Universe
    datasets_config = [
        # (workspace, project, version)
        ("yolov8-testing-suwzp", "bear-dataset", 1),
    ]

    all_train_images = []
    all_val_images = []

    for workspace, project, version in datasets_config:
        print(f"\nDownloading {workspace}/{project} v{version}...")
        try:
            proj = rf.workspace(workspace).project(project)
            ds = proj.version(version).download("yolov8", location=f"datasets/raw/{project}")
            print(f"  Downloaded to: datasets/raw/{project}")

            # Collect paths
            raw_dir = Path(f"datasets/raw/{project}")
            for split_dir in ["train", "valid", "test"]:
                img_dir = raw_dir / split_dir / "images"
                if img_dir.exists():
                    if split_dir in ("train", "test"):
                        all_train_images.append((img_dir, raw_dir / split_dir / "labels"))
                    else:
                        all_val_images.append((img_dir, raw_dir / split_dir / "labels"))
        except Exception as e:
            print(f"  Warning: Failed to download {project}: {e}")

    # Merge into unified dataset
    merge_datasets(all_train_images, all_val_images)


def merge_datasets(train_sources, val_sources):
    """Merge multiple datasets into unified YOLO format."""
    for split, sources in [("train", train_sources), ("val", val_sources)]:
        img_out = DATASET_DIR / split / "images"
        lbl_out = DATASET_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_dir, lbl_dir in sources:
            for img_file in sorted(img_dir.glob("*")):
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    continue

                # Rename to avoid conflicts
                new_name = f"{count:06d}{img_file.suffix}"
                shutil.copy2(img_file, img_out / new_name)
                shutil.copy2(lbl_file, lbl_out / f"{count:06d}.txt")
                count += 1

        print(f"  {split}: {count} images")

    # Write data.yaml
    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"names:\n"
        f"  0: bear\n"
    )
    print(f"\nDataset config: {data_yaml}")


def add_local_images(image_dir: str):
    """Add local annotated images to the dataset.

    Expected structure:
      image_dir/
        images/   (*.jpg)
        labels/   (*.txt in YOLO format: class_id cx cy w h)
    """
    image_dir = Path(image_dir)
    img_dir = image_dir / "images"
    lbl_dir = image_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Error: Expected {img_dir} and {lbl_dir} directories")
        sys.exit(1)

    # Add to training set
    train_img_out = DATASET_DIR / "train" / "images"
    train_lbl_out = DATASET_DIR / "train" / "labels"
    train_img_out.mkdir(parents=True, exist_ok=True)
    train_lbl_out.mkdir(parents=True, exist_ok=True)

    existing = len(list(train_img_out.glob("*")))
    count = 0

    for img_file in sorted(img_dir.glob("*")):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        if not lbl_file.exists():
            continue

        idx = existing + count
        new_name = f"{idx:06d}{img_file.suffix}"
        shutil.copy2(img_file, train_img_out / new_name)
        shutil.copy2(lbl_file, train_lbl_out / f"{idx:06d}.txt")
        count += 1

    print(f"Added {count} local images to training set")


def train(data_yaml: str, epochs: int, imgsz: int, batch: int, model_base: str):
    """Fine-tune YOLOv8 on bear dataset."""
    from ultralytics import YOLO

    print(f"\nTraining YOLOv8 bear detector")
    print(f"  Base model: {model_base}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")

    model = YOLO(model_base)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="bear_detector",
        patience=30,
        save=True,
        plots=True,
    )

    best_model = Path("runs/detect/bear_detector/weights/best.pt")
    if best_model.exists():
        print(f"\nBest model saved to: {best_model}")
        print(f"\nTo use in FindingKuma:")
        print(f"  python3 scripts/detect.py <image> --model {best_model}")
    return results


def evaluate(model_path: str, data_yaml: str):
    """Evaluate trained model."""
    from ultralytics import YOLO

    print(f"\nEvaluating model: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    print(f"\nResults:")
    print(f"  mAP50:    {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")


def main():
    parser = argparse.ArgumentParser(description="FindingKuma - Train bear detector")
    parser.add_argument("--download", action="store_true", help="Download datasets from Roboflow")
    parser.add_argument("--add-local", type=str, help="Add local annotated images (dir with images/ and labels/)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained model")
    parser.add_argument("--data", type=str, default=str(DATASET_DIR / "data.yaml"), help="Dataset YAML path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model or trained model path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    if not any([args.download, args.add_local, args.train, args.eval]):
        parser.print_help()
        sys.exit(1)

    if args.download:
        download_datasets()

    if args.add_local:
        add_local_images(args.add_local)

    if args.train:
        train(args.data, args.epochs, args.imgsz, args.batch, args.model)

    if args.eval:
        evaluate(args.model, args.data)


if __name__ == "__main__":
    main()
