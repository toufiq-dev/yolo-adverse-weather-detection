#!/usr/bin/env python3
"""
DAWN Dataset Downloader
DAWN数据集下载脚本

Downloads the DAWN (Detection in Adverse Weather Nature) dataset from
Roboflow Universe in YOLO format.

Usage:
    python scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

    Or set the environment variable:
    export ROBOFLOW_API_KEY=your_key_here
    python scripts/download_dataset.py
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "dawn_dataset"

# Roboflow workspace/project for the DAWN dataset
# Multiple DAWN datasets exist on Roboflow; this one matches the thesis
# (7 classes: Bicycle, Bus, Car, Motorcycle, Person, Train, Truck)
ROBOFLOW_WORKSPACE = "louisa"
ROBOFLOW_PROJECT = "dawn-weather-detection"
ROBOFLOW_VERSION = 1  # Latest version


def download_with_roboflow(api_key: str, output_dir: Path, format: str = "yolov8"):
    """
    Download dataset using the roboflow Python package.

    Args:
        api_key: Roboflow API key.
        output_dir: Where to save the dataset.
        format: Export format (yolov8 for YOLOv5/v8/v11 compatible).
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error(
            "roboflow package not installed. Install it with:\n"
            "  pip install roboflow"
        )
        sys.exit(1)

    logger.info(f"Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    logger.info(f"Accessing workspace: {ROBOFLOW_WORKSPACE}")
    workspace = rf.workspace(ROBOFLOW_WORKSPACE)

    logger.info(f"Accessing project: {ROBOFLOW_PROJECT}")
    project = workspace.project(ROBOFLOW_PROJECT)

    # List available versions
    versions = project.versions()
    logger.info(f"Available versions: {[v.version for v in versions]}")

    # Download the specified version
    version = project.version(ROBOFLOW_VERSION)
    logger.info(f"Downloading version {ROBOFLOW_VERSION} in {format} format...")

    dataset = version.download(format)
    logger.info(f"Downloaded to: {dataset.location}")

    return dataset.location


def restructure_dataset(download_dir: str, output_dir: Path):
    """
    Restructure the downloaded dataset into clean YOLO format.

    Roboflow exports typically have:
        download_dir/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/ (or val/)
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── data.yaml

    We restructure to:
        output_dir/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── data.yaml
    """
    download_path = Path(download_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find the split directories (train, valid/val, test)
    split_map = {}
    for d in download_path.iterdir():
        if d.is_dir():
            name = d.name.lower()
            if name in ("train", "training"):
                split_map["train"] = d
            elif name in ("valid", "val", "validation"):
                split_map["valid"] = d
            elif name in ("test", "testing"):
                split_map["test"] = d

    logger.info(f"Found splits: {list(split_map.keys())}")

    # Copy each split
    for split_name, split_dir in split_map.items():
        out_split = output_path / split_name
        out_split.mkdir(exist_ok=True)

        # Handle both nested (images/, labels/) and flat structures
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if images_dir.exists() and labels_dir.exists():
            # Nested structure - copy as-is
            out_images = out_split / "images"
            out_labels = out_split / "labels"
            if out_images.exists():
                shutil.rmtree(out_images)
            if out_labels.exists():
                shutil.rmtree(out_labels)
            shutil.copytree(images_dir, out_images)
            shutil.copytree(labels_dir, out_labels)
        else:
            # Flat structure - separate images and labels
            out_images = out_split / "images"
            out_labels = out_split / "labels"
            out_images.mkdir(exist_ok=True)
            out_labels.mkdir(exist_ok=True)

            for f in split_dir.iterdir():
                if f.is_file():
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        shutil.copy2(f, out_images / f.name)
                    elif f.suffix.lower() == ".txt":
                        shutil.copy2(f, out_labels / f.name)

        # Count files
        n_images = len(list((out_split / "images").glob("*"))) if (out_split / "images").exists() else 0
        n_labels = len(list((out_split / "labels").glob("*"))) if (out_split / "labels").exists() else 0
        logger.info(f"  {split_name}: {n_images} images, {n_labels} labels")

    # Copy data.yaml
    yaml_files = list(download_path.glob("*.yaml")) + list(download_path.glob("*.yml"))
    if yaml_files:
        shutil.copy2(yaml_files[0], output_path / "data.yaml")
        logger.info(f"Copied dataset YAML: {yaml_files[0].name}")

    logger.info(f"Dataset restructured to: {output_path}")


def create_test_split(output_dir: Path, test_ratio: float = 0.1):
    """
    Create a test split if one doesn't exist.
    Uses the 70/20/10 rule from train + valid data.
    """
    test_dir = output_dir / "test"
    if test_dir.exists() and list((test_dir / "images").glob("*")):
        logger.info("Test split already exists, skipping.")
        return

    logger.info(f"Creating test split ({test_ratio*100:.0f}% of data)...")

    import random

    # Collect all images from train and valid
    all_images = []
    for split in ["train", "valid"]:
        img_dir = output_dir / split / "images"
        if img_dir.exists():
            for img in img_dir.glob("*"):
                all_images.append((split, img))

    if not all_images:
        logger.error("No images found to create test split from.")
        return

    # Random sample for test
    random.seed(42)
    n_test = max(1, int(len(all_images) * test_ratio))
    test_images = random.sample(all_images, n_test)

    # Create test directories
    test_images_dir = output_dir / "test" / "images"
    test_labels_dir = output_dir / "test" / "labels"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for split, img_path in test_images:
        # Copy image
        shutil.copy2(img_path, test_images_dir / img_path.name)

        # Copy corresponding label
        label_path = output_dir / split / "labels" / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, test_labels_dir / (img_path.stem + ".txt"))
            moved += 1

    logger.info(f"Created test split: {moved} images with labels")


def print_dataset_summary(output_dir: Path):
    """Print a summary of the dataset structure."""
    print("\n" + "=" * 60)
    print("DAWN Dataset Summary")
    print("=" * 60)

    total_images = 0
    total_labels = 0

    for split in ["train", "valid", "test"]:
        split_dir = output_dir / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        total_images += n_img
        total_labels += n_lbl

        status = "✅" if n_img > 0 else "❌"
        print(f"  {status} {split:8s}: {n_img:5d} images, {n_lbl:5d} labels")

    print(f"  {'─' * 40}")
    print(f"  Total:    {total_images:5d} images, {total_labels:5d} labels")

    # Check class distribution if labels exist
    class_counts = {}
    class_names = {0: "Bicycle", 1: "Bus", 2: "Car", 3: "Motorcycle",
                   4: "Person", 5: "Train", 6: "Truck"}

    for split in ["train", "valid", "test"]:
        lbl_dir = output_dir / split / "labels"
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    if class_counts:
        print(f"\n  Class Distribution:")
        for cls_id in sorted(class_counts.keys()):
            name = class_names.get(cls_id, f"Class_{cls_id}")
            print(f"    {name:12s}: {class_counts[cls_id]:6d} instances")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download DAWN dataset from Roboflow")
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--create-test", action="store_true", default=True,
        help="Create test split if missing (default: True)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="Ratio of data for test split (default: 0.1)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, just restructure existing files",
    )
    parser.add_argument(
        "--download-dir", type=str, default=None,
        help="Directory with already-downloaded Roboflow export (with --skip-download)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    if not args.skip_download:
        # Get API key
        api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            logger.error(
                "No API key provided. Either:\n"
                "  1. Use --api-key YOUR_KEY\n"
                "  2. Set ROBOFLOW_API_KEY environment variable\n"
                "\nGet a free key at: https://app.roboflow.com/settings/api"
            )
            sys.exit(1)

        # Download
        download_dir = download_with_roboflow(api_key, output_dir)
    else:
        if not args.download_dir:
            logger.error("--download-dir required when using --skip-download")
            sys.exit(1)
        download_dir = args.download_dir

    # Restructure
    restructure_dataset(download_dir, output_dir)

    # Create test split if needed
    if args.create_test:
        create_test_split(output_dir, args.test_ratio)

    # Print summary
    print_dataset_summary(output_dir)

    logger.info(f"\nDataset ready at: {output_dir}")
    logger.info("You can now train models with:")
    logger.info(f"  python scripts/run_experiment.py --dataset-dir {output_dir}")


if __name__ == "__main__":
    main()
