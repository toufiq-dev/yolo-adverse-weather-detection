#!/usr/bin/env python3
"""
YOLO Model Training Pipeline
用于YOLO模型训练的完整流程

Supports training YOLOv8x, YOLOv11n/s/m/l/x, and YOLOv12x
on the DAWN dataset for adverse weather object detection.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Supported model variants
MODEL_VARIANTS = {
    "yolov8x": "yolov8x.pt",
    "yolov11n": "yolo11n.pt",
    "yolov11s": "yolo11s.pt",
    "yolov11m": "yolo11m.pt",
    "yolov11l": "yolo11l.pt",
    "yolov11x": "yolo11x.pt",
    "yolov12x": "yolo12x.pt",
}


def load_config(config_path: str = "src/config/model_config.yaml") -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(
    model_name: str,
    data_yaml: str,
    config: Optional[dict] = None,
    project: str = "runs/train",
    experiment_name: Optional[str] = None,
) -> dict:
    """
    Train a YOLO model variant.

    Args:
        model_name: Key from MODEL_VARIANTS (e.g., 'yolov8x', 'yolov11n').
        data_yaml: Path to YOLO-format dataset YAML.
        config: Training config dict. If None, loads from default config.
        project: Base directory for saving training runs.
        experiment_name: Name for this training run directory.

    Returns:
        dict with training results summary.
    """
    if model_name not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_VARIANTS.keys())}"
        )

    if config is None:
        config = load_config()

    training_cfg = config.get("training", {})

    # Resolve pretrained weight
    weights = MODEL_VARIANTS[model_name]
    exp_name = experiment_name or model_name

    logger.info(f"Training {model_name} with weights={weights}")

    # Load model
    model = YOLO(weights)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=training_cfg.get("epochs", 120),
        batch=training_cfg.get("batch_size", 4),
        imgsz=training_cfg.get("image_size", 640),
        lr0=training_cfg.get("learning_rate", 0.000909),
        optimizer=training_cfg.get("optimizer", "AdamW"),
        project=project,
        name=exp_name,
        exist_ok=True,
        patience=30,
        save=True,
        verbose=True,
    )

    summary = {
        "model_name": model_name,
        "weights": weights,
        "epochs": training_cfg.get("epochs", 120),
        "results_dir": os.path.join(project, exp_name),
    }

    logger.info(f"Training complete: {summary}")
    return summary


def train_all_models(
    data_yaml: str,
    config_path: str = "src/config/model_config.yaml",
    model_names: Optional[list] = None,
    project: str = "runs/train",
) -> list:
    """
    Train all (or selected) YOLO model variants sequentially.

    Args:
        data_yaml: Path to YOLO-format dataset YAML.
        config_path: Path to model config YAML.
        model_names: List of model names to train. If None, trains all.
        project: Base directory for saving runs.

    Returns:
        List of training result dicts.
    """
    config = load_config(config_path)
    names = model_names or list(MODEL_VARIANTS.keys())
    results = []

    for name in names:
        logger.info(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
        try:
            result = train_model(name, data_yaml, config, project)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            results.append({"model_name": name, "error": str(e)})

    return results


def main():
    """CLI entry point for training."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train YOLO models on DAWN dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="src/config/dawn_dataset.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to train (e.g., yolov8x). If omitted, trains all.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Output directory for training runs",
    )
    args = parser.parse_args()

    if args.model:
        results = train_all_models(
            args.data, args.config, [args.model], args.project
        )
    else:
        results = train_all_models(args.data, args.config, project=args.project)

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['model_name']}: FAILED - {r['error']}")
        else:
            print(f"  {r['model_name']}: OK -> {r['results_dir']}")


if __name__ == "__main__":
    main()
