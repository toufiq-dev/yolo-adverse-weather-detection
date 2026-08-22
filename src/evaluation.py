#!/usr/bin/env python3
"""
Model Evaluation Metrics
模型评估指标计算

Computes precision, recall, F1-score, accuracy, mAP@50, and mAP@50-95
for YOLO model predictions on the DAWN dataset.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Class names from DAWN dataset
DAWN_CLASSES = ["Bicycle", "Bus", "Car", "Motorcycle", "Person", "Train", "Truck"]


def evaluate_model(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.6,
) -> dict:
    """
    Evaluate a trained YOLO model and return all metrics.

    Args:
        model_path: Path to trained model weights (.pt).
        data_yaml: Path to YOLO-format dataset YAML.
        split: Dataset split to evaluate ('val' or 'test').
        imgsz: Input image size.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        dict with metrics: precision, recall, f1, accuracy, mAP50, mAP50_95.
    """
    logger.info(f"Evaluating {model_path} on {split} set")

    model = YOLO(model_path)

    # Run validation/evaluation
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=True,
    )

    # Extract metrics from ultralytics Results object
    metrics = {
        "precision": float(results.box.mp),      # mean precision
        "recall": float(results.box.mr),          # mean recall
        "f1": float(results.box.f1),             # F1 at default threshold
        "mAP50": float(results.box.map50),       # mAP at IoU=0.50
        "mAP50_95": float(results.box.map),      # mAP at IoU=0.50:0.95
    }

    # Compute accuracy: (TP + TN) / (TP + TN + FP + FN)
    # Using confusion matrix values
    tp = results.box.targets.shape[0] if hasattr(results.box, "targets") else 0
    # Approximate accuracy from confusion matrix
    conf_matrix = results.confusion_matrix
    if conf_matrix is not None:
        cm = conf_matrix.matrix
        total = cm.sum()
        correct = np.diag(cm).sum()
        metrics["accuracy"] = float(correct / total) if total > 0 else 0.0
    else:
        # Fallback: compute from precision and recall (rough estimate)
        p, r = metrics["precision"], metrics["recall"]
        metrics["accuracy"] = p * r / (p + r - p * r + 1e-8) if (p + r) > 0 else 0.0

    # Per-class metrics
    per_class = {}
    if hasattr(results, "names") and results.names:
        class_names = results.names
        for cls_idx, cls_name in class_names.items():
            try:
                per_class[cls_name] = {
                    "precision": float(results.box.p[cls_idx]),
                    "recall": float(results.box.r[cls_idx]),
                    "f1": float(results.box.f1[cls_idx]),
                    "ap50": float(results.box.ap50[cls_idx]),
                    "ap50_95": float(results.box.ap[cls_idx]),
                }
            except (IndexError, AttributeError):
                pass
    metrics["per_class"] = per_class

    # Save confusion matrix plot if available
    if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
        try:
            results.confusion_matrix.plot(save_dir=str(Path(model_path).parent / "confusion_matrix"))
        except Exception as e:
            logger.warning(f"Could not save confusion matrix: {e}")

    logger.info(f"Evaluation results: { {k: v for k, v in metrics.items() if k != 'per_class'} }")
    return metrics


def evaluate_all_models(
    model_dir: str,
    data_yaml: str,
    split: str = "test",
    model_names: Optional[list] = None,
) -> dict:
    """
    Evaluate multiple trained models and return comparison dict.

    Args:
        model_dir: Directory containing sub-folders with trained weights.
        data_yaml: Path to dataset YAML.
        split: Dataset split to evaluate.
        model_names: Specific models to evaluate. If None, evaluates all found.

    Returns:
        dict mapping model_name -> metrics dict.
    """
    model_dir = Path(model_dir)
    all_results = {}

    if model_names:
        search_names = model_names
    else:
        # Auto-discover model directories
        search_names = [
            d.name for d in model_dir.iterdir()
            if d.is_dir() and (d / "weights").exists()
        ]

    for name in search_names:
        # Look for best.pt in standard ultralytics output structure
        weight_candidates = [
            model_dir / name / "weights" / "best.pt",
            model_dir / name / "weights" / "last.pt",
        ]

        model_path = None
        for wc in weight_candidates:
            if wc.exists():
                model_path = wc
                break

        if model_path is None:
            logger.warning(f"No weights found for {name} in {model_dir}")
            continue

        logger.info(f"Evaluating {name}...")
        try:
            metrics = evaluate_model(str(model_path), data_yaml, split)
            all_results[name] = metrics
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")
            all_results[name] = {"error": str(e)}

    return all_results


def format_results_table(results: dict) -> str:
    """
    Format evaluation results as a markdown table.

    Args:
        results: dict mapping model_name -> metrics.

    Returns:
        Markdown-formatted table string.
    """
    header = "| Model | Precision | Recall | F1-Score | Accuracy | mAP@50 | mAP@50-95 |"
    sep = "|-------|-----------|--------|----------|----------|--------|-----------|"
    rows = [header, sep]

    for name, m in results.items():
        if "error" in m:
            rows.append(f"| {name} | ERROR | - | - | - | - | - |")
        else:
            rows.append(
                f"| {name} "
                f"| {m['precision']*100:.2f}% "
                f"| {m['recall']*100:.2f}% "
                f"| {m['f1']*100:.2f}% "
                f"| {m['accuracy']*100:.2f}% "
                f"| {m['mAP50']*100:.2f}% "
                f"| {m['mAP50_95']*100:.2f}% |"
            )

    return "\n".join(rows)


def save_results(results: dict, output_path: str):
    """Save evaluation results to JSON."""
    # Convert numpy types to Python floats for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"Results saved to {output_path}")


def main():
    """CLI entry point for evaluation."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate trained YOLO models")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="src/config/dawn_dataset.yaml", help="Dataset YAML")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    metrics = evaluate_model(args.model, args.data, args.split)

    print("\n" + format_results_table({"model": metrics}))

    if args.output:
        save_results(metrics, args.output)


if __name__ == "__main__":
    main()
