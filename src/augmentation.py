#!/usr/bin/env python3
"""
Weather-Specific Data Augmentation
天气特定的数据增强

Implements fog, rain, snow, and sandstorm augmentation using Albumentations,
preserving YOLO bounding box annotations.
"""

import os
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
from albumentations.core.bbox_utils import BboxParams

logger = logging.getLogger(__name__)

# Albumentations transform definitions for weather-specific augmentation


def get_fog_transform(strength: float = 0.5) -> A.Compose:
    """
    Simulate fog effect using Gaussian blur + brightness reduction + white overlay.

    Args:
        strength: Fog intensity (0.0 to 1.0).

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=strength, alpha_coef=0.3, p=1.0),
    ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))


def get_rain_transform(strength: float = 0.5) -> A.Compose:
    """
    Simulate rain effect using motion blur + noise.

    Args:
        strength: Rain intensity (0.0 to 1.0).

    Returns:
        Albumentations Compose transform.
    """
    slant_lower = int(-20 * strength)
    slant_upper = int(20 * strength)
    return A.Compose([
        A.RandomRain(
            slant_lower=slant_lower,
            slant_upper=slant_upper,
            drop_length=int(20 * strength),
            drop_width=1,
            drop_color=(200, 200, 200),
            p=1.0,
        ),
    ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))


def get_snow_transform(strength: float = 0.5) -> A.Compose:
    """
    Simulate snow effect.

    Args:
        strength: Snow intensity (0.0 to 1.0).

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.RandomSnow(
            snow_point_lower=0.1,
            snow_point_upper=strength * 0.5,
            brightness_coeff=2.0,
            p=1.0,
        ),
    ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))


def get_sandstorm_transform(strength: float = 0.5) -> A.Compose:
    """
    Simulate sandstorm using pixel dust + color shift + noise.

    Args:
        strength: Sandstorm intensity (0.0 to 1.0).

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.RandomGamma(gamma_limit=(int(60 - 20 * strength), int(60 + 20 * strength)), p=0.8),
        A.GaussNoise(var_limit=(10.0 * strength, 50.0 * strength), p=0.9),
        A.ColorJitter(brightness=0.3 * strength, contrast=0.2 * strength,
                       saturation=0.3 * strength, hue=0.1, p=0.9),
        A.ToGray(p=0.2 * strength),
    ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))


WEATHER_TRANSFORMS = {
    "fog": get_fog_transform,
    "rain": get_rain_transform,
    "snow": get_snow_transform,
    "sandstorm": get_sandstorm_transform,
}


def load_yolo_labels(label_path: str) -> list:
    """
    Load YOLO-format bounding box labels from a text file.

    Returns:
        List of (class_id, x_center, y_center, w, h) tuples.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                coords = list(map(float, parts[1:5]))
                boxes.append([cls] + coords)
    return boxes


def save_yolo_labels(labels: list, output_path: str):
    """Save YOLO-format labels."""
    with open(output_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")


def augment_dataset(
    input_dir: str,
    output_dir: str,
    weather_types: Optional[list] = None,
    strength: float = 0.5,
    copies_per_type: int = 1,
    seed: int = 42,
):
    """
    Augment a YOLO dataset with weather-specific effects.

    Args:
        input_dir: Root of YOLO dataset (contains images/ and labels/).
        output_dir: Output directory for augmented dataset.
        weather_types: List of weather types to apply. Default: all.
        strength: Augmentation strength (0.0 to 1.0).
        copies_per_type: Number of augmented copies per weather type.
        seed: Random seed for reproducibility.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    weather_types = weather_types or list(WEATHER_TRANSFORMS.keys())

    images_in = input_dir / "images"
    labels_in = input_dir / "labels"

    if not images_in.exists():
        raise FileNotFoundError(f"Images directory not found: {images_in}")

    # Create output directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [f for f in images_in.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    logger.info(f"Augmenting {len(image_files)} images with weather: {weather_types}")

    for img_path in image_files:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Could not read: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load corresponding labels
        label_path = labels_in / (img_path.stem + ".txt")
        raw_labels = load_yolo_labels(str(label_path))

        # Separate class IDs from coordinates
        class_labels = [int(l[0]) for l in raw_labels]
        bboxes = [l[1:5] for l in raw_labels]

        for weather in weather_types:
            transform_fn = WEATHER_TRANSFORMS[weather]
            transform = transform_fn(strength=strength)

            for copy_idx in range(copies_per_type):
                try:
                    augmented = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels,
                    )

                    aug_image = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_classes = augmented["class_labels"]

                    # Save augmented image
                    out_name = f"{img_path.stem}_{weather}_c{copy_idx}{img_path.suffix}"
                    out_img_path = output_dir / "images" / out_name
                    cv2.imwrite(
                        str(out_img_path),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                    )

                    # Save augmented labels
                    out_label_path = output_dir / "labels" / (out_name.rsplit(".", 1)[0] + ".txt")
                    new_labels = []
                    for cls, bbox in zip(aug_classes, aug_bboxes):
                        new_labels.append([cls] + list(bbox))
                    save_yolo_labels(new_labels, str(out_label_path))

                except Exception as e:
                    logger.warning(f"Augmentation failed for {img_path.name} ({weather}): {e}")

    logger.info(f"Augmentation complete. Output: {output_dir}")


def create_augmented_dataset_yaml(
    original_yaml: str,
    augmented_dirs: list,
    output_yaml: str,
):
    """
    Create a new dataset YAML that includes augmented data paths.

    Args:
        original_yaml: Path to original dataset YAML.
        augmented_dirs: List of augmented dataset directories.
        output_yaml: Path to write new YAML.
    """
    with open(original_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Append augmented paths to existing train/val/test paths
    for split_key in ["train", "val", "test"]:
        if split_key in config and isinstance(config[split_key], str):
            original_path = config[split_key]
            augmented_paths = [os.path.join(d, split_key) for d in augmented_dirs]
            config[split_key] = [original_path] + augmented_paths
        elif split_key in config and isinstance(config[split_key], list):
            for d in augmented_dirs:
                config[split_key].append(os.path.join(d, split_key))

    with open(output_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Augmented dataset YAML saved to {output_yaml}")


def main():
    """CLI entry point for data augmentation."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Augment DAWN dataset with weather effects")
    parser.add_argument("--input", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--weather", type=str, nargs="+", default=["fog", "rain", "snow", "sandstorm"],
        help="Weather types to simulate",
    )
    parser.add_argument("--strength", type=float, default=0.5, help="Augmentation strength (0-1)")
    parser.add_argument("--copies", type=int, default=1, help="Copies per weather type")
    args = parser.parse_args()

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        weather_types=args.weather,
        strength=args.strength,
        copies_per_type=args.copies,
    )


if __name__ == "__main__":
    main()
