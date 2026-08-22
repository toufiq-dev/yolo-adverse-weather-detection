#!/usr/bin/env python3
"""
End-to-End Experiment Runner
端到端实验运行脚本

Orchestrates the full pipeline:
1. Data augmentation (weather simulation)
2. Model training (all YOLO variants)
3. Model evaluation (compute metrics)
4. Figure generation (plots for thesis)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation import augment_dataset
from src.training import train_all_models, load_config
from src.evaluation import evaluate_all_models, format_results_table, save_results
from src.visualization import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(args):
    """Execute the full experiment pipeline."""
    config = load_config(args.config)

    # ── Step 1: Data Augmentation ──────────────────────────────────────
    if not args.skip_augment:
        logger.info("=" * 60)
        logger.info("Step 1: Data Augmentation")
        logger.info("=" * 60)

        augment_dataset(
            input_dir=args.dataset_dir,
            output_dir=args.augmented_dir,
            weather_types=args.weather,
            strength=args.augment_strength,
            copies_per_type=args.augment_copies,
        )
    else:
        logger.info("Skipping data augmentation (--skip-augment)")

    # ── Step 2: Model Training ─────────────────────────────────────────
    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("Step 2: Model Training")
        logger.info("=" * 60)

        model_names = args.models.split(",") if args.models else None
        training_results = train_all_models(
            data_yaml=args.data_yaml,
            config_path=args.config,
            model_names=model_names,
            project=args.train_dir,
        )

        logger.info("Training results:")
        for r in training_results:
            logger.info(f"  {r.get('model_name', '?')}: {r.get('results_dir', r.get('error', 'unknown'))}")
    else:
        logger.info("Skipping training (--skip-train)")

    # ── Step 3: Model Evaluation ───────────────────────────────────────
    if not args.skip_eval:
        logger.info("=" * 60)
        logger.info("Step 3: Model Evaluation")
        logger.info("=" * 60)

        eval_results = evaluate_all_models(
            model_dir=args.train_dir,
            data_yaml=args.data_yaml,
            split=args.eval_split,
        )

        # Print results table
        table = format_results_table(eval_results)
        logger.info(f"\n{table}")

        # Save results to JSON
        os.makedirs(args.output_dir, exist_ok=True)
        results_json = os.path.join(args.output_dir, "evaluation_results.json")
        save_results(eval_results, results_json)
    else:
        logger.info("Skipping evaluation (--skip-eval)")
        # Try to load existing results
        results_json = os.path.join(args.output_dir, "evaluation_results.json")
        if os.path.exists(results_json):
            with open(results_json) as f:
                eval_results = json.load(f)
        else:
            logger.error("No results available. Cannot generate figures.")
            return

    # ── Step 4: Figure Generation ──────────────────────────────────────
    if not args.skip_figures:
        logger.info("=" * 60)
        logger.info("Step 4: Figure Generation")
        logger.info("=" * 60)

        fig_dir = os.path.join(args.output_dir, "figures")
        generate_all_figures(eval_results, fig_dir)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run full YOLO experiment pipeline")

    # Paths
    parser.add_argument("--config", type=str, default="src/config/model_config.yaml")
    parser.add_argument("--data-yaml", type=str, default="src/config/dawn_dataset.yaml")
    parser.add_argument("--dataset-dir", type=str, default="data/dawn_dataset",
                        help="Root of YOLO-format dataset")
    parser.add_argument("--augmented-dir", type=str, default="data/dawn_augmented",
                        help="Output for augmented data")
    parser.add_argument("--train-dir", type=str, default="runs/train",
                        help="Output for training runs")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output for evaluation results and figures")

    # Augmentation options
    parser.add_argument("--weather", type=str, nargs="+",
                        default=["fog", "rain", "snow", "sandstorm"])
    parser.add_argument("--augment-strength", type=float, default=0.5)
    parser.add_argument("--augment-copies", type=int, default=1)

    # Training options
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (e.g., yolov8x,yolov11n)")

    # Evaluation options
    parser.add_argument("--eval-split", type=str, default="test", choices=["val", "test"])

    # Skip flags
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
