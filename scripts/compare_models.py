#!/usr/bin/env python3
"""
Model Comparison Script
模型比较脚本

Produces the final comparison tables (Table 4.8) and figures
(Figs 4.37, 4.38) for the thesis paper from saved evaluation results.
"""

import os
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import format_results_table
from src.visualization import (
    plot_comparison_bars,
    plot_metric_radar,
    plot_metricwise_comparison,
    generate_all_figures,
    set_plot_style,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_thesis_table(results: dict):
    """Print the thesis-formatted performance comparison table (Table 4.8)."""
    print("\n" + "=" * 80)
    print("Table 4.8: All YOLO Models Performance Analysis")
    print("=" * 80)
    print(format_results_table(results))
    print("=" * 80)

    # Also identify the best model
    best_model = None
    best_f1 = 0
    for name, m in results.items():
        if "error" not in m and m.get("f1", 0) > best_f1:
            best_f1 = m["f1"]
            best_model = name

    if best_model:
        print(f"\n★ Best Model (by F1-Score): {best_model} ({best_f1*100:.2f}%)")


def generate_thesis_figures(results: dict, output_dir: str = "docs/figures/thesis"):
    """Generate the key thesis figures from comparison results."""
    set_plot_style()

    # Figure 4.37: YOLO Models Result Comparison (grouped bar chart)
    logger.info("Generating Fig. 4.37: YOLO Models Result Comparison")
    plot_comparison_bars(
        results, output_dir,
        filename="fig_4_37_yolo_models_result_comparison.png",
    )

    # Figure 4.38: YOLO Models Metric-wise Result Comparison
    logger.info("Generating Fig. 4.38: YOLO Models Metric-wise Comparison")
    plot_metricwise_comparison(
        results, output_dir,
        filename="fig_4_38_metricwise_result_comparison.png",
    )

    # Additional: Radar chart (supplementary)
    logger.info("Generating supplementary radar chart")
    plot_metric_radar(
        results, output_dir,
        filename="fig_supplementary_radar.png",
    )

    logger.info(f"All thesis figures saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate thesis comparison tables and figures")
    parser.add_argument(
        "--results", type=str, default="results/evaluation_results.json",
        help="Path to evaluation results JSON",
    )
    parser.add_argument(
        "--output", type=str, default="docs/figures/thesis",
        help="Output directory for thesis figures",
    )
    args = parser.parse_args()

    # Load results
    if not os.path.exists(args.results):
        logger.error(f"Results file not found: {args.results}")
        logger.error("Run the experiment first: python scripts/run_experiment.py")
        sys.exit(1)

    with open(args.results) as f:
        results = json.load(f)

    # Print comparison table
    print_thesis_table(results)

    # Generate thesis figures
    generate_thesis_figures(results, args.output)


if __name__ == "__main__":
    main()
