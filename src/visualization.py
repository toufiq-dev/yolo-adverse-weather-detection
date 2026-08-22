#!/usr/bin/env python3
"""
Visualization Module
可视化模块

Generates performance comparison plots, per-model metric graphs,
and the final comparison figures used in the thesis.
"""

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Thesis color palette
MODEL_COLORS = {
    "YOLOv11n": "#2196F3",
    "YOLOv11s": "#4CAF50",
    "YOLOv11m": "#FF9800",
    "YOLOv11l": "#9C27B0",
    "YOLOv11x": "#F44336",
    "YOLOv12x": "#00BCD4",
    "YOLOv8x":  "#E91E63",
}

METRICS = ["precision", "recall", "f1", "accuracy", "mAP50", "mAP50_95"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "accuracy": "Accuracy",
    "mAP50": "mAP@50",
    "mAP50_95": "mAP@50-95",
}


def set_plot_style():
    """Set consistent plot style for thesis figures."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "grid.alpha": 0.3,
    })
    sns.set_style("whitegrid")


def plot_model_metrics(
    model_name: str,
    metrics: dict,
    output_dir: str,
):
    """
    Plot per-model performance metrics as a bar chart.

    Args:
        model_name: Name of the model.
        metrics: Metrics dict with keys from METRICS.
        output_dir: Directory to save the plot.
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    metric_keys = [m for m in METRICS if m in metrics]
    values = [metrics[m] * 100 for m in metric_keys]
    labels = [METRIC_LABELS.get(m, m) for m in metric_keys]

    color = MODEL_COLORS.get(model_name, "#666666")
    bars = ax.bar(labels, values, color=color, alpha=0.85, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Performance Metrics for {model_name}")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_metrics.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_comparison_bars(
    all_results: dict,
    output_dir: str,
    filename: str = "models_comparison.png",
):
    """
    Create a grouped bar chart comparing all models across all metrics.

    Args:
        all_results: dict mapping model_name -> metrics dict.
        output_dir: Directory to save the plot.
        filename: Output filename.
    """
    set_plot_style()

    models = list(all_results.keys())
    metric_keys = [m for m in METRICS if m in all_results.get(models[0], {})]
    n_metrics = len(metric_keys)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, model_name in enumerate(models):
        m = all_results[model_name]
        values = [m.get(metric, 0) * 100 for metric in metric_keys]
        color = MODEL_COLORS.get(model_name, "#666666")
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_keys])
    ax.set_ylabel("Score (%)")
    ax.set_title("YOLO Models Performance Comparison")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_metric_radar(
    all_results: dict,
    output_dir: str,
    filename: str = "models_radar.png",
):
    """
    Create a radar/spider chart comparing all models.

    Args:
        all_results: dict mapping model_name -> metrics dict.
        output_dir: Directory to save the plot.
        filename: Output filename.
    """
    set_plot_style()

    models = list(all_results.keys())
    metric_keys = [m for m in METRICS if m in all_results.get(models[0], {})]
    n_metrics = len(metric_keys)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for model_name in models:
        m = all_results[model_name]
        values = [m.get(metric, 0) * 100 for metric in metric_keys]
        values += values[:1]  # close polygon
        color = MODEL_COLORS.get(model_name, "#666666")
        ax.plot(angles, values, "o-", label=model_name, color=color, linewidth=2, markersize=6)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_keys], fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Model Performance Radar Chart", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_metricwise_comparison(
    all_results: dict,
    output_dir: str,
    filename: str = "metricwise_comparison.png",
):
    """
    Create a horizontal bar chart with each metric as a separate subplot.

    Args:
        all_results: dict mapping model_name -> metrics dict.
        output_dir: Directory to save.
        filename: Output filename.
    """
    set_plot_style()

    models = list(all_results.keys())
    metric_keys = [m for m in METRICS if m in all_results.get(models[0], {})]
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6), sharey=True)

    for idx, metric in enumerate(metric_keys):
        ax = axes[idx]
        values = [all_results[mdl].get(metric, 0) * 100 for mdl in models]
        colors = [MODEL_COLORS.get(mdl, "#666666") for mdl in models]

        # Sort by value
        sorted_pairs = sorted(zip(models, values, colors), key=lambda x: x[1])
        s_models, s_values, s_colors = zip(*sorted_pairs)

        ax.barh(s_models, s_values, color=s_colors, alpha=0.85)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12)
        ax.set_xlim(0, 105)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        for i, v in enumerate(s_values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    fig.suptitle("Metric-wise Model Comparison", fontsize=16, y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_training_curves(
    results_csv: str,
    model_name: str,
    output_dir: str,
):
    """
    Plot training and validation loss curves from an ultralytics results CSV.

    Args:
        results_csv: Path to results.csv from training run.
        model_name: Name of the model.
        output_dir: Directory to save plots.
    """
    set_plot_style()

    import pandas as pd

    df = pd.read_csv(results_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training & Validation box loss
    if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
        axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="Train", color="#2196F3")
        axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="Val", color="#F44336")
        axes[0, 0].set_title("Box Loss")
        axes[0, 0].legend()

    # Training & Validation cls loss
    if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
        axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="Train", color="#2196F3")
        axes[0, 1].plot(df["epoch"], df["val/cls_loss"], label="Val", color="#F44336")
        axes[0, 1].set_title("Classification Loss")
        axes[0, 1].legend()

    # mAP@50
    map_col = "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else "map50"
    if map_col in df.columns:
        axes[1, 0].plot(df["epoch"], df[map_col] * 100, color="#4CAF50", linewidth=2)
        axes[1, 0].set_title("mAP@50")
        axes[1, 0].set_ylabel("%")

    # Precision & Recall
    prec_col = "metrics/precision(B)" if "metrics/precision(B)" in df.columns else "precision"
    rec_col = "metrics/recall(B)" if "metrics/recall(B)" in df.columns else "recall"
    if prec_col in df.columns:
        axes[1, 1].plot(df["epoch"], df[prec_col] * 100, label="Precision", color="#2196F3")
    if rec_col in df.columns:
        axes[1, 1].plot(df["epoch"], df[rec_col] * 100, label="Recall", color="#FF9800")
    axes[1, 1].set_title("Precision & Recall")
    axes[1, 1].set_ylabel("%")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves: {model_name}", fontsize=16, y=1.01)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def generate_all_figures(
    results: dict,
    output_dir: str = "docs/figures/generated",
):
    """
    Generate all thesis figures from evaluation results.

    Args:
        results: dict mapping model_name -> metrics dict.
        output_dir: Directory for all generated figures.
    """
    set_plot_style()

    os.makedirs(output_dir, exist_ok=True)

    # Per-model metric bar charts
    for model_name, metrics in results.items():
        plot_model_metrics(model_name, metrics, output_dir)

    # Grouped comparison bar chart
    plot_comparison_bars(results, output_dir)

    # Radar chart
    plot_metric_radar(results, output_dir)

    # Metric-wise horizontal bar chart
    plot_metricwise_comparison(results, output_dir)

    logger.info(f"All figures generated in {output_dir}")


def main():
    """CLI entry point for generating figures."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate thesis performance figures")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output", type=str, default="docs/figures/generated")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    generate_all_figures(results, args.output)
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()
