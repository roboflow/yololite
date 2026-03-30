#!/usr/bin/env python3
"""Visualize SAB benchmark results.

Generates four plot types from SAB per-variant CSVs:

1. Heatmap of mAP per dataset per variant (one per metric, per engine)
2. Median mAP vs latency Pareto curves (3x2 grid: engines x metrics,
   with separate curves for standard vs edge variants)
3. Distribution of mAP@50 across datasets (box plots, per variant per engine)
4. Distribution of latency across datasets (box plots, per variant per engine)
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yololite.benchmark._variants import (
    EDGE_VARIANTS,
    STANDARD_VARIANTS,
    VARIANT_ORDER,
)

# Short size labels for plot annotations
SIZE_LABELS = {"n": "N", "s": "S", "m": "M", "l": "L", "xl": "XL"}

RUNTIMES = ["ONNX-CPU", "TRT-fp32", "TRT-fp16"]

STD_COLOR = "#4C72B0"
EDGE_COLOR = "#DD8452"


def _size_label(variant: str) -> str:
    """Extract size label from variant name: 'yololite-edge-xl' -> 'XL'."""
    size = variant.split("-")[-1]
    return SIZE_LABELS.get(size, size.upper())


def _load_sab_data(results_dir: str) -> pd.DataFrame | None:
    sab_dir = os.path.join(results_dir, "sab")
    csvs = sorted(glob.glob(os.path.join(sab_dir, "bench_results_*.csv")))
    csvs = [c for c in csvs if "combined" not in os.path.basename(c)]
    if not csvs:
        return None
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df["mAP50_95"].notna()]
    present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]
    df["variant"] = pd.Categorical(df["variant"], categories=present, ordered=True)
    return df


# ── Plot 1: Heatmaps ────────────────────────────────────────────────────────

def plot_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    """Heatmap of mAP per dataset x variant, one per (metric, engine)."""
    if "dataset" not in df.columns or df["dataset"].nunique() <= 1:
        print("  Skipping heatmaps (single dataset).")
        return

    present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]

    for metric, title in [("mAP50", "mAP@50"), ("mAP50_95", "mAP@50:95")]:
        if metric not in df.columns:
            continue
        for rt in RUNTIMES:
            rt_df = df[df["runtime"] == rt]
            if rt_df.empty:
                continue

            pivot = rt_df.pivot_table(
                index="dataset", columns="variant", values=metric,
                aggfunc="first", observed=True,
            )
            pivot = pivot.reindex(columns=[v for v in present if v in pivot.columns])
            if pivot.empty:
                continue
            # Sort rows by first variant column descending
            if pivot.columns[0] in pivot.columns:
                pivot = pivot.sort_values(by=pivot.columns[0], ascending=False)

            fig, ax = plt.subplots(figsize=(
                max(8, len(pivot.columns) * 1.2),
                max(12, len(pivot) * 0.18),
            ))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=6)
            ax.set_title(f"{title} per dataset x variant ({rt})", fontweight="bold", fontsize=13)
            fig.colorbar(im, ax=ax, shrink=0.5, label=title)
            fig.tight_layout()

            rt_slug = rt.lower().replace("-", "_")
            fname = f"heatmap_{metric}_{rt_slug}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close(fig)
            print(f"  Saved: {fname}")


# ── Plot 2: mAP vs latency Pareto ───────────────────────────────────────────

def _gather_curve(df_rt: pd.DataFrame, variants: list[str], metric: str):
    """Return (latencies, maps, labels) for a list of variants, in order."""
    lats, maps, labels = [], [], []
    for v in variants:
        vdf = df_rt[df_rt["variant"] == v]
        if vdf.empty:
            continue
        lats.append(vdf["latency_median_ms"].median())
        maps.append(vdf[metric].median())
        labels.append(_size_label(v))
    return lats, maps, labels


def plot_map_vs_latency(df: pd.DataFrame, out_dir: str) -> None:
    """3x2 grid: rows = engines, cols = mAP@50 / mAP@50:95.

    Each subplot has two curves: YOLOLite (standard) and YOLOLite Edge,
    connected by lines with size labels at each point.
    """
    available_rt = [rt for rt in RUNTIMES if rt in df["runtime"].values]
    if not available_rt:
        return

    metrics = [("mAP50", "mAP@50"), ("mAP50_95", "mAP@50:95")]
    nrows = len(available_rt)
    ncols = len(metrics)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for row, rt in enumerate(available_rt):
        rt_df = df[df["runtime"] == rt]
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]

            for variants, color, label in [
                (STANDARD_VARIANTS, STD_COLOR, "YOLOLite"),
                (EDGE_VARIANTS, EDGE_COLOR, "YOLOLite Edge"),
            ]:
                lats, maps, size_labels = _gather_curve(rt_df, variants, metric)
                if not lats:
                    continue
                ax.plot(lats, maps, color=color, marker="o", markersize=8,
                        linewidth=2, zorder=3, label=label)
                for lat, mval, sl in zip(lats, maps, size_labels):
                    ax.annotate(
                        sl, (lat, mval), fontsize=9, fontweight="bold",
                        color=color, textcoords="offset points",
                        xytext=(6, 6), alpha=0.85,
                    )

            ax.set_xlabel("Latency [ms]", fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(f"{metric_label} ({rt})", fontweight="bold", fontsize=12)
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(alpha=0.2)

    fig.suptitle("Object Detection: mAP vs Latency", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "map_vs_latency.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: map_vs_latency.png")


# ── Plot 3: mAP distribution ────────────────────────────────────────────────

def plot_map_distribution(df: pd.DataFrame, out_dir: str) -> None:
    """Box plots of mAP@50 across datasets, per variant, one subplot per engine."""
    available_rt = [rt for rt in RUNTIMES if rt in df["runtime"].values]
    if not available_rt:
        return

    present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]
    nrt = len(available_rt)
    fig, axes = plt.subplots(1, nrt, figsize=(6 * nrt, 6), squeeze=False)

    for i, rt in enumerate(available_rt):
        ax = axes[0, i]
        rt_df = df[df["runtime"] == rt]

        data = []
        tick_labels = []
        colors = []
        for v in present:
            vals = rt_df[rt_df["variant"] == v]["mAP50"].dropna().values
            if len(vals) == 0:
                continue
            data.append(vals)
            tick_labels.append(_size_label(v))
            colors.append(EDGE_COLOR if "edge" in v else STD_COLOR)

        if not data:
            continue

        bp = ax.boxplot(
            data, tick_labels=tick_labels, patch_artist=True,
            showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_title(f"mAP@50 ({rt})", fontweight="bold", fontsize=12)
        ax.set_ylabel("mAP@50", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=STD_COLOR, alpha=0.7, label="YOLOLite"),
        Patch(facecolor=EDGE_COLOR, alpha=0.7, label="YOLOLite Edge"),
    ]
    axes[0, -1].legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.suptitle("mAP@50 Distribution Across Datasets", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "map_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: map_distribution.png")


# ── Plot 4: Latency distribution ────────────────────────────────────────────

def plot_latency_distribution(df: pd.DataFrame, out_dir: str) -> None:
    """Box plots of latency across datasets, per variant, one subplot per engine."""
    available_rt = [rt for rt in RUNTIMES if rt in df["runtime"].values]
    if not available_rt:
        return

    present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]
    nrt = len(available_rt)
    fig, axes = plt.subplots(1, nrt, figsize=(6 * nrt, 6), squeeze=False)

    for i, rt in enumerate(available_rt):
        ax = axes[0, i]
        rt_df = df[df["runtime"] == rt]

        data = []
        tick_labels = []
        colors = []
        for v in present:
            vals = rt_df[rt_df["variant"] == v]["latency_median_ms"].dropna().values
            if len(vals) == 0:
                continue
            data.append(vals)
            tick_labels.append(_size_label(v))
            colors.append(EDGE_COLOR if "edge" in v else STD_COLOR)

        if not data:
            continue

        bp = ax.boxplot(
            data, tick_labels=tick_labels, patch_artist=True,
            showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_title(f"Latency ({rt})", fontweight="bold", fontsize=12)
        ax.set_ylabel("Latency [ms]", fontsize=11)
        ax.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=STD_COLOR, alpha=0.7, label="YOLOLite"),
        Patch(facecolor=EDGE_COLOR, alpha=0.7, label="YOLOLite Edge"),
    ]
    axes[0, -1].legend(handles=legend_elements, fontsize=9, loc="upper right")

    fig.suptitle("Inference Latency Distribution Across Datasets", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: latency_distribution.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize SAB benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str,
                        default="rf100vl_benchmark_results",
                        help="Results directory containing sab/ CSVs")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = _load_sab_data(args.results_dir)
    if df is None or df.empty:
        print("No SAB benchmark CSVs found. Nothing to plot.")
        return

    available_rt = [rt for rt in RUNTIMES if rt in df["runtime"].values]
    present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]
    print(f"Loaded {len(df)} results: {len(present)} variants, "
          f"{len(available_rt)} engines")
    if "dataset" in df.columns:
        print(f"  Datasets: {df['dataset'].nunique()}")
    print()

    print("Heatmaps:")
    plot_heatmaps(df, out_dir)
    print("\nmAP vs Latency:")
    plot_map_vs_latency(df, out_dir)
    print("\nmAP Distribution:")
    plot_map_distribution(df, out_dir)
    print("\nLatency Distribution:")
    plot_latency_distribution(df, out_dir)

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
