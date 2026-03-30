#!/usr/bin/env python3
"""Visualize benchmark results from per-variant CSVs.

Works with both RF100-VL (multi-dataset) and COCO (single-dataset) results.
Generates training metric plots (1-5) and SAB latency/accuracy plots (6-10).
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yololite.benchmark._variants import VARIANT_ORDER

# ── Configuration ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.environ.get("RF100VL_RESULTS_DIR", "rf100vl_benchmark_results"),
        help="Results directory containing per-variant CSVs",
    )
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir
    OUT_DIR = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load training CSVs ───────────────────────────────────────────────────
    csvs = sorted(
        glob.glob(os.path.join(RESULTS_DIR, "results_*.csv"))
        + glob.glob(os.path.join(RESULTS_DIR, "train_results_*.csv"))
    )
    # Exclude combined CSVs
    csvs = [c for c in csvs if "combined" not in os.path.basename(c)]

    if csvs:
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        # Only keep rows with mAP50 (training results with metrics)
        if "mAP50" in df.columns:
            df = df[df["mAP50"].notna()]
        present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]
        df["variant"] = pd.Categorical(df["variant"], categories=present, ordered=True)
        HAS_TRAINING = len(present) > 0 and "mAP50" in df.columns
    else:
        print("No training CSVs found -- skipping training plots (1-5).\n")
        HAS_TRAINING = False

    if HAS_TRAINING:
        multi_dataset = "dataset" in df.columns and df["dataset"].nunique() > 1

        # ── Summary table ────────────────────────────────────────────────────
        metrics_cols = [c for c in ["mAP50", "mAP50_95", "precision", "recall"] if c in df.columns]
        if metrics_cols:
            summary = (
                df.groupby("variant", observed=True)[metrics_cols]
                .agg(["mean", "median", "std", "count"])
            )
            summary.to_csv(os.path.join(OUT_DIR, "summary_stats.csv"))
            print(summary.to_string())
            print()

        # ── Color palette ────────────────────────────────────────────────────
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        variant_colors = {v: base_colors[i] for i, v in enumerate(present)}

        # ── 1. Median metrics bar chart ──────────────────────────────────────
        if metrics_cols:
            fig, axes = plt.subplots(1, len(metrics_cols), figsize=(4.5 * len(metrics_cols), 5))
            if len(metrics_cols) == 1:
                axes = [axes]
            for ax, metric in zip(axes, metrics_cols):
                means = df.groupby("variant", observed=True)[metric].median()
                stds = df.groupby("variant", observed=True)[metric].std()
                colors = [variant_colors[v] for v in means.index]
                ax.bar(range(len(means)), means, yerr=stds, color=colors,
                       capsize=3, edgecolor="white", linewidth=0.5)
                ax.set_xticks(range(len(means)))
                ax.set_xticklabels(means.index, rotation=45, ha="right", fontsize=8)
                ax.set_title(metric, fontweight="bold")
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle("Median metrics across datasets (+/- std)", fontweight="bold", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "median_metrics_bar.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/median_metrics_bar.png")

        # ── 2. Box plots per metric ──────────────────────────────────────────
        if metrics_cols:
            ncols = min(len(metrics_cols), 2)
            nrows = (len(metrics_cols) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
            axes_flat = [axes] if len(metrics_cols) == 1 else axes.flat
            for ax, metric in zip(axes_flat, metrics_cols):
                data = [df[df["variant"] == v][metric].dropna().values for v in present]
                bp = ax.boxplot(data, tick_labels=present, patch_artist=True, showfliers=True,
                                flierprops=dict(marker=".", markersize=3, alpha=0.4))
                for patch, v in zip(bp["boxes"], present):
                    patch.set_facecolor(variant_colors[v])
                    patch.set_alpha(0.7)
                ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
                ax.set_title(metric, fontweight="bold")
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle("Metric distributions across datasets", fontweight="bold", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "metric_boxplots.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/metric_boxplots.png")

        # ── 3 & 4. Per-dataset heatmaps (only for multi-dataset) ────────────
        if multi_dataset:
            for metric, title in [("mAP50", "mAP@50"), ("mAP50_95", "mAP@50:95")]:
                if metric not in df.columns:
                    continue
                pivot = df.pivot_table(index="dataset", columns="variant", values=metric,
                                       aggfunc="first", observed=True)
                pivot = pivot.reindex(columns=present)
                if present[0] in pivot.columns:
                    pivot = pivot.sort_values(by=present[0], ascending=False)

                fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.2),
                                                max(12, len(pivot) * 0.18)))
                im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=6)
                ax.set_title(f"{title} per dataset x variant", fontweight="bold", fontsize=13)
                fig.colorbar(im, ax=ax, shrink=0.5, label=title)
                fig.tight_layout()
                fname = f"heatmap_{metric}.png"
                fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
                print(f"Saved: {OUT_DIR}/{fname}")

        # ── 5. Edge vs standard variant comparison ───────────────────────────
        pairs = []
        for v in present:
            if "edge" not in v:
                edge_v = v.replace("yololite-", "yololite-edge-")
                if edge_v in present:
                    pairs.append((v, edge_v))

        if pairs and "mAP50" in df.columns:
            compare_metrics = [m for m in ["mAP50", "mAP50_95"] if m in df.columns]
            fig, axes = plt.subplots(1, len(compare_metrics), figsize=(7 * len(compare_metrics), 5))
            if len(compare_metrics) == 1:
                axes = [axes]
            for ax, metric in zip(axes, compare_metrics):
                x_labels = []
                std_means = []
                edge_means = []
                for std_v, edge_v in pairs:
                    size = std_v.split("-")[-1]
                    x_labels.append(size)
                    std_means.append(df[df["variant"] == std_v][metric].median())
                    edge_means.append(df[df["variant"] == edge_v][metric].median())

                x = np.arange(len(x_labels))
                w = 0.35
                ax.bar(x - w / 2, std_means, w, label="Standard", color="#4C72B0")
                ax.bar(x + w / 2, edge_means, w, label="Edge", color="#DD8452")
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=10)
                ax.set_title(metric, fontweight="bold")
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle("Standard vs Edge variants (median across datasets)",
                         fontweight="bold", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "standard_vs_edge.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/standard_vs_edge.png")

    # ── SAB benchmark plots ──────────────────────────────────────────────────
    SAB_DIR = os.path.join(RESULTS_DIR, "sab")
    bench_csvs = sorted(glob.glob(os.path.join(SAB_DIR, "bench_results_*.csv")))
    bench_csvs = [c for c in bench_csvs if "combined" not in os.path.basename(c)]

    if bench_csvs:
        bdf = pd.concat([pd.read_csv(c) for c in bench_csvs], ignore_index=True)
        bdf = bdf[bdf["mAP50_95"].notna()]
        bdf_present = [v for v in VARIANT_ORDER if v in bdf["variant"].unique()]
        bdf["variant"] = pd.Categorical(bdf["variant"], categories=bdf_present, ordered=True)

        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        if not HAS_TRAINING:
            variant_colors = {v: base_colors[i] for i, v in enumerate(bdf_present)}

        available_runtimes = [rt for rt in ["TRT-fp16", "TRT-fp32", "ONNX-CPU"]
                              if rt in bdf["runtime"].values]
        best_runtime = available_runtimes[0] if available_runtimes else None

        # ── 6. Median mAP bar chart from SAB (fallback if no training data) ─
        if not HAS_TRAINING:
            onnx_cpu = bdf[bdf["runtime"] == "ONNX-CPU"]
            if not onnx_cpu.empty:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                for ax, metric in zip(axes, ["mAP50", "mAP50_95"]):
                    medians = onnx_cpu.groupby("variant", observed=True)[metric].median()
                    stds = onnx_cpu.groupby("variant", observed=True)[metric].std()
                    colors = [variant_colors.get(v, "gray") for v in medians.index]
                    ax.bar(range(len(medians)), medians, yerr=stds, color=colors,
                           capsize=3, edgecolor="white", linewidth=0.5)
                    ax.set_xticks(range(len(medians)))
                    ax.set_xticklabels(medians.index, rotation=45, ha="right", fontsize=8)
                    ax.set_title(metric, fontweight="bold")
                    ax.set_ylim(0, 1)
                    ax.grid(axis="y", alpha=0.3)
                fig.suptitle("Median metrics (ONNX-CPU, +/- std)", fontweight="bold", fontsize=13)
                fig.tight_layout()
                fig.savefig(os.path.join(OUT_DIR, "median_metrics_bar.png"), dpi=150)
                print(f"Saved: {OUT_DIR}/median_metrics_bar.png")

        # ── 7. Per-dataset heatmap from SAB (fallback) ──────────────────────
        if not HAS_TRAINING and "dataset" in bdf.columns and bdf["dataset"].nunique() > 1:
            onnx_cpu = bdf[bdf["runtime"] == "ONNX-CPU"]
            if not onnx_cpu.empty:
                pivot95 = onnx_cpu.pivot_table(index="dataset", columns="variant",
                                               values="mAP50_95", aggfunc="first",
                                               observed=True)
                pivot95 = pivot95.reindex(columns=bdf_present)
                if bdf_present[0] in pivot95.columns:
                    pivot95 = pivot95.sort_values(by=bdf_present[0], ascending=False)

                fig, ax = plt.subplots(figsize=(max(8, len(bdf_present) * 1.2),
                                                max(12, len(pivot95) * 0.18)))
                im = ax.imshow(pivot95.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
                ax.set_xticks(range(len(pivot95.columns)))
                ax.set_xticklabels(pivot95.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(pivot95.index)))
                ax.set_yticklabels(pivot95.index, fontsize=6)
                ax.set_title("mAP@50:95 per dataset x variant (ONNX-CPU)",
                             fontweight="bold", fontsize=13)
                fig.colorbar(im, ax=ax, shrink=0.5, label="mAP@50:95")
                fig.tight_layout()
                fig.savefig(os.path.join(OUT_DIR, "heatmap_mAP50_95.png"), dpi=150)
                print(f"Saved: {OUT_DIR}/heatmap_mAP50_95.png")

        # ── 8. mAP vs latency -- one subplot per runtime ────────────────────
        metric_colors = {"mAP50": "#E24A33", "mAP50_95": "#348ABD"}
        all_runtimes = ["ONNX-CPU", "TRT-fp32", "TRT-fp16"]
        if available_runtimes:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            for ax, rt in zip(axes, all_runtimes):
                rt_df = bdf[bdf["runtime"] == rt]
                if rt_df.empty:
                    ax.set_title(rt, fontweight="bold")
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12, color="gray")
                    ax.set_xlabel("Median latency (ms)")
                    ax.set_ylabel("Median mAP")
                    ax.grid(alpha=0.3)
                    continue
                for metric, label in [("mAP50", "mAP@50"), ("mAP50_95", "mAP@50:95")]:
                    lats, maps, labels = [], [], []
                    for v in bdf_present:
                        vdf = rt_df[rt_df["variant"] == v]
                        if vdf.empty:
                            continue
                        lats.append(vdf["latency_median_ms"].median())
                        maps.append(vdf[metric].median())
                        labels.append(v)
                    color = metric_colors[metric]
                    ax.scatter(lats, maps, s=120, c=color, marker="o",
                               edgecolors="black", linewidths=0.5, zorder=3, label=label)
                    for lat, mval, lbl in zip(lats, maps, labels):
                        ax.annotate(lbl, (lat, mval), fontsize=6,
                                    textcoords="offset points", xytext=(6, 4), alpha=0.7)
                ax.set_title(rt, fontweight="bold")
                ax.set_xlabel("Median latency (ms)")
                ax.set_ylabel("Median mAP")
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            fig.suptitle("mAP vs latency -- median across datasets", fontweight="bold", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "map_vs_latency_pareto.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/map_vs_latency_pareto.png")

        # ── 9. Latency bar chart ────────────────────────────────────────────
        if best_runtime:
            rt_df = bdf[bdf["runtime"] == best_runtime]
            if not rt_df.empty:
                fig, ax = plt.subplots(figsize=(12, 5))
                lat_medians = rt_df.groupby("variant", observed=True)["latency_median_ms"].median()
                lat_stds = rt_df.groupby("variant", observed=True)["latency_median_ms"].std()
                colors = [variant_colors.get(v, "gray") for v in lat_medians.index]
                ax.bar(range(len(lat_medians)), lat_medians, yerr=lat_stds, color=colors,
                       capsize=3, edgecolor="white", linewidth=0.5)
                ax.set_xticks(range(len(lat_medians)))
                ax.set_xticklabels(lat_medians.index, rotation=45, ha="right", fontsize=9)
                ax.set_ylabel("Median latency (ms)")
                ax.set_title(f"{best_runtime} inference latency per variant", fontweight="bold")
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                fname = f"latency_bar_{best_runtime.lower().replace('-', '_')}.png"
                fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
                print(f"Saved: {OUT_DIR}/{fname}")

        # ── 10. Runtime comparison ──────────────────────────────────────────
        if len(available_runtimes) > 1:
            fig, ax = plt.subplots(figsize=(14, 5))
            x = np.arange(len(bdf_present))
            width = 0.8 / len(available_runtimes)
            rt_colors = {"ONNX-CPU": "#4C72B0", "TRT-fp32": "#55A868", "TRT-fp16": "#DD8452"}
            for i, rt in enumerate(available_runtimes):
                rtdf = bdf[bdf["runtime"] == rt]
                means = [rtdf[rtdf["variant"] == v]["mAP50_95"].median() for v in bdf_present]
                ax.bar(x + i * width - 0.4 + width / 2, means, width,
                       label=rt, color=rt_colors.get(rt, "gray"))
            ax.set_xticks(x)
            ax.set_xticklabels(bdf_present, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Median mAP@50:95")
            ax.set_title("Median mAP@50:95 across inference engines", fontweight="bold")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "runtime_comparison.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/runtime_comparison.png")
    else:
        print("\nNo SAB benchmark CSVs found -- skipping latency plots.")

    plt.close("all")
    print(f"\nAll plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
