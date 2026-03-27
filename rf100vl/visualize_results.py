#!/usr/bin/env python3
"""Visualize RF100-VL benchmark results from per-variant CSVs."""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.environ.get("RF100VL_RESULTS_DIR", "rf100vl_benchmark_results")
OUT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Canonical variant ordering (small → large)
VARIANT_ORDER = [
    "yololite-n", "yololite-edge-n",
    "yololite-s", "yololite-edge-s",
    "yololite-m", "yololite-edge-m",
    "yololite-l", "yololite-edge-l",
    "yololite-xl", "yololite-edge-xl",
]

# ── Load all per-variant training CSVs (optional) ────────────────────────────
# Support both old (results_*.csv) and new (train_results_*.csv) naming
csvs = sorted(
    glob.glob(os.path.join(RESULTS_DIR, "results_*.csv"))
    + glob.glob(os.path.join(RESULTS_DIR, "train_results_*.csv"))
)
if csvs:
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df["mAP50"].notna()]  # drop failed runs
    present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]
    df["variant"] = pd.Categorical(df["variant"], categories=present, ordered=True)
    HAS_TRAINING = True
else:
    print("No training CSVs found — skipping training plots (1–5).\n")
    HAS_TRAINING = False

if HAS_TRAINING:
    # ── Summary table ────────────────────────────────────────────────────────
    summary = (
        df.groupby("variant", observed=True)[["mAP50", "mAP50_95", "precision", "recall"]]
        .agg(["mean", "median", "std", "count"])
    )
    summary.to_csv(os.path.join(OUT_DIR, "summary_stats.csv"))
    print(summary.to_string())
    print()

    # ── Color palette ────────────────────────────────────────────────────────
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    variant_colors = {v: base_colors[i] for i, v in enumerate(present)}

    # ── 1. Mean metrics bar chart ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, ["mAP50", "mAP50_95", "precision", "recall"]):
        means = df.groupby("variant", observed=True)[metric].median()
        stds = df.groupby("variant", observed=True)[metric].std()
        colors = [variant_colors[v] for v in means.index]
        bars = ax.bar(range(len(means)), means, yerr=stds, color=colors,
                      capsize=3, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Median metrics across RF100-VL datasets (± std)", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "median_metrics_bar.png"), dpi=150)
    print(f"Saved: {OUT_DIR}/median_metrics_bar.png")

    # ── 2. Box plots per metric ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, metric in zip(axes.flat, ["mAP50", "mAP50_95", "precision", "recall"]):
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
    fig.suptitle("Metric distributions across RF100-VL datasets", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "metric_boxplots.png"), dpi=150)
    print(f"Saved: {OUT_DIR}/metric_boxplots.png")

    # ── 3. Per-dataset heatmap (mAP50) ───────────────────────────────────────
    pivot = df.pivot_table(index="dataset", columns="variant", values="mAP50", aggfunc="first", observed=True)
    pivot = pivot.reindex(columns=present)
    pivot = pivot.sort_values(by=present[0], ascending=False) if present[0] in pivot.columns else pivot

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.2), max(12, len(pivot) * 0.18)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_title("mAP@50 per dataset × variant", fontweight="bold", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.5, label="mAP@50")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_mAP50.png"), dpi=150)
    print(f"Saved: {OUT_DIR}/heatmap_mAP50.png")

    # ── 4. Per-dataset heatmap (mAP50:95) ────────────────────────────────────
    pivot95 = df.pivot_table(index="dataset", columns="variant", values="mAP50_95", aggfunc="first", observed=True)
    pivot95 = pivot95.reindex(columns=present)
    pivot95 = pivot95.sort_values(by=present[0], ascending=False) if present[0] in pivot95.columns else pivot95

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.2), max(12, len(pivot95) * 0.18)))
    im = ax.imshow(pivot95.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot95.columns)))
    ax.set_xticklabels(pivot95.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot95.index)))
    ax.set_yticklabels(pivot95.index, fontsize=6)
    ax.set_title("mAP@50:95 per dataset × variant", fontweight="bold", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.5, label="mAP@50:95")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_mAP50_95.png"), dpi=150)
    print(f"Saved: {OUT_DIR}/heatmap_mAP50_95.png")

    # ── 5. Edge vs standard variant comparison ───────────────────────────────
    # Pair up standard and edge variants
    pairs = []
    for v in present:
        if "edge" not in v:
            edge_v = v.replace("yololite-", "yololite-edge-")
            if edge_v in present:
                pairs.append((v, edge_v))

    if pairs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, metric in zip(axes, ["mAP50", "mAP50_95"]):
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
        fig.suptitle("Standard vs Edge variants (median across datasets)", fontweight="bold", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "standard_vs_edge.png"), dpi=150)
        print(f"Saved: {OUT_DIR}/standard_vs_edge.png")

# ── 8–10. Latency plots (from SAB benchmark CSVs) ───────────────────────────
bench_csvs = sorted(glob.glob(os.path.join(RESULTS_DIR, "bench_results_*.csv")))
# Exclude the combined CSV to avoid double-counting
bench_csvs = [c for c in bench_csvs if "combined" not in os.path.basename(c)]
if bench_csvs:
    bdf = pd.concat([pd.read_csv(c) for c in bench_csvs], ignore_index=True)
    bdf = bdf[bdf["mAP50_95"].notna()]
    bdf_present = [v for v in VARIANT_ORDER if v in bdf["variant"].unique()]
    bdf["variant"] = pd.Categorical(bdf["variant"], categories=bdf_present, ordered=True)

    # Color palette (shared with training plots if available, otherwise built from SAB data)
    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if not HAS_TRAINING:
        variant_colors = {v: base_colors[i] for i, v in enumerate(bdf_present)}

    # Pick the best available runtime for mAP/latency plots
    available_runtimes = [rt for rt in ["TRT-fp16", "TRT-fp32", "ONNX-CPU"] if rt in bdf["runtime"].values]
    best_runtime = available_runtimes[0] if available_runtimes else None

    # ── 6. Median mAP bar chart (from SAB bench results) ────────────────────
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
            fig.suptitle("Median metrics across RF100-VL datasets (ONNX-CPU, ± std)", fontweight="bold", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "median_metrics_bar.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/median_metrics_bar.png")

    # ── 7. Per-dataset heatmap from SAB (mAP50:95) ──────────────────────────
    if not HAS_TRAINING:
        onnx_cpu = bdf[bdf["runtime"] == "ONNX-CPU"]
        if not onnx_cpu.empty:
            pivot95 = onnx_cpu.pivot_table(index="dataset", columns="variant", values="mAP50_95",
                                           aggfunc="first", observed=True)
            pivot95 = pivot95.reindex(columns=bdf_present)
            pivot95 = pivot95.sort_values(by=bdf_present[0], ascending=False) if bdf_present[0] in pivot95.columns else pivot95

            fig, ax = plt.subplots(figsize=(max(8, len(bdf_present) * 1.2), max(12, len(pivot95) * 0.18)))
            im = ax.imshow(pivot95.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot95.columns)))
            ax.set_xticklabels(pivot95.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot95.index)))
            ax.set_yticklabels(pivot95.index, fontsize=6)
            ax.set_title("mAP@50:95 per dataset x variant (ONNX-CPU)", fontweight="bold", fontsize=13)
            fig.colorbar(im, ax=ax, shrink=0.5, label="mAP@50:95")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "heatmap_mAP50_95.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/heatmap_mAP50_95.png")

    # ── 8. mAP vs latency Pareto curve ───────────────────────────────────────
    if best_runtime:
        rt_df = bdf[bdf["runtime"] == best_runtime]
        if not rt_df.empty:
            fig, ax = plt.subplots(figsize=(10, 7))
            for v in bdf_present:
                vdf = rt_df[rt_df["variant"] == v]
                if vdf.empty:
                    continue
                med_map = vdf["mAP50_95"].median()
                med_lat = vdf["latency_median_ms"].median()
                is_edge = "edge" in v
                marker = "D" if is_edge else "o"
                color = variant_colors.get(v, "gray")
                ax.scatter(med_lat, med_map, s=140, c=[color], marker=marker,
                           edgecolors="black", linewidths=0.5, zorder=3)
                ax.annotate(v, (med_lat, med_map), fontsize=7,
                            textcoords="offset points", xytext=(6, 6))
            ax.set_xlabel(f"Median {best_runtime} latency (ms)")
            ax.set_ylabel("Median mAP@50:95")
            ax.set_title(f"mAP vs latency — median across datasets ({best_runtime})", fontweight="bold")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "map_vs_latency_pareto.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/map_vs_latency_pareto.png")

    # ── 9. Latency bar chart ─────────────────────────────────────────────────
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
            fig.savefig(os.path.join(OUT_DIR, f"latency_bar_{best_runtime.lower().replace('-', '_')}.png"), dpi=150)
            print(f"Saved: {OUT_DIR}/latency_bar_{best_runtime.lower().replace('-', '_')}.png")

    # ── 10. Runtime comparison (mAP consistency across engines) ──────────────
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
        ax.set_title("Median mAP@50:95 across inference engines (should be consistent)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "runtime_comparison.png"), dpi=150)
        print(f"Saved: {OUT_DIR}/runtime_comparison.png")
else:
    print("\nNo SAB benchmark CSVs found — skipping latency plots.")

plt.close("all")
print(f"\nAll plots saved to: {OUT_DIR}/")
