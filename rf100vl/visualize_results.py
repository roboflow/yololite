#!/usr/bin/env python3
"""Visualize RF100-VL benchmark results from per-variant CSVs."""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "/dev/shm/rf100vl_benchmark_results"
OUT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all per-variant CSVs ────────────────────────────────────────────────
csvs = sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*.csv")))
df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
df = df[df["mAP50"].notna()]  # drop failed runs

# Canonical variant ordering (small → large)
VARIANT_ORDER = [
    "yololite-n", "yololite-edge-n",
    "yololite-s", "yololite-edge-s",
    "yololite-m", "yololite-edge-m",
    "yololite-l", "yololite-edge-l",
    "yololite-xl", "yololite-edge-xl",
]
present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]
df["variant"] = pd.Categorical(df["variant"], categories=present, ordered=True)

# ── Summary table ────────────────────────────────────────────────────────────
summary = (
    df.groupby("variant", observed=True)[["mAP50", "mAP50_95", "precision", "recall"]]
    .agg(["mean", "median", "std", "count"])
)
summary.to_csv(os.path.join(OUT_DIR, "summary_stats.csv"))
print(summary.to_string())
print()

# ── Color palette ────────────────────────────────────────────────────────────
base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
variant_colors = {v: base_colors[i] for i, v in enumerate(present)}

# ── 1. Mean metrics bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, metric in zip(axes, ["mAP50", "mAP50_95", "precision", "recall"]):
    means = df.groupby("variant", observed=True)[metric].mean()
    stds = df.groupby("variant", observed=True)[metric].std()
    colors = [variant_colors[v] for v in means.index]
    bars = ax.bar(range(len(means)), means, yerr=stds, color=colors,
                  capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha="right", fontsize=8)
    ax.set_title(metric, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Mean metrics across RF100-VL datasets (± std)", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mean_metrics_bar.png"), dpi=150)
print(f"Saved: {OUT_DIR}/mean_metrics_bar.png")

# ── 2. Box plots per metric ─────────────────────────────────────────────────
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

# ── 3. Per-dataset heatmap (mAP50) ──────────────────────────────────────────
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

# ── 4. Per-dataset heatmap (mAP50:95) ───────────────────────────────────────
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

# ── 5. Training time comparison ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
means_t = df.groupby("variant", observed=True)["train_s"].mean() / 3600  # hours
stds_t = df.groupby("variant", observed=True)["train_s"].std() / 3600
colors = [variant_colors[v] for v in means_t.index]
ax.bar(range(len(means_t)), means_t, yerr=stds_t, color=colors,
       capsize=3, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(means_t)))
ax.set_xticklabels(means_t.index, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Training time (hours)")
ax.set_title("Mean training time per variant (100 epochs)", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_time.png"), dpi=150)
print(f"Saved: {OUT_DIR}/training_time.png")

# ── 6. Accuracy vs speed scatter ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for v in present:
    vdf = df[df["variant"] == v]
    mean_map = vdf["mAP50_95"].mean()
    mean_time = vdf["train_s"].mean() / 3600
    ax.scatter(mean_time, mean_map, s=120, c=[variant_colors[v]],
               edgecolors="black", linewidths=0.5, zorder=3)
    ax.annotate(v, (mean_time, mean_map), fontsize=7,
                textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Mean training time (hours)")
ax.set_ylabel("Mean mAP@50:95")
ax.set_title("Accuracy vs training time", fontweight="bold")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "accuracy_vs_speed.png"), dpi=150)
print(f"Saved: {OUT_DIR}/accuracy_vs_speed.png")

# ── 7. Edge vs standard variant comparison ───────────────────────────────────
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
            std_means.append(df[df["variant"] == std_v][metric].mean())
            edge_means.append(df[df["variant"] == edge_v][metric].mean())

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
    fig.suptitle("Standard vs Edge variants", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "standard_vs_edge.png"), dpi=150)
    print(f"Saved: {OUT_DIR}/standard_vs_edge.png")

plt.close("all")
print(f"\nAll plots saved to: {OUT_DIR}/")
