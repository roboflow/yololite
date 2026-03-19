#!/usr/bin/env python3
"""Benchmark all yololite variants supported in roboflow-train against RF100-VL.

Downloads all RF100-VL datasets (yolov8 format) and trains every yololite
variant for 100 epochs with batch_size=16.  Results (mAP@50 and mAP@50:95) are
collected into a pandas DataFrame and saved as both CSV and a readable table.

Concurrency: training jobs run in parallel via a process pool, one variant at
a time.  Each variant specifies how many concurrent jobs fit per GPU (based on
VRAM footprint), so small models get higher parallelism than large ones.
Adjust NUM_GPUS and per-variant jobs_per_gpu to match your hardware.
"""

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files as _pkg_files
from pathlib import Path

import pandas as pd
import torch

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS_DIR = "./rf100vl_datasets"
RESULTS_DIR = "./rf100vl_benchmark_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
DATASETS_FORMAT = "yolov8"
IMG_SIZE = 640

# rf100-vl standard benchmarking parameters
EPOCHS = 100
BATCH_SIZE = 16
MAX_DETS = 500  # max detections per image for COCO eval on test split

# ── GPU concurrency ───────────────────────────────────────────────────────────
NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = os.cpu_count() or 120

# All yololite variants supported in roboflow-train.
# Each entry: (config_subdir, config_yaml, jobs_per_gpu).
# jobs_per_gpu = floor(80 GB / estimated peak VRAM per training job) - 1.
# Variants are processed one at a time so different-sized models never share
# a GPU; this makes VRAM budgeting predictable.
YOLOLITE_VARIANTS = {
    "yololite-n":       ("v2_models", "yololite_n.yaml",  6),   # ~11 GB peak
    "yololite-edge-n":  ("models",    "edge_n.yaml",      6),   # ~11 GB peak
    "yololite-s":       ("v2_models", "yololite_s.yaml",  4),   # ~14 GB peak
    "yololite-edge-s":  ("models",    "edge_s.yaml",      4),   # ~14 GB peak
    "yololite-m":       ("v2_models", "yololite_m.yaml",  4),   # ~16 GB peak
    "yololite-edge-m":  ("models",    "edge_m.yaml",      4),   # ~16 GB peak
    "yololite-l":       ("v2_models", "yololite_l.yaml",  2),   # ~30 GB peak
    "yololite-edge-l":  ("models",    "edge_l.yaml",      2),   # ~30 GB peak
    "yololite-xl":      ("models",    "yololite_xl.yaml", 1),   # ~52 GB peak
    "yololite-edge-xl": ("models",    "edge_xl.yaml",     1),   # ~52 GB peak
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_datasets() -> list[str]:
    """Download all RF100-VL datasets and return list of dataset directories."""
    from rf100vl import download_rf100vl

    print(f"\n{'='*70}")
    print("Downloading RF100-VL datasets …")
    print(f"  destination: {DATASETS_DIR}")
    print(f"  format:      {DATASETS_FORMAT}")
    print(f"{'='*70}\n")

    dataset_list = download_rf100vl(
        path=DATASETS_DIR,
        model_format=DATASETS_FORMAT,
        overwrite=False,
    )

    # Collect actual dataset directories on disk (each should have a data.yaml)
    dataset_dirs = sorted(
        str(p.parent)
        for p in Path(DATASETS_DIR).rglob("data.yaml")
    )
    print(f"\nFound {len(dataset_dirs)} datasets with data.yaml")
    return dataset_dirs


def _find_data_yaml(dataset_dir: str) -> str:
    """Return the path to data.yaml inside a dataset directory."""
    # Roboflow yolov8 format puts data.yaml at the root of each dataset
    candidate = os.path.join(dataset_dir, "data.yaml")
    if os.path.isfile(candidate):
        return candidate
    # Fallback: search one level deeper
    for p in Path(dataset_dir).glob("*/data.yaml"):
        return str(p)
    raise FileNotFoundError(f"No data.yaml found in {dataset_dir}")


def _find_test_folder(dataset_dir: str) -> str | None:
    """Return the path to the test split folder (containing images/ and labels/)."""
    for name in ("test", "Test"):
        candidate = os.path.join(dataset_dir, name)
        if os.path.isdir(os.path.join(candidate, "images")) and os.path.isdir(os.path.join(candidate, "labels")):
            return candidate
    return None


def run_single_training(
    variant_name: str,
    dataset_dir: str,
    dataset_name: str,
    device: str = "0",
    max_concurrent: int = 1,
) -> dict:
    """Train one yololite variant on one dataset, then evaluate on test split."""
    import torch

    # Cap PyTorch intra-op thread pool to avoid oversubscription.
    # set_num_interop_threads cannot be called after fork, so we rely on
    # OMP_NUM_THREADS / MKL_NUM_THREADS env vars set in _worker() instead.
    torch.set_num_threads(max(1, NUM_CPUS // max_concurrent))

    from yololite.scripts.args.build_args import load_configs
    from yololite.tools.train import run_training
    from yololite.tools.evaluate import evaluate_on_folder

    configs_root = _pkg_files("yololite").joinpath("configs")
    subdir, yaml_file, _jobs_per_gpu = YOLOLITE_VARIANTS[variant_name]
    model_yaml = str(configs_root / subdir / yaml_file)
    train_yaml = str(configs_root / "train" / "standard_train.yaml")
    data_yaml = _find_data_yaml(dataset_dir)

    log_dir = os.path.join(
        RESULTS_DIR, "runs", dataset_name, variant_name,
    )
    os.makedirs(log_dir, exist_ok=True)

    config = load_configs(
        model_yaml=model_yaml,
        train_yaml=train_yaml,
        data_yaml=data_yaml,
        log_dir=log_dir,
    )

    # Override training hyperparameters
    config["training"]["epochs"] = EPOCHS
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["img_size"] = IMG_SIZE
    config["training"]["save_every"] = EPOCHS + 1  # no periodic saves
    config["training"]["save_by"] = "AP"
    config["training"]["num_workers"] = max(1, NUM_CPUS // max_concurrent)
    config["training"]["device"] = device

    t0 = time.time()
    train_result = run_training(config)
    elapsed = time.time() - t0

    # Evaluate the best checkpoint on the test split with max_dets=500
    best_ckpt = train_result["best_checkpoint"]
    test_folder = _find_test_folder(dataset_dir)

    if test_folder is not None:
        eval_log_dir = os.path.join(log_dir, "test_eval")
        os.makedirs(eval_log_dir, exist_ok=True)
        test_metrics = evaluate_on_folder(
            weights=best_ckpt,
            test_folder=test_folder,
            batch_size=BATCH_SIZE,
            device=device,
            max_dets=MAX_DETS,
            log_dir=eval_log_dir,
        )
    else:
        # Fall back to training's val-split metrics if no test split exists
        test_metrics = train_result.get("best_metrics", {})

    return {
        "dataset": dataset_name,
        "variant": variant_name,
        "mAP50": test_metrics.get("mAP", test_metrics.get("AP50", 0.0)),
        "mAP50_95": test_metrics.get("mAP_50_95", test_metrics.get("AP", 0.0)),
        "precision": test_metrics.get("precision", 0.0),
        "recall": test_metrics.get("recall", 0.0),
        "epochs_completed": train_result.get("epochs_completed", 0),
        "elapsed_s": round(elapsed, 1),
    }


def _worker(args: tuple) -> dict:
    """Top-level worker function for ProcessPoolExecutor (must be picklable)."""
    variant_name, dataset_dir, dataset_name, device, max_concurrent = args
    # Set thread limits before torch is imported in this process
    threads_per_job = str(max(1, NUM_CPUS // max_concurrent))
    os.environ["OMP_NUM_THREADS"] = threads_per_job
    os.environ["MKL_NUM_THREADS"] = threads_per_job
    try:
        return run_single_training(variant_name, dataset_dir, dataset_name, device, max_concurrent)
    except Exception as e:
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "variant": variant_name,
            "mAP50": None,
            "mAP50_95": None,
            "epochs_completed": 0,
            "elapsed_s": 0,
            "error": str(e),
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Download datasets
    dataset_dirs = download_datasets()
    if not dataset_dirs:
        print("ERROR: No datasets found after download. Check ROBOFLOW_API_KEY.")
        sys.exit(1)

    total = len(dataset_dirs) * len(YOLOLITE_VARIANTS)

    print(f"\n{'='*70}")
    print(f"Benchmark matrix: {len(dataset_dirs)} datasets × "
          f"{len(YOLOLITE_VARIANTS)} variants = {total} training runs")
    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  "
          f"GPUs: {NUM_GPUS}")
    print(f"{'='*70}\n")

    # Load any existing partial results so we can skip completed runs
    completed = set()
    rows = []
    if os.path.isfile(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        rows = existing.to_dict("records")
        for r in rows:
            completed.add((r["dataset"], r["variant"]))
        print(f"Resuming: {len(completed)} runs already completed, "
              f"{total - len(completed)} remaining.\n")

    done_count = len(completed)

    # 2. Process one variant at a time so different-sized models never share
    #    a GPU.  Each variant gets its own pool sized to its VRAM footprint.
    for variant, (_subdir, _yaml, jobs_per_gpu) in YOLOLITE_VARIANTS.items():
        max_concurrent = NUM_GPUS * jobs_per_gpu

        # Build jobs for this variant: (variant, dataset_dir, dataset_name, device)
        variant_jobs = []
        for i, ddir in enumerate(dataset_dirs):
            dname = Path(ddir).name
            if (dname, variant) in completed:
                continue
            device = str(i % NUM_GPUS)
            variant_jobs.append((variant, ddir, dname, device, max_concurrent))

        if not variant_jobs:
            print(f"[{variant}] all {len(dataset_dirs)} datasets already done, skipping.")
            continue

        print(f"\n{'─'*70}")
        print(f"[{variant}] {len(variant_jobs)} datasets remaining  |  "
              f"jobs_per_gpu={jobs_per_gpu}  |  max_concurrent={max_concurrent}")
        print(f"{'─'*70}")

        # 3. Run training jobs for this variant
        with ProcessPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {pool.submit(_worker, job): job for job in variant_jobs}

            for future in as_completed(futures):
                result = future.result()
                rows.append(result)
                done_count += 1

                status = "OK" if result.get("error") is None else f"FAIL: {result['error']}"
                print(
                    f"[{done_count}/{total}] {result['dataset']} / "
                    f"{result['variant']}  —  "
                    f"mAP50={result['mAP50']}  mAP50:95={result['mAP50_95']}  "
                    f"({result['elapsed_s']}s)  [{status}]"
                )

                # Save incrementally after each run
                df = pd.DataFrame(rows)
                df.to_csv(RESULTS_CSV, index=False)

    # 4. Build and save the final results table
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)

    # Pivot tables for quick comparison
    if not df.empty and "error" not in df.columns or not df.get("error", pd.Series()).any():
        for metric in ["mAP50", "mAP50_95", "precision", "recall"]:
            pivot = df.pivot_table(
                index="dataset", columns="variant", values=metric, aggfunc="first",
            )
            pivot_path = os.path.join(RESULTS_DIR, f"pivot_{metric}.csv")
            pivot.to_csv(pivot_path)
            print(f"\nPivot table saved: {pivot_path}")

        # Mean across datasets per variant
        summary = (
            df.groupby("variant")[["mAP50", "mAP50_95", "precision", "recall"]]
            .mean()
            .sort_values("mAP50", ascending=False)
        )
        summary_path = os.path.join(RESULTS_DIR, "summary_by_variant.csv")
        summary.to_csv(summary_path)
        print(f"Summary saved:     {summary_path}")
        print(f"\n{summary.to_string()}")

    print(f"\nFull results:      {RESULTS_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
