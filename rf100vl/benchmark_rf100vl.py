#!/usr/bin/env python3
"""Benchmark all yololite variants supported in roboflow-train against RF100-VL.

Downloads all RF100-VL datasets (yolov8 format) and trains every yololite
variant for 100 epochs with batch_size=16.  Results (mAP@50 and mAP@50:95) are
collected into a pandas DataFrame and saved as both CSV and a readable table.

Concurrency: multiple training jobs run in parallel via a process pool.  With a
single A100-40GB the default is 2 concurrent jobs — adjust MAX_CONCURRENT_JOBS
below if you have more (or fewer) resources.
"""

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files as _pkg_files
from pathlib import Path

import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS_DIR = "./rf100vl_datasets"
RESULTS_DIR = "./rf100vl_benchmark_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
DATASETS_FORMAT = "yolov8"
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
MAX_DETS = 500  # max detections per image for COCO eval on test split
MAX_CONCURRENT_JOBS = 2  # conservative for 1× A100-40GB; increase if more GPUs

# All yololite variants supported in roboflow-train (maps name → config path)
YOLOLITE_VARIANTS = {
    "yololite-n": ("v2_models", "yololite_n.yaml"),
    "yololite-s": ("v2_models", "yololite_s.yaml"),
    "yololite-m": ("v2_models", "yololite_m.yaml"),
    "yololite-l": ("v2_models", "yololite_l.yaml"),
    "yololite-xl": ("models", "yololite_xl.yaml"),
    "yololite-edge-n": ("models", "edge_n.yaml"),
    "yololite-edge-s": ("models", "edge_s.yaml"),
    "yololite-edge-m": ("models", "edge_m.yaml"),
    "yololite-edge-l": ("models", "edge_l.yaml"),
    "yololite-edge-xl": ("models", "edge_xl.yaml"),
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
) -> dict:
    """Train one yololite variant on one dataset, then evaluate on test split."""
    from yololite.scripts.args.build_args import load_configs
    from yololite.tools.train import run_training
    from yololite.tools.evaluate import evaluate_on_folder

    configs_root = _pkg_files("yololite").joinpath("configs")
    subdir, yaml_file = YOLOLITE_VARIANTS[variant_name]
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
    config["training"]["num_workers"] = 4
    config["training"]["device"] = "0"

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
            device="0",
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
    variant_name, dataset_dir, dataset_name = args
    try:
        return run_single_training(variant_name, dataset_dir, dataset_name)
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

    # 2. Build the job matrix: (variant, dataset_dir, dataset_name)
    jobs = []
    for ddir in dataset_dirs:
        dname = Path(ddir).name
        for variant in YOLOLITE_VARIANTS:
            jobs.append((variant, ddir, dname))

    total = len(jobs)
    print(f"\n{'='*70}")
    print(f"Benchmark matrix: {len(dataset_dirs)} datasets × "
          f"{len(YOLOLITE_VARIANTS)} variants = {total} training runs")
    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  "
          f"Concurrent jobs: {MAX_CONCURRENT_JOBS}")
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

    remaining_jobs = [j for j in jobs if (j[2], j[0]) not in completed]

    if not remaining_jobs:
        print("All runs already completed!")
    else:
        # 3. Run training jobs concurrently
        with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as pool:
            futures = {pool.submit(_worker, job): job for job in remaining_jobs}
            done_count = len(completed)

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
            .sort_values("mAP50_95", ascending=False)
        )
        summary_path = os.path.join(RESULTS_DIR, "summary_by_variant.csv")
        summary.to_csv(summary_path)
        print(f"Summary saved:     {summary_path}")
        print(f"\n{summary.to_string()}")

    print(f"\nFull results:      {RESULTS_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
