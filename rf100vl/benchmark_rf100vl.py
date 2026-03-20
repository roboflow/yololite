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

import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files as _pkg_files
from pathlib import Path

import pandas as pd

# NOTE: torch is intentionally NOT imported at module level.  Spawned worker
# processes re-import this module, and any module-level torch import would
# initialize CUDA before the pool initializer can set CUDA_VISIBLE_DEVICES.
# Use _get_num_gpus() in the main process; workers import torch after the
# initializer has restricted them to a single GPU.

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS_DIR = "/dev/shm/rf100vl_datasets"
RESULTS_DIR = "/dev/shm/rf100vl_benchmark_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")  # combined final
GCS_BUCKET = "gs://rf-detr-rf100-vl/yololite-benchmark"
DATASETS_FORMAT = "yolov8"
IMG_SIZE = 640

# rf100-vl standard benchmarking parameters
EPOCHS = 100
BATCH_SIZE = 16
# NOTE: COCO eval uses the default maxDets=100.  We cannot raise it (e.g. to
# 500) because pycocotools' _summarize() hardcodes maxDets=100 as the default
# parameter for AP@0.50:0.95 (stats[0]).  Changing E.params.maxDets to include
# a value other than [1, 10, 100] causes stats[0] to look up maxDets=100 in
# the list, and if the list is [1, 10, 500] it returns -1 (not found).  Even
# keeping 100 in a 4-element list [1, 10, 100, 500] shifts which index
# _summarize references for AR stats.  A proper fix would require patching
# pycocotools' summarize() method.

# ── GPU concurrency ───────────────────────────────────────────────────────────
NUM_CPUS = os.cpu_count() or 120
_SPAWN_CTX = multiprocessing.get_context("spawn")


def _get_num_gpus() -> int:
    import torch
    return torch.cuda.device_count()

# All yololite variants supported in roboflow-train.
# Each entry: (config_subdir, config_yaml, jobs_per_gpu).
# jobs_per_gpu = floor(80 GB / estimated peak VRAM per training job) - 1.
# Variants are processed one at a time so different-sized models never share
# a GPU; this makes VRAM budgeting predictable.
YOLOLITE_VARIANTS = {
    "yololite-n":       ("v2_models", "yololite_n.yaml",  4),   # ~11 GB peak
    "yololite-edge-n":  ("models",    "edge_n.yaml",      4),   # ~11 GB peak
    "yololite-s":       ("v2_models", "yololite_s.yaml",  4),   # ~14 GB peak
    "yololite-edge-s":  ("models",    "edge_s.yaml",      4),   # ~14 GB peak
    "yololite-m":       ("v2_models", "yololite_m.yaml",  4),   # ~16 GB peak
    "yololite-edge-m":  ("models",    "edge_m.yaml",      4),   # ~16 GB peak
    "yololite-l":       ("v2_models", "yololite_l.yaml",  2),   # ~30 GB peak
    "yololite-edge-l":  ("models",    "edge_l.yaml",      2),   # ~30 GB peak
    "yololite-xl":      ("models",    "yololite_xl.yaml", 1),   # ~52 GB peak
    "yololite-edge-xl": ("models",    "edge_xl.yaml",     1),   # ~52 GB peak
}


# ── GCS upload ────────────────────────────────────────────────────────────────

def sync_results_to_gcs() -> None:
    """Rsync the entire RESULTS_DIR to GCS.
    Failures are logged but never abort the benchmark."""
    try:
        subprocess.run(
            ["gcloud", "storage", "rsync", "-r", RESULTS_DIR, GCS_BUCKET],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        print("WARNING: gcloud CLI not found — skipping GCS sync")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: GCS sync failed: {e.stderr.strip()}")


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
    train_elapsed = time.time() - t0

    # Evaluate the best checkpoint on the test split
    best_ckpt = train_result["best_checkpoint"]
    test_folder = _find_test_folder(dataset_dir)

    if test_folder is not None:
        eval_log_dir = os.path.join(log_dir, "test_eval")
        os.makedirs(eval_log_dir, exist_ok=True)
        t1 = time.time()
        test_metrics = evaluate_on_folder(
            weights=best_ckpt,
            test_folder=test_folder,
            batch_size=BATCH_SIZE,
            device=device,
            log_dir=eval_log_dir,
        )
        eval_elapsed = time.time() - t1
    else:
        raise FileNotFoundError(
            f"No test split found in {dataset_dir} — benchmark requires a test split"
        )

    return {
        "dataset": dataset_name,
        "variant": variant_name,
        "mAP50": test_metrics.get("mAP", test_metrics.get("AP50", 0.0)),
        "mAP50_95": test_metrics.get("mAP_50_95", test_metrics.get("AP", 0.0)),
        "precision": test_metrics.get("precision", 0.0),
        "recall": test_metrics.get("recall", 0.0),
        "epochs_completed": train_result.get("epochs_completed", 0),
        "train_s": round(train_elapsed, 1),
        "eval_s": round(eval_elapsed, 1),
    }


def _pool_initializer(gpu_id: int, max_concurrent: int) -> None:
    """Called once per worker process before any task runs.  Sets
    CUDA_VISIBLE_DEVICES so the subsequent torch import only sees one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    threads = str(max(1, NUM_CPUS // max_concurrent))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads


def _worker(args: tuple) -> dict:
    """Top-level worker function for ProcessPoolExecutor (must be picklable)."""
    variant_name, dataset_dir, dataset_name, max_concurrent = args
    try:
        # device="0" because CUDA_VISIBLE_DEVICES remaps the assigned GPU
        return run_single_training(variant_name, dataset_dir, dataset_name, "0", max_concurrent)
    except Exception as e:
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "variant": variant_name,
            "mAP50": None,
            "mAP50_95": None,
            "epochs_completed": 0,
            "train_s": 0,
            "eval_s": 0,
            "error": str(e),
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def _variant_csv(variant: str) -> str:
    """Return the per-variant CSV path."""
    return os.path.join(RESULTS_DIR, f"results_{variant}.csv")


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
    num_gpus = _get_num_gpus()

    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  "
          f"GPUs: {num_gpus}")
    print(f"{'='*70}\n")

    # Load any existing partial results from per-variant CSVs so we can
    # resume.  Each variant has its own CSV: results_<variant>.csv
    completed = set()
    variant_rows: dict[str, list[dict]] = {v: [] for v in YOLOLITE_VARIANTS}
    for variant in YOLOLITE_VARIANTS:
        csv_path = _variant_csv(variant)
        if os.path.isfile(csv_path):
            existing = pd.read_csv(csv_path)
            variant_rows[variant] = existing.to_dict("records")
            for r in variant_rows[variant]:
                completed.add((r["dataset"], r["variant"]))

    if completed:
        print(f"Resuming: {len(completed)} runs already completed, "
              f"{total - len(completed)} remaining.\n")

    done_count = len(completed)

    # 2. Process one variant at a time so different-sized models never share
    #    a GPU.  Each variant gets its own pool sized to its VRAM footprint.
    #    Per-GPU pools with CUDA_VISIBLE_DEVICES set via initializer ensure
    #    each worker only ever sees one GPU — no stray contexts on GPU 0.
    for variant, (_subdir, _yaml, jobs_per_gpu) in YOLOLITE_VARIANTS.items():
        max_concurrent = num_gpus * jobs_per_gpu

        # Build jobs for this variant (no device — pools handle GPU assignment)
        variant_jobs = []
        for ddir in dataset_dirs:
            dname = Path(ddir).name
            if (dname, variant) in completed:
                continue
            variant_jobs.append((variant, ddir, dname, max_concurrent))

        if not variant_jobs:
            print(f"[{variant}] all {len(dataset_dirs)} datasets already done, skipping.")
            continue

        print(f"\n{'─'*70}")
        print(f"[{variant}] {len(variant_jobs)} datasets remaining  |  "
              f"jobs_per_gpu={jobs_per_gpu}  |  max_concurrent={max_concurrent}")
        print(f"{'─'*70}")

        variant_csv = _variant_csv(variant)

        # 3. Distribute jobs round-robin across GPUs, one pool per GPU.
        #    Each pool's initializer sets CUDA_VISIBLE_DEVICES before any
        #    task runs.  Because torch is NOT imported at module level,
        #    CUDA is not yet initialized when the initializer executes.
        gpu_jobs: list[list[tuple]] = [[] for _ in range(num_gpus)]
        for i, job in enumerate(variant_jobs):
            gpu_jobs[i % num_gpus].append(job)

        pools = []
        for gpu_id in range(num_gpus):
            pools.append(ProcessPoolExecutor(
                max_workers=jobs_per_gpu,
                mp_context=_SPAWN_CTX,
                initializer=_pool_initializer,
                initargs=(gpu_id, max_concurrent),
            ))

        try:
            futures = {}
            for gpu_id, jobs in enumerate(gpu_jobs):
                for job in jobs:
                    futures[pools[gpu_id].submit(_worker, job)] = job

            for future in as_completed(futures):
                result = future.result()
                variant_rows[variant].append(result)
                done_count += 1

                status = "OK" if result.get("error") is None else f"FAIL: {result['error']}"
                print(
                    f"[{done_count}/{total}] {result['dataset']} / "
                    f"{result['variant']}  —  "
                    f"mAP50={result['mAP50']}  mAP50:95={result['mAP50_95']}  "
                    f"(train={result['train_s']}s  eval={result['eval_s']}s)  [{status}]"
                )

                # Save per-variant CSV incrementally
                pd.DataFrame(variant_rows[variant]).to_csv(variant_csv, index=False)
        finally:
            # Shut down pools one at a time to avoid semaphore cleanup races
            # (Python 3.10 bug where concurrent finalizers try to sem_unlink
            # the same named semaphore twice).
            for pool in pools:
                pool.shutdown(wait=True)

        # Sync entire results folder to GCS after each variant completes
        sync_results_to_gcs()

    # 4. Build combined results from all per-variant CSVs
    all_rows = [r for rows in variant_rows.values() for r in rows]
    df = pd.DataFrame(all_rows)
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

    sync_results_to_gcs()

    print(f"\nFull results:      {RESULTS_CSV}")
    print(f"GCS results:       {GCS_BUCKET}/")
    print("Done.")


if __name__ == "__main__":
    main()
