#!/usr/bin/env python3
"""Train all yololite variants on RF100-VL and export each to decoded ONNX.

Downloads all RF100-VL datasets (yolov8 format) and trains every yololite
variant for 100 epochs with batch_size=16.  After training, the best
checkpoint is exported to a decoded ONNX file for downstream benchmarking
via single_artifact_benchmarking.

Concurrency: training jobs run in parallel via a process pool, one variant at
a time.  Each variant specifies how many concurrent jobs fit per GPU (based on
VRAM footprint), so small models get higher parallelism than large ones.
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

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS_DIR = os.environ.get("RF100VL_DATASETS_DIR", "rf100vl_datasets")
RESULTS_DIR = os.environ.get("RF100VL_RESULTS_DIR", "rf100vl_benchmark_results")
ONNX_DIR = os.path.join(RESULTS_DIR, "onnx")
GCS_BUCKET = "gs://rf-detr-rf100-vl/yololite-benchmark"
DATASETS_FORMAT = "yolov8"
IMG_SIZE = 640

EPOCHS = 100
BATCH_SIZE = 16

# ── GPU concurrency ───────────────────────────────────────────────────────────
NUM_CPUS = os.cpu_count() or 120
_SPAWN_CTX = multiprocessing.get_context("spawn")


def _get_num_gpus() -> int:
    import torch
    return torch.cuda.device_count()


# All yololite variants supported in roboflow-train.
# Each entry: (config_subdir, config_yaml, jobs_per_gpu).
YOLOLITE_VARIANTS = {
    "yololite-n":       ("v2_models", "yololite_n.yaml",  4),
    "yololite-edge-n":  ("models",    "edge_n.yaml",      4),
    "yololite-s":       ("v2_models", "yololite_s.yaml",  4),
    "yololite-edge-s":  ("models",    "edge_s.yaml",      4),
    "yololite-m":       ("v2_models", "yololite_m.yaml",  4),
    "yololite-edge-m":  ("models",    "edge_m.yaml",      4),
    "yololite-l":       ("v2_models", "yololite_l.yaml",  2),
    "yololite-edge-l":  ("models",    "edge_l.yaml",      2),
    "yololite-xl":      ("models",    "yololite_xl.yaml", 1),
    "yololite-edge-xl": ("models",    "edge_xl.yaml",     1),
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
    """Download all RF100-VL datasets in yolov8 format and return directory list."""
    from rf100vl import download_rf100vl

    print(f"\n{'='*70}")
    print("Downloading RF100-VL datasets …")
    print(f"  destination: {DATASETS_DIR}")
    print(f"  format:      {DATASETS_FORMAT}")
    print(f"{'='*70}\n")

    download_rf100vl(
        path=DATASETS_DIR,
        model_format=DATASETS_FORMAT,
        overwrite=False,
    )

    dataset_dirs = sorted(
        str(p.parent)
        for p in Path(DATASETS_DIR).rglob("data.yaml")
    )
    print(f"\nFound {len(dataset_dirs)} datasets with data.yaml")
    return dataset_dirs


def _find_data_yaml(dataset_dir: str) -> str:
    candidate = os.path.join(dataset_dir, "data.yaml")
    if os.path.isfile(candidate):
        return candidate
    for p in Path(dataset_dir).glob("*/data.yaml"):
        return str(p)
    raise FileNotFoundError(f"No data.yaml found in {dataset_dir}")


def _onnx_path(dataset_name: str, variant_name: str) -> str:
    return os.path.join(ONNX_DIR, dataset_name, f"{variant_name}.onnx")


def run_single_training(
    variant_name: str,
    dataset_dir: str,
    dataset_name: str,
    device: str = "0",
    max_concurrent: int = 1,
) -> dict:
    """Train one yololite variant on one dataset, then export to ONNX."""
    import torch

    torch.set_num_threads(max(1, NUM_CPUS // max_concurrent))

    from yololite.scripts.args.build_args import load_configs
    from yololite.tools.train import run_training
    from yololite.export.export_onnx import export_decoded_onnx

    configs_root = _pkg_files("yololite").joinpath("configs")
    subdir, yaml_file, _jobs_per_gpu = YOLOLITE_VARIANTS[variant_name]
    model_yaml = str(configs_root / subdir / yaml_file)
    train_yaml = str(configs_root / "train" / "standard_train.yaml")
    data_yaml = _find_data_yaml(dataset_dir)

    log_dir = os.path.join(RESULTS_DIR, "runs", dataset_name, variant_name)
    os.makedirs(log_dir, exist_ok=True)

    config = load_configs(
        model_yaml=model_yaml,
        train_yaml=train_yaml,
        data_yaml=data_yaml,
        log_dir=log_dir,
    )

    config["training"]["epochs"] = EPOCHS
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["img_size"] = IMG_SIZE
    config["training"]["save_every"] = EPOCHS + 1  # no periodic saves
    config["training"]["save_by"] = "AP"
    config["training"]["num_workers"] = max(1, NUM_CPUS // max_concurrent)
    config["training"]["device"] = device

    # ── Train ──
    t0 = time.time()
    train_result = run_training(config)
    train_elapsed = time.time() - t0

    # ── Export to ONNX ──
    best_ckpt = train_result["best_checkpoint"]
    onnx_out = _onnx_path(dataset_name, variant_name)
    os.makedirs(os.path.dirname(onnx_out), exist_ok=True)

    t1 = time.time()
    export_decoded_onnx(
        checkpoint_path=best_ckpt,
        img_size=IMG_SIZE,
        out_path=onnx_out,
    )
    export_elapsed = time.time() - t1

    return {
        "dataset": dataset_name,
        "variant": variant_name,
        "epochs_completed": train_result.get("epochs_completed", 0),
        "train_s": round(train_elapsed, 1),
        "export_s": round(export_elapsed, 1),
        "best_checkpoint": best_ckpt,
        "onnx_path": onnx_out,
    }


# ── Pool management ──────────────────────────────────────────────────────────

def _pool_initializer(gpu_queue: multiprocessing.Queue, max_concurrent: int) -> None:
    """Grab a GPU ID from the queue and restrict this worker to it."""
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    threads = str(max(1, NUM_CPUS // max_concurrent))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads


def _worker(args: tuple) -> dict:
    variant_name, dataset_dir, dataset_name, max_concurrent = args
    try:
        return run_single_training(variant_name, dataset_dir, dataset_name, "0", max_concurrent)
    except Exception as e:
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "variant": variant_name,
            "epochs_completed": 0,
            "train_s": 0,
            "export_s": 0,
            "error": str(e),
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def _variant_csv(variant: str) -> str:
    return os.path.join(RESULTS_DIR, f"train_results_{variant}.csv")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ONNX_DIR, exist_ok=True)

    # 1. Download datasets
    dataset_dirs = download_datasets()
    if not dataset_dirs:
        print("ERROR: No datasets found after download. Check ROBOFLOW_API_KEY.")
        sys.exit(1)

    total = len(dataset_dirs) * len(YOLOLITE_VARIANTS)
    num_gpus = _get_num_gpus()

    print(f"\n{'='*70}")
    print(f"Training matrix: {len(dataset_dirs)} datasets × "
          f"{len(YOLOLITE_VARIANTS)} variants = {total} runs")
    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  GPUs: {num_gpus}")
    print(f"{'='*70}\n")

    # 2. Load existing partial results for resume
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

    # 3. Process one variant at a time
    for variant, (_subdir, _yaml, jobs_per_gpu) in YOLOLITE_VARIANTS.items():
        max_concurrent = num_gpus * jobs_per_gpu

        variant_jobs = []
        for ddir in dataset_dirs:
            dname = Path(ddir).name
            if (dname, variant) in completed:
                continue
            # Also skip if ONNX already exists (export succeeded but CSV wasn't written)
            if os.path.isfile(_onnx_path(dname, variant)):
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

        gpu_queue = _SPAWN_CTX.Queue()
        for gpu_id in range(num_gpus):
            for _ in range(jobs_per_gpu):
                gpu_queue.put(gpu_id)

        pool = ProcessPoolExecutor(
            max_workers=max_concurrent,
            mp_context=_SPAWN_CTX,
            initializer=_pool_initializer,
            initargs=(gpu_queue, max_concurrent),
        )

        try:
            futures = {pool.submit(_worker, job): job for job in variant_jobs}

            for future in as_completed(futures):
                result = future.result()
                variant_rows[variant].append(result)
                done_count += 1

                status = "OK" if result.get("error") is None else f"FAIL: {result['error']}"
                print(
                    f"[{done_count}/{total}] {result['dataset']} / "
                    f"{result['variant']}  "
                    f"(train={result['train_s']}s  export={result.get('export_s', 0)}s)  "
                    f"[{status}]"
                )

                pd.DataFrame(variant_rows[variant]).to_csv(variant_csv, index=False)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        sync_results_to_gcs()

    # 4. Summary
    all_rows = [r for rows in variant_rows.values() for r in rows]
    df = pd.DataFrame(all_rows)
    combined_csv = os.path.join(RESULTS_DIR, "train_results_combined.csv")
    df.to_csv(combined_csv, index=False)

    sync_results_to_gcs()

    print(f"\nTraining + export complete.")
    print(f"  Combined CSV: {combined_csv}")
    print(f"  ONNX models:  {ONNX_DIR}/")
    print(f"  GCS:          {GCS_BUCKET}/")


if __name__ == "__main__":
    main()
