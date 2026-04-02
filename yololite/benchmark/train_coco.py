#!/usr/bin/env python3
"""Train all yololite variants on COCO 2017 and export each to decoded ONNX.

Downloads the COCO 2017 dataset (train/val splits) in YOLO format and trains
every yololite variant.  After training, the best checkpoint is exported to a
decoded ONNX file.

Each variant is trained sequentially when only one GPU is available.  When
multiple GPUs are available, variants run in parallel up to JOBS_PER_GPU
concurrent jobs per GPU.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# NOTE: torch is intentionally NOT imported at module level.

from yololite.benchmark._io import (
    find_data_yaml,
    load_training_config,
    load_variant_csv,
    save_variant_csv,
)
from yololite.benchmark._pool import NUM_CPUS, cap_threads, get_num_gpus, run_variant_pool
from yololite.benchmark._variants import YOLOLITE_VARIANTS

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 640
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 8
JOBS_PER_GPU = 2

ROBOFLOW_WORKSPACE = "microsoft"
ROBOFLOW_PROJECT = "coco"
ROBOFLOW_VERSION = 9
DATASETS_FORMAT = "yolov8"


# ── COCO download ───────────────────────────────────────────────────────────

def download_coco(datasets_dir: str) -> str:
    """Download COCO via Roboflow API in yolov8 format and return the dataset directory."""
    from roboflow import Roboflow

    print(f"\n{'='*70}")
    print("Downloading COCO dataset via Roboflow API ...")
    print(f"  workspace: {ROBOFLOW_WORKSPACE}")
    print(f"  project:   {ROBOFLOW_PROJECT}")
    print(f"  version:   {ROBOFLOW_VERSION}")
    print(f"  format:    {DATASETS_FORMAT}")
    print(f"  destination: {datasets_dir}")
    print(f"{'='*70}\n")

    # Check if already downloaded (data.yaml present)
    try:
        data_yaml = find_data_yaml(datasets_dir)
        print(f"Dataset already present at {datasets_dir}, skipping download.")
        return str(Path(data_yaml).parent)
    except FileNotFoundError:
        pass

    rf = Roboflow()
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(DATASETS_FORMAT, location=datasets_dir, overwrite=False)

    dataset_dir = dataset.location
    print(f"\nCOCO dataset ready at: {dataset_dir}")
    return dataset_dir


# ── Training ─────────────────────────────────────────────────────────────────

def run_single_training(
    variant_name: str,
    dataset_dir: str,
    results_dir: str,
    onnx_dir: str,
    device: str = "0",
    max_concurrent: int = 1,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Train one yololite variant on COCO, then export to ONNX."""
    cap_threads(max_concurrent)

    from yololite.export.export_onnx import export_decoded_onnx
    from yololite.tools.train import run_training

    data_yaml = find_data_yaml(dataset_dir)
    log_dir = os.path.join(results_dir, "runs", variant_name)
    os.makedirs(log_dir, exist_ok=True)

    config = load_training_config(
        variant_name,
        data_yaml,
        log_dir,
        epochs=epochs,
        batch_size=batch_size,
        img_size=IMG_SIZE,
        device=device,
        save_every=50,
        num_workers=max(1, NUM_CPUS // max_concurrent),
    )

    # ── Train ──
    t0 = time.time()
    train_result = run_training(config)
    train_elapsed = time.time() - t0

    # ── Export to ONNX ──
    best_ckpt = train_result["best_checkpoint"]
    onnx_out = os.path.join(onnx_dir, f"{variant_name}.onnx")
    os.makedirs(os.path.dirname(onnx_out), exist_ok=True)

    t1 = time.time()
    export_decoded_onnx(
        checkpoint_path=best_ckpt,
        img_size=IMG_SIZE,
        out_path=onnx_out,
    )
    export_elapsed = time.time() - t1

    return {
        "variant": variant_name,
        "epochs_completed": train_result.get("epochs_completed", 0),
        "best_metrics": train_result.get("best_metrics", {}),
        "train_s": round(train_elapsed, 1),
        "export_s": round(export_elapsed, 1),
        "best_checkpoint": best_ckpt,
        "onnx_path": onnx_out,
    }


# ── Pool management ──────────────────────────────────────────────────────────

def _worker(args: tuple) -> dict:
    (variant_name, dataset_dir, results_dir, onnx_dir,
     max_concurrent, epochs, batch_size) = args
    try:
        return run_single_training(
            variant_name, dataset_dir, results_dir, onnx_dir,
            "0", max_concurrent, epochs, batch_size,
        )
    except Exception as e:
        traceback.print_exc()
        return {
            "variant": variant_name,
            "epochs_completed": 0,
            "train_s": 0,
            "export_s": 0,
            "error": str(e),
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train yololite variants on COCO 2017 and export to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--datasets-dir", type=str, default="coco_dataset",
                        help="COCO dataset directory")
    parser.add_argument("--results-dir", type=str, default="coco_benchmark_results",
                        help="Results output directory")
    args = parser.parse_args()

    results_dir = args.results_dir
    onnx_dir = os.path.join(results_dir, "onnx")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)

    # 1. Download COCO
    dataset_dir = download_coco(args.datasets_dir)

    num_gpus = get_num_gpus()
    total = len(YOLOLITE_VARIANTS)

    print(f"\n{'='*70}")
    print(f"Training {total} yololite variants on COCO")
    print(f"Epochs: {args.epochs}  |  Batch size: {args.batch_size}  |  GPUs: {num_gpus}")
    print(f"{'='*70}\n")

    # Helper closures over results_dir / onnx_dir
    def variant_csv(variant: str) -> str:
        return os.path.join(results_dir, f"train_results_{variant}.csv")

    def onnx_path(variant: str) -> str:
        return os.path.join(onnx_dir, f"{variant}.onnx")

    # 2. Load existing partial results for resume
    completed: set[str] = set()
    results: dict[str, dict] = {}
    for variant in YOLOLITE_VARIANTS:
        rows = load_variant_csv(variant_csv(variant))
        if rows:
            results[variant] = rows[0]
            completed.add(variant)

    if completed:
        print(f"Resuming: {len(completed)}/{total} variants already completed.\n")

    # 3. Build work queue
    pending = [
        v for v in YOLOLITE_VARIANTS
        if v not in completed and not os.path.isfile(onnx_path(v))
    ]

    if not pending:
        print("All variants already trained.")
    else:
        max_concurrent = num_gpus * JOBS_PER_GPU
        jobs = [
            (variant, dataset_dir, results_dir, onnx_dir,
             max_concurrent, args.epochs, args.batch_size)
            for variant in pending
        ]

        def _on_result(result: dict) -> None:
            variant = result["variant"]
            results[variant] = result

            status = "OK" if result.get("error") is None else f"FAIL: {result['error']}"
            metrics = result.get("best_metrics", {})
            ap_str = f"  AP={metrics.get('AP', 0):.3f}" if metrics else ""
            print(
                f"[{len(results)}/{total}] {variant}  "
                f"(train={result['train_s']}s  export={result.get('export_s', 0)}s)"
                f"{ap_str}  [{status}]"
            )
            save_variant_csv([result], variant_csv(variant))

        run_variant_pool(
            jobs, _worker, num_gpus, jobs_per_gpu=JOBS_PER_GPU,
            on_result=_on_result,
        )

    # 4. Summary
    all_rows = list(results.values())
    combined_csv = os.path.join(results_dir, "train_results_combined.csv")
    pd.DataFrame(all_rows).to_csv(combined_csv, index=False)

    print(f"\nTraining + export complete.")
    print(f"  Combined CSV: {combined_csv}")
    print(f"  ONNX models:  {onnx_dir}/")


if __name__ == "__main__":
    main()
