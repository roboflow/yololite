#!/usr/bin/env python3
"""Train all yololite variants on COCO and export each to decoded ONNX.

Downloads the COCO 2017 dataset (train/val splits) in YOLO format and trains
every yololite variant supported in roboflow-train.  After training, the best
checkpoint is exported to a decoded ONNX file.

Each variant is trained sequentially.  When multiple GPUs are available, one
variant is assigned to each GPU and variants run in parallel up to the number
of GPUs.
"""

import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files as _pkg_files
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yaml

# NOTE: torch is intentionally NOT imported at module level.  Spawned worker
# processes re-import this module, and any module-level torch import would
# initialize CUDA before the pool initializer can set CUDA_VISIBLE_DEVICES.

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS_DIR = os.environ.get("COCO_DATASETS_DIR", "coco_dataset")
RESULTS_DIR = os.environ.get("COCO_RESULTS_DIR", "coco_benchmark_results")
ONNX_DIR = os.path.join(RESULTS_DIR, "onnx")
IMG_SIZE = 640

EPOCHS = int(os.environ.get("COCO_EPOCHS", 300))
BATCH_SIZE = int(os.environ.get("COCO_BATCH_SIZE", 16))

# ── COCO download URLs ──────────────────────────────────────────────────────
COCO_TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_LABELS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
)
COCO_YAML_URL = (
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/"
    "ultralytics/cfg/datasets/coco.yaml"
)

# ── GPU concurrency ───────────────────────────────────────────────────────────
_SPAWN_CTX = multiprocessing.get_context("spawn")


def _get_num_gpus() -> int:
    import torch
    return torch.cuda.device_count()


# All yololite variants supported in roboflow-train.
# Each entry: (config_subdir, config_yaml).
YOLOLITE_VARIANTS = {
    "yololite-n":       ("v2_models", "yololite_n.yaml"),
    "yololite-edge-n":  ("models",    "edge_n.yaml"),
    "yololite-s":       ("v2_models", "yololite_s.yaml"),
    "yololite-edge-s":  ("models",    "edge_s.yaml"),
    "yololite-m":       ("v2_models", "yololite_m.yaml"),
    "yololite-edge-m":  ("models",    "edge_m.yaml"),
    "yololite-l":       ("v2_models", "yololite_l.yaml"),
    "yololite-edge-l":  ("models",    "edge_l.yaml"),
    "yololite-xl":      ("models",    "yololite_xl.yaml"),
    "yololite-edge-xl": ("models",    "edge_xl.yaml"),
}


# ── COCO download ────────────────────────────────────────────────────────────

def _download_and_extract(url: str, dest_dir: str) -> None:
    """Download a zip from *url* into *dest_dir* and extract it."""
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    zip_path = os.path.join(dest_dir, filename)

    if not os.path.isfile(zip_path):
        print(f"  Downloading {filename} …")
        subprocess.run(
            ["curl", "-L", "-o", zip_path, url],
            check=True,
        )
    else:
        print(f"  {filename} already downloaded, skipping.")

    print(f"  Extracting {filename} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _fetch_coco_class_names() -> list[str]:
    """Fetch the canonical COCO class names from the ultralytics coco.yaml."""
    with urlopen(COCO_YAML_URL) as resp:
        coco_cfg = yaml.safe_load(resp.read())
    names = coco_cfg["names"]
    # ultralytics uses {0: "person", 1: "bicycle", ...} — convert to list
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def download_coco() -> str:
    """Download COCO 2017 in YOLO format and return the dataset directory.

    Final layout::

        {DATASETS_DIR}/
        ├── images/
        │   ├── train2017/   (118k images)
        │   └── val2017/     (5k images)
        ├── labels/
        │   ├── train2017/   (YOLO .txt labels)
        │   └── val2017/
        └── data.yaml
    """
    print(f"\n{'='*70}")
    print("Downloading COCO 2017 dataset …")
    print(f"  destination: {DATASETS_DIR}")
    print(f"{'='*70}\n")

    images_dir = os.path.join(DATASETS_DIR, "images")
    labels_dir = os.path.join(DATASETS_DIR, "labels")

    # Download images
    if not os.path.isdir(os.path.join(images_dir, "train2017")):
        _download_and_extract(COCO_TRAIN_IMAGES_URL, images_dir)
    else:
        print("  train2017 images already present, skipping.")

    if not os.path.isdir(os.path.join(images_dir, "val2017")):
        _download_and_extract(COCO_VAL_IMAGES_URL, images_dir)
    else:
        print("  val2017 images already present, skipping.")

    # Download YOLO-format labels (from ultralytics assets)
    if not (
        os.path.isdir(os.path.join(labels_dir, "train2017"))
        and os.path.isdir(os.path.join(labels_dir, "val2017"))
    ):
        _download_and_extract(COCO_LABELS_URL, DATASETS_DIR)

        # The zip extracts into a nested coco/ subdirectory — move labels up
        nested_labels = os.path.join(DATASETS_DIR, "coco", "labels")
        if os.path.isdir(nested_labels) and not os.path.isdir(labels_dir):
            shutil.move(nested_labels, labels_dir)

        # Clean up nested coco directory
        nested_coco = os.path.join(DATASETS_DIR, "coco")
        if os.path.isdir(nested_coco):
            shutil.rmtree(nested_coco)
    else:
        print("  YOLO labels already present, skipping.")

    # Write data.yaml (yololite format: names as list, directory paths)
    data_yaml_path = os.path.join(DATASETS_DIR, "data.yaml")
    if not os.path.isfile(data_yaml_path):
        class_names = _fetch_coco_class_names()
        data = {
            "train": os.path.abspath(os.path.join(images_dir, "train2017")),
            "val": os.path.abspath(os.path.join(images_dir, "val2017")),
            "labels": {
                "train": os.path.abspath(os.path.join(labels_dir, "train2017")),
                "val": os.path.abspath(os.path.join(labels_dir, "val2017")),
            },
            "nc": len(class_names),
            "names": class_names,
        }
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\nCOCO dataset ready at {DATASETS_DIR}")
    return DATASETS_DIR


# ── Training ──────────────────────────────────────────────────────────────────

def run_single_training(
    variant_name: str,
    dataset_dir: str,
    device: str = "0",
    max_concurrent: int = 1,
) -> dict:
    """Train one yololite variant on COCO, then export to ONNX."""
    import torch

    torch.set_num_threads(max(1, os.cpu_count() // max_concurrent))

    from yololite.scripts.args.build_args import load_configs
    from yololite.tools.train import run_training
    from yololite.export.export_onnx import export_decoded_onnx

    configs_root = _pkg_files("yololite").joinpath("configs")
    subdir, yaml_file = YOLOLITE_VARIANTS[variant_name]
    model_yaml = str(configs_root / subdir / yaml_file)
    train_yaml = str(configs_root / "train" / "standard_train.yaml")
    data_yaml = os.path.join(dataset_dir, "data.yaml")

    log_dir = os.path.join(RESULTS_DIR, "runs", variant_name)
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
    config["training"]["save_every"] = 50
    config["training"]["save_by"] = "AP"
    config["training"]["device"] = device

    # ── Train ──
    t0 = time.time()
    train_result = run_training(config)
    train_elapsed = time.time() - t0

    # ── Export to ONNX ──
    best_ckpt = train_result["best_checkpoint"]
    onnx_out = _onnx_path(variant_name)
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

def _pool_initializer(gpu_queue: multiprocessing.Queue, max_workers: int) -> None:
    """Restrict this worker to a single GPU and cap CPU threads."""
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    threads = str(max(1, os.cpu_count() // max_workers))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads


def _worker(args: tuple) -> dict:
    variant_name, dataset_dir, max_concurrent = args
    try:
        return run_single_training(variant_name, dataset_dir, "0", max_concurrent)
    except Exception as e:
        traceback.print_exc()
        return {
            "variant": variant_name,
            "epochs_completed": 0,
            "train_s": 0,
            "export_s": 0,
            "error": str(e),
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _onnx_path(variant_name: str) -> str:
    return os.path.join(ONNX_DIR, f"{variant_name}.onnx")


def _variant_csv(variant: str) -> str:
    return os.path.join(RESULTS_DIR, f"train_results_{variant}.csv")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ONNX_DIR, exist_ok=True)

    # 1. Download COCO
    dataset_dir = download_coco()

    num_gpus = _get_num_gpus()
    total = len(YOLOLITE_VARIANTS)

    print(f"\n{'='*70}")
    print(f"Training {total} yololite variants on COCO")
    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  GPUs: {num_gpus}")
    print(f"{'='*70}\n")

    # 2. Load existing partial results for resume
    completed = set()
    results: dict[str, dict] = {}
    for variant in YOLOLITE_VARIANTS:
        csv_path = _variant_csv(variant)
        if os.path.isfile(csv_path):
            existing = pd.read_csv(csv_path)
            if not existing.empty:
                results[variant] = existing.to_dict("records")[0]
                completed.add(variant)

    if completed:
        print(f"Resuming: {len(completed)} variants already completed, "
              f"{total - len(completed)} remaining.\n")

    # 3. Build work queue (skip completed variants)
    pending = [
        v for v in YOLOLITE_VARIANTS
        if v not in completed and not os.path.isfile(_onnx_path(v))
    ]

    if not pending:
        print("All variants already trained.")
    elif num_gpus <= 1:
        # Sequential: one variant at a time
        for variant in pending:
            print(f"\n{'─'*70}")
            print(f"[{variant}] Starting COCO training")
            print(f"{'─'*70}")

            result = _worker((variant, dataset_dir, 1))
            results[variant] = result

            status = "OK" if result.get("error") is None else f"FAIL: {result['error']}"
            metrics = result.get("best_metrics", {})
            ap_str = f"  AP={metrics.get('AP', 0):.3f}" if metrics else ""
            print(
                f"[{len(results)}/{total}] {variant}  "
                f"(train={result['train_s']}s  export={result.get('export_s', 0)}s)"
                f"{ap_str}  [{status}]"
            )

            pd.DataFrame([result]).to_csv(_variant_csv(variant), index=False)
    else:
        # Parallel: one variant per GPU
        gpu_queue = _SPAWN_CTX.Queue()
        for gpu_id in range(num_gpus):
            gpu_queue.put(gpu_id)

        pool = ProcessPoolExecutor(
            max_workers=num_gpus,
            mp_context=_SPAWN_CTX,
            initializer=_pool_initializer,
            initargs=(gpu_queue, num_gpus),
        )

        try:
            futures = {
                pool.submit(_worker, (variant, dataset_dir, num_gpus)): variant
                for variant in pending
            }

            for future in as_completed(futures):
                result = future.result()
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

                pd.DataFrame([result]).to_csv(_variant_csv(variant), index=False)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # 4. Summary
    all_rows = list(results.values())
    df = pd.DataFrame(all_rows)
    combined_csv = os.path.join(RESULTS_DIR, "train_results_combined.csv")
    df.to_csv(combined_csv, index=False)

    print(f"\nTraining + export complete.")
    print(f"  Combined CSV: {combined_csv}")
    print(f"  ONNX models:  {ONNX_DIR}/")


if __name__ == "__main__":
    main()
