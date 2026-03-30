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
import zipfile
from urllib.request import urlopen, urlretrieve

import pandas as pd
import yaml

# NOTE: torch is intentionally NOT imported at module level.

from yololite.benchmark._io import (
    load_training_config,
    load_variant_csv,
    save_variant_csv,
)
from yololite.benchmark._pool import NUM_CPUS, cap_threads, get_num_gpus, run_variant_pool
from yololite.benchmark._variants import YOLOLITE_VARIANTS

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 640
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 16
JOBS_PER_GPU = 2

# ── COCO download URLs ──────────────────────────────────────────────────────
COCO_TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_LABELS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "coco2017labels-segments.zip"
)
COCO_YAML_URL = (
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/"
    "ultralytics/cfg/datasets/coco.yaml"
)


# ── COCO download ───────────────────────────────────────────────────────────

def _download_and_extract(url: str, dest_dir: str) -> None:
    """Download a zip file and extract it into *dest_dir*."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_name = url.rsplit("/", 1)[-1]
    zip_path = os.path.join(dest_dir, zip_name)

    if not os.path.isfile(zip_path):
        print(f"  Downloading {zip_name} ...")
        urlretrieve(url, zip_path)
    else:
        print(f"  {zip_name} already downloaded, skipping.")

    print(f"  Extracting {zip_name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _fetch_coco_class_names() -> list[str]:
    """Fetch canonical 80-class COCO names from the ultralytics YAML."""
    with urlopen(COCO_YAML_URL) as resp:
        data = yaml.safe_load(resp.read())
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    return list(names)


def download_coco(datasets_dir: str) -> str:
    """Download COCO 2017 train/val and return the dataset directory."""
    print(f"\n{'='*70}")
    print("Downloading COCO 2017 dataset ...")
    print(f"  destination: {datasets_dir}")
    print(f"{'='*70}\n")

    images_dir = os.path.join(datasets_dir, "images")
    labels_dir = os.path.join(datasets_dir, "labels")

    # Train images
    if not os.path.isdir(os.path.join(images_dir, "train2017")):
        _download_and_extract(COCO_TRAIN_IMAGES_URL, images_dir)
    else:
        print("  train2017 images already present, skipping.")

    # Val images
    if not os.path.isdir(os.path.join(images_dir, "val2017")):
        _download_and_extract(COCO_VAL_IMAGES_URL, images_dir)
    else:
        print("  val2017 images already present, skipping.")

    # YOLO-format labels
    if not (
        os.path.isdir(os.path.join(labels_dir, "train2017"))
        and os.path.isdir(os.path.join(labels_dir, "val2017"))
    ):
        _download_and_extract(COCO_LABELS_URL, datasets_dir)
        # The zip extracts to coco/labels/ — move if needed
        nested = os.path.join(datasets_dir, "coco", "labels")
        if os.path.isdir(nested) and not os.path.isdir(labels_dir):
            os.rename(nested, labels_dir)
    else:
        print("  YOLO labels already present, skipping.")

    # data.yaml
    data_yaml_path = os.path.join(datasets_dir, "data.yaml")
    if not os.path.isfile(data_yaml_path):
        names = _fetch_coco_class_names()
        data = {
            "train": os.path.abspath(os.path.join(images_dir, "train2017")),
            "val": os.path.abspath(os.path.join(images_dir, "val2017")),
            "labels": {
                "train": os.path.abspath(os.path.join(labels_dir, "train2017")),
                "val": os.path.abspath(os.path.join(labels_dir, "val2017")),
            },
            "nc": len(names),
            "names": names,
        }
        with open(data_yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"  Created {data_yaml_path} ({len(names)} classes)")
    else:
        print(f"  {data_yaml_path} already exists, skipping.")

    print(f"\nCOCO dataset ready at: {datasets_dir}")
    return datasets_dir


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

    data_yaml = os.path.join(dataset_dir, "data.yaml")
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
