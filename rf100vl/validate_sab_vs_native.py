#!/usr/bin/env python3
"""Validate SAB benchmarking results against yololite's native evaluation.

Runs both evaluation paths on a small subset of (variant, dataset) pairs:
  1. Yololite's native evaluate_on_folder() with the PyTorch checkpoint
  2. SAB's ONNX-CPU evaluation with the exported ONNX model

Compares mAP50 and mAP50:95 between the two and flags any pair where the
delta exceeds a threshold (default 1%).  This catches preprocessing
mismatches, class ID bugs, or NMS parameter differences that would
silently corrupt benchmark results.

Usage:
    python validate_sab_vs_native.py [--variants yololite-n] [--datasets circuit-voltages,bees]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = "/dev/shm/rf100vl_benchmark_results"
DATASETS_DIR = "/dev/shm/rf100vl_datasets"  # YOLOv8 format (for native eval)
COCO_DATASETS_DIR = "/dev/shm/rf100vl_datasets_coco"  # COCO format (for SAB eval)
ONNX_DIR = os.path.join(RESULTS_DIR, "onnx")
BATCH_SIZE = 16
DELTA_THRESHOLD = 0.01  # 1% mAP difference triggers a warning


def run_native_eval(checkpoint_path: str, dataset_dir: str) -> dict:
    """Run yololite's native evaluation on the YOLOv8-format test split."""
    from yololite.tools.evaluate import evaluate_on_folder

    test_folder = None
    for name in ("test", "Test"):
        candidate = os.path.join(dataset_dir, name)
        if os.path.isdir(os.path.join(candidate, "images")):
            test_folder = candidate
            break

    if test_folder is None:
        raise FileNotFoundError(f"No test split found in {dataset_dir}")

    eval_log_dir = os.path.join(RESULTS_DIR, "validation_logs", "native")
    os.makedirs(eval_log_dir, exist_ok=True)

    metrics = evaluate_on_folder(
        weights=checkpoint_path,
        test_folder=test_folder,
        batch_size=BATCH_SIZE,
        device="0",
        log_dir=eval_log_dir,
        num_workers=4,
    )

    return {
        "mAP50": metrics.get("mAP", metrics.get("AP50", 0.0)),
        "mAP50_95": metrics.get("mAP_50_95", metrics.get("AP", 0.0)),
    }


def run_sab_eval(onnx_path: str, coco_dataset_dir: str) -> dict:
    """Run SAB's ONNX-CPU evaluation on the COCO-format test split."""
    from sab.models.benchmark_yololite import YoloLiteONNXCPUInference
    from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifact

    ann_path = os.path.join(coco_dataset_dir, "test", "_annotations.coco.json")
    img_dir = os.path.join(coco_dataset_dir, "test")

    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"No COCO annotations at {ann_path}")

    request = ArtifactBenchmarkRequest(
        onnx_path=onnx_path,
        inference_class=YoloLiteONNXCPUInference,
        max_dets=500,
    )

    accuracy_stats, latency_stats, throttled = run_benchmark_on_artifact(
        request, img_dir, ann_path
    )

    return {
        "mAP50": accuracy_stats[1] if len(accuracy_stats) > 1 else 0.0,
        "mAP50_95": accuracy_stats[0] if len(accuracy_stats) > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate SAB vs native yololite eval")
    parser.add_argument(
        "--variants", type=str, default="yololite-n",
        help="Comma-separated list of variants to validate",
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset names (default: auto-select 3-5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DELTA_THRESHOLD,
        help=f"mAP delta threshold for warnings (default {DELTA_THRESHOLD})",
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]

    # Auto-select datasets if not specified: pick a few with varying properties
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    else:
        # Find datasets that have both ONNX and COCO data for the first variant
        v = variants[0]
        candidates = []
        for p in sorted(Path(ONNX_DIR).glob(f"*/{v}.onnx")):
            dname = p.parent.name
            if os.path.isfile(os.path.join(COCO_DATASETS_DIR, dname, "test", "_annotations.coco.json")):
                candidates.append(dname)
        dataset_names = candidates[:5]
        print(f"Auto-selected {len(dataset_names)} datasets: {dataset_names}")

    if not dataset_names:
        print("ERROR: No datasets available for validation.")
        sys.exit(1)

    # Run comparisons
    rows = []
    for variant in variants:
        for dataset_name in dataset_names:
            onnx = os.path.join(ONNX_DIR, dataset_name, f"{variant}.onnx")
            yolo_dir = os.path.join(DATASETS_DIR, dataset_name)
            coco_dir = os.path.join(COCO_DATASETS_DIR, dataset_name)

            # Find the best checkpoint
            ckpt = os.path.join(
                RESULTS_DIR, "runs", dataset_name, variant, "weights", "best_model_state.pt"
            )

            if not os.path.isfile(onnx):
                print(f"SKIP {variant}/{dataset_name}: no ONNX file")
                continue
            if not os.path.isfile(ckpt):
                print(f"SKIP {variant}/{dataset_name}: no checkpoint")
                continue
            if not os.path.isdir(coco_dir):
                print(f"SKIP {variant}/{dataset_name}: no COCO dataset")
                continue

            print(f"\n{'─'*60}")
            print(f"Validating {variant} / {dataset_name}")
            print(f"{'─'*60}")

            print("  Running native eval …")
            native = run_native_eval(ckpt, yolo_dir)

            print("  Running SAB ONNX-CPU eval …")
            sab = run_sab_eval(onnx, coco_dir)

            delta_50 = abs(native["mAP50"] - sab["mAP50"])
            delta_50_95 = abs(native["mAP50_95"] - sab["mAP50_95"])

            flag = ""
            if delta_50 > args.threshold or delta_50_95 > args.threshold:
                flag = " *** MISMATCH ***"

            print(f"  Native   mAP50={native['mAP50']:.4f}  mAP50:95={native['mAP50_95']:.4f}")
            print(f"  SAB      mAP50={sab['mAP50']:.4f}  mAP50:95={sab['mAP50_95']:.4f}")
            print(f"  Delta    mAP50={delta_50:.4f}  mAP50:95={delta_50_95:.4f}{flag}")

            rows.append({
                "dataset": dataset_name,
                "variant": variant,
                "native_mAP50": native["mAP50"],
                "sab_mAP50": sab["mAP50"],
                "native_mAP50_95": native["mAP50_95"],
                "sab_mAP50_95": sab["mAP50_95"],
                "delta_mAP50": delta_50,
                "delta_mAP50_95": delta_50_95,
            })

    # Summary
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(RESULTS_DIR, "validation_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*60}")
        print("Validation Summary")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"\nSaved to: {csv_path}")

        mismatches = df[(df["delta_mAP50"] > args.threshold) | (df["delta_mAP50_95"] > args.threshold)]
        if not mismatches.empty:
            print(f"\nWARNING: {len(mismatches)} pairs exceeded {args.threshold:.0%} delta threshold!")
        else:
            print(f"\nAll pairs within {args.threshold:.0%} threshold.")
    else:
        print("\nNo validation pairs were run.")


if __name__ == "__main__":
    main()
