#!/usr/bin/env python3
"""Run SAB benchmarking on COCO val2017 for all yololite variants.

For each yololite variant with an exported ONNX model (from train_coco.py),
runs SAB benchmarking with ONNX-CPU, TRT-fp32, and TRT-fp16 engines.

Expects:
  - ONNX models at  <results-dir>/onnx/{variant}.onnx
  - COCO val2017 images + instances_val2017.json annotations

Produces per-variant CSVs: <results-dir>/sab/bench_results_{variant}.csv
"""

import argparse
import os
import sys
import zipfile
from urllib.request import urlretrieve

import pandas as pd

from yololite.benchmark._io import (
    build_completed_set,
    load_variant_csv,
    save_variant_csv,
)
from yololite.benchmark._variants import VARIANT_NAMES

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_coco_val(coco_dir: str) -> tuple[str, str]:
    """Ensure COCO val2017 images and annotations exist.

    Returns (image_dir, annotations_json).
    """
    images_dir = os.path.join(coco_dir, "images")
    val_img_dir = os.path.join(images_dir, "val2017")
    ann_json = os.path.join(coco_dir, "annotations", "instances_val2017.json")

    if not os.path.isdir(val_img_dir):
        print("Downloading COCO val2017 images ...")
        os.makedirs(images_dir, exist_ok=True)
        zip_path = os.path.join(images_dir, "val2017.zip")
        if not os.path.isfile(zip_path):
            urlretrieve(COCO_VAL_IMAGES_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(images_dir)

    if not os.path.isfile(ann_json):
        print("Downloading COCO annotations ...")
        os.makedirs(os.path.join(coco_dir, "annotations"), exist_ok=True)
        zip_path = os.path.join(coco_dir, "annotations_trainval2017.zip")
        if not os.path.isfile(zip_path):
            urlretrieve(COCO_ANN_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(coco_dir)

    if not os.path.isfile(ann_json):
        print(f"ERROR: annotations not found at {ann_json}")
        sys.exit(1)

    return val_img_dir, ann_json


def _onnx_path(onnx_dir: str, variant_name: str) -> str:
    return os.path.join(onnx_dir, f"{variant_name}.onnx")


def _variant_csv(sab_dir: str, variant: str) -> str:
    return os.path.join(sab_dir, f"bench_results_{variant}.csv")


def benchmark_single(
    variant_name: str,
    onnx_path: str,
    image_dir: str,
    annotations_path: str,
) -> list[dict]:
    """Run SAB benchmarking for one variant on COCO val.

    Returns a list of result dicts, one per inference engine.
    """
    from sab.models.benchmark_yololite import (
        YoloLiteONNXCPUInference,
        YoloLiteTRTInference,
    )
    from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifact

    engines = [
        ("ONNX-CPU", ArtifactBenchmarkRequest(
            onnx_path=onnx_path,
            inference_class=YoloLiteONNXCPUInference,
            max_dets=500,
        )),
        ("TRT-fp32", ArtifactBenchmarkRequest(
            onnx_path=onnx_path,
            inference_class=YoloLiteTRTInference,
            needs_fp16=False,
            max_dets=500,
        )),
        ("TRT-fp16", ArtifactBenchmarkRequest(
            onnx_path=onnx_path,
            inference_class=YoloLiteTRTInference,
            needs_fp16=True,
            max_dets=500,
        )),
    ]

    rows = []
    for runtime_name, request in engines:
        try:
            accuracy_stats, latency_stats, throttled = run_benchmark_on_artifact(
                request, image_dir, annotations_path
            )
            rows.append({
                "variant": variant_name,
                "runtime": runtime_name,
                "fp16": request.needs_fp16,
                "mAP50": accuracy_stats[1] if len(accuracy_stats) > 1 else None,
                "mAP50_95": accuracy_stats[0] if len(accuracy_stats) > 0 else None,
                "AP75": accuracy_stats[2] if len(accuracy_stats) > 2 else None,
                "AP_s": accuracy_stats[3] if len(accuracy_stats) > 3 else None,
                "AP_m": accuracy_stats[4] if len(accuracy_stats) > 4 else None,
                "AP_l": accuracy_stats[5] if len(accuracy_stats) > 5 else None,
                "AR_maxdets": accuracy_stats[8] if len(accuracy_stats) > 8 else None,
                "latency_median_ms": latency_stats.get("median"),
                "latency_p95_ms": latency_stats.get("p95"),
                "throttled": throttled,
            })
        except Exception as e:
            print(f"  ERROR [{runtime_name}]: {e}")
            rows.append({
                "variant": variant_name,
                "runtime": runtime_name,
                "fp16": request.needs_fp16,
                "error": str(e),
            })

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run SAB benchmarking on COCO val2017 for yololite variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, default="coco_benchmark_results",
                        help="Results output directory")
    parser.add_argument("--coco-dir", type=str, default="coco_dataset",
                        help="COCO dataset directory")
    args = parser.parse_args()

    results_dir = args.results_dir
    onnx_dir = os.path.join(results_dir, "onnx")
    sab_dir = os.path.join(results_dir, "sab")

    os.makedirs(sab_dir, exist_ok=True)

    # 1. Ensure COCO val data exists
    image_dir, annotations_path = _ensure_coco_val(args.coco_dir)

    # 2. Discover available ONNX models
    available: list[tuple[str, str]] = []
    for variant in VARIANT_NAMES:
        onnx = _onnx_path(onnx_dir, variant)
        if os.path.isfile(onnx):
            available.append((variant, onnx))

    print(f"\n{'='*70}")
    print(f"Benchmarking: {len(available)} ONNX models on COCO val2017")
    print(f"{'='*70}\n")

    if not available:
        print("ERROR: No ONNX models found. Run train_coco first.")
        sys.exit(1)

    # 3. Load existing partial results for resume
    variant_rows: dict[str, list[dict]] = {v: [] for v in VARIANT_NAMES}
    for variant in VARIANT_NAMES:
        variant_rows[variant] = load_variant_csv(_variant_csv(sab_dir, variant))
    completed = build_completed_set(
        variant_rows, key_fields=("variant", "runtime"),
    )

    if completed:
        print(f"Resuming: {len(completed)} benchmark runs already completed.\n")

    done_count = len(completed)
    total = len(available) * 3  # 3 engines per variant

    # 4. Iterate variants
    for variant, onnx_path in available:
        already_done = {
            rt for rt in ("ONNX-CPU", "TRT-fp32", "TRT-fp16")
            if (variant, rt) in completed
        }
        if len(already_done) == 3:
            continue

        print(f"\n  Benchmarking {variant} ...")

        rows = benchmark_single(variant, onnx_path, image_dir, annotations_path)

        variant_csv = _variant_csv(sab_dir, variant)
        for row in rows:
            if row.get("runtime") not in already_done:
                variant_rows[variant].append(row)
                done_count += 1
                if row.get("error") is None:
                    print(
                        f"    [{done_count}/{total}] {row['runtime']}  "
                        f"mAP50={row.get('mAP50', '?'):.4f}  "
                        f"latency={row.get('latency_median_ms', '?')}ms"
                    )
                else:
                    print(f"    [{done_count}/{total}] {row['runtime']}  FAIL: {row['error']}")

        save_variant_csv(variant_rows[variant], variant_csv)

    # 5. Combined output
    all_rows = [r for rows in variant_rows.values() for r in rows]
    df = pd.DataFrame(all_rows)
    combined_csv = os.path.join(sab_dir, "bench_results_combined.csv")
    df.to_csv(combined_csv, index=False)

    print(f"\nBenchmarking complete.")
    print(f"  Combined CSV: {combined_csv}")


if __name__ == "__main__":
    main()
