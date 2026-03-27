#!/usr/bin/env python3
"""Orchestrate single_artifact_benchmarking across all RF100-VL datasets.

For each yololite variant with exported ONNX models (from train_and_export.py),
runs SAB benchmarking with ONNX-CPU, TRT-fp32, and TRT-fp16 engines.

Expects:
  - ONNX models at  RESULTS_DIR/onnx/{dataset}/{variant}.onnx
  - COCO-format datasets at COCO_DATASETS_DIR/{dataset}/test/
    (downloaded by this script if missing)

Produces per-variant CSVs: RESULTS_DIR/bench_results_{variant}.csv
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
RESULTS_DIR = os.environ.get("RF100VL_RESULTS_DIR", "rf100vl_benchmark_results")
ONNX_DIR = os.path.join(RESULTS_DIR, "onnx")
COCO_DATASETS_DIR = os.environ.get("RF100VL_COCO_DIR", "rf100vl_datasets_coco")
GCS_BUCKET = "gs://rf-detr-rf100-vl/yololite-benchmark"

# Same variant list as train_and_export.py (only names needed here)
YOLOLITE_VARIANTS = [
    "yololite-n", "yololite-edge-n",
    "yololite-s", "yololite-edge-s",
    "yololite-m", "yololite-edge-m",
    "yololite-l", "yololite-edge-l",
    "yololite-xl", "yololite-edge-xl",
]


# ── GCS upload ────────────────────────────────────────────────────────────────

def sync_results_to_gcs() -> None:
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

def download_coco_datasets() -> list[str]:
    """Download RF100-VL datasets in COCO format for SAB evaluation."""
    from rf100vl import download_rf100vl

    print(f"\n{'='*70}")
    print("Downloading RF100-VL datasets (COCO format) …")
    print(f"  destination: {COCO_DATASETS_DIR}")
    print(f"{'='*70}\n")

    download_rf100vl(
        path=COCO_DATASETS_DIR,
        model_format="coco",
        overwrite=False,
    )

    # Find all dataset directories that have a test split with annotations
    dataset_dirs = sorted(
        str(p.parent.parent)
        for p in Path(COCO_DATASETS_DIR).rglob("test/_annotations.coco.json")
    )
    print(f"\nFound {len(dataset_dirs)} COCO datasets with test annotations")
    return dataset_dirs


def _find_coco_test(dataset_dir: str) -> tuple[str, str] | None:
    """Return (image_dir, annotations_json) for the COCO test split."""
    ann_path = os.path.join(dataset_dir, "test", "_annotations.coco.json")
    img_dir = os.path.join(dataset_dir, "test")
    if os.path.isfile(ann_path) and os.path.isdir(img_dir):
        return img_dir, ann_path
    return None


def _onnx_path(dataset_name: str, variant_name: str) -> str:
    return os.path.join(ONNX_DIR, dataset_name, f"{variant_name}.onnx")


def _variant_csv(variant: str) -> str:
    return os.path.join(RESULTS_DIR, f"bench_results_{variant}.csv")


def benchmark_single(
    variant_name: str,
    dataset_name: str,
    onnx_path: str,
    image_dir: str,
    annotations_path: str,
) -> list[dict]:
    """Run SAB benchmarking for one (variant, dataset) pair.

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
                "dataset": dataset_name,
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
                "dataset": dataset_name,
                "variant": variant_name,
                "runtime": runtime_name,
                "fp16": request.needs_fp16,
                "error": str(e),
            })

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Download COCO-format datasets
    coco_dataset_dirs = download_coco_datasets()
    if not coco_dataset_dirs:
        print("ERROR: No COCO datasets found. Check ROBOFLOW_API_KEY.")
        sys.exit(1)

    coco_by_name = {Path(d).name: d for d in coco_dataset_dirs}

    # 2. Discover available ONNX models, keyed by dataset
    datasets_with_models: dict[str, list[tuple[str, str]]] = {}  # dataset -> [(variant, onnx)]
    for dataset_name in sorted(coco_by_name.keys()):
        for variant in YOLOLITE_VARIANTS:
            onnx = _onnx_path(dataset_name, variant)
            if os.path.isfile(onnx):
                datasets_with_models.setdefault(dataset_name, []).append((variant, onnx))

    total_models = sum(len(v) for v in datasets_with_models.values())
    print(f"\n{'='*70}")
    print(f"Benchmarking: {total_models} ONNX models across "
          f"{len(datasets_with_models)} datasets")
    print(f"{'='*70}\n")

    if not datasets_with_models:
        print("ERROR: No ONNX models found. Run train_and_export.py first.")
        sys.exit(1)

    # 3. Load existing partial results for resume
    completed = set()
    variant_rows: dict[str, list[dict]] = {v: [] for v in YOLOLITE_VARIANTS}
    for variant in YOLOLITE_VARIANTS:
        csv_path = _variant_csv(variant)
        if os.path.isfile(csv_path):
            existing = pd.read_csv(csv_path)
            variant_rows[variant] = existing.to_dict("records")
            for r in variant_rows[variant]:
                completed.add((r["dataset"], r["variant"], r["runtime"]))

    if completed:
        print(f"Resuming: {len(completed)} benchmark runs already completed.\n")

    done_count = len(completed)
    total = total_models * 3  # 3 engines per (variant, dataset)

    # 4. Iterate datasets first, then variants within each dataset
    for dataset_name, variant_onnx_list in list(datasets_with_models.items())[:3]:
        coco_dir = coco_by_name.get(dataset_name)
        if coco_dir is None:
            print(f"SKIP {dataset_name}: no COCO dataset found")
            continue

        test_info = _find_coco_test(coco_dir)
        if test_info is None:
            print(f"SKIP {dataset_name}: no test split with annotations")
            continue

        image_dir, annotations_path = test_info

        # Filter to variants that still need work
        pending = [
            (variant, onnx_path) for variant, onnx_path in variant_onnx_list
            if not all(
                (dataset_name, variant, rt) in completed
                for rt in ("ONNX-CPU", "TRT-fp32", "TRT-fp16")
            )
        ]

        if not pending:
            continue

        print(f"\n{'─'*70}")
        print(f"[{dataset_name}] {len(pending)} variants remaining")
        print(f"{'─'*70}")

        for variant, onnx_path in pending:
            already_done = {
                rt for rt in ("ONNX-CPU", "TRT-fp32", "TRT-fp16")
                if (dataset_name, variant, rt) in completed
            }
            if len(already_done) == 3:
                continue

            print(f"\n  Benchmarking {dataset_name} / {variant} …")

            rows = benchmark_single(
                variant, dataset_name, onnx_path, image_dir, annotations_path
            )

            variant_csv = _variant_csv(variant)
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

            # Save per-variant CSV incrementally
            pd.DataFrame(variant_rows[variant]).to_csv(variant_csv, index=False)

        sync_results_to_gcs()

    # 5. Combined output
    all_rows = [r for rows in variant_rows.values() for r in rows]
    df = pd.DataFrame(all_rows)
    combined_csv = os.path.join(RESULTS_DIR, "bench_results_combined.csv")
    df.to_csv(combined_csv, index=False)

    sync_results_to_gcs()

    print(f"\nBenchmarking complete.")
    print(f"  Combined CSV: {combined_csv}")
    print(f"  GCS:          {GCS_BUCKET}/")


if __name__ == "__main__":
    main()
