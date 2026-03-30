#!/usr/bin/env python3
"""Run SAB benchmarking across all RF100-VL datasets.

For each yololite variant with exported ONNX models (from train_rf100vl.py),
runs SAB benchmarking with ONNX-CPU, TRT-fp32, and TRT-fp16 engines.

Expects:
  - ONNX models at  <results-dir>/onnx/{dataset}/{variant}.onnx
  - COCO-format datasets at <coco-datasets-dir>/{dataset}/test/
    (downloaded by this script if missing)

Produces per-variant CSVs: <results-dir>/sab/bench_results_{variant}.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from yololite.benchmark._io import (
    build_completed_set,
    load_variant_csv,
    save_variant_csv,
)
from yololite.benchmark._variants import VARIANT_NAMES


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_coco_datasets(coco_datasets_dir: str) -> list[str]:
    """Download RF100-VL datasets in COCO format for SAB evaluation."""
    from rf100vl import download_rf100vl

    print(f"\n{'='*70}")
    print("Downloading RF100-VL datasets (COCO format) ...")
    print(f"  destination: {coco_datasets_dir}")
    print(f"{'='*70}\n")

    download_rf100vl(
        path=coco_datasets_dir,
        model_format="coco",
        overwrite=False,
    )

    dataset_dirs = sorted(
        str(p.parent.parent)
        for p in Path(coco_datasets_dir).rglob("test/_annotations.coco.json")
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


def _onnx_path(onnx_dir: str, dataset_name: str, variant_name: str) -> str:
    return os.path.join(onnx_dir, dataset_name, f"{variant_name}.onnx")


def _variant_csv(sab_dir: str, variant: str) -> str:
    return os.path.join(sab_dir, f"bench_results_{variant}.csv")


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
    parser = argparse.ArgumentParser(
        description="Run SAB benchmarking across all RF100-VL datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, default="rf100vl_benchmark_results",
                        help="Results output directory")
    parser.add_argument("--coco-datasets-dir", type=str, default="rf100vl_datasets_coco",
                        help="COCO datasets directory")
    args = parser.parse_args()

    results_dir = args.results_dir
    onnx_dir = os.path.join(results_dir, "onnx")
    sab_dir = os.path.join(results_dir, "sab")
    coco_datasets_dir = args.coco_datasets_dir

    os.makedirs(sab_dir, exist_ok=True)

    # 1. Download COCO-format datasets
    coco_dataset_dirs = download_coco_datasets(coco_datasets_dir)
    if not coco_dataset_dirs:
        print("ERROR: No COCO datasets found. Check ROBOFLOW_API_KEY.")
        sys.exit(1)

    coco_by_name = {Path(d).name: d for d in coco_dataset_dirs}

    # 2. Discover available ONNX models
    datasets_with_models: dict[str, list[tuple[str, str]]] = {}
    for dataset_name in sorted(coco_by_name.keys()):
        for variant in VARIANT_NAMES:
            onnx = _onnx_path(onnx_dir, dataset_name, variant)
            if os.path.isfile(onnx):
                datasets_with_models.setdefault(dataset_name, []).append((variant, onnx))

    total_models = sum(len(v) for v in datasets_with_models.values())
    print(f"\n{'='*70}")
    print(f"Benchmarking: {total_models} ONNX models across "
          f"{len(datasets_with_models)} datasets")
    print(f"{'='*70}\n")

    if not datasets_with_models:
        print("ERROR: No ONNX models found. Run train_rf100vl first.")
        sys.exit(1)

    # 3. Load existing partial results for resume
    variant_rows: dict[str, list[dict]] = {v: [] for v in VARIANT_NAMES}
    for variant in VARIANT_NAMES:
        variant_rows[variant] = load_variant_csv(_variant_csv(sab_dir, variant))
    completed = build_completed_set(
        variant_rows, key_fields=("dataset", "variant", "runtime"),
    )

    if completed:
        print(f"Resuming: {len(completed)} benchmark runs already completed.\n")

    done_count = len(completed)
    total = total_models * 3  # 3 engines per (variant, dataset)

    # 4. Iterate datasets, then variants
    for dataset_name, variant_onnx_list in list(datasets_with_models.items()):
        coco_dir = coco_by_name.get(dataset_name)
        if coco_dir is None:
            print(f"SKIP {dataset_name}: no COCO dataset found")
            continue

        test_info = _find_coco_test(coco_dir)
        if test_info is None:
            print(f"SKIP {dataset_name}: no test split with annotations")
            continue

        image_dir, annotations_path = test_info

        pending = [
            (variant, onnx_path) for variant, onnx_path in variant_onnx_list
            if not all(
                (dataset_name, variant, rt) in completed
                for rt in ("ONNX-CPU", "TRT-fp32", "TRT-fp16")
            )
        ]

        if not pending:
            continue

        print(f"\n{'---'*23}")
        print(f"[{dataset_name}] {len(pending)} variants remaining")
        print(f"{'---'*23}")

        for variant, onnx_path in pending:
            already_done = {
                rt for rt in ("ONNX-CPU", "TRT-fp32", "TRT-fp16")
                if (dataset_name, variant, rt) in completed
            }
            if len(already_done) == 3:
                continue

            print(f"\n  Benchmarking {dataset_name} / {variant} ...")

            rows = benchmark_single(
                variant, dataset_name, onnx_path, image_dir, annotations_path
            )

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
