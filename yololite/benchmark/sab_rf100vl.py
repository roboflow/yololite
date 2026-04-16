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
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from yololite.benchmark._io import (
    build_completed_set,
    load_variant_csv,
    save_variant_csv,
)
from yololite.benchmark._variants import VARIANT_NAMES

RUNTIMES = ("ONNX-CPU", "TRT-fp32", "TRT-fp16")
_RESULT_MARKER = "__SAB_RESULT__"
MAX_DETS = 500


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


def _run_one_engine_in_process(
    runtime_name: str, fp16: bool, onnx_path: str, image_dir: str, annotations_path: str,
) -> dict:
    """Run a single benchmark in this process and return a result dict."""
    from sab.models.benchmark_yololite import (
        YoloLiteONNXCPUInference,
        YoloLiteTRTInference,
    )
    from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifact

    inference_class = (
        YoloLiteONNXCPUInference if runtime_name == "ONNX-CPU" else YoloLiteTRTInference
    )
    request = ArtifactBenchmarkRequest(
        onnx_path=onnx_path,
        inference_class=inference_class,
        needs_fp16=fp16,
        max_dets=MAX_DETS,
    )
    accuracy_stats, latency_stats, throttled = run_benchmark_on_artifact(
        request, image_dir, annotations_path
    )
    return {
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
    }


def _run_one_engine_subprocess(
    runtime_name: str, fp16: bool, onnx_path: str, image_dir: str, annotations_path: str,
) -> dict:
    """Spawn a subprocess to run one engine. Subprocess exit guarantees FD cleanup."""
    cmd = [
        sys.executable, "-m", "yololite.benchmark.sab_rf100vl", "--worker",
        "--runtime", runtime_name,
        "--onnx-path", onnx_path,
        "--image-dir", image_dir,
        "--annotations-path", annotations_path,
    ]
    if fp16:
        cmd.append("--fp16")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"worker exit {result.returncode}\nstderr:\n{result.stderr[-2000:]}"
        )

    for line in result.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    raise RuntimeError(f"worker produced no result line\nstdout:\n{result.stdout[-2000:]}")


def benchmark_single(
    variant_name: str,
    dataset_name: str,
    onnx_path: str,
    image_dir: str,
    annotations_path: str,
) -> list[dict]:
    """Run all 3 engines for one (variant, dataset) pair, each in a subprocess."""
    engines = [
        ("ONNX-CPU", False),
        ("TRT-fp32", False),
        ("TRT-fp16", True),
    ]

    rows = []
    for runtime_name, fp16 in engines:
        try:
            stats = _run_one_engine_subprocess(
                runtime_name, fp16, onnx_path, image_dir, annotations_path
            )
            rows.append({
                "dataset": dataset_name,
                "variant": variant_name,
                "runtime": runtime_name,
                "fp16": fp16,
                **stats,
            })
        except Exception as e:
            print(f"  ERROR [{runtime_name}]: {e}")
            rows.append({
                "dataset": dataset_name,
                "variant": variant_name,
                "runtime": runtime_name,
                "fp16": fp16,
                "error": str(e),
            })

    return rows


def worker_main():
    """Entry point for the --worker subprocess mode."""
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--runtime", required=True, choices=RUNTIMES)
    p.add_argument("--onnx-path", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--annotations-path", required=True)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    result = _run_one_engine_in_process(
        args.runtime, args.fp16, args.onnx_path, args.image_dir, args.annotations_path
    )
    print(_RESULT_MARKER + json.dumps(result))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if "--worker" in sys.argv:
        worker_main()
        return

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
