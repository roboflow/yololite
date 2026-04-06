#!/usr/bin/env python3
"""Train all yololite variants on RF100-VL and export each to decoded ONNX.

Downloads all RF100-VL datasets (yolov8 format) and trains every yololite
variant for a configurable number of epochs.  After training, the best
checkpoint is exported to a decoded ONNX file for downstream SAB benchmarking.

Concurrency: training jobs run in parallel via a process pool, one variant at
a time.  Each variant specifies how many concurrent jobs fit per GPU (based on
VRAM footprint), so small models get higher parallelism than large ones.

Optionally initializes from pre-trained weights (e.g. COCO) via
``--pretrained-dir``.
"""

import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd

# NOTE: torch is intentionally NOT imported at module level.

from yololite.benchmark._io import (
    build_completed_set,
    find_data_yaml,
    load_training_config,
    load_variant_csv,
    save_variant_csv,
)
from yololite.benchmark._pool import NUM_CPUS, cap_threads, get_num_gpus, run_variant_pool
from yololite.benchmark._variants import YOLOLITE_VARIANTS

# ── Constants ────────────────────────────────────────────────────────────────
DATASETS_FORMAT = "yolov8"
IMG_SIZE = 640

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_datasets(datasets_dir: str, rf20: bool = False) -> list[str]:
    """Download RF100-VL (or RF20-VL subset) datasets in yolov8 format."""
    if rf20:
        from rf100vl import download_rf20vl_full as _download
        label = "RF20-VL"
    else:
        from rf100vl import download_rf100vl as _download
        label = "RF100-VL"

    print(f"\n{'='*70}")
    print(f"Downloading {label} datasets ...")
    print(f"  destination: {datasets_dir}")
    print(f"  format:      {DATASETS_FORMAT}")
    print(f"{'='*70}\n")

    _download(
        path=datasets_dir,
        model_format=DATASETS_FORMAT,
        overwrite=False,
    )

    dataset_dirs = sorted(
        str(p.parent)
        for p in Path(datasets_dir).rglob("data.yaml")
    )
    print(f"\nFound {len(dataset_dirs)} datasets with data.yaml")
    return dataset_dirs


def _onnx_path(onnx_dir: str, dataset_name: str, variant_name: str) -> str:
    return os.path.join(onnx_dir, dataset_name, f"{variant_name}.onnx")


def _variant_csv(results_dir: str, variant: str) -> str:
    return os.path.join(results_dir, f"train_results_{variant}.csv")


def _resolve_pretrained_weights(weights_dir: str) -> dict[str, str]:
    """Discover best checkpoints per variant in a results directory.

    Scans ``weights_dir/runs/{variant}/*/weights/best_model_state.pt`` and
    returns ``{variant_name: checkpoint_path}``.
    """
    found: dict[str, str] = {}
    runs_dir = os.path.join(weights_dir, "runs")
    if not os.path.isdir(runs_dir):
        return found
    for variant in YOLOLITE_VARIANTS:
        pattern = os.path.join(runs_dir, variant, "*", "weights", "best_model_state.pt")
        matches = sorted(Path(runs_dir).glob(f"{variant}/*/weights/best_model_state.pt"))
        if matches:
            found[variant] = str(matches[-1])  # latest run
    return found


def _strip_training_state(checkpoint_path: str) -> str:
    """Load a checkpoint, remove training_state, save to a temp file.

    This ensures that ``run_training()`` treats the checkpoint as a
    weights-only warm-start (no optimizer/scheduler/epoch restoration).
    Returns the path to the stripped temp file.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        ckpt.pop("training_state", None)
    # Write to a temp file in the same filesystem for speed
    fd, tmp_path = tempfile.mkstemp(suffix=".pt", prefix="pretrained_")
    os.close(fd)
    torch.save(ckpt, tmp_path)
    return tmp_path


# ── Training ─────────────────────────────────────────────────────────────────

def run_single_training(
    variant_name: str,
    dataset_dir: str,
    dataset_name: str,
    results_dir: str,
    onnx_dir: str,
    device: str = "0",
    max_concurrent: int = 1,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pretrained_path: str | None = None,
) -> dict:
    """Train one yololite variant on one dataset, then export to ONNX."""
    cap_threads(max_concurrent)

    from yololite.export.export_onnx import export_decoded_onnx
    from yololite.tools.train import run_training

    data_yaml = find_data_yaml(dataset_dir)
    log_dir = os.path.join(results_dir, "runs", dataset_name, variant_name)
    os.makedirs(log_dir, exist_ok=True)

    # pretrained_path is already stripped by the main process
    config = load_training_config(
        variant_name,
        data_yaml,
        log_dir,
        epochs=epochs,
        batch_size=batch_size,
        img_size=IMG_SIZE,
        device=device,
        save_every=epochs + 1,  # no periodic saves
        num_workers=max(1, NUM_CPUS // max_concurrent),
        resume=pretrained_path,
    )

    # ── Train ──
    t0 = time.time()
    train_result = run_training(config)
    train_elapsed = time.time() - t0

    # ── Export to ONNX ──
    best_ckpt = train_result["best_checkpoint"]
    onnx_out = _onnx_path(onnx_dir, dataset_name, variant_name)
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

def _worker(args: tuple) -> dict:
    (variant_name, dataset_dir, dataset_name, results_dir, onnx_dir,
     max_concurrent, epochs, batch_size, pretrained_path) = args
    try:
        return run_single_training(
            variant_name, dataset_dir, dataset_name, results_dir, onnx_dir,
            "0", max_concurrent, epochs, batch_size, pretrained_path,
        )
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

def main():
    parser = argparse.ArgumentParser(
        description="Train yololite variants on RF100-VL and export to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--datasets-dir", type=str, default="rf100vl_datasets",
                        help="RF100-VL dataset directory")
    parser.add_argument("--results-dir", type=str, default="rf100vl_benchmark_results",
                        help="Results output directory")
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default=None,
        help="Results directory with pre-trained checkpoints (e.g. coco_benchmark_results)",
    )
    parser.add_argument(
        "--rf20", action="store_true",
        help="Use RF20-VL (20 datasets) instead of full RF100-VL",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    onnx_dir = os.path.join(results_dir, "onnx")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(onnx_dir, exist_ok=True)

    # 1. Download datasets
    dataset_dirs = download_datasets(args.datasets_dir, rf20=args.rf20)
    if not dataset_dirs:
        print("ERROR: No datasets found after download. Check ROBOFLOW_API_KEY.")
        sys.exit(1)

    # 2. Resolve pre-trained weights
    pretrained_weights: dict[str, str] = {}
    if args.pretrained_dir:
        pretrained_weights = _resolve_pretrained_weights(args.pretrained_dir)
        print(f"\nPre-trained checkpoints found for {len(pretrained_weights)}/{len(YOLOLITE_VARIANTS)} variants")
        for v, p in pretrained_weights.items():
            print(f"  {v}: {p}")
        missing = set(YOLOLITE_VARIANTS) - set(pretrained_weights)
        if missing:
            print(f"  WARNING: no checkpoint for: {', '.join(sorted(missing))} — training from scratch")
        print()

    total = len(dataset_dirs) * len(YOLOLITE_VARIANTS)
    num_gpus = get_num_gpus()

    print(f"\n{'='*70}")
    print(f"Training matrix: {len(dataset_dirs)} datasets x "
          f"{len(YOLOLITE_VARIANTS)} variants = {total} runs")
    print(f"Epochs: {args.epochs}  |  Batch size: {args.batch_size}  |  GPUs: {num_gpus}")
    if pretrained_weights:
        print(f"Pre-trained: {args.pretrained_dir}")
    print(f"{'='*70}\n")

    # 3. Load existing partial results for resume
    variant_rows: dict[str, list[dict]] = {v: [] for v in YOLOLITE_VARIANTS}
    for variant in YOLOLITE_VARIANTS:
        variant_rows[variant] = load_variant_csv(_variant_csv(results_dir, variant))
    completed = build_completed_set(variant_rows)

    if completed:
        print(f"Resuming: {len(completed)} runs already completed, "
              f"{total - len(completed)} remaining.\n")

    done_count = len(completed)

    # 4. Process one variant at a time
    for variant, (_subdir, _yaml, jobs_per_gpu) in YOLOLITE_VARIANTS.items():
        max_concurrent = num_gpus * jobs_per_gpu
        pretrained_path = pretrained_weights.get(variant)

        # Strip training_state once in the main process, share the
        # stripped file across all workers for this variant.
        stripped_path = None
        if pretrained_path is not None:
            stripped_path = _strip_training_state(pretrained_path)

        variant_jobs = []
        for ddir in dataset_dirs:
            dname = Path(ddir).name
            if (dname, variant) in completed:
                continue
            if os.path.isfile(_onnx_path(onnx_dir, dname, variant)):
                continue
            variant_jobs.append((
                variant, ddir, dname, results_dir, onnx_dir,
                max_concurrent, args.epochs, args.batch_size, stripped_path,
            ))

        if not variant_jobs:
            print(f"[{variant}] all {len(dataset_dirs)} datasets already done, skipping.")
            continue

        print(f"\n{'---'*23}")
        print(f"[{variant}] {len(variant_jobs)} datasets remaining  |  "
              f"jobs_per_gpu={jobs_per_gpu}  |  max_concurrent={max_concurrent}")
        if pretrained_path:
            print(f"  pretrained: {pretrained_path}")
        print(f"{'---'*23}")

        variant_csv = _variant_csv(results_dir, variant)

        def _on_result(result: dict) -> None:
            nonlocal done_count
            variant_rows[variant].append(result)
            done_count += 1

            status = "OK" if result.get("error") is None else f"FAIL: {result['error']}"
            print(
                f"[{done_count}/{total}] {result['dataset']} / "
                f"{result['variant']}  "
                f"(train={result['train_s']}s  export={result.get('export_s', 0)}s)  "
                f"[{status}]"
            )
            save_variant_csv(variant_rows[variant], variant_csv)

        run_variant_pool(
            variant_jobs, _worker, num_gpus, jobs_per_gpu,
            on_result=_on_result,
        )

        # Clean up stripped checkpoint
        if stripped_path is not None and os.path.isfile(stripped_path):
            os.unlink(stripped_path)

    # 5. Summary
    all_rows = [r for rows in variant_rows.values() for r in rows]
    df = pd.DataFrame(all_rows)
    combined_csv = os.path.join(results_dir, "train_results_combined.csv")
    df.to_csv(combined_csv, index=False)

    print(f"\nTraining + export complete.")
    print(f"  Combined CSV: {combined_csv}")
    print(f"  ONNX models:  {onnx_dir}/")


if __name__ == "__main__":
    main()
