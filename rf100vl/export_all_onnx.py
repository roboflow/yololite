#!/usr/bin/env python3
"""Export all trained yololite checkpoints from the benchmark_rf100vl.py run to ONNX.

Scans RESULTS_DIR/runs/{dataset}/{variant}/*/weights/best_model_state.pt and
exports each to ONNX_DIR/{dataset}/{variant}.onnx using the decoded format.
Skips any that already exist. Runs exports in parallel across CPU cores.
"""

import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RESULTS_DIR = os.environ.get("RF100VL_RESULTS_DIR", "rf100vl_benchmark_results")
ONNX_DIR = os.path.join(RESULTS_DIR, "onnx")
IMG_SIZE = 640
MAX_WORKERS = os.cpu_count()


def _export_one(ckpt_str: str, out_path: str) -> tuple[str, bool, str]:
    """Export a single checkpoint. Returns (out_path, success, error_msg)."""
    try:
        from yololite.export.export_onnx import export_decoded_onnx
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        export_decoded_onnx(
            checkpoint_path=ckpt_str,
            img_size=IMG_SIZE,
            out_path=out_path,
        )
        return out_path, True, ""
    except Exception as e:
        return out_path, False, f"{e}\n{traceback.format_exc()}"


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)

    checkpoints = sorted(Path(RESULTS_DIR, "runs").rglob("weights/best_model_state.pt"))
    print(f"Found {len(checkpoints)} checkpoints")

    # Build work list, skipping existing
    jobs = []
    skipped = 0
    for ckpt in checkpoints:
        # Path: runs/{dataset}/{variant}/{run_id}/weights/best_model_state.pt
        variant = ckpt.parent.parent.parent.name
        dataset = ckpt.parent.parent.parent.parent.name
        out_path = os.path.join(ONNX_DIR, dataset, f"{variant}.onnx")
        if os.path.isfile(out_path):
            skipped += 1
            continue
        jobs.append((str(ckpt), out_path, dataset, variant))

    print(f"{skipped} already exported, {len(jobs)} remaining, "
          f"using {MAX_WORKERS} workers")

    if not jobs:
        print("Nothing to do.")
        return

    done = 0
    failed = 0
    total = len(jobs)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_export_one, ckpt_str, out_path): (dataset, variant)
            for ckpt_str, out_path, dataset, variant in jobs
        }
        for future in as_completed(futures):
            dataset, variant = futures[future]
            out_path, success, err = future.result()
            if success:
                done += 1
                print(f"  [{done + failed}/{total}] OK  {dataset}/{variant}")
            else:
                failed += 1
                print(f"  [{done + failed}/{total}] FAIL {dataset}/{variant}: {err}")

    print(f"\nDone: {done} exported, {skipped} skipped, {failed} failed "
          f"(total {len(checkpoints)} checkpoints)")


if __name__ == "__main__":
    main()
