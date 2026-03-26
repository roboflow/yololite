#!/usr/bin/env python3
"""Export all trained yololite checkpoints from the benchmark_rf100vl.py run to ONNX.

Scans RESULTS_DIR/runs/{dataset}/{variant}/*/weights/best_model_state.pt and
exports each to ONNX_DIR/{dataset}/{variant}.onnx using the decoded format.
Skips any that already exist.
"""

import os
import sys
import traceback
from pathlib import Path

RESULTS_DIR = "/dev/shm/rf100vl_benchmark_results"
ONNX_DIR = os.path.join(RESULTS_DIR, "onnx")
IMG_SIZE = 640


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)

    # Find all best checkpoints
    checkpoints = sorted(Path(RESULTS_DIR, "runs").rglob("weights/best_model_state.pt"))
    print(f"Found {len(checkpoints)} checkpoints")

    # Lazy import so the script starts fast
    from yololite.export.export_onnx import export_decoded_onnx

    done = 0
    skipped = 0
    failed = 0

    for ckpt in checkpoints:
        # Path: runs/{dataset}/{variant}/{run_id}/weights/best_model_state.pt
        variant = ckpt.parent.parent.parent.name
        dataset = ckpt.parent.parent.parent.parent.name

        out_path = os.path.join(ONNX_DIR, dataset, f"{variant}.onnx")
        if os.path.isfile(out_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"[{done + skipped + failed + 1}/{len(checkpoints)}] "
              f"{dataset}/{variant} → {out_path}")

        try:
            export_decoded_onnx(
                checkpoint_path=str(ckpt),
                img_size=IMG_SIZE,
                out_path=out_path,
            )
            done += 1
        except Exception as e:
            traceback.print_exc()
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\nDone: {done} exported, {skipped} skipped, {failed} failed "
          f"(total {len(checkpoints)} checkpoints)")


if __name__ == "__main__":
    main()
