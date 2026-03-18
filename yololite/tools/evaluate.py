# train.py
import os, sys
from pathlib import Path
import torch

from yololite.tools.infer import load_model_names_imgsize_from_ckpt
from torch.utils.data import DataLoader

from yololite.scripts.data.dataset import YoloDataset
from yololite.scripts.data.augment import get_val_transform
from yololite.scripts.helpers.helpers import yolo_collate
from yololite.scripts.helpers.evaluate import evaluate_model
import argparse

def _next_run_dir(base: str) -> str:
    """
    Skapa och returnera nästa lediga run-mapp som en numerisk subdir under 'base'.
    Ex: base='runs' -> 'runs/1', 'runs/2', ...
        base='runs/weeds' -> 'runs/weeds/1', ...
    """
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        cand = root / str(n)
        try:
            cand.mkdir(parents=False, exist_ok=False)
            return str(cand.resolve())
        except FileExistsError:
            n += 1


def evaluate_on_folder(
    weights: str,
    test_folder: str,
    batch_size: int = 8,
    device: str = "0",
    max_dets: int = 100,
    no_letterbox: bool = False,
    log_dir: str | None = None,
) -> dict:
    """Evaluate a checkpoint on a folder with images/ and labels/ subdirectories.

    Returns the metrics dict from evaluate_model (keys: mAP, mAP_50_95, precision, recall, …).
    """
    dev = f"cuda:{device}" if device != "cpu" and torch.cuda.is_available() else "cpu"

    model, names, meta_img_size = load_model_names_imgsize_from_ckpt(weights, dev)
    test_images = os.path.join(test_folder, "images")
    test_labels = os.path.join(test_folder, "labels")

    if log_dir is None:
        log_dir_parent = os.path.join("runs/evaluate")
        os.makedirs(log_dir_parent, exist_ok=True)
        log_dir = _next_run_dir(log_dir_parent)

    val_resize = 1.0 if no_letterbox else 0.0
    test_ds = YoloDataset(
        test_images,
        test_labels,
        img_size=meta_img_size,
        is_train=False,
        transforms=get_val_transform(meta_img_size, val_resize),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=yolo_collate,
        drop_last=False,
    )

    return evaluate_model(
        model=model, val_loader=test_loader, log_dir=log_dir,
        NUM_CLASSES=len(names), DEVICE=dev, IMG_SIZE=meta_img_size,
        batch_size=batch_size, class_names=names, max_dets=max_dets,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to checkpoint (.pt/.pth)")
    ap.add_argument("--test_folder", required=True, help="path to test/val folder")
    ap.add_argument("--img_size", type=int, default=0, help="Override meta.img_size)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--no_letterbox", action="store_true", help="Use pure resize instead of letterbox")
    ap.add_argument("--max_dets", type=int, default=100, help="Max detections per image for COCO eval (default 100)")
    args = ap.parse_args()

    evaluate_on_folder(
        weights=args.weights,
        test_folder=args.test_folder,
        batch_size=args.batch_size,
        device=args.device,
        max_dets=args.max_dets,
        no_letterbox=args.no_letterbox,
    )
    
    

if __name__ == "__main__":
    main()
