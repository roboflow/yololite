import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, csv

def build_curves_from_coco(coco_images, coco_anns, coco_dets,
                           out_dir, iou=0.50, steps=201):
    """
    Bygger:
      - PR-kurva (rankad efter score)
      - Precision/Recall/F1 vs confidence (0..1)
    och sparar CSV + PNG i `out_dir`. Returnerar en summary-dict.

    Param:
      coco_images: [ { "id": int, "width": int, "height": int, ... }, ... ] (ej krav)
      coco_anns:   [ { "image_id": int, "category_id": int, "bbox": [x,y,w,h], ... }, ... ]
      coco_dets:   [ { "image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float }, ... ]
      out_dir:     str eller Path (skapas vid behov)
      iou:         IoU-tröskel för matchning (default 0.50)
      steps:       antal tröskelsteg mellan 0..1 för P/R/F1 (default 201)

    Obs:
      - Matchning sker PER KLASS (category_id), girigt mot o-matchade GT vid vald IoU.
      - För vettig PR-kurva: se till att dina `coco_dets` innehåller *låga* konfidenser också
        (t.ex. eval-decode med conf_th=0.001, per-class NMS, maxDets=100).
    """

    

    # ---------------- helpers ----------------
    def build_gt_index(anns):
        # (img_id, cat_id) -> np.ndarray of shape (N, 4) in xywh format
        d = {}
        for a in anns:
            key = (int(a["image_id"]), int(a["category_id"]))
            d.setdefault(key, []).append(a["bbox"])
        return {k: np.array(v, dtype=np.float64) for k, v in d.items()}

    def iou_one_vs_many(det_bbox, gt_arr):
        """IoU of one xywh box against an (N,4) array of xywh boxes."""
        dx, dy, dw, dh = det_bbox
        gx, gy, gw, gh = gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], gt_arr[:, 3]
        ix1 = np.maximum(dx, gx)
        iy1 = np.maximum(dy, gy)
        ix2 = np.minimum(dx + dw, gx + gw)
        iy2 = np.minimum(dy + dh, gy + gh)
        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih
        ua = np.maximum(0.0, dw * dh) + np.maximum(0.0, gw * gh) - inter
        return np.where(ua > 0, inter / ua, 0.0)

    # ---------------- PR curve (score-rankad svep) ----------------
    gt_index = build_gt_index(coco_anns)
    matched_flags = {k: np.zeros(len(v), dtype=bool) for k, v in gt_index.items()}
    total_gt = sum(len(v) for v in gt_index.values())

    dets_sorted = sorted(coco_dets, key=lambda x: float(x.get("score", 0.0)), reverse=True)

    tps, fps = [], []
    for d in dets_sorted:
        key = (int(d["image_id"]), int(d["category_id"]))
        gt_arr = gt_index.get(key)
        if gt_arr is None:
            fps.append(1.0); tps.append(0.0)
            continue
        flags = matched_flags[key]
        ious = iou_one_vs_many(d["bbox"], gt_arr)
        ious[flags] = -1.0  # mask already-matched GTs
        best_j = int(np.argmax(ious))
        best_iou = ious[best_j]
        if best_iou >= iou:
            flags[best_j] = True
            tps.append(1.0); fps.append(0.0)
        else:
            fps.append(1.0); tps.append(0.0)

    if len(tps) == 0:
        # inga prediktioner: skriv tomma filer och returnera
        _confs = np.linspace(0.0, 1.0, steps)
        return {
            "iou": float(iou),
            "best_f1": 0.0,
            "best_conf": 0.0,
            "precision_at_best": 0.0,
            "recall_at_best": 0.0,
            "confs": _confs,
            "P_curve": np.zeros(steps),
            "R_curve": np.zeros(steps),
            "F1_curve": np.zeros(steps),
        }

    tps = np.array(tps); fps = np.array(fps)
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls_rank = cum_tp / max(1, total_gt)
    precisions_rank = cum_tp / np.maximum(1, cum_tp + cum_fp)


    # ---------------- P/R/F1 vs confidence (derived from ranked pass) ----
    # Instead of re-running greedy matching 201 times, reuse the single
    # score-ranked matching pass above.  For each confidence threshold we
    # just need the cumulative TP/FP up to the last detection whose score
    # >= threshold.  This is O(steps * log(n)) instead of O(steps * n * m).
    scores_sorted = np.array(
        [float(d.get("score", 0.0)) for d in dets_sorted]
    )
    confs = np.linspace(0.0, 1.0, steps)
    # For each threshold, find the last index where score >= thr.
    # searchsorted on the reversed (ascending) scores gives us the count
    # of detections below the threshold.
    scores_asc = scores_sorted[::-1]
    # n_above[i] = number of detections with score >= confs[i]
    n_above = len(scores_sorted) - np.searchsorted(scores_asc, confs, side="left")

    safe_idx = np.clip(n_above - 1, 0, len(cum_tp) - 1)
    tp_at_thr = cum_tp[safe_idx]
    n_safe = np.maximum(n_above, 1)
    P_curve = np.where(n_above > 0, tp_at_thr / n_safe, 0.0)
    R_curve = np.where(n_above > 0, tp_at_thr / max(1, total_gt), 0.0)
    pr_sum = P_curve + R_curve
    F1_curve = np.where(pr_sum > 0, 2 * P_curve * R_curve / np.maximum(pr_sum, 1e-16), 0.0)

    P_curve = np.array(P_curve); R_curve = np.array(R_curve); F1_curve = np.array(F1_curve)
    best_idx = int(np.argmax(F1_curve))
    # Välj din fasta tröskel (ex 0.50 eller 0.575)
    fixed_conf = 0.50
    idx = int(np.argmin(np.abs(confs - fixed_conf)))

    P_fixed = float(P_curve[idx])
    R_fixed = float(R_curve[idx])
    F1_fixed = float(F1_curve[idx])
    

    summary = {
        "iou": float(iou),
        "best_f1": float(F1_curve[best_idx]),
        "best_conf": float(confs[best_idx]),
        "precision_at_best": float(P_curve[best_idx]),
        "recall_at_best": float(R_curve[best_idx]),
        "fixed_conf": fixed_conf,
        "precision_at_fixed_conf": P_fixed,
        "recall_at_fixed_conf": R_fixed,
        "f1_at_fixed_conf": F1_fixed,
        "P_curve": P_curve,
        "R_curve": R_curve,
        "F1_curve": F1_curve,
        "confs": confs,
        "best_idx": best_idx
                
    }
    
    return summary
