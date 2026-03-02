# tools/export_onnx.py
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= sys.path & imports =========
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ========= util: prints =========
def log(msg: str, verbose: bool):
    print(msg, flush=True) if verbose else None

def must(msg: str):
    print(msg, flush=True)


# ========= run-dir =========
def next_run_dir(base: str) -> str:
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


# ========= decoded wrapper (utan NMS) =========
class AFDecode(nn.Module):
    def __init__(self, img_size: int, center_mode: str = "v8", wh_mode: str = "softplus"):
        super().__init__()
        self.img_size = int(img_size)
        self.center_mode = center_mode
        self.wh_mode = wh_mode

    @staticmethod
    def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
        x, y, w, h = xywh.unbind(-1)
        return torch.stack([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], dim=-1)

    def _decode_level(self, p: torch.Tensor):
        # p: [B,A,S,S,D] eller [B,S,S,D] (tolkas A=1)
        if p.dim() == 4:
            p = p.unsqueeze(1)  # [B,1,S,S,D]
        B, A, S, _, D = p.shape
        cell = float(self.img_size) / float(S)

        gy, gx = torch.meshgrid(torch.arange(S, device=p.device),
                                torch.arange(S, device=p.device), indexing="ij")
        gx = gx.float(); gy = gy.float()

        tx = p[..., 0]; ty = p[..., 1]
        tw = p[..., 2]; th = p[..., 3]
        tobj = p[..., 4]
        tcls = p[..., 5:]

        if self.center_mode == "v8":
            px = ((tx.sigmoid() * 2.0 - 0.5) + gx) * cell
            py = ((ty.sigmoid() * 2.0 - 0.5) + gy) * cell
        else:
            px = (tx.sigmoid() + gx) * cell
            py = (ty.sigmoid() + gy) * cell

        if   self.wh_mode == "v8":
            pw = (tw.sigmoid() * 2).pow(2) * cell
            ph = (th.sigmoid() * 2).pow(2) * cell
        elif self.wh_mode == "softplus":
            pw = F.softplus(tw) * cell
            ph = F.softplus(th) * cell
        else:
            pw = tw.clamp(-4, 4).exp() * cell
            ph = th.clamp(-4, 4).exp() * cell

        xyxy = self._xywh_to_xyxy(torch.stack([px, py, pw, ph], dim=-1))
        xyxy[..., 0::2] = xyxy[..., 0::2].clamp(0, self.img_size - 1)
        xyxy[..., 1::2] = xyxy[..., 1::2].clamp(0, self.img_size - 1)

        xyxy = xyxy.reshape(B, -1, 4)
        obj  = tobj.reshape(B, -1, 1)
        cls  = tcls.reshape(B, -1, tcls.shape[-1])
        return xyxy, obj, cls

    def forward(self, preds):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        boxes, objs, clss = [], [], []
        for p in preds:
            b, o, c = self._decode_level(p)
            boxes.append(b); objs.append(o); clss.append(c)
        boxes = torch.cat(boxes, dim=1)
        obj   = torch.cat(objs,  dim=1)
        cls   = torch.cat(clss,  dim=1)
        return boxes, obj, cls


# ========= build model from meta/config =========
def build_model_from_meta(meta: dict) -> nn.Module:
    try:
        from yololite.scripts.model.model_v2 import YOLOLiteMS, YOLOLiteMS_CPU
    except ImportError:
        from scripts.model.model_v2 import YOLOLiteMS, YOLOLiteMS_CPU

    cfg  = meta.get("config", {}) or {}
    mcfg = cfg.get("model", {}) or {}
    tcfg = cfg.get("training", {}) or {}
    print(cfg)
    arch        = (meta.get("arch") or mcfg.get("arch") or "YOLOLiteMS").lower()
    backbone    = (meta.get("backbone") or mcfg.get("backbone") or "resnet18")
    num_classes = int(meta.get("num_classes") or mcfg.get("num_classes") or 80)
    num_anchors_per_level = tuple(meta.get("num_anchors_per_level") or (1,1,1))
    fpn_channels   = int(mcfg.get("fpn_channels", 128))
    depth_multiple = float(mcfg.get("depth_multiple", 1.0))
    width_multiple = float(mcfg.get("width_multiple", 1.0))
    head_depth     = int(mcfg.get("head_depth", 1))

    img_size = int(tcfg.get("img_size", meta.get("img_size", 640)))
    use_p6 = cfg["training"]["use_p6"]
    use_p2 = cfg["training"]["use_p2"]
    if arch == "yololitems":
        model = YOLOLiteMS(
            backbone=backbone,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            head_depth=head_depth,
            num_anchors_per_level=num_anchors_per_level,
            use_p6=use_p6,
            use_p2=use_p2
        )
    elif arch == "yololitems_cpu":
        model = YOLOLiteMS_CPU(
            backbone=backbone,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            head_depth=head_depth,
            num_anchors_per_level=num_anchors_per_level,
            use_p6=use_p6,
            use_p2=use_p2
        )
    else:
        raise ValueError(f"Okänd arch i meta/config: {arch}")
    return model


# ========= load ckpt =========
def load_model_from_ckpt(weights: str, device: torch.device, verbose: bool) -> Tuple[nn.Module, dict]:
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Viktfil hittas inte: {weights}")
    must(f"• Läser checkpoint: {weights}")
    ckpt = torch.load(weights, map_location=device)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "meta" in ckpt):
        raise RuntimeError("Checkpoint saknar 'state_dict'/'meta' – spara via save_checkpoint_state(...).")
    meta = ckpt["meta"] or {}
    log(f"  meta.keys: {list(meta.keys())}", verbose)
    model = build_model_from_meta(meta)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:    must(f"  [load_state_dict] saknade nycklar: {len(missing)}")
    if unexpected: must(f"  [load_state_dict] oväntade nycklar: {len(unexpected)}")
    model.to(device).eval()
    must(f"• Modell byggd: arch={meta.get('arch')} backbone={meta.get('backbone')} nc={meta.get('num_classes')}")
    return model, meta


# ========= programmatic export helpers =========
def export_decoded_onnx(
    checkpoint_path: str,
    img_size: int,
    out_path: str,
    opset: int = 17,
    center_mode: str = "v8",
    wh_mode: str = "softplus",
) -> None:
    """Export a yololite checkpoint to a decoded ONNX file (no NMS).

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint saved by ``save_checkpoint_state()``.
    img_size:
        Spatial resolution (H == W) used during training.
    out_path:
        Destination ``.onnx`` file path.
    opset:
        ONNX opset version (default 17).
    center_mode / wh_mode:
        Decoding parameters forwarded to ``AFDecode``.
    """
    device = torch.device("cpu")
    model, meta = load_model_from_ckpt(checkpoint_path, device=device, verbose=False)

    class _DecodedWrapper(nn.Module):
        def __init__(self, core, size, center_mode, wh_mode):
            super().__init__()
            self.core = core
            self.decode = AFDecode(img_size=size, center_mode=center_mode, wh_mode=wh_mode)

        def forward(self, x):
            return self.decode(self.core(x))

    wrapper = _DecodedWrapper(model, img_size, center_mode, wh_mode).eval()
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            opset_version=opset,
            input_names=["images"],
            output_names=["boxes_xyxy", "obj_logits", "cls_logits"],
            dynamic_axes={
                "images": {0: "batch"},
                "boxes_xyxy": {0: "batch"},
                "obj_logits": {0: "batch"},
                "cls_logits": {0: "batch"},
            },
            do_constant_folding=True,
        )


def export_raw_onnx(
    checkpoint_path: str,
    img_size: int,
    out_path: str,
    opset: int = 17,
    dynamic_batch: bool = False,
    dynamic_shape: bool = False,
    device: str = "cpu",
    half: bool = False,
    verbose: bool = False,
) -> None:
    """Export a yololite checkpoint to a raw ONNX file (one output tensor per FPN level).

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint saved by ``save_checkpoint_state()``.
    img_size:
        Spatial resolution (H == W).
    out_path:
        Destination ``.onnx`` file path.
    opset:
        ONNX opset version (default 17).
    dynamic_batch:
        Export with a dynamic batch dimension.
    dynamic_shape:
        Export with dynamic H/W dimensions (implies dynamic_batch).
    device:
        ``"cpu"`` or a CUDA device index string (e.g. ``"0"``).
    half:
        Export with FP16 weights (requires CUDA device).
    verbose:
        Print extra diagnostic information.
    """
    device_t = torch.device(f"cuda:0" if device != "cpu" and torch.cuda.is_available() else "cpu")
    model, meta = load_model_from_ckpt(checkpoint_path, device=device_t, verbose=verbose)

    if half and device_t.type == "cuda":
        model.half()

    dtype = torch.float16 if (half and device_t.type == "cuda") else torch.float32
    dummy = torch.zeros(1, 3, img_size, img_size, device=device_t, dtype=dtype)

    class _RawWrapper(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
        def forward(self, x):
            y = self.core(x)
            return tuple(y) if isinstance(y, (list, tuple)) else (y,)

    wrapper = _RawWrapper(model)

    with torch.inference_mode():
        outs = wrapper(dummy)
    output_names = [f"p{i}" for i in range(len(outs))]
    log(f"  output_names={output_names}", verbose)

    dynamic_axes = None
    if dynamic_batch or dynamic_shape:
        dynamic_axes = {"images": {0: "batch"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}
        if dynamic_shape:
            dynamic_axes["images"][2] = "height"
            dynamic_axes["images"][3] = "width"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            opset_version=opset,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )


# ========= top-level callable (used by both CLI and programmatic callers) =========
def run_export(
    weights: str,
    out: str = None,
    img_size: int = 640,
    format: str = "decoded",
    opset: int = 17,
    device: str = "cpu",
    half: bool = False,
    simplify: bool = False,
    dynamic_batch: bool = False,
    dynamic_shape: bool = False,
    center_mode: str = "v8",
    wh_mode: str = "softplus",
    verbose: bool = False,
) -> str:
    """Export a yololite checkpoint to ONNX.

    Parameters
    ----------
    weights:
        Path to a ``.pt`` checkpoint.
    out:
        Output ``.onnx`` path.  Auto-generated under ``runs/export/`` if omitted.
    img_size:
        Spatial resolution (H == W).  Overrides the value stored in the checkpoint.
    format:
        ``"decoded"`` (boxes/obj/cls, no NMS) or ``"raw"`` (per-level head tensors).
    opset:
        ONNX opset version.
    device:
        ``"cpu"`` or CUDA device index (e.g. ``"0"``).  Only relevant for raw format.
    half:
        FP16 export (raw format + CUDA only).
    simplify:
        Run ``onnxsim`` on the exported file.
    dynamic_batch:
        Dynamic batch dimension (raw format; decoded is always dynamic).
    dynamic_shape:
        Dynamic H/W dimensions (raw format only).
    center_mode / wh_mode:
        Decoding parameters (decoded format only).
    verbose:
        Print extra diagnostic information.

    Returns
    -------
    str
        Absolute path to the exported ``.onnx`` file.
    """
    device_t = torch.device("cuda:0" if device != "cpu" and torch.cuda.is_available() else "cpu")
    must(f"• Device: {device_t}")

    model, meta = load_model_from_ckpt(weights, device_t, verbose=verbose)

    meta_img_size = int(meta.get("img_size", img_size))
    resolved_size = int(img_size) if img_size else meta_img_size
    must(f"• img_size: {resolved_size}")

    if half and device_t.type == "cuda":
        must("• FP16: ON")

    # Sanity-check forward pass
    dtype = torch.float16 if (half and device_t.type == "cuda") else torch.float32
    dummy = torch.zeros(1, 3, resolved_size, resolved_size, device=device_t, dtype=dtype)
    with torch.inference_mode():
        y = model(dummy)
    if isinstance(y, (list, tuple)):
        must(f"• Torrkörning OK: {len(y)} utgång(ar) (raw head-nivåer).")
    else:
        must("• Torrkörning OK: 1 utgång (monolitisk).")

    export_dir = next_run_dir("runs/export")
    out_path = Path(out) if out else Path(export_dir) / ("model_decoded.onnx" if format == "decoded" else "model.onnx")
    must(f"• Export-katalog: {export_dir}")
    must(f"• Skriver: {out_path}")

    try:
        if format == "raw":
            if dynamic_shape:
                must("• Dynamic shape: ON (råt läge)")
            export_raw_onnx(
                checkpoint_path=weights,
                img_size=resolved_size,
                out_path=str(out_path),
                opset=opset,
                dynamic_batch=dynamic_batch,
                dynamic_shape=dynamic_shape,
                device=device,
                half=half,
                verbose=verbose,
            )
            must("✓ ONNX export (raw) klar")
        else:  # decoded
            if dynamic_shape:
                must("! Ignorerar --dynamic-shape i decoded-läge (kräver fast img_size).")
            export_decoded_onnx(
                checkpoint_path=weights,
                img_size=resolved_size,
                out_path=str(out_path),
                opset=opset,
                center_mode=center_mode,
                wh_mode=wh_mode,
            )
            must("✓ ONNX export (decoded) klar")
    except Exception as e:
        raise RuntimeError(f"ONNX-export ({format}) misslyckades: {e}") from e

    if simplify:
        try:
            import onnx, onnxsim
            model_onnx = onnx.load(str(out_path))
            model_simplified, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(model_simplified, str(out_path))
                must("✓ ONNX simplified")
            else:
                must("! onnxsim returnerade ok=False (sparar osimplifierad)")
        except Exception as e:
            must(f"! onnxsim misslyckades: {e}")

    must("Klart.")
    return str(out_path)


# ========= CLI =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to checkpoint (.pt/.pth)")
    ap.add_argument("--out", default=None, help="save path (.onnx). Default: runs/export/<n>/model(.onnx|_decoded.onnx)")
    ap.add_argument("--img-size", type=int, default=640, help="(H=W)")
    ap.add_argument("--device", default="cpu", help="'cpu' or cuda index '0'")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--half", action="store_true", help="FP16 (requires CUDA for dummy)")
    ap.add_argument("--simplify", action="store_true", help="onnxsim after export")
    ap.add_argument("--dynamic-batch", action="store_true", help="Dynamic batch-dimension (raw format)")
    ap.add_argument("--dynamic-shape", action="store_true", help="Dynamic H/W (raw format only)")
    ap.add_argument("--format", choices=["raw", "decoded"], default="decoded",
                    help="raw = raw data per nivå; decoded = boxes/obj/cls")
    ap.add_argument("--center-mode", default="v8", choices=["v8", "sigmoid"], help="Decode-center (decoded)")
    ap.add_argument("--wh-mode", default="softplus", choices=["softplus", "v8", "exp"], help="Decode-wh (decoded)")
    ap.add_argument("--verbose", action="store_true", help="Detailed logg")
    args = ap.parse_args()

    run_export(
        weights=args.weights,
        out=args.out,
        img_size=args.img_size,
        format=args.format,
        opset=args.opset,
        device=args.device,
        half=args.half,
        simplify=args.simplify,
        dynamic_batch=args.dynamic_batch,
        dynamic_shape=args.dynamic_shape,
        center_mode=args.center_mode,
        wh_mode=args.wh_mode,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
