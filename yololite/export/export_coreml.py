"""Export a YOLOLite checkpoint to CoreML (.mlpackage) with baked-in pre/post processing.

Output tensors match the RF-DETR CoreML convention for roboflow-swift compatibility:
    - boxes:  [1, K, 4]  -- cxcywh normalised coordinates (values in 0-1)
    - scores: [1, K]     -- confidence (sigmoid applied)
    - labels: [1, K]     -- class indices (int, from argmax)
"""

import argparse
import logging

import torch
import torch.nn as nn

from .export_onnx import AFDecode, load_model_from_ckpt

logger = logging.getLogger(__name__)


class YOLOLiteCoreML(nn.Module):
    """Wrapper that bakes preprocessing, inference, decoding, and postprocessing
    into a single forward pass suitable for CoreML export.

    Parameters
    ----------
    model:
        The underlying YOLOLite model (already loaded & eval-mode).
    img_size:
        Spatial resolution (H == W) the model expects.
    num_classes:
        Number of object classes.
    center_mode / wh_mode:
        Decoding parameters forwarded to ``AFDecode``.
    top_k:
        Maximum number of detections to return.
    """

    def __init__(
        self,
        model: nn.Module,
        img_size: int,
        num_classes: int,
        center_mode: str = "v8",
        wh_mode: str = "softplus",
        top_k: int = 300,
    ):
        super().__init__()
        self.model = model
        self.decode = AFDecode(img_size=img_size, center_mode=center_mode, wh_mode=wh_mode)
        self.img_size = img_size
        self.num_classes = num_classes
        self.top_k = top_k

        # ImageNet normalisation buffers (NCHW broadcasting shape)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor  [1, 3, H, W]
            Raw uint8-range image (0-255, float32).

        Returns
        -------
        boxes  : Tensor [1, K, 4]  -- cxcywh normalised
        scores : Tensor [1, K]     -- confidence
        labels : Tensor [1, K]     -- class indices (int64)
        """
        # --- Preprocessing ---
        x = x / 255.0
        x = (x - self.mean) / self.std

        # --- Inference ---
        raw = self.model(x)

        # --- Decode ---
        boxes_xyxy, obj_logits, cls_logits = self.decode(raw)
        # boxes_xyxy: [B, N, 4], obj_logits: [B, N, 1], cls_logits: [B, N, C]

        # --- Postprocessing ---
        obj_conf = torch.sigmoid(obj_logits)          # [B, N, 1]
        cls_prob = torch.sigmoid(cls_logits)           # [B, N, C]
        cls_max, cls_idx = cls_prob.max(dim=-1)        # [B, N], [B, N]
        confidence = obj_conf.squeeze(-1) * cls_max    # [B, N]

        # Top-K selection
        k = min(self.top_k, confidence.shape[-1])
        topk_scores, topk_indices = torch.topk(confidence, k, dim=-1)  # [B, K]

        # Gather boxes and labels
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, 4)  # [B, K, 4]
        topk_boxes = torch.gather(boxes_xyxy, 1, topk_indices_exp)      # [B, K, 4]
        topk_labels = torch.gather(cls_idx, 1, topk_indices)            # [B, K]

        # Convert xyxy pixel coords to normalised cxcywh
        x1 = topk_boxes[..., 0]
        y1 = topk_boxes[..., 1]
        x2 = topk_boxes[..., 2]
        y2 = topk_boxes[..., 3]
        cx = (x1 + x2) / (2.0 * self.img_size)
        cy = (y1 + y2) / (2.0 * self.img_size)
        w = (x2 - x1) / float(self.img_size)
        h = (y2 - y1) / float(self.img_size)
        boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)  # [B, K, 4]

        return boxes_cxcywh, topk_scores, topk_labels


def export_coreml(
    checkpoint_path: str,
    img_size: int = 640,
    out_path: str = "weights.mlpackage",
    center_mode: str = "v8",
    wh_mode: str = "softplus",
) -> str:
    """Export a YOLOLite checkpoint to CoreML (.mlpackage).

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint saved by ``save_checkpoint_state()``.
    img_size:
        Spatial resolution (H == W).
    out_path:
        Destination ``.mlpackage`` path.
    center_mode / wh_mode:
        Decoding parameters forwarded to ``AFDecode``.

    Returns
    -------
    str
        Absolute path to the saved ``.mlpackage``.
    """
    import coremltools as ct

    device = torch.device("cpu")
    model, meta = load_model_from_ckpt(checkpoint_path, device=device, verbose=False)

    num_classes = int(meta.get("num_classes", 80))
    logger.info("Loaded model: arch=%s backbone=%s num_classes=%d",
                meta.get("arch"), meta.get("backbone"), num_classes)

    wrapper = YOLOLiteCoreML(
        model=model,
        img_size=img_size,
        num_classes=num_classes,
        center_mode=center_mode,
        wh_mode=wh_mode,
    ).eval()

    # Step 1: Trace
    logger.info("Step 1/3: Tracing model (img_size=%d)...", img_size)
    sample_input = torch.randint(
        0, 256, (1, 3, img_size, img_size), dtype=torch.float32,
    )
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_input)
    logger.info("✅ Model traced")

    # Step 2: Convert to CoreML
    logger.info("Step 2/3: Converting to CoreML (.mlpackage)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image_input",
                shape=(1, 3, img_size, img_size),
            ),
        ],
        outputs=[
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores"),
            ct.TensorType(name="labels"),
        ],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )
    logger.info("✅ CoreML conversion complete")

    # Step 3: Save
    logger.info("Step 3/3: Saving CoreML model...")
    mlmodel.save(out_path)
    logger.info("✅ CoreML model saved to %s", out_path)
    return str(out_path)


# ========= CLI =========
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Export a YOLOLite checkpoint to CoreML (.mlpackage).",
    )
    ap.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint (.pt/.pth)",
    )
    ap.add_argument(
        "--img-size", type=int, default=640,
        help="Spatial resolution (H=W) (default: 640)",
    )
    ap.add_argument(
        "--out-path", default="weights.mlpackage",
        help="Output .mlpackage path (default: weights.mlpackage)",
    )
    ap.add_argument(
        "--center-mode", default="v8", choices=["v8", "sigmoid"],
        help="Decode center mode (default: v8)",
    )
    ap.add_argument(
        "--wh-mode", default="softplus", choices=["softplus", "v8", "exp"],
        help="Decode wh mode (default: softplus)",
    )
    args = ap.parse_args()

    export_coreml(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        out_path=args.out_path,
        center_mode=args.center_mode,
        wh_mode=args.wh_mode,
    )


if __name__ == "__main__":
    main()
