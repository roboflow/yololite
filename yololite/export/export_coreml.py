"""Export a YOLOLite checkpoint to CoreML (.mlpackage) with baked-in NMS.

The exported model is a pipeline:
    1. Detection model: preprocessing → backbone → decode → per-class confidence
    2. CoreML NMS: filters detections by confidence and IoU overlap

VNCoreML returns VNRecognizedObjectObservation objects with class labels,
matching the YOLOv5 detection path in roboflow-swift.  Runtime thresholds
are adjustable via VNCoreMLModel.featureProvider (iouThreshold,
confidenceThreshold).
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

    Outputs ``[N, 4]`` normalised cxcywh coordinates and ``[N, C]`` per-class
    confidence scores — the format expected by CoreML's NMS layer.
    """

    def __init__(
        self,
        model: nn.Module,
        img_size: int,
        num_classes: int,
        center_mode: str = "v8",
        wh_mode: str = "softplus",
    ):
        super().__init__()
        self.model = model
        self.decode = AFDecode(img_size=img_size, center_mode=center_mode, wh_mode=wh_mode)
        self.img_size = img_size
        self.num_classes = num_classes

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
        coordinates : Tensor [N, 4]  -- cxcywh normalised (0-1)
        confidence  : Tensor [N, C]  -- per-class confidence
        """
        # --- Preprocessing ---
        x = x / 255.0
        x = (x - self.mean) / self.std

        # --- Inference ---
        raw = self.model(x)

        # --- Decode ---
        boxes_xyxy, obj_logits, cls_logits = self.decode(raw)
        # boxes_xyxy: [B, N, 4], obj_logits: [B, N, 1], cls_logits: [B, N, C]

        # --- Per-class confidence ---
        obj_conf = torch.sigmoid(obj_logits)          # [B, N, 1]
        cls_prob = torch.sigmoid(cls_logits)           # [B, N, C]
        confidence = obj_conf * cls_prob              # [B, N, C]

        # Convert xyxy pixel coords to normalised cxcywh
        img = float(self.img_size)
        x1 = boxes_xyxy[..., 0]
        y1 = boxes_xyxy[..., 1]
        x2 = boxes_xyxy[..., 2]
        y2 = boxes_xyxy[..., 3]
        coordinates = torch.stack([
            (x1 + x2) / (2.0 * img),
            (y1 + y2) / (2.0 * img),
            (x2 - x1) / img,
            (y2 - y1) / img,
        ], dim=-1)  # [B, N, 4]

        # Squeeze batch dim — CoreML NMS expects [N, 4] and [N, C]
        return coordinates.squeeze(0), confidence.squeeze(0)


def _build_nms_pipeline(det_model, class_names, num_detections, num_classes,
                        iou_threshold=0.45, confidence_threshold=0.25):
    """Wrap a detection CoreML model with NMS in a pipeline.

    The resulting pipeline accepts an image plus optional iouThreshold /
    confidenceThreshold scalars, and returns filtered coordinates and
    confidence arrays.  VNCoreML interprets this as object detection and
    returns VNRecognizedObjectObservation results.
    """
    import coremltools as ct
    from coremltools.proto import Model_pb2

    det_spec = det_model.get_spec()

    # Rename detection outputs and set explicit shapes for NMS compatibility.
    # Use FLOAT32 to match the actual neuralnetwork computation dtype.
    _DTYPE = Model_pb2.ArrayFeatureType.FLOAT32
    det_outputs = [
        ("coordinates", [num_detections, 4]),
        ("confidence", [num_detections, num_classes]),
    ]
    for i, (name, shape) in enumerate(det_outputs):
        out = det_spec.description.output[i]
        out.name = name
        out.type.multiArrayType.dataType = _DTYPE
        out.type.multiArrayType.shape[:] = shape

    # --- NMS model spec ---
    nms_spec = Model_pb2.Model()
    nms_spec.specificationVersion = det_spec.specificationVersion

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = "confidence"
    nms.coordinatesInputFeatureName = "coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = iou_threshold
    nms.confidenceThreshold = confidence_threshold

    for name in class_names:
        nms.stringClassLabels.vector.append(name)

    def _add_array_features(desc_list, features):
        for feat_name, shape in features:
            feat = desc_list.add()
            feat.name = feat_name
            feat.type.multiArrayType.dataType = _DTYPE
            for s in shape:
                feat.type.multiArrayType.shape.append(s)

    _add_array_features(nms_spec.description.input, det_outputs)
    for feat_name in ["iouThreshold", "confidenceThreshold"]:
        inp = nms_spec.description.input.add()
        inp.name = feat_name
        inp.type.doubleType.MergeFromString(b"")
    _add_array_features(nms_spec.description.output, det_outputs)

    # --- Pipeline spec ---
    pipeline_spec = Model_pb2.Model()
    pipeline_spec.specificationVersion = det_spec.specificationVersion

    # Pipeline input: image (from detection model)
    pipeline_spec.description.input.add().CopyFrom(det_spec.description.input[0])

    # Pipeline threshold inputs (optional, with defaults from NMS spec)
    for feat_name in ["iouThreshold", "confidenceThreshold"]:
        inp = pipeline_spec.description.input.add()
        inp.name = feat_name
        inp.type.doubleType.MergeFromString(b"")

    # Pipeline outputs = NMS outputs
    for out in nms_spec.description.output:
        pipeline_spec.description.output.add().CopyFrom(out)

    # Add models to pipeline
    pipeline_spec.pipeline.models.add().CopyFrom(det_spec)
    pipeline_spec.pipeline.models.add().CopyFrom(nms_spec)

    return ct.models.MLModel(pipeline_spec)


def export_coreml(
    checkpoint_path: str,
    img_size: int = 640,
    out_path: str = "weights.mlpackage",
    center_mode: str = "v8",
    wh_mode: str = "softplus",
    iou_threshold: float = 0.45,
    confidence_threshold: float = 0.25,
) -> str:
    """Export a YOLOLite checkpoint to CoreML (.mlpackage) with NMS.

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
    iou_threshold:
        Default NMS IoU threshold (overridable at runtime).
    confidence_threshold:
        Default NMS confidence threshold (overridable at runtime).

    Returns
    -------
    str
        Absolute path to the saved ``.mlpackage``.
    """
    import coremltools as ct

    device = torch.device("cpu")
    model, meta = load_model_from_ckpt(checkpoint_path, device=device, verbose=False)

    num_classes = int(meta.get("num_classes", 80))
    class_names = meta.get("names") or [str(i) for i in range(num_classes)]
    if len(class_names) != num_classes:
        logger.warning("class_names length (%d) != num_classes (%d), padding",
                       len(class_names), num_classes)
        class_names = list(class_names) + [str(i) for i in range(len(class_names), num_classes)]

    logger.info("Loaded model: arch=%s backbone=%s num_classes=%d classes=%s",
                meta.get("arch"), meta.get("backbone"), num_classes, class_names)

    wrapper = YOLOLiteCoreML(
        model=model,
        img_size=img_size,
        num_classes=num_classes,
        center_mode=center_mode,
        wh_mode=wh_mode,
    ).eval()

    # Step 1: Trace
    logger.info("Step 1/4: Tracing model (img_size=%d)...", img_size)
    sample_input = torch.randint(
        0, 256, (1, 3, img_size, img_size), dtype=torch.float32,
    )
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_input)
        # Determine num_detections from traced output
        coords_out, _ = traced(sample_input)
        num_detections = coords_out.shape[0]
    logger.info("✅ Model traced (%d detections)", num_detections)

    # Step 2: Convert to CoreML (neuralnetwork for NMS pipeline compatibility)
    logger.info("Step 2/4: Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, img_size, img_size),
            ),
        ],
        outputs=[
            ct.TensorType(name="coordinates"),
            ct.TensorType(name="confidence"),
        ],
        convert_to="neuralnetwork",
    )
    logger.info("✅ CoreML conversion complete")

    # Step 3: Add NMS pipeline
    logger.info("Step 3/4: Adding NMS pipeline (iou=%.2f, conf=%.2f)...",
                iou_threshold, confidence_threshold)
    pipeline_model = _build_nms_pipeline(
        mlmodel, class_names, num_detections, num_classes,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )
    logger.info("✅ NMS pipeline built")

    # Step 4: Save
    logger.info("Step 4/4: Saving CoreML model...")
    pipeline_model.save(out_path)
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
    ap.add_argument(
        "--iou-threshold", type=float, default=0.45,
        help="Default NMS IoU threshold (default: 0.45)",
    )
    ap.add_argument(
        "--confidence-threshold", type=float, default=0.25,
        help="Default NMS confidence threshold (default: 0.25)",
    )
    args = ap.parse_args()

    export_coreml(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        out_path=args.out_path,
        center_mode=args.center_mode,
        wh_mode=args.wh_mode,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
