"""Export a YOLOLite checkpoint to TensorFlow.js format.

Pipeline:
    1. Export raw ONNX (per-level NCHW head outputs, no decode)
    2. onnx2tf -> TF SavedModel  (backbone/FPN/heads are standard conv ops)
    3. Wrap SavedModel with TF decode layer (meshgrid + box math + concat)
    4. Validate wrapped model vs decoded ONNX
    5. Wrap decoded model with NMS (tf.image.non_max_suppression)
    6. TF SavedModel -> TFJS

The decode is implemented in pure TF ops in the wrapper — never processed
by onnx2tf — avoiding its buggy NCW->NWC transposition on non-spatial
tensors entirely.  NMS is applied after decode so the TFJS model outputs
filtered detections directly.
"""

import argparse
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_CALIBRATION_FILE = "calibration_image_sample_data_20x128x128x3_float32.npy"


def _ensure_calibration_data() -> None:
    """Generate onnx2tf calibration npy in cwd if missing."""
    import numpy as np

    local_path = Path(_CALIBRATION_FILE)
    if local_path.exists():
        return
    np.save(local_path, np.random.rand(20, 128, 128, 3).astype(np.float32))


def _export_nchw_heads_onnx(checkpoint_path: str, img_size: int,
                            out_path: str, opset: int = 17) -> list:
    """Export per-level head outputs as 4D NCHW tensors [B, D, S, S].

    Returns the list of strides for each output level.
    """
    import torch
    import torch.nn as nn
    from .export_onnx import load_model_from_ckpt

    device = torch.device("cpu")
    model, _ = load_model_from_ckpt(checkpoint_path, device=device, verbose=False)

    class _NCHWHeadsWrapper(nn.Module):
        """Wraps model to output per-level [B, D, S, S] NCHW tensors.

        Avoids the 5D view+permute in _forward_head by concatenating
        box/obj/cls conv outputs along the channel dimension directly.
        """
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, x):
            feats = self.core.backbone(x)

            if self.core.use_p2:
                c2, c3, c4, c5 = feats
            else:
                c3, c4, c5 = feats

            p5 = self.core.smooth5(self.core.lateral5(c5))
            p4 = self.core.smooth4(self.core._upsample_add(p5, self.core.lateral4(c4)))
            p3 = self.core.smooth3(self.core._upsample_add(p4, self.core.lateral3(c3)))

            levels = []

            if self.core.use_p2:
                p2 = self.core.smooth2(self.core._upsample_add(p3, self.core.lateral2(c2)))
                levels.append(self._head_nchw(p2, self.core.head2))

            levels.append(self._head_nchw(p3, self.core.head3))
            levels.append(self._head_nchw(p4, self.core.head4))
            levels.append(self._head_nchw(p5, self.core.head5))

            if self.core.use_p6:
                p6 = self.core.smooth6(self.core.p6_act(self.core.p6_bn(self.core.p6_down(p5))))
                levels.append(self._head_nchw(p6, self.core.head6))

            return tuple(levels)

        @staticmethod
        def _head_nchw(p, head_dict):
            """Run head and return [B, D, S, S] in NCHW — no 5D reshape."""
            p = head_dict["trunk"](p)
            box = head_dict["out"]["box"](p)  # [B, A*4, S, S]
            obj = head_dict["out"]["obj"](p)  # [B, A*1, S, S]
            cls = head_dict["out"]["cls"](p)  # [B, A*C, S, S]
            return torch.cat([box, obj, cls], dim=1)  # [B, D, S, S]

    wrapper = _NCHWHeadsWrapper(model).eval()
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)

    with torch.inference_mode():
        outs = wrapper(dummy)

    output_names = [f"level_{i}" for i in range(len(outs))]

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, out_path,
            opset_version=opset,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes={"images": {0: "batch"}},
            do_constant_folding=True,
            external_data=False,
        )

    logger.info("Exported NCHW heads ONNX: %d levels, strides=%s",
                len(outs), model.get_strides())


def _wrap_with_decode(tf_model_dir: str, wrapped_dir: str,
                      img_size: int, num_classes: int,
                      center_mode: str, wh_mode: str) -> None:
    """Wrap raw-heads SavedModel with TF decode → boxes/obj/cls outputs.

    Implements the same decode logic as AFDecode but in pure TF ops.
    The inner model outputs per-level NHWC tensors [B, S, S, D] (after
    onnx2tf's NCHW→NHWC conversion).  This wrapper decodes each level
    and concatenates into [B, N_total, 4/1/nc].
    """
    import tensorflow as tf
    inner = tf.saved_model.load(tf_model_dir)
    inner_fn = inner.signatures["serving_default"]
    input_spec = list(inner_fn.structured_input_signature[1].values())[0]

    # Sort output keys to match level order (level_0, level_1, ...)
    output_keys = sorted(inner_fn.structured_outputs.keys())

    _img_size = float(img_size)
    _nc = num_classes
    _center_mode = center_mode
    _wh_mode = wh_mode

    class DecodeWrapper(tf.Module):
        def __init__(self):
            super().__init__()
            self._inner_fn = inner_fn

        @tf.function(input_signature=[input_spec])
        def serve(self, images):
            results = self._inner_fn(images=images)

            all_boxes, all_obj, all_cls = [], [], []

            for key in output_keys:
                level = results[key]  # [B, S, S, D] after NHWC conversion
                S = tf.shape(level)[1]
                cell = _img_size / tf.cast(S, tf.float32)

                # Slice features from last dim: [box(4), obj(1), cls(nc)]
                tx = level[..., 0]
                ty = level[..., 1]
                tw = level[..., 2]
                th = level[..., 3]
                obj = level[..., 4:5]
                cls = level[..., 5:]

                # Grid
                gy = tf.cast(tf.range(S), tf.float32)
                gx = tf.cast(tf.range(S), tf.float32)
                grid_y, grid_x = tf.meshgrid(gy, gx, indexing='ij')

                # Decode centers
                if _center_mode == "v8":
                    px = (tf.sigmoid(tx) * 2.0 - 0.5 + grid_x) * cell
                    py = (tf.sigmoid(ty) * 2.0 - 0.5 + grid_y) * cell
                else:
                    px = (tf.sigmoid(tx) + grid_x) * cell
                    py = (tf.sigmoid(ty) + grid_y) * cell

                # Decode width/height
                if _wh_mode == "v8":
                    pw = tf.pow(tf.sigmoid(tw) * 2.0, 2) * cell
                    ph = tf.pow(tf.sigmoid(th) * 2.0, 2) * cell
                elif _wh_mode == "softplus":
                    pw = tf.math.softplus(tw) * cell
                    ph = tf.math.softplus(th) * cell
                else:
                    pw = tf.exp(tf.clip_by_value(tw, -4.0, 4.0)) * cell
                    ph = tf.exp(tf.clip_by_value(th, -4.0, 4.0)) * cell

                # xywh -> xyxy
                x1 = px - pw * 0.5
                y1 = py - ph * 0.5
                x2 = px + pw * 0.5
                y2 = py + ph * 0.5

                # Clamp
                x1 = tf.clip_by_value(x1, 0.0, _img_size - 1)
                y1 = tf.clip_by_value(y1, 0.0, _img_size - 1)
                x2 = tf.clip_by_value(x2, 0.0, _img_size - 1)
                y2 = tf.clip_by_value(y2, 0.0, _img_size - 1)

                # Stack and reshape: [B, S, S, 4] -> [B, S*S, 4]
                boxes = tf.stack([x1, y1, x2, y2], axis=-1)
                boxes = tf.reshape(boxes, [-1, S * S, 4])
                obj = tf.reshape(obj, [-1, S * S, 1])
                cls = tf.reshape(cls, [-1, S * S, _nc])

                all_boxes.append(boxes)
                all_obj.append(obj)
                all_cls.append(cls)

            return {
                "boxes_xyxy": tf.concat(all_boxes, axis=1),
                "obj_logits": tf.concat(all_obj, axis=1),
                "cls_logits": tf.concat(all_cls, axis=1),
            }

    wrapper = DecodeWrapper()
    tf.saved_model.save(
        wrapper, wrapped_dir,
        signatures={"serving_default": wrapper.serve},
    )
    logger.info("Wrapped SavedModel with decode at %s", wrapped_dir)


def _wrap_with_nms(decoded_dir: str, nms_dir: str, img_size: int,
                   max_detections: int = 500) -> None:
    """Wrap a decoded SavedModel with NMS post-processing.

    Takes the decode wrapper outputs (boxes_xyxy, obj_logits, cls_logits)
    and applies sigmoid, confidence scoring, and tf.image.non_max_suppression.

    Outputs:
        boxes:   [M, 4]  -- xyxy pixel coordinates of kept detections
        scores:  [M]     -- confidence scores
        classes: [M]     -- class indices (int32)
    where M <= max_detections is variable.
    """
    import tensorflow as tf

    inner = tf.saved_model.load(decoded_dir)
    inner_fn = inner.signatures["serving_default"]
    input_spec = list(inner_fn.structured_input_signature[1].values())[0]

    _img_size = float(img_size)
    _max_det = max_detections

    class NMSWrapper(tf.Module):
        def __init__(self):
            super().__init__()
            self._inner_fn = inner_fn

        @tf.function(input_signature=[
            input_spec,
            tf.TensorSpec([], tf.float32, name="iou_threshold"),
            tf.TensorSpec([], tf.float32, name="score_threshold"),
        ])
        def serve(self, images, iou_threshold, score_threshold):
            results = self._inner_fn(images=images)
            boxes_xyxy = results["boxes_xyxy"]    # [B, N, 4]
            obj_logits = results["obj_logits"]    # [B, N, 1]
            cls_logits = results["cls_logits"]    # [B, N, C]

            # Sigmoid + confidence (batch dim squeezed — single image)
            obj_conf = tf.sigmoid(tf.squeeze(obj_logits[0], axis=-1))  # [N]
            cls_prob = tf.sigmoid(cls_logits[0])                       # [N, C]
            cls_max = tf.reduce_max(cls_prob, axis=-1)                 # [N]
            confidence = obj_conf * cls_max                            # [N]
            class_indices = tf.cast(tf.argmax(cls_prob, axis=-1), tf.int32)  # [N]

            # NMS expects [y1, x1, y2, x2] normalised — convert from xyxy pixel
            raw_boxes = boxes_xyxy[0]  # [N, 4] as x1,y1,x2,y2
            x1, y1, x2, y2 = (raw_boxes[:, i] for i in range(4))
            nms_boxes = tf.stack([y1, x1, y2, x2], axis=-1) / _img_size

            selected = tf.image.non_max_suppression(
                nms_boxes, confidence,
                max_output_size=_max_det,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )

            return {
                "boxes": tf.gather(raw_boxes, selected),
                "scores": tf.gather(confidence, selected),
                "classes": tf.gather(class_indices, selected),
            }

    wrapper = NMSWrapper()
    tf.saved_model.save(
        wrapper, nms_dir,
        signatures={"serving_default": wrapper.serve},
    )
    logger.info("Wrapped decoded model with NMS at %s", nms_dir)


def _validate_against_onnx(decoded_onnx_path: str, tf_model_dir: str) -> None:
    """Compare decoded ONNX vs wrapped TF SavedModel outputs."""
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    sess = ort.InferenceSession(decoded_onnx_path, providers=["CPUExecutionProvider"])
    onnx_input = sess.get_inputs()[0]
    static_shape = [d if isinstance(d, int) else 1 for d in onnx_input.shape]
    _, c, h, w = static_shape

    np.random.seed(42)
    nchw_data = np.random.rand(1, c, h, w).astype(np.float32)
    nhwc_data = nchw_data.transpose(0, 2, 3, 1)

    onnx_outputs = sess.run(None, {onnx_input.name: nchw_data})

    tf_model = tf.saved_model.load(tf_model_dir)
    tf_fn = tf_model.signatures["serving_default"]
    tf_results = tf_fn(images=tf.constant(nhwc_data))
    tf_outputs = [tf_results[k].numpy() for k in sorted(tf_results.keys())]

    onnx_by_shape = sorted(onnx_outputs, key=lambda x: x.shape)
    tf_by_shape = sorted(tf_outputs, key=lambda x: x.shape)

    for i, (o, t) in enumerate(zip(onnx_by_shape, tf_by_shape)):
        if o.shape != t.shape:
            raise RuntimeError(
                f"TFJS validation failed: output {i} shape mismatch: "
                f"ONNX {o.shape} vs TF {t.shape}"
            )
        max_err = np.max(np.abs(o - t))
        mean_err = np.mean(np.abs(o - t))
        logger.info(
            "  output %d %s: max_err=%.6f mean_err=%.6f",
            i, o.shape, max_err, mean_err,
        )
        if max_err > 0.05:
            raise RuntimeError(
                f"TFJS validation failed: output {i} max error {max_err:.6f} "
                f"exceeds tolerance 0.05"
            )

    logger.info("Validation passed: ONNX and TF outputs match within tolerance")


def export_tfjs(
    checkpoint_path: str,
    out_dir: str,
    img_size: int = 640,
    quantize: bool = True,
    shard_size_bytes: int = 1_048_576,
    center_mode: str = "v8",
    wh_mode: str = "softplus",
    max_detections: int = 500,
) -> str:
    """Convert a yololite checkpoint to TensorFlow.js graph model format.

    Exports a raw NCHW-heads ONNX (standard conv ops only), converts via
    onnx2tf, wraps with a TF decode layer, validates against decoded ONNX,
    then adds NMS.  The resulting TFJS model outputs filtered detections
    (boxes, scores, classes) with NMS thresholds configurable at runtime
    via model inputs (iou_threshold, score_threshold scalars).

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint.
    out_dir:
        Directory where the TFJS artifacts will be written.
    img_size:
        Spatial resolution (H == W).
    quantize:
        Apply uint8 weight quantization.
    shard_size_bytes:
        Maximum weight shard file size in bytes.
    center_mode / wh_mode:
        Decoding parameters forwarded to the TF decode wrapper.
    max_detections:
        Maximum number of detections after NMS.
    """
    out_dir = str(Path(out_dir).resolve())

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    import onnx2tf
    from tensorflowjs.converters.tf_saved_model_conversion_v2 import convert_tf_saved_model
    from .export_onnx import export_decoded_onnx

    _ensure_calibration_data()

    import torch
    meta = torch.load(checkpoint_path, map_location="cpu").get("meta", {})
    num_classes = int(meta.get("num_classes", 80))

    with tempfile.TemporaryDirectory(prefix="yololite_tfjs_") as tmp:
        # Step 1: Export raw NCHW heads ONNX
        raw_onnx = str(Path(tmp) / "heads_nchw.onnx")
        logger.info("Step 1/6: Exporting NCHW heads ONNX...")
        _export_nchw_heads_onnx(checkpoint_path, img_size, raw_onnx)
        try:
            import onnx, onnxsim
            m = onnx.load(raw_onnx)
            m_sim, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m_sim, raw_onnx)
        except Exception:
            pass
        logger.info("✅ NCHW heads ONNX exported")

        # Step 2: Convert ONNX to TF SavedModel
        tf_model_dir = str(Path(tmp) / "tf_savedmodel")
        logger.info("Step 2/6: Converting ONNX to TensorFlow SavedModel...")
        onnx2tf.convert(
            input_onnx_file_path=raw_onnx,
            output_folder_path=tf_model_dir,
            not_use_onnxsim=True,
            verbosity="error",
            output_integer_quantized_tflite=False,
            quant_type="per-tensor",
            custom_input_op_name_np_data_path=None,
            enable_batchmatmul_unfold=True,
            output_signaturedefs=True,
            disable_group_convolution=True,
            # onnx2tf 2.4.0 made flatbuffer_direct the default backend. That backend
            # writes TFLite only, so the SavedModel step 3 loads below never appears.
            # tf_converter is also the only path that converts the convnextv2_tiny
            # backbone of yololite-l; flatbuffer_direct rejects it on GELU.
            tflite_backend="tf_converter",
        )
        logger.info("✅ TensorFlow SavedModel created")

        # Step 3: Wrap with TF decode layer
        wrapped_dir = str(Path(tmp) / "tf_decoded")
        logger.info("Step 3/6: Wrapping SavedModel with decode layer...")
        _wrap_with_decode(
            tf_model_dir, wrapped_dir, img_size, num_classes,
            center_mode, wh_mode,
        )
        logger.info("✅ Decode wrapper applied")

        # Step 4: Validate against decoded ONNX
        logger.info("Step 4/6: Validating wrapped model vs decoded ONNX...")
        decoded_onnx = str(Path(tmp) / "decoded.onnx")
        export_decoded_onnx(
            checkpoint_path=checkpoint_path, img_size=img_size,
            out_path=decoded_onnx,
        )
        _validate_against_onnx(decoded_onnx, wrapped_dir)
        logger.info("✅ Validation passed")

        # Step 5: Wrap with NMS
        nms_dir = str(Path(tmp) / "tf_nms")
        logger.info("Step 5/6: Adding NMS (max=%d, thresholds configurable at runtime)...",
                     max_detections)
        _wrap_with_nms(wrapped_dir, nms_dir, img_size, max_detections=max_detections)
        logger.info("✅ NMS wrapper applied")

        # Step 6: Convert to TensorFlow.js
        logger.info("Step 6/6: Converting SavedModel to TensorFlow.js...")
        convert_tf_saved_model(
            nms_dir,
            out_dir,
            signature_def="serving_default",
            saved_model_tags="serve",
            quantization_dtype_map={"uint8": "*"} if quantize else None,
            weight_shard_size_bytes=shard_size_bytes,
        )
        logger.info("✅ TensorFlow.js model exported to %s", out_dir)

    return out_dir


# ========= CLI =========
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Export a yololite checkpoint to TensorFlow.js format.",
    )
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--out-dir", required=True, help="Output directory for TFJS artifacts")
    ap.add_argument("--img-size", type=int, default=640, help="Image size (default: 640)")
    ap.add_argument("--no-quantize", action="store_true", help="Disable uint8 weight quantization")
    ap.add_argument("--shard-size-bytes", type=int, default=1_048_576)
    args = ap.parse_args()

    export_tfjs(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        img_size=args.img_size,
        quantize=not args.no_quantize,
        shard_size_bytes=args.shard_size_bytes,
    )


if __name__ == "__main__":
    main()
