"""
PP-OCRv6 Text Detection via PaddleX/PaddleOCR APIs.

Handles:
- Creating the text detector across PaddleX versions
- Parsing various output formats (dt_polys, rec_polys, boxes, etc.)
- Filtering boxes by size/aspect ratio/det_score
"""
from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _get_paddle_device(cfg: dict) -> str:
    """Determine Paddle device from config or auto-detect."""
    import os

    device = os.environ.get("OCR_V2_PADDLE_DEVICE") or cfg.get("paddle_device")
    if device:
        return device
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            return "gpu:0"
    except Exception:
        pass
    return "cpu"


def create_detector(cfg: dict):
    """Create a PP-OCRv6 text detector.

    Returns:
        Detector model object with a predict(image) method.
    """
    import os
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    # Stub modelscope to avoid import issues
    _install_modelscope_stub()

    device = _get_paddle_device(cfg)
    logger.info(f"Creating PP-OCRv6 detector on device={device}")

    try:
        from paddleocr import TextDetection

        return TextDetection(
            model_name=cfg["det_model_name"],
            device=device,
            engine="paddle_static",
            limit_side_len=cfg["det_limit_side_len"],
            limit_type=cfg["det_limit_type"],
            thresh=cfg["det_thresh"],
            box_thresh=cfg["det_box_thresh"],
            unclip_ratio=cfg["det_unclip_ratio"],
            enable_mkldnn=False,
            cpu_threads=4,
        )
    except (ImportError, AttributeError) as exc:
        logger.info(f"PaddleOCR TextDetection unavailable ({exc}); trying paddlex.TextDetection.")

    try:
        from paddlex import TextDetection

        return TextDetection(
            model_name=cfg["det_model_name"],
            device=device,
            engine="paddle_static",
            limit_side_len=cfg["det_limit_side_len"],
            limit_type=cfg["det_limit_type"],
            thresh=cfg["det_thresh"],
            box_thresh=cfg["det_box_thresh"],
            unclip_ratio=cfg["det_unclip_ratio"],
            enable_mkldnn=False,
            cpu_threads=4,
        )
    except (ImportError, AttributeError) as exc:
        logger.info(f"PaddleX TextDetection unavailable ({exc}); trying PaddleOCR end-to-end detector.")

    try:
        return _PaddleOCRDetector(cfg)
    except Exception as exc:
        logger.info(f"PaddleOCR end-to-end detector unavailable ({exc}); trying paddlex.create_model.")

    try:
        from paddlex import create_model

        model = create_model(model_name=cfg["det_model_name"])
        return _PaddleXCreateModelDetector(model)
    except Exception as exc:
        logger.info(f"paddlex.create_model detector unavailable ({exc}); falling back to PaddleOCR.")

    return _PaddleOCRDetector(cfg)


class _PaddleXCreateModelDetector:
    """Adapter for PaddleX 3.x create_model() detection models."""

    def __init__(self, model):
        self.model = model

    def predict(self, img_rgb: np.ndarray):
        try:
            result = self.model.predict(input=img_rgb, batch_size=1)
        except TypeError:
            result = self.model.predict(img_rgb)
        return _materialize_prediction(result)


class _PaddleOCRDetector:
    """Adapter using PaddleOCR 3.x end-to-end OCR output as detection source."""

    def __init__(self, cfg: dict):
        from paddleocr import PaddleOCR

        self.engine = PaddleOCR(
            ocr_version="PP-OCRv6",
            lang="vi",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=cfg["det_limit_side_len"],
            text_det_thresh=cfg["det_thresh"],
            text_det_box_thresh=cfg["det_box_thresh"],
            text_det_unclip_ratio=cfg["det_unclip_ratio"],
            text_rec_score_thresh=0.0,
        )

    def predict(self, img_rgb: np.ndarray):
        return _materialize_prediction(self.engine.predict(img_rgb))


def _materialize_prediction(result):
    if isinstance(result, (list, tuple, dict)):
        return result
    try:
        return list(result)
    except TypeError:
        return result


def _install_modelscope_stub() -> None:
    """Avoid ModelScope importing torch during PaddleOCR import."""
    import sys
    import types
    if "modelscope" in sys.modules:
        return
    stub = types.ModuleType("modelscope")
    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("ModelScope download is disabled.")
    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub


def parse_text_detection_output(det_output) -> tuple[list[np.ndarray], list[float]]:
    """Parse TextDetection output into (boxes, det_scores).

    Supports various output field names:
    - dt_polys / rec_polys / boxes
    - dt_scores / scores

    Args:
        det_output: Raw output from TextDetection.predict()

    Returns:
        (boxes, det_scores) where boxes is list of np.ndarray [4, 2]
    """
    if det_output is None:
        return [], []

    # Handle list-wrapped output
    data = det_output
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return [], []
        # If it's a list of dicts, take the first
        if isinstance(data[0], dict):
            data = data[0]
        # If it's a list of results, take the first
        elif hasattr(data[0], '__getitem__') and not isinstance(data[0], np.ndarray):
            data = data[0]

    if hasattr(data, "json"):
        try:
            data = data.json
        except Exception:
            try:
                data = data.json()
            except Exception:
                pass

    if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
        data = data["res"]

    # Extract polygons
    polys = None
    for key in ("dt_polys", "rec_polys", "boxes", "polys"):
        if isinstance(data, dict) and key in data:
            polys = data[key]
            break
        elif hasattr(data, key):
            polys = getattr(data, key)
            break

    if polys is None:
        # Try treating data itself as list of polygons
        if isinstance(data, (list, tuple)) and len(data) > 0:
            first = data[0]
            if isinstance(first, (list, np.ndarray)) and np.array(first).ndim >= 2:
                polys = data
            else:
                return [], []
        else:
            return [], []

    # Extract scores
    scores = None
    for key in ("dt_scores", "scores"):
        if isinstance(data, dict) and key in data:
            scores = data[key]
            break
        elif hasattr(data, key):
            scores = getattr(data, key)
            break

    # Convert to standard format
    boxes = []
    det_scores = []

    for i, poly in enumerate(polys):
        poly_np = np.array(poly, dtype=np.float32)
        if poly_np.ndim == 1:
            # Flat [x1,y1,x2,y2,...] → reshape
            if len(poly_np) == 8:
                poly_np = poly_np.reshape(4, 2)
            elif len(poly_np) == 4:
                # bbox format [x1, y1, x2, y2]
                x1, y1, x2, y2 = poly_np
                poly_np = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            else:
                continue
        elif poly_np.ndim == 2:
            if poly_np.shape[0] != 4 or poly_np.shape[1] != 2:
                # Polygon with more than 4 points → take bounding rect
                if poly_np.shape[1] == 2 and poly_np.shape[0] > 4:
                    x1, y1 = poly_np.min(axis=0)
                    x2, y2 = poly_np.max(axis=0)
                    poly_np = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                else:
                    continue

        boxes.append(poly_np)

        if scores is not None and i < len(scores):
            det_scores.append(float(scores[i]))
        else:
            det_scores.append(1.0)

    return boxes, det_scores


def _valid_box(box: np.ndarray, det_score: float, cfg: dict) -> bool:
    """Check if a detected box passes filtering criteria."""
    if det_score < cfg.get("drop_low_det_score_below", 0.0):
        return False

    xs = box[:, 0]
    ys = box[:, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())

    if width < cfg.get("min_box_width", 10):
        return False
    if height < cfg.get("min_box_height", 8):
        return False

    aspect = min(width, height) / max(width, height) if max(width, height) > 0 else 0
    if aspect < cfg.get("min_box_aspect_ratio", 0.25):
        return False

    return True


def _box_to_bbox_xyxy(box: np.ndarray) -> list[int]:
    """Convert polygon [4,2] to bbox [x1, y1, x2, y2]."""
    xs = box[:, 0]
    ys = box[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def detect_lines(detector, img_rgb: np.ndarray, cfg: dict) -> tuple[list[dict], float]:
    """Run text detection and return filtered line items.

    Args:
        detector: TextDetection model
        img_rgb: Input image as RGB numpy array
        cfg: Configuration dict

    Returns:
        (line_items, det_time_sec)
        Each line_item has: line_id, det_idx, box, bbox_xyxy, det_score
    """
    t0 = time.time()

    # TextDetection expects file path or numpy array
    try:
        det_output = detector.predict(img_rgb)
    except OSError as exc:
        if _looks_like_gpu_runtime_error(exc) and _get_paddle_device(cfg) != "cpu":
            logger.warning(f"GPU detector failed ({exc}); retrying detection on CPU.")
            cpu_cfg = dict(cfg)
            cpu_cfg["paddle_device"] = "cpu"
            detector = create_detector(cpu_cfg)
            det_output = detector.predict(img_rgb)
        else:
            raise
    det_time = time.time() - t0

    boxes, det_scores = parse_text_detection_output(det_output)

    logger.info(f"Detection found {len(boxes)} raw boxes in {det_time:.3f}s")

    line_items = []
    line_id = 0
    for det_idx, (box, det_score) in enumerate(zip(boxes, det_scores)):
        if not _valid_box(box, det_score, cfg):
            continue

        line_items.append({
            "line_id": line_id,
            "det_idx": det_idx,
            "box": box.tolist(),
            "bbox_xyxy": _box_to_bbox_xyxy(box),
            "det_score": det_score,
        })
        line_id += 1

    logger.info(f"After filtering: {len(line_items)} valid lines (dropped {len(boxes) - len(line_items)})")
    return line_items, det_time


def _looks_like_gpu_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cudnn" in message or "cuda" in message or "gpu" in message
