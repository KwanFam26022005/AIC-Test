"""
VietOCR recognition wrapper.

Handles:
- Creating the VietOCR predictor
- Robust confidence extraction from multiple output formats
- Detecting flat confidence
"""
from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def create_vietocr_predictor(cfg: dict):
    """Create a VietOCR predictor.

    Args:
        cfg: Config dict with vietocr_config, vietocr_device, vietocr_beamsearch

    Returns:
        VietOCR Predictor object
    """
    import torch
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

    config_name = cfg.get("vietocr_config", "vgg_transformer")
    vietocr_cfg = Cfg.load_config_from_name(config_name)

    # Device
    device = cfg.get("vietocr_device")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vietocr_cfg["device"] = device

    # Beamsearch
    if cfg.get("vietocr_beamsearch", True):
        vietocr_cfg["predictor"]["beamsearch"] = True
    else:
        vietocr_cfg["predictor"]["beamsearch"] = False

    predictor = Predictor(vietocr_cfg)
    logger.info(f"VietOCR predictor created: config={config_name}, device={device}")
    return predictor


def vietocr_predict_with_conf(
    predictor,
    crop: np.ndarray | Image.Image,
    use_return_prob: bool = True,
) -> tuple[str, float | None, str]:
    """Run VietOCR prediction with robust confidence extraction.

    Handles multiple output formats:
    - (text, prob)          → return_prob_valid
    - [text, prob]          → return_prob_list_valid
    - text only             → return_prob_no_conf
    - missing confidence    → return_prob_missing
    - exception/type error  → fallback predict(crop)

    Args:
        predictor: VietOCR Predictor
        crop: Input image (numpy RGB or PIL Image)
        use_return_prob: Whether to try return_prob=True

    Returns:
        (text, rec_conf, rec_conf_source)
        rec_conf is None if confidence could not be extracted
    """
    # Convert numpy to PIL if needed
    if isinstance(crop, np.ndarray):
        pil_img = Image.fromarray(crop)
    else:
        pil_img = crop

    text = ""
    rec_conf = None
    source = "unknown"

    if use_return_prob:
        try:
            out = predictor.predict(pil_img, return_prob=True)

            if isinstance(out, tuple) and len(out) == 2:
                text, prob = out
                if isinstance(prob, (float, int)):
                    rec_conf = float(prob)
                    source = "return_prob_valid"
                elif hasattr(prob, 'item'):
                    rec_conf = float(prob.item())
                    source = "return_prob_valid"
                else:
                    text = str(out[0])
                    rec_conf = None
                    source = "return_prob_no_conf"

            elif isinstance(out, list) and len(out) == 2:
                text = str(out[0])
                try:
                    rec_conf = float(out[1])
                    source = "return_prob_list_valid"
                except (TypeError, ValueError):
                    rec_conf = None
                    source = "return_prob_no_conf"

            elif isinstance(out, str):
                text = out
                rec_conf = None
                source = "return_prob_no_conf"

            else:
                text = str(out)
                rec_conf = None
                source = "return_prob_missing"

        except (TypeError, AttributeError) as e:
            # return_prob not supported, fallback
            logger.debug(f"return_prob failed ({e}), falling back to predict()")
            try:
                text = predictor.predict(pil_img)
                if not isinstance(text, str):
                    text = str(text)
                rec_conf = None
                source = "return_prob_missing"
            except Exception as e2:
                logger.warning(f"VietOCR predict fallback also failed: {e2}")
                text = ""
                rec_conf = None
                source = "error"
    else:
        try:
            text = predictor.predict(pil_img)
            if not isinstance(text, str):
                text = str(text)
            rec_conf = None
            source = "no_return_prob"
        except Exception as e:
            logger.warning(f"VietOCR predict failed: {e}")
            text = ""
            rec_conf = None
            source = "error"

    return text.strip(), rec_conf, source


def recognize_all_lines(
    predictor,
    line_items: list[dict],
    cfg: dict,
) -> float:
    """Run VietOCR on all line crops.

    Modifies line_items in place with:
    - vietocr_text, rec_conf, rec_conf_source

    Args:
        predictor: VietOCR Predictor
        line_items: List of line_item dicts with persp_crop_path
        cfg: Config dict

    Returns:
        Total recognition time in seconds
    """
    import cv2

    use_prob = cfg.get("use_vietocr_return_prob", True)
    total_time = 0.0

    for i, line in enumerate(line_items):
        crop_path = line.get("persp_crop_path")
        if not crop_path:
            line["vietocr_text"] = ""
            line["rec_conf"] = None
            line["rec_conf_source"] = "no_crop"
            continue

        # Load crop
        try:
            crop_bgr = cv2.imread(crop_path)
            if crop_bgr is None:
                raise FileNotFoundError(f"Cannot read: {crop_path}")
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Line {line.get('line_id')}: cannot load crop {crop_path}: {e}")
            line["vietocr_text"] = ""
            line["rec_conf"] = None
            line["rec_conf_source"] = "load_error"
            continue

        t0 = time.time()
        text, conf, source = vietocr_predict_with_conf(predictor, crop_rgb, use_prob)
        total_time += time.time() - t0

        line["vietocr_text"] = text
        line["rec_conf"] = conf
        line["rec_conf_source"] = source

    logger.info(f"VietOCR recognized {len(line_items)} lines in {total_time:.2f}s "
                f"({total_time / max(1, len(line_items)):.3f}s/line)")
    return total_time
