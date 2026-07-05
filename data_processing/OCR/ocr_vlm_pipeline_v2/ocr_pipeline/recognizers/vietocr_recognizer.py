"""
VietOCR recognition wrapper with batch support.

Handles:
- Creating the VietOCR predictor
- Robust confidence extraction from multiple output formats
- Batch CNN encoding + sequential decode for throughput
- Greedy / beamsearch selection
"""
from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def create_vietocr_predictor(cfg: dict):
    """Create a VietOCR predictor.

    Args:
        cfg: Config dict with vietocr_config, vietocr_device, vietocr_beamsearch

    Returns:
        VietOCR Predictor object
    """
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
    vietocr_cfg["predictor"]["beamsearch"] = bool(
        cfg.get("vietocr_beamsearch", False)  # Default: greedy (faster)
    )

    predictor = Predictor(vietocr_cfg)
    logger.info(
        f"VietOCR predictor created: config={config_name}, device={device}, "
        f"beamsearch={vietocr_cfg['predictor']['beamsearch']}"
    )
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


# ── Batch inference ──────────────────────────────────────────────────


def _load_crop_pil(crop_path: str) -> Image.Image | None:
    """Load a crop image as PIL RGB. Returns None on failure."""
    try:
        img = Image.open(crop_path).convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"Cannot load crop {crop_path}: {e}")
        return None


def _preprocess_for_vietocr(
    pil_img: Image.Image,
    predictor,
) -> torch.Tensor:
    """Resize + normalize a single PIL image using VietOCR's internal pipeline.

    Returns a tensor of shape [1, C, H, W].
    """
    # Use predictor's internal preprocessing (process_input / resize)
    try:
        # VietOCR Predictor stores a translate function & image processing
        from vietocr.tool.translate import process_input
        img_tensor = process_input(pil_img, predictor.config["dataset"]["image_height"],
                                   predictor.config["dataset"]["image_min_width"],
                                   predictor.config["dataset"]["image_max_width"],
                                   )
    except (ImportError, AttributeError, KeyError):
        # Fallback: manual resize
        h = predictor.config.get("dataset", {}).get("image_height", 32)
        w_orig, h_orig = pil_img.size
        ratio = h / h_orig
        new_w = max(int(w_orig * ratio), 32)
        pil_img = pil_img.resize((new_w, h), Image.BILINEAR)
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        img_tensor = transform(pil_img).unsqueeze(0)

    return img_tensor


def _batch_encode_cnn(
    tensors: list[torch.Tensor],
    model,
    device: str,
    batch_size: int = 16,
) -> list[torch.Tensor]:
    """Batch-encode images through VietOCR's CNN encoder.

    Groups images into batches, pads to max width within batch,
    and runs CNN encoder in batch. Returns list of encoded memories.

    Args:
        tensors: List of [1, C, H, W] tensors (varying W)
        model: VietOCR model (has .cnn and .transformer)
        device: torch device string
        batch_size: Number of images per batch

    Returns:
        List of (memory, src) tuples for each image
    """
    results = []

    for start in range(0, len(tensors), batch_size):
        batch_tensors = tensors[start:start + batch_size]
        if not batch_tensors:
            continue

        # Pad to max width in this batch
        max_w = max(t.shape[3] for t in batch_tensors)
        padded = []
        for t in batch_tensors:
            if t.shape[3] < max_w:
                pad = torch.zeros(1, t.shape[1], t.shape[2], max_w - t.shape[3],
                                  dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad], dim=3)
            padded.append(t)

        batch = torch.cat(padded, dim=0).to(device)  # [N, C, H, W]

        with torch.no_grad():
            src = model.cnn(batch)
            memories = model.transformer.forward_encoder(src)

        # Extract per-image memory
        for i in range(len(batch_tensors)):
            try:
                memory = model.transformer.get_memory(memories, i)
            except (AttributeError, TypeError):
                # Fallback: slice directly from memories
                memory = memories[:, i:i+1, :]
            results.append(memory)

    return results


def _greedy_decode_from_memory(
    memory: torch.Tensor,
    model,
    device: str,
    max_seq_length: int = 128,
    sos_token: int = 1,
    eos_token: int = 2,
) -> tuple[list[int], float]:
    """Greedy decode a single memory into token IDs + avg log-prob.

    Args:
        memory: Encoded memory from CNN+encoder [T, 1, E]
        model: VietOCR model
        device: torch device
        max_seq_length: Maximum decoding length
        sos_token, eos_token: Special token IDs

    Returns:
        (token_ids, avg_log_prob)
    """
    from torch.nn.functional import log_softmax

    translated = [sos_token]
    total_log_prob = 0.0
    num_tokens = 0

    with torch.no_grad():
        for _ in range(max_seq_length):
            tgt = torch.LongTensor(translated).unsqueeze(1).to(device)  # [T_out, 1]
            output, _ = model.transformer.forward_decoder(tgt, memory)
            logits = output[-1, 0, :]  # last timestep
            log_probs = log_softmax(logits, dim=-1)
            next_token = log_probs.argmax().item()
            total_log_prob += log_probs[next_token].item()
            num_tokens += 1

            if next_token == eos_token:
                break
            translated.append(next_token)

    avg_log_prob = total_log_prob / max(num_tokens, 1)
    # Convert log-prob to prob for compatibility
    prob = float(np.exp(avg_log_prob))

    return translated[1:], prob  # skip SOS


def _is_batch_decode_suspicious(
    text: str,
    prob: float | None,
    cfg: dict,
    max_seq_length: int,
) -> bool:
    """Detect outputs from the custom batch decoder that should be rechecked.

    The custom fast path is experimental. When it misses EOS it tends to emit
    long low-diversity strings such as "IIII..." or "DpDPP...". Re-run those
    crops through VietOCR's official Predictor API before scoring/indexing.
    """
    from ..structural_filter import classify_noise_text

    stripped = (text or "").strip()
    if not stripped:
        return False

    if classify_noise_text(stripped, cfg):
        return True

    max_len_ratio = float(cfg.get("vietocr_batch_max_output_len_ratio", 0.75))
    if len(stripped) >= int(max_seq_length * max_len_ratio):
        return True

    min_len = int(cfg.get("vietocr_batch_fallback_min_text_len", 12))
    prob_threshold = float(cfg.get("vietocr_batch_fallback_prob_below", 0.35))
    if prob is not None and prob < prob_threshold and len(stripped) >= min_len:
        return True

    return False


def recognize_all_lines_batch(
    predictor,
    line_items: list[dict],
    cfg: dict,
) -> float:
    """Batch VietOCR recognition: batch CNN encode → sequential decode.

    This is significantly faster than per-crop predict() because:
    1. CNN forward pass is batched (GPU parallelism)
    2. torch.no_grad() eliminates autograd overhead
    3. All images are pre-loaded before inference starts

    Args:
        predictor: VietOCR Predictor
        line_items: List of line_item dicts with persp_crop_path
        cfg: Config dict

    Returns:
        Total recognition time in seconds
    """
    batch_size = cfg.get("vietocr_batch_size", 16)
    max_seq = cfg.get("vietocr_max_seq_length", 128)
    use_batch = cfg.get("vietocr_use_batch", False)

    if not use_batch:
        return recognize_all_lines(predictor, line_items, cfg)

    model = predictor.model
    vocab = predictor.vocab
    device = predictor.device

    # ── Phase 1: Load all crops ──────────────────────────────────────
    t_load_start = time.time()
    valid_indices = []
    pil_images = []

    for i, line in enumerate(line_items):
        crop_path = line.get("persp_crop_path")
        if not crop_path:
            line["vietocr_text"] = ""
            line["rec_conf"] = None
            line["rec_conf_source"] = "no_crop"
            continue

        pil_img = _load_crop_pil(crop_path)
        if pil_img is None:
            line["vietocr_text"] = ""
            line["rec_conf"] = None
            line["rec_conf_source"] = "load_error"
            continue

        valid_indices.append(i)
        pil_images.append(pil_img)
    pil_by_line_idx = dict(zip(valid_indices, pil_images))

    t_load = time.time() - t_load_start

    if not valid_indices:
        logger.info("VietOCR batch: no valid crops to recognize")
        return 0.0

    # ── Phase 2: Preprocess all images ───────────────────────────────
    t_prep_start = time.time()
    tensors = []
    for pil_img in pil_images:
        try:
            t = _preprocess_for_vietocr(pil_img, predictor)
            tensors.append(t)
        except Exception as e:
            logger.warning(f"Preprocess failed: {e}")
            tensors.append(None)
    t_prep = time.time() - t_prep_start

    # ── Phase 3: Batch CNN encode ────────────────────────────────────
    t_encode_start = time.time()
    valid_tensors = []
    valid_tensor_indices = []
    for idx, t in zip(valid_indices, tensors):
        if t is not None:
            valid_tensors.append(t)
            valid_tensor_indices.append(idx)

    memories = _batch_encode_cnn(valid_tensors, model, device, batch_size)
    t_encode = time.time() - t_encode_start

    # ── Phase 4: Sequential decode each memory ──────────────────────
    t_decode_start = time.time()
    fallback_count = 0
    sos_token = 1
    eos_token = 2

    # Try to get actual token IDs from vocab
    try:
        if hasattr(vocab, 'SOS_TOKEN'):
            sos_token = vocab.SOS_TOKEN
        if hasattr(vocab, 'EOS_TOKEN'):
            eos_token = vocab.EOS_TOKEN
    except Exception:
        pass

    for mem_idx, line_idx in enumerate(valid_tensor_indices):
        memory = memories[mem_idx]

        token_ids, prob = _greedy_decode_from_memory(
            memory, model, device, max_seq, sos_token, eos_token
        )

        # Convert token IDs to text
        try:
            text = vocab.decode(token_ids)
        except Exception:
            text = "".join(vocab.lookup_tokens(token_ids))

        source = "batch_greedy"
        conf = prob
        if cfg.get("vietocr_batch_noise_fallback", True) and _is_batch_decode_suspicious(
            text, prob, cfg, max_seq
        ):
            fallback_count += 1
            pil_img = pil_by_line_idx.get(line_idx)
            if pil_img is not None:
                fb_text, fb_conf, fb_source = vietocr_predict_with_conf(
                    predictor,
                    pil_img,
                    cfg.get("use_vietocr_return_prob", True),
                )
                if fb_text:
                    text = fb_text
                    conf = fb_conf
                    source = f"batch_noise_fallback_{fb_source}"
                else:
                    source = "batch_greedy_noise_unfixed"

        line_items[line_idx]["vietocr_text"] = text.strip()
        line_items[line_idx]["rec_conf"] = conf
        line_items[line_idx]["rec_conf_source"] = source

    t_decode = time.time() - t_decode_start

    # Fill remaining invalid entries
    for idx in valid_indices:
        if "vietocr_text" not in line_items[idx]:
            line_items[idx]["vietocr_text"] = ""
            line_items[idx]["rec_conf"] = None
            line_items[idx]["rec_conf_source"] = "batch_error"

    total_time = t_load + t_prep + t_encode + t_decode
    logger.info(
        f"VietOCR batch recognized {len(valid_tensor_indices)} lines in {total_time:.2f}s "
        f"(load={t_load:.2f}s prep={t_prep:.2f}s encode={t_encode:.2f}s "
        f"decode={t_decode:.2f}s batch_size={batch_size} fallback={fallback_count})"
    )
    return total_time


def recognize_all_lines(
    predictor,
    line_items: list[dict],
    cfg: dict,
) -> float:
    """Run VietOCR on all line crops (sequential fallback).

    Modifies line_items in place with:
    - vietocr_text, rec_conf, rec_conf_source

    Args:
        predictor: VietOCR Predictor
        line_items: List of line_item dicts with persp_crop_path
        cfg: Config dict

    Returns:
        Total recognition time in seconds
    """
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
