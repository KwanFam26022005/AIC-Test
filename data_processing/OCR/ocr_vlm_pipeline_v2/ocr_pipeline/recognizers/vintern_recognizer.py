"""
Vintern VLM recognizer for line and group fallback OCR.

Handles:
- Loading Vintern-1B-v3_5 model and tokenizer
- dynamic_preprocess for tile-based input
- Line-level OCR fallback with agreement checking
- Group-level OCR fallback
"""
from __future__ import annotations

import contextlib
import difflib
import logging
import math
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

logger = logging.getLogger(__name__)

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is listed in requirements.
    tqdm = None


@contextlib.contextmanager
def _quiet_transformers_generation():
    """Temporarily silence repetitive Transformers generation warnings."""
    old_verbosity = None
    try:
        from transformers.utils import logging as hf_logging

        old_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    except Exception:
        hf_logging = None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*")
            warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
            warnings.filterwarnings("ignore", message=".*Starting from v4\\.46.*")
            yield
    finally:
        if old_verbosity is not None and hf_logging is not None:
            hf_logging.set_verbosity(old_verbosity)


def _progress(iterable, *, total: int, desc: str, enabled: bool):
    """Return a tqdm progress bar when available and enabled."""
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="crop", dynamic_ncols=True)


def _has_flash_attn() -> bool:
    """Check if flash_attn is installed."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _install_flash_attn_stub() -> None:
    """Install flash_attn stub if not available."""
    import importlib.machinery
    import sys
    import types

    if "flash_attn" in sys.modules:
        return

    def _unavailable(*args, **kwargs):
        raise RuntimeError("flash_attn is not installed; set attn_implementation='eager'.")

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn.flash_attn_func = _unavailable
    flash_attn.flash_attn_varlen_func = _unavailable

    flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
    flash_attn_interface.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", loader=None)
    flash_attn_interface.flash_attn_func = _unavailable
    flash_attn_interface.flash_attn_varlen_func = _unavailable

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding.index_first_axis = _unavailable
    bert_padding.pad_input = _unavailable
    bert_padding.unpad_input = _unavailable

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface
    sys.modules["flash_attn.bert_padding"] = bert_padding


def load_vintern_model(cfg: dict):
    """Load Vintern model and tokenizer.

    Args:
        cfg: Config dict

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    model_id = cfg.get("vintern_model_id", "5CD-AI/Vintern-1B-v3_5")

    # Quantization
    quant = cfg.get("vintern_quantization", "4bit_nf4")
    quant_config = None
    if quant == "4bit_nf4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Attention implementation
    attn_impl = cfg.get("vintern_attn_implementation")
    if attn_impl is None:
        attn_impl = "flash_attention_2" if _has_flash_attn() else "eager"
    if attn_impl == "eager" and not _has_flash_attn():
        _install_flash_attn_stub()

    # Device map
    device = cfg.get("vintern_device")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_map = {"": device}

    logger.info(f"Loading Vintern: {model_id}, device={device}, attn={attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    ).eval()

    logger.info(f"Vintern loaded successfully: {model_id}")
    return model, tokenizer


def _build_transform(input_size: int):
    """Build the image transform for Vintern tiles."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    input_size: int = 448,
    max_tiles: int = 6,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Preprocess image into tiles for Vintern.

    Steps:
    1. Choose tile layout based on aspect ratio
    2. Resize and split into tiles of input_size × input_size
    3. Optionally add a thumbnail tile

    Args:
        image: PIL Image
        input_size: Tile size (448 for Vintern)
        max_tiles: Maximum number of tiles
        use_thumbnail: Whether to add a downscaled full image tile

    Returns:
        List of PIL Image tiles
    """
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h

    # Generate target ratios
    target_ratios = set()
    for n in range(1, max_tiles + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_tiles:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best ratio
    best_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, input_size
    )

    target_w = best_ratio[0] * input_size
    target_h = best_ratio[1] * input_size
    num_blocks = best_ratio[0] * best_ratio[1]

    # Resize image
    resized = image.resize((target_w, target_h))

    # Split into tiles
    tiles = []
    for i in range(best_ratio[1]):  # rows
        for j in range(best_ratio[0]):  # cols
            box = (
                j * input_size,
                i * input_size,
                (j + 1) * input_size,
                (i + 1) * input_size,
            )
            tiles.append(resized.crop(box))

    # Add thumbnail
    if use_thumbnail and len(tiles) != 1:
        thumbnail = image.resize((input_size, input_size))
        tiles.append(thumbnail)

    return tiles


def _preprocess_for_vintern(
    image_path: str | Path,
    input_size: int = 448,
    max_tiles: int = 6,
) -> torch.Tensor:
    """Load and preprocess an image for Vintern inference.

    Args:
        image_path: Path to image
        input_size: Tile size
        max_tiles: Max number of tiles

    Returns:
        Tensor of shape [num_tiles, 3, input_size, input_size]
    """
    image = Image.open(image_path).convert("RGB")
    tiles = dynamic_preprocess(image, input_size, max_tiles)
    transform = _build_transform(input_size)
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values


def vintern_ocr_crop(
    model,
    tokenizer,
    crop_path: str | Path,
    prompt: str,
    cfg: dict,
) -> str:
    """Run Vintern OCR on a single crop image.

    Args:
        model: Vintern model
        tokenizer: Vintern tokenizer
        crop_path: Path to crop image
        prompt: OCR prompt
        cfg: Config dict

    Returns:
        Extracted text
    """
    input_size = cfg.get("vintern_input_size", 448)
    max_tiles = cfg.get("vintern_max_tiles", 6)
    max_new_tokens = cfg.get("vintern_max_new_tokens", 256)

    # Preprocess
    pixel_values = _preprocess_for_vintern(crop_path, input_size, max_tiles)

    # Determine device from model
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device=device, dtype=torch.float16)

    # Build generation config
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": cfg.get("vintern_do_sample", False),
    }
    if generation_config["do_sample"]:
        generation_config["temperature"] = cfg.get("vintern_temperature", 0.0)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id
    if eos_token_id is not None:
        generation_config["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        generation_config["pad_token_id"] = pad_token_id

    # Format prompt for Vintern chat
    question = f"<image>\n{prompt}"

    # Use model.chat if available (Vintern-specific API)
    try:
        with _quiet_transformers_generation():
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
            )
        if isinstance(response, tuple):
            response = response[0]
        return response.strip()
    except Exception as e:
        logger.warning(f"Vintern chat failed: {e}")
        return ""


def vintern_ocr_line(
    model,
    tokenizer,
    crop_path: str,
    cfg: dict,
) -> str:
    """OCR a line crop using Vintern line prompt."""
    prompt = cfg.get("vintern_line_prompt",
                     "Hãy đọc chính xác toàn bộ chữ trong ảnh crop này.\n"
                     "Chỉ trả về nội dung OCR, không giải thích.\n"
                     "Giữ nguyên tiếng Việt có dấu nếu có.")
    return vintern_ocr_crop(model, tokenizer, crop_path, prompt, cfg)


def vintern_ocr_group(
    model,
    tokenizer,
    crop_path: str,
    cfg: dict,
) -> str:
    """OCR a group crop using Vintern group prompt."""
    prompt = cfg.get("vintern_group_prompt",
                     "Hãy đọc chính xác toàn bộ chữ trong ảnh crop này theo đúng từng dòng.\n"
                     "Nếu ảnh có nhiều dòng chữ, hãy xuống dòng giữa các dòng.\n"
                     "Chỉ trả về nội dung OCR, không giải thích.")
    return vintern_ocr_crop(model, tokenizer, crop_path, prompt, cfg)


def compute_agreement_similarity(text_a: str, text_b: str) -> float:
    """Compute text similarity between two OCR outputs."""
    a = (text_a or "").strip().lower()
    b = (text_b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def decide_line_final_text(
    line_item: dict,
    vintern_text: str,
    cfg: dict,
    wordset: set[str],
    base_to_variants: dict[str, set[str]],
    wordlist_enabled: bool,
) -> None:
    """Decide final text for a line after Vintern fallback.

    Compares VietOCR and Vintern outputs:
    - High agreement → pick better composite score, need_review=False
    - Disagreement + Vintern better → override, need_review=True
    - Vintern not good enough → keep VietOCR, keep_for_index=False

    Modifies line_item in place.
    """
    from ..scoring.wordlist import extract_tokens, get_eval_tokens, compute_lex_ratio, compute_diacritic_susp
    from ..scoring.features import compute_charset_penalty, compute_repetition_penalty, compute_composite_score

    line_item["vintern_text"] = vintern_text

    vietocr_text = line_item.get("vietocr_text", "")
    old_score = line_item.get("composite_score", 0.0)

    # Compute Vintern composite score
    v_tokens = extract_tokens(vintern_text)
    v_eval = get_eval_tokens(v_tokens)
    v_lex = compute_lex_ratio(v_eval, wordset, wordlist_enabled,
                              cfg.get("neutral_lex_ratio_if_no_wordlist", 0.50))
    v_dia, _ = compute_diacritic_susp(v_eval, base_to_variants, wordset, wordlist_enabled)
    v_charset = compute_charset_penalty(vintern_text)
    v_rep = compute_repetition_penalty(vintern_text, v_tokens)

    # Use det_score from original detection
    det_score = line_item.get("det_score", 0.0)
    # Vintern doesn't have rec_conf, use a default
    v_rec_conf = 0.70

    v_composite, _ = compute_composite_score(
        v_rec_conf, det_score, v_lex, v_dia, v_charset, v_rep, cfg
    )
    line_item["vintern_composite_score"] = v_composite

    # Compute agreement
    similarity = compute_agreement_similarity(vietocr_text, vintern_text)
    line_item["agreement_similarity"] = similarity

    agree_threshold = cfg.get("vintern_agreement_threshold", 0.82)
    min_accept = cfg.get("vintern_min_composite_accept", 0.55)
    margin = cfg.get("vintern_override_margin", 0.05)

    if similarity >= agree_threshold:
        # Agreement: pick better score
        if v_composite >= old_score:
            line_item["final_text"] = vintern_text
            line_item["final_source"] = "vintern_line_agree"
        else:
            line_item["final_text"] = vietocr_text
            line_item["final_source"] = "vietocr_agree"
        line_item["keep_for_index"] = True
        line_item["need_review"] = False

    elif v_composite >= min_accept and v_composite >= old_score + margin:
        # Disagreement but Vintern is better
        line_item["final_text"] = vintern_text
        line_item["final_source"] = "vintern_line_override"
        line_item["keep_for_index"] = True
        line_item["need_review"] = True  # disagreement → still review

    else:
        # Vintern not good enough, keep VietOCR
        line_item["final_text"] = vietocr_text
        line_item["final_source"] = "vietocr_vintern_insufficient"
        line_item["keep_for_index"] = False
        line_item["need_review"] = True


def decide_group_final_text(
    group: dict,
    vintern_text: str,
    line_items: list[dict],
    cfg: dict,
    wordset: set[str],
    base_to_variants: dict[str, set[str]],
    wordlist_enabled: bool,
) -> None:
    """Decide final text for a group after Vintern group fallback.

    Accept conditions (plan section 15.4):
    - group_vintern_composite_score >= 0.55
    - OR soft rule: len >= threshold, lex >= 0.45, charset <= 0.05

    Modifies group in place.
    """
    from ..scoring.wordlist import extract_tokens, get_eval_tokens, compute_lex_ratio, compute_diacritic_susp
    from ..scoring.features import compute_charset_penalty, compute_repetition_penalty, compute_composite_score

    group["group_vintern_text"] = vintern_text

    # Compute Vintern group composite score
    v_tokens = extract_tokens(vintern_text)
    v_eval = get_eval_tokens(v_tokens)
    v_lex = compute_lex_ratio(v_eval, wordset, wordlist_enabled,
                              cfg.get("neutral_lex_ratio_if_no_wordlist", 0.50))
    v_dia, _ = compute_diacritic_susp(v_eval, base_to_variants, wordset, wordlist_enabled)
    v_charset = compute_charset_penalty(vintern_text)
    v_rep = compute_repetition_penalty(vintern_text, v_tokens)
    v_rec_conf = 0.70
    v_det_score = group.get("mean_det_score", 0.0)

    v_composite, _ = compute_composite_score(
        v_rec_conf, v_det_score, v_lex, v_dia, v_charset, v_rep, cfg
    )
    group["group_vintern_composite_score"] = v_composite

    # Agreement with existing review text
    old_review = group.get("group_text_review", "")
    similarity = compute_agreement_similarity(old_review, vintern_text)
    group["group_vintern_agreement_similarity"] = similarity

    # Acceptance logic
    min_composite = cfg.get("group_vintern_min_composite_accept", 0.55)
    min_len_abs = cfg.get("group_vintern_min_text_len_abs", 6)
    min_len_ratio = cfg.get("group_vintern_min_text_len_ratio", 0.6)
    accept_lex = cfg.get("group_vintern_accept_lex_ratio", 0.45)
    accept_charset = cfg.get("group_vintern_accept_max_charset", 0.05)
    agree_threshold = cfg.get("group_vintern_agreement_threshold", 0.82)

    # Hard rule
    hard_accept = v_composite >= min_composite

    # Soft rule
    review_len = len(old_review) if old_review else 0
    soft_accept = (
        len(vintern_text) >= max(min_len_abs, int(review_len * min_len_ratio))
        and v_lex >= accept_lex
        and v_charset <= accept_charset
    )

    if hard_accept or soft_accept:
        group["group_text_clean"] = vintern_text
        group["group_text"] = vintern_text
        group["group_final_source"] = "vintern_group_fallback"
        group["group_keep_for_index"] = True
        group["need_review"] = similarity < agree_threshold
    else:
        # Reject: keep existing
        group["group_final_source"] = "vintern_group_rejected"
        group["group_keep_for_index"] = False
        group["need_review"] = True


def run_vintern_line_fallback(
    model,
    tokenizer,
    line_items: list[dict],
    cfg: dict,
    wordset: set[str],
    base_to_variants: dict[str, set[str]],
    wordlist_enabled: bool,
) -> float:
    """Run Vintern fallback on top priority VLM candidates.

    Args:
        model, tokenizer: Vintern model
        line_items: All line items
        cfg: Config dict
        wordset, base_to_variants, wordlist_enabled: Wordlist data

    Returns:
        Total Vintern line inference time
    """
    if not cfg.get("use_vintern_line_fallback", True):
        return 0.0

    # Select and rank candidates
    candidates = [l for l in line_items if l.get("send_to_vintern", False)]
    candidates.sort(key=lambda x: -x.get("priority", 0))

    max_cand = cfg.get("vintern_max_candidates", 8)
    candidates = candidates[:max_cand]

    if not candidates:
        logger.info("No Vintern line candidates")
        return 0.0

    logger.info(f"Running Vintern line fallback on {len(candidates)} candidates")
    total_time = 0.0
    progress_enabled = cfg.get("vintern_progress", True)
    progress = _progress(
        candidates,
        total=len(candidates),
        desc="VLM lines",
        enabled=progress_enabled,
    )

    for done, line in enumerate(progress, start=1):
        crop_path = line.get("persp_crop_path")
        if not crop_path:
            continue

        t0 = time.time()
        vtext = vintern_ocr_line(model, tokenizer, crop_path, cfg)
        elapsed = time.time() - t0
        total_time += elapsed

        logger.debug(f"  Line {line.get('line_id')}: Vintern={vtext!r} ({elapsed:.2f}s)")

        decide_line_final_text(line, vtext, cfg, wordset, base_to_variants, wordlist_enabled)
        if progress_enabled and tqdm is not None:
            avg_time = total_time / done
            progress.set_postfix_str(
                f"last={elapsed:.1f}s avg={avg_time:.1f}s total={total_time:.1f}s"
            )

    logger.info(f"Vintern line fallback done: {len(candidates)} lines in {total_time:.2f}s")
    return total_time


def run_vintern_group_fallback(
    model,
    tokenizer,
    groups: list[dict],
    line_items: list[dict],
    cfg: dict,
    wordset: set[str],
    base_to_variants: dict[str, set[str]],
    wordlist_enabled: bool,
) -> float:
    """Run Vintern group fallback on priority groups.

    Args:
        model, tokenizer: Vintern model
        groups: All groups
        line_items: All line items
        cfg: Config dict
        wordset, base_to_variants, wordlist_enabled: Wordlist data

    Returns:
        Total Vintern group inference time
    """
    if not cfg.get("use_vintern_group_fallback", True):
        return 0.0

    min_lines = cfg.get("vintern_group_min_lines", 2)
    max_groups = cfg.get("vintern_group_max_candidates", 4)

    # Select groups that need Vintern
    candidates = []
    for group in groups:
        if not group.get("group_crop_path"):
            continue
        if group.get("num_lines", 0) < min_lines:
            continue

        # Check if group has at least 1 suspicious line
        has_suspicious = False
        for idx in group.get("line_indices", []):
            if idx < len(line_items):
                line = line_items[idx]
                if line.get("need_review") or line.get("send_to_vintern"):
                    has_suspicious = True
                    break

        if has_suspicious:
            candidates.append(group)

    # Sort by group_vlm_priority
    candidates.sort(key=lambda g: -g.get("group_vlm_priority", 0))
    candidates = candidates[:max_groups]

    if not candidates:
        logger.info("No Vintern group candidates")
        return 0.0

    logger.info(f"Running Vintern group fallback on {len(candidates)} groups")
    total_time = 0.0
    progress_enabled = cfg.get("vintern_progress", True)
    progress = _progress(
        candidates,
        total=len(candidates),
        desc="VLM groups",
        enabled=progress_enabled,
    )

    for done, group in enumerate(progress, start=1):
        crop_path = group["group_crop_path"]

        t0 = time.time()
        gv_text = vintern_ocr_group(model, tokenizer, crop_path, cfg)
        elapsed = time.time() - t0
        total_time += elapsed

        logger.debug(f"  Group {group.get('group_id')}: Vintern={gv_text!r} ({elapsed:.2f}s)")

        decide_group_final_text(
            group, gv_text, line_items, cfg,
            wordset, base_to_variants, wordlist_enabled
        )
        if progress_enabled and tqdm is not None:
            avg_time = total_time / done
            progress.set_postfix_str(
                f"last={elapsed:.1f}s avg={avg_time:.1f}s total={total_time:.1f}s"
            )

    logger.info(f"Vintern group fallback done: {len(candidates)} groups in {total_time:.2f}s")
    return total_time
