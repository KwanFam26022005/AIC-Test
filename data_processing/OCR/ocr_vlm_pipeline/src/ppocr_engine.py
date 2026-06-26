from __future__ import annotations


def _install_modelscope_stub() -> None:
    """Avoid ModelScope importing torch during PaddleOCR import on Colab.

    PaddleX imports `modelscope` at module import time even when the selected
    model source is HuggingFace/BOS. On Colab this can crash before PaddleOCR
    starts if the runtime has a torch/NCCL mismatch. The OCR stage does not need
    ModelScope, so a tiny stub is enough and lets PaddleX use other sources.
    """
    import sys
    import types

    if "modelscope" in sys.modules:
        return

    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError(
            "ModelScope download is disabled in this OCR runtime. "
            "Use PADDLE_PDX_MODEL_SOURCE=huggingface or bos."
        )

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub


def create_ppocr_engine(cfg):
    import os

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", cfg.ppocr.get("model_source", "huggingface"))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if cfg.ppocr.get("stub_modelscope", True):
        _install_modelscope_stub()

    from paddleocr import PaddleOCR

    kwargs = {
        "lang": cfg.ppocr.lang,
        "use_doc_orientation_classify": bool(cfg.ppocr.use_doc_orientation_classify),
        "use_doc_unwarping": bool(cfg.ppocr.use_doc_unwarping),
        "use_textline_orientation": bool(cfg.ppocr.use_textline_orientation),
        "text_det_limit_side_len": int(cfg.ppocr.text_det_limit_side_len),
        "text_det_thresh": float(cfg.ppocr.text_det_thresh),
        "text_det_box_thresh": float(cfg.ppocr.text_det_box_thresh),
        "text_det_unclip_ratio": float(cfg.ppocr.text_det_unclip_ratio),
        "text_rec_score_thresh": float(cfg.ppocr.text_rec_score_thresh),
    }
    return PaddleOCR(**kwargs)


def normalize_ppocr_result(result) -> list[dict]:
    if not result:
        return []
    item = result[0] if isinstance(result, list) and len(result) == 1 else result
    if isinstance(item, dict):
        texts = item.get("rec_texts") or item.get("texts") or []
        scores = item.get("rec_scores") or item.get("scores") or []
        polys = item.get("rec_polys") or item.get("dt_polys") or item.get("polys") or []
        rows = []
        for idx, text in enumerate(texts):
            poly = polys[idx].tolist() if hasattr(polys[idx], "tolist") else polys[idx]
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            rows.append(
                {
                    "line_idx": idx,
                    "poly": [[int(p[0]), int(p[1])] for p in poly],
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "ocr_text": str(text),
                    "confidence": float(scores[idx]) if idx < len(scores) else 0.0,
                }
            )
        return rows
    rows = []
    for idx, line in enumerate(item):
        poly, rec = line
        text, score = rec
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        rows.append({"line_idx": idx, "poly": poly, "bbox": [min(xs), min(ys), max(xs), max(ys)], "ocr_text": text, "confidence": float(score)})
    return rows
