from __future__ import annotations


def create_ppocr_engine(cfg):
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

