"""Default retrieval query set and route heuristics for Phase 5."""

from __future__ import annotations

from .retrieval_export import normalized_text


def build_default_queries(video_id: str) -> list[dict]:
    """Return a small sanity-check query set for offline retrieval."""
    queries = [
        {
            "query_id": "q_ocr_001",
            "video_id": video_id,
            "query": "on screen text hoi nghi",
            "route": "ocr",
            "target_unit_types": ["frame", "shot"],
            "expected_terms_any": ["hoi", "nghi"],
            "notes": "OCR-oriented query",
        },
        {
            "query_id": "q_ocr_002",
            "video_id": video_id,
            "query": "on screen text thanh pho",
            "route": "ocr",
            "target_unit_types": ["frame", "shot"],
            "expected_terms_any": ["thanh", "pho"],
            "notes": "OCR-oriented query",
        },
        {
            "query_id": "q_obj_001",
            "video_id": video_id,
            "query": "person screen nguoi man hinh",
            "route": "object_visual",
            "target_unit_types": ["frame", "shot"],
            "expected_terms_any": ["person", "screen", "nguoi"],
            "notes": "Object/visual query",
        },
        {
            "query_id": "q_obj_002",
            "video_id": video_id,
            "query": "visible objects person people nguoi",
            "route": "object_visual",
            "target_unit_types": ["frame", "shot"],
            "expected_terms_any": ["person", "people", "nguoi"],
            "notes": "Object/visual query",
        },
        {
            "query_id": "q_audio_001",
            "video_id": video_id,
            "query": "spoken content chuyen doi so",
            "route": "audio",
            "target_unit_types": ["shot", "event_step"],
            "expected_terms_any": ["chuyen", "doi", "so"],
            "notes": "Audio/topic query",
        },
        {
            "query_id": "q_audio_002",
            "video_id": video_id,
            "query": "spoken content quan ly van hanh",
            "route": "audio",
            "target_unit_types": ["shot", "event_step"],
            "expected_terms_any": ["quan", "ly", "van", "hanh"],
            "notes": "Audio/topic query",
        },
        {
            "query_id": "q_shot_001",
            "video_id": video_id,
            "query": "shot representative frames person screen trinh bay",
            "route": "general",
            "target_unit_types": ["shot", "event_step"],
            "expected_terms_any": ["trinh", "bay", "screen", "person"],
            "notes": "Shot-level query",
        },
        {
            "query_id": "q_shot_002",
            "video_id": video_id,
            "query": "scene transition change then continuation",
            "route": "general",
            "target_unit_types": ["shot", "event_step"],
            "expected_terms_any": ["transition", "change", "scene"],
            "notes": "Temporal/context query",
        },
        {
            "query_id": "q_general_001",
            "video_id": video_id,
            "query": "scene shows screen visible objects",
            "route": "general",
            "target_unit_types": ["frame", "shot"],
            "expected_terms_any": ["screen", "hien", "thi"],
            "notes": "General mixed query",
        },
        {
            "query_id": "q_general_002",
            "video_id": video_id,
            "query": "person speaking presenting spoken content",
            "route": "general",
            "target_unit_types": ["frame", "shot", "event_step"],
            "expected_terms_any": ["speaking", "presenting", "person"],
            "notes": "General mixed query",
        },
        {
            "query_id": "q_trake_001",
            "video_id": video_id,
            "query": "start beginning middle continuation trinh bay",
            "route": "trake",
            "target_unit_types": ["event_step", "shot"],
            "expected_terms_any": ["start", "beginning", "middle", "continuation"],
            "notes": "TRAKE/order query",
        },
        {
            "query_id": "q_trake_002",
            "video_id": video_id,
            "query": "end completion transition change",
            "route": "trake",
            "target_unit_types": ["event_step", "shot"],
            "expected_terms_any": ["end", "completion", "transition", "change"],
            "notes": "TRAKE/order query",
        },
    ]
    return queries


def infer_route(query: str) -> str:
    text = normalized_text(query)
    if _contains_any(text, ["chu", "text", "logo", "bang", "tieu de"]):
        return "ocr"
    if _contains_any(text, ["noi", "phat bieu", "am thanh", "giong", "thao luan"]):
        return "audio"
    if _contains_any(text, ["nguoi", "xe", "man hinh", "micro", "microphone"]):
        return "object_visual"
    if _contains_any(text, ["dau tien", "sau do", "cuoi cung", "truoc khi", "tiep theo"]):
        return "trake"
    return "general"


def normalize_queries(rows: list[dict], video_id: str) -> list[dict]:
    """Fill missing query metadata with stable defaults."""
    results: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        query = row.get("query", "") or row.get("text", "") or ""
        route = row.get("route") or infer_route(query)
        item = dict(row)
        item["query_id"] = item.get("query_id") or f"q_{idx:04d}"
        item["video_id"] = item.get("video_id") or video_id
        item["query"] = query
        item["route"] = route
        item["target_unit_types"] = item.get("target_unit_types") or []
        item["expected_terms_any"] = item.get("expected_terms_any") or []
        results.append(item)
    return results


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)

