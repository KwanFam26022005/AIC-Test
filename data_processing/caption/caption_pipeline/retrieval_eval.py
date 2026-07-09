"""Offline lexical retrieval evaluator for Phase 5."""

from __future__ import annotations

import math
from typing import Any

from .retrieval_export import normalized_text, tokenize
from .text_utils import truncate


ROUTE_FIELD_BONUS = {
    "ocr": {"ocr_text": 1.8, "all_text": 1.1},
    "audio": {"audio_text": 1.8, "temporal_caption": 1.2, "trake_text": 1.1},
    "object_visual": {"object_text": 1.7, "caption_text": 1.2, "scene_text": 1.1},
    "trake": {
        "trake_text": 2.0,
        "current_observation": 1.5,
        "event_caption": 1.4,
        "before_context": 1.1,
        "after_context": 1.1,
    },
    "general": {},
}

ROUTE_UNIT_BONUS = {
    "ocr": {"frame": 0.4, "shot": 0.2},
    "audio": {"shot": 0.3, "event_step": 0.2},
    "object_visual": {"frame": 0.3, "shot": 0.2},
    "trake": {"event_step": 0.7, "shot": 0.3},
    "general": {},
}


def run_local_retrieval(
    corpus: list[dict],
    queries: list[dict],
    top_k: int,
) -> list[dict]:
    """Score all queries against the local corpus and return ranked hits."""
    results: list[dict] = []
    for query in queries:
        scored = _score_query(corpus, query)
        for rank, item in enumerate(scored[:top_k], start=1):
            result = {
                "query_id": query.get("query_id", ""),
                "query": query.get("query", ""),
                "route": query.get("route", "general"),
                "rank": rank,
                "score": round(item["score"], 6),
                "document_id": item["doc"].get("document_id", ""),
                "unit_type": item["doc"].get("unit_type", ""),
                "video_id": item["doc"].get("video_id", ""),
                "unit_id": item["doc"].get("unit_id", ""),
                "frame_id": item["doc"].get("frame_id", ""),
                "shot_id": item["doc"].get("shot_id", ""),
                "event_id": item["doc"].get("event_id", ""),
                "start_sec": item["doc"].get("start_sec", 0.0),
                "end_sec": item["doc"].get("end_sec", 0.0),
                "matched_fields": item["matched_fields"],
                "matched_terms": item["matched_terms"],
                "snippet": _make_snippet(item["doc"], item["matched_terms"]),
            }
            if "step_order" in item["doc"]:
                result["step_order"] = item["doc"].get("step_order")
            results.append(result)
    return results


def _score_query(corpus: list[dict], query: dict) -> list[dict]:
    query_text = query.get("query", "") or ""
    query_terms = tokenize(query_text)
    route = query.get("route", "general") or "general"
    target_units = set(query.get("target_unit_types") or [])
    scored: list[dict[str, Any]] = []

    for doc in corpus:
        unit_type = doc.get("unit_type", "")
        if target_units and unit_type not in target_units:
            continue
        score, matched_fields, matched_terms = _score_doc(doc, query_terms, query_text, route)
        if score <= 0:
            continue
        scored.append({
            "doc": doc,
            "score": score,
            "matched_fields": matched_fields,
            "matched_terms": matched_terms,
        })

    scored.sort(
        key=lambda item: (
            -item["score"],
            item["doc"].get("start_sec", 0.0),
            item["doc"].get("document_id", ""),
        )
    )
    return scored


def _score_doc(
    doc: dict,
    query_terms: list[str],
    query_text: str,
    route: str,
) -> tuple[float, list[str], list[str]]:
    fields = doc.get("fields") or {}
    boosts = dict(doc.get("boosts") or {})
    for field, bonus in ROUTE_FIELD_BONUS.get(route, {}).items():
        boosts[field] = boosts.get(field, 1.0) * bonus

    matched_fields: list[str] = []
    matched_terms: set[str] = set()
    score = 0.0
    for field, text in fields.items():
        if not text:
            continue
        field_terms = tokenize(text)
        if not field_terms:
            continue
        field_term_set = set(field_terms)
        hits = [term for term in query_terms if term in field_term_set]
        if not hits:
            continue
        weight = boosts.get(field, 1.0)
        tf_score = sum(1.0 + math.log(1.0 + field_terms.count(term)) for term in hits)
        coverage = len(hits) / max(len(query_terms), 1)
        score += weight * (tf_score + coverage)
        matched_fields.append(field)
        matched_terms.update(hits)

    phrase = normalized_text(query_text)
    search_text = normalized_text(doc.get("search_text", ""))
    if phrase and phrase in search_text:
        score += 3.0

    unit_bonus = ROUTE_UNIT_BONUS.get(route, {}).get(doc.get("unit_type", ""), 0.0)
    score += unit_bonus
    if doc.get("unit_type") == "event_step" and route == "trake":
        score += _temporal_role_bonus(doc, query_terms)

    return score, matched_fields, sorted(matched_terms)


def _temporal_role_bonus(doc: dict, query_terms: list[str]) -> float:
    action_state = doc.get("action_state", "")
    temporal_role = doc.get("temporal_role", "")
    terms = set(query_terms)
    bonus = 0.0
    if {"dau", "tien"} & terms and (action_state == "start" or temporal_role == "beginning"):
        bonus += 1.0
    if {"cuoi", "cung"} & terms and (action_state == "end" or temporal_role == "completion"):
        bonus += 1.0
    if "sau" in terms and temporal_role in {"continuation", "change"}:
        bonus += 0.5
    return bonus


def _make_snippet(doc: dict, matched_terms: list[str]) -> str:
    fields = doc.get("fields") or {}
    preferred = [
        "trake_text",
        "current_observation",
        "caption_text",
        "temporal_caption",
        "ocr_text",
        "audio_text",
        "object_text",
        "all_text",
    ]
    for field in preferred:
        text = fields.get(field, "") or ""
        if not text:
            continue
        normalized = normalized_text(text)
        if any(term in normalized for term in matched_terms):
            return truncate(text, 280)
    return truncate(doc.get("search_text", ""), 280)
