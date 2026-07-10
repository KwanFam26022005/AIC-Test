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

SCORING_PROFILES = {"default", "route_aware", "rrf", "route_gated_rrf"}

ROUTE_AWARE_FIELD_MULTIPLIER = {
    "ocr": {
        "ocr_text": 2.4,
        "caption_text": 0.65,
        "temporal_caption": 0.65,
        "trake_text": 0.55,
        "audio_text": 0.35,
        "all_text": 0.2,
    },
    "audio": {
        "audio_text": 2.8,
        "ocr_text": 0.45,
        "caption_text": 0.55,
        "temporal_caption": 0.75,
        "trake_text": 0.65,
        "all_text": 0.2,
    },
    "object_visual": {
        "caption_text": 1.8,
        "object_text": 1.6,
        "scene_text": 1.3,
        "ocr_text": 0.45,
        "audio_text": 0.25,
        "trake_text": 0.9,
        "all_text": 0.25,
    },
    "trake": {
        "trake_text": 1.8,
        "current_observation": 1.7,
        "event_caption": 1.5,
        "temporal_caption": 1.1,
        "caption_text": 0.75,
        "ocr_text": 0.45,
        "audio_text": 0.45,
        "all_text": 0.2,
    },
    "general": {
        "caption_text": 1.25,
        "temporal_caption": 1.1,
        "trake_text": 1.1,
        "object_text": 1.1,
        "scene_text": 1.0,
        "ocr_text": 0.9,
        "audio_text": 0.9,
        "all_text": 0.25,
    },
}

ROUTE_AWARE_UNIT_BONUS = {
    "ocr": {"frame": 0.8, "shot": 0.5},
    "audio": {"event_step": 0.9, "shot": 0.5},
    "object_visual": {"frame": 0.8, "shot": 0.6},
    "trake": {"event_step": 1.1, "shot": 0.4},
    "general": {"event_step": 0.2, "shot": 0.2, "frame": 0.1},
}

ROUTE_AWARE_GLOBAL_STOP_TERMS = {
    "a", "an", "and", "are", "at", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "with",
}

ROUTE_AWARE_ROUTE_STOP_TERMS = {
    "audio": {"audio", "content", "spoken", "speech"},
    "object_visual": {
        "frame", "frames", "object", "objects", "representative",
        "scene", "show", "shows", "visible",
    },
    "ocr": {"screen", "text", "on"},
    "trake": {"shot", "scene"},
    "general": {"frame", "frames", "representative", "scene", "show", "shows"},
}

RRF_K = 60
RRF_CHANNEL_DEPTH = 100

RRF_CHANNELS = {
    "visual": {
        "fields": (
            "caption_text",
            "temporal_caption",
            "object_text",
            "scene_text",
            "current_observation",
            "scene",
        ),
        "field_weights": {
            "caption_text": 1.4,
            "temporal_caption": 1.1,
            "object_text": 1.25,
            "scene_text": 1.0,
            "current_observation": 1.15,
            "scene": 0.9,
        },
    },
    "ocr": {
        "fields": ("ocr_text",),
        "field_weights": {"ocr_text": 1.0},
    },
    "audio": {
        "fields": ("audio_text",),
        "field_weights": {"audio_text": 1.0},
    },
    "trake": {
        "fields": (
            "trake_text",
            "event_caption",
            "current_observation",
            "before_context",
            "after_context",
            "temporal_caption",
        ),
        "field_weights": {
            "trake_text": 1.35,
            "event_caption": 1.15,
            "current_observation": 1.15,
            "before_context": 0.7,
            "after_context": 0.7,
            "temporal_caption": 0.8,
        },
    },
    "all_text": {
        "fields": ("all_text",),
        "field_weights": {"all_text": 1.0},
    },
}

RRF_ROUTE_CHANNEL_WEIGHTS = {
    "audio": {
        "audio": 1.45,
        "trake": 0.55,
        "visual": 0.35,
        "ocr": 0.30,
        "all_text": 0.20,
    },
    "ocr": {
        "ocr": 1.50,
        "visual": 0.45,
        "trake": 0.35,
        "audio": 0.20,
        "all_text": 0.20,
    },
    "object_visual": {
        "visual": 1.50,
        "trake": 0.55,
        "ocr": 0.35,
        "audio": 0.15,
        "all_text": 0.20,
    },
    "trake": {
        "trake": 1.50,
        "visual": 0.55,
        "audio": 0.25,
        "ocr": 0.20,
        "all_text": 0.20,
    },
    "general": {
        "visual": 1.00,
        "trake": 0.90,
        "ocr": 0.75,
        "audio": 0.75,
        "all_text": 0.30,
    },
}

# Phase 11 uses hard modality gates. A routed query cannot gain rank from a
# verbose caption copied into an unrelated channel such as audio or OCR.
ROUTE_GATED_RRF_CHANNEL_WEIGHTS = {
    "audio": {"audio": 1.0},
    "ocr": {"ocr": 1.0},
    "object_visual": {"visual": 1.0},
    "trake": {"trake": 1.0},
    "general": {
        "visual": 1.0,
        "trake": 0.85,
        "ocr": 0.65,
        "audio": 0.65,
    },
}

RRF_CHANNEL_UNIT_BONUS = {
    "visual": {"frame": 0.25, "shot": 0.20, "event_step": 0.10},
    "ocr": {"frame": 0.30, "shot": 0.20},
    "audio": {"shot": 0.25, "event_step": 0.20},
    "trake": {"event_step": 0.35, "shot": 0.15},
    "all_text": {},
}


def run_local_retrieval(
    corpus: list[dict],
    queries: list[dict],
    top_k: int,
    scoring_profile: str = "default",
) -> list[dict]:
    """Score all queries against the local corpus and return ranked hits."""
    _validate_scoring_profile(scoring_profile)
    results: list[dict] = []
    for query in queries:
        scored = _score_query(corpus, query, scoring_profile=scoring_profile)
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
            if item.get("ranking_channels"):
                result["ranking_channels"] = item.get("ranking_channels")
            results.append(result)
    return results


def _score_query(
    corpus: list[dict],
    query: dict,
    scoring_profile: str,
) -> list[dict]:
    if scoring_profile in {"rrf", "route_gated_rrf"}:
        return _score_query_rrf(
            corpus,
            query,
            route_gated=scoring_profile == "route_gated_rrf",
        )

    query_text = query.get("query", "") or ""
    route = query.get("route", "general") or "general"
    query_terms = _query_terms_for_profile(tokenize(query_text), route, scoring_profile)
    target_units = set(query.get("target_unit_types") or [])
    scored: list[dict[str, Any]] = []

    for doc in corpus:
        unit_type = doc.get("unit_type", "")
        if target_units and unit_type not in target_units:
            continue
        score, matched_fields, matched_terms = _score_doc(
            doc,
            query_terms,
            query_text,
            route,
            scoring_profile,
        )
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


def _score_query_rrf(
    corpus: list[dict],
    query: dict,
    route_gated: bool = False,
) -> list[dict[str, Any]]:
    query_text = query.get("query", "") or ""
    route = query.get("route", "general") or "general"
    query_terms = _rrf_query_terms(query_text, route, route_gated)
    if not query_terms:
        return []

    target_units = set(query.get("target_unit_types") or [])
    weight_profiles = (
        ROUTE_GATED_RRF_CHANNEL_WEIGHTS
        if route_gated
        else RRF_ROUTE_CHANNEL_WEIGHTS
    )
    channel_weights = weight_profiles.get(route, weight_profiles["general"])
    by_doc: dict[str, dict[str, Any]] = {}

    for channel_name, channel_weight in channel_weights.items():
        if channel_weight <= 0:
            continue
        channel = RRF_CHANNELS[channel_name]
        channel_scored: list[dict[str, Any]] = []
        for doc in corpus:
            unit_type = doc.get("unit_type", "")
            if target_units and unit_type not in target_units:
                continue
            raw_score, matched_fields, matched_terms = _score_doc_channel(
                doc,
                query_terms,
                query_text,
                channel_name,
                channel,
            )
            if raw_score <= 0:
                continue
            channel_scored.append({
                "doc": doc,
                "raw_score": raw_score,
                "matched_fields": matched_fields,
                "matched_terms": matched_terms,
            })

        channel_scored.sort(
            key=lambda item: (
                -item["raw_score"],
                item["doc"].get("start_sec", 0.0),
                item["doc"].get("document_id", ""),
            )
        )

        for rank, item in enumerate(channel_scored[:RRF_CHANNEL_DEPTH], start=1):
            doc = item["doc"]
            document_id = doc.get("document_id", "")
            if not document_id:
                continue
            aggregate = by_doc.setdefault(document_id, {
                "doc": doc,
                "score": 0.0,
                "raw_score": 0.0,
                "matched_fields": set(),
                "matched_terms": set(),
                "ranking_channels": [],
            })
            aggregate["score"] += channel_weight / (RRF_K + rank)
            aggregate["raw_score"] = max(aggregate["raw_score"], item["raw_score"])
            aggregate["matched_fields"].update(item["matched_fields"])
            aggregate["matched_terms"].update(item["matched_terms"])
            aggregate["ranking_channels"].append(f"{channel_name}:{rank}")

    scored: list[dict[str, Any]] = []
    for aggregate in by_doc.values():
        scored.append({
            "doc": aggregate["doc"],
            "score": aggregate["score"],
            "raw_score": aggregate["raw_score"],
            "matched_fields": sorted(aggregate["matched_fields"]),
            "matched_terms": sorted(aggregate["matched_terms"]),
            "ranking_channels": aggregate["ranking_channels"],
        })

    scored.sort(
        key=lambda item: (
            -item["score"],
            -item["raw_score"],
            item["doc"].get("start_sec", 0.0),
            item["doc"].get("document_id", ""),
        )
    )
    return scored


def _score_doc_channel(
    doc: dict,
    query_terms: list[str],
    query_text: str,
    channel_name: str,
    channel: dict[str, Any],
) -> tuple[float, list[str], list[str]]:
    fields = doc.get("fields") or {}
    doc_boosts = doc.get("boosts") or {}
    channel_fields = channel.get("fields") or ()
    field_weights = channel.get("field_weights") or {}
    matched_fields: list[str] = []
    matched_terms: set[str] = set()
    score = 0.0

    for field in channel_fields:
        text = fields.get(field, "") or ""
        if not text:
            continue
        field_terms = tokenize(text)
        if not field_terms:
            continue
        field_term_set = set(field_terms)
        hits = [term for term in query_terms if term in field_term_set]
        if not hits:
            continue
        weight = doc_boosts.get(field, 1.0) * field_weights.get(field, 1.0)
        tf_score = sum(1.0 + math.log(1.0 + field_terms.count(term)) for term in hits)
        coverage = len(hits) / max(len(query_terms), 1)
        score += weight * (tf_score + coverage)
        matched_fields.append(field)
        matched_terms.update(hits)

    phrase = normalized_text(query_text)
    channel_text = normalized_text(" ".join(fields.get(field, "") or "" for field in channel_fields))
    if phrase and phrase in channel_text:
        score += 2.0

    # Unit-type preferences are tie-breakers for lexical matches, not matches
    # by themselves. The old behavior admitted every frame/shot into a channel.
    if score <= 0:
        return 0.0, [], []

    unit_type = doc.get("unit_type", "")
    score += RRF_CHANNEL_UNIT_BONUS.get(channel_name, {}).get(unit_type, 0.0)
    if channel_name == "trake" and unit_type == "event_step":
        score += _temporal_role_bonus(doc, query_terms)

    return score, matched_fields, sorted(matched_terms)


def _rrf_query_terms(query_text: str, route: str, route_gated: bool) -> list[str]:
    terms = tokenize(query_text)
    if not route_gated:
        return terms
    stop_terms = set(ROUTE_AWARE_GLOBAL_STOP_TERMS)
    stop_terms.update(ROUTE_AWARE_ROUTE_STOP_TERMS.get(route, set()))
    filtered = [term for term in terms if term not in stop_terms]
    return filtered or terms

def _score_doc(
    doc: dict,
    query_terms: list[str],
    query_text: str,
    route: str,
    scoring_profile: str,
) -> tuple[float, list[str], list[str]]:
    fields = doc.get("fields") or {}
    boosts = _boosts_for_profile(doc, route, scoring_profile)
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

    unit_bonus = _unit_bonus_for_profile(route, doc.get("unit_type", ""), scoring_profile)
    score += unit_bonus
    if doc.get("unit_type") == "event_step" and route == "trake":
        score += _temporal_role_bonus(doc, query_terms)

    return score, matched_fields, sorted(matched_terms)


def _validate_scoring_profile(scoring_profile: str) -> None:
    if scoring_profile not in SCORING_PROFILES:
        allowed = ", ".join(sorted(SCORING_PROFILES))
        raise ValueError(
            f"Invalid scoring_profile '{scoring_profile}'. Expected one of: {allowed}"
        )


def _query_terms_for_profile(
    query_terms: list[str],
    route: str,
    scoring_profile: str,
) -> list[str]:
    if scoring_profile != "route_aware":
        return query_terms
    stop_terms = set(ROUTE_AWARE_GLOBAL_STOP_TERMS)
    stop_terms.update(ROUTE_AWARE_ROUTE_STOP_TERMS.get(route, set()))
    filtered = [term for term in query_terms if term not in stop_terms]
    return filtered or query_terms


def _boosts_for_profile(
    doc: dict,
    route: str,
    scoring_profile: str,
) -> dict[str, float]:
    boosts = dict(doc.get("boosts") or {})
    if scoring_profile != "route_aware":
        return boosts
    for field, multiplier in ROUTE_AWARE_FIELD_MULTIPLIER.get(route, {}).items():
        boosts[field] = boosts.get(field, 1.0) * multiplier
    return boosts


def _unit_bonus_for_profile(route: str, unit_type: str, scoring_profile: str) -> float:
    if scoring_profile == "route_aware":
        return ROUTE_AWARE_UNIT_BONUS.get(route, {}).get(unit_type, 0.0)
    return ROUTE_UNIT_BONUS.get(route, {}).get(unit_type, 0.0)


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
