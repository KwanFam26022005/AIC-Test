"""Retrieval corpus and Elasticsearch bulk export helpers for Phase 5."""

from __future__ import annotations

import logging
from typing import Any

from .text_utils import join_texts, normalize_whitespace, remove_accents

logger = logging.getLogger(__name__)

RETRIEVAL_CORPUS_SCHEMA = "caption_retrieval_corpus_v1"

COMPACT_INDEX_NAME = "aic_caption_compact_v1"
EVENT_STEP_INDEX_NAME = "aic_caption_event_step_v1"


def build_retrieval_corpus(
    compact_docs: list[dict],
    event_step_docs: list[dict],
) -> list[dict]:
    """Build a local-search corpus from compact docs and event-step docs."""
    rows: list[dict] = []
    for doc in compact_docs:
        rows.append(_compact_to_corpus_doc(doc))
    for doc in event_step_docs:
        rows.append(_event_step_to_corpus_doc(doc))
    logger.info(
        "Retrieval corpus: %d docs (%d compact + %d event_step)",
        len(rows), len(compact_docs), len(event_step_docs),
    )
    return rows


def build_compact_es_bulk(compact_docs: list[dict]) -> list[dict]:
    """Build Elasticsearch bulk action/doc rows for compact search docs."""
    rows: list[dict] = []
    for doc in compact_docs:
        document_id = _compact_document_id(doc)
        body = dict(doc)
        body["document_id"] = document_id
        rows.append({"index": {"_index": COMPACT_INDEX_NAME, "_id": document_id}})
        rows.append(body)
    return rows


def build_event_step_es_bulk(event_step_docs: list[dict]) -> list[dict]:
    """Build Elasticsearch bulk action/doc rows for event-step docs."""
    rows: list[dict] = []
    for doc in event_step_docs:
        document_id = _event_step_document_id(doc)
        body = dict(doc)
        body["document_id"] = document_id
        rows.append({"index": {"_index": EVENT_STEP_INDEX_NAME, "_id": document_id}})
        rows.append(body)
    return rows


def compact_es_mapping() -> dict[str, Any]:
    """Return a conservative ES mapping for compact frame/shot search docs."""
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "aic_text": {
                        "type": "standard",
                        "stopwords": "_none_",
                    },
                },
            },
        },
        "mappings": {
            "dynamic": True,
            "properties": {
                "schema_version": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "unit_type": {"type": "keyword"},
                "unit_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "frame_id": {"type": "keyword"},
                "timestamp_sec": {"type": "float"},
                "start_sec": {"type": "float"},
                "end_sec": {"type": "float"},
                "caption_text": {"type": "text", "analyzer": "aic_text"},
                "ocr_text": {"type": "text", "analyzer": "aic_text"},
                "audio_text": {"type": "text", "analyzer": "aic_text"},
                "object_text": {"type": "text", "analyzer": "aic_text"},
                "scene_text": {"type": "text", "analyzer": "aic_text"},
                "trake_text": {"type": "text", "analyzer": "aic_text"},
                "all_text": {"type": "text", "analyzer": "aic_text"},
            },
        },
    }


def event_step_es_mapping() -> dict[str, Any]:
    """Return a conservative ES mapping for TRAKE event-step docs."""
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "aic_text": {
                        "type": "standard",
                        "stopwords": "_none_",
                    },
                },
            },
        },
        "mappings": {
            "dynamic": True,
            "properties": {
                "schema_version": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "unit_type": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "event_id": {"type": "keyword"},
                "shot_id": {"type": "keyword"},
                "step_order": {"type": "integer"},
                "start_sec": {"type": "float"},
                "end_sec": {"type": "float"},
                "action_state": {"type": "keyword"},
                "temporal_role": {"type": "keyword"},
                "actors": {"type": "keyword"},
                "actions": {"type": "keyword"},
                "objects_involved": {"type": "keyword"},
                "event_caption": {"type": "text", "analyzer": "aic_text"},
                "current_observation": {"type": "text", "analyzer": "aic_text"},
                "before_context": {"type": "text", "analyzer": "aic_text"},
                "after_context": {"type": "text", "analyzer": "aic_text"},
                "scene": {"type": "text", "analyzer": "aic_text"},
                "trake_text": {"type": "text", "analyzer": "aic_text"},
                "trake_text_search": {"type": "text", "analyzer": "aic_text"},
            },
        },
    }


def _compact_to_corpus_doc(doc: dict) -> dict[str, Any]:
    unit_type = doc.get("unit_type", "")
    unit_id = doc.get("unit_id", "")
    fields = {
        "caption_text": doc.get("caption_text", "") or "",
        "temporal_caption": doc.get("temporal_caption", "") or "",
        "ocr_text": doc.get("ocr_text", "") or "",
        "audio_text": doc.get("audio_text", "") or "",
        "object_text": doc.get("object_text", "") or "",
        "scene_text": doc.get("scene_text", "") or "",
        "trake_text": doc.get("trake_text", "") or "",
        "all_text": doc.get("all_text", "") or "",
    }
    search_text = join_texts(*fields.values())
    return {
        "schema_version": RETRIEVAL_CORPUS_SCHEMA,
        "document_id": _compact_document_id(doc),
        "source": "compact_search_index",
        "unit_type": unit_type,
        "video_id": doc.get("video_id", ""),
        "unit_id": unit_id,
        "frame_id": doc.get("frame_id", "") or (unit_id if unit_type == "frame" else ""),
        "shot_id": unit_id if unit_type == "shot" else "",
        "event_id": "",
        "timestamp_sec": doc.get("timestamp_sec", 0.0),
        "start_sec": doc.get("start_sec", 0.0),
        "end_sec": doc.get("end_sec", 0.0),
        "search_text": search_text,
        "fields": fields,
        "terms": tokenize(search_text),
        "boosts": _boosts_for_unit(unit_type),
    }


def _event_step_to_corpus_doc(doc: dict) -> dict[str, Any]:
    source_text = doc.get("source_text") or {}
    fields = {
        "event_caption": doc.get("event_caption", "") or "",
        "current_observation": doc.get("current_observation", "") or "",
        "before_context": doc.get("before_context", "") or "",
        "after_context": doc.get("after_context", "") or "",
        "scene": doc.get("scene", "") or "",
        "trake_text": doc.get("trake_text", "") or "",
        "caption_text": source_text.get("caption_text", "") or "",
        "temporal_caption": source_text.get("temporal_caption", "") or "",
        "ocr_text": source_text.get("merged_ocr_text", "") or "",
        "audio_text": source_text.get("merged_audio_text", "") or "",
        "object_text": source_text.get("object_text", "") or "",
    }
    search_text = join_texts(*fields.values())
    return {
        "schema_version": RETRIEVAL_CORPUS_SCHEMA,
        "document_id": _event_step_document_id(doc),
        "source": "event_step_index",
        "unit_type": "event_step",
        "video_id": doc.get("video_id", ""),
        "unit_id": doc.get("event_id", ""),
        "frame_id": "",
        "shot_id": doc.get("shot_id", ""),
        "event_id": doc.get("event_id", ""),
        "timestamp_sec": doc.get("start_sec", 0.0),
        "start_sec": doc.get("start_sec", 0.0),
        "end_sec": doc.get("end_sec", 0.0),
        "step_order": doc.get("step_order", 0),
        "action_state": doc.get("action_state", ""),
        "temporal_role": doc.get("temporal_role", ""),
        "search_text": search_text,
        "fields": fields,
        "terms": tokenize(search_text),
        "boosts": _boosts_for_unit("event_step"),
    }


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text for local lexical retrieval."""
    normalized = normalize_whitespace(remove_accents((text or "").lower()))
    terms: list[str] = []
    seen: set[str] = set()
    for token in normalized.split():
        clean = token.strip(".,;:!?\"'()[]{}<>/\\|+-=*#`~")
        if clean and len(clean) >= 2 and clean not in seen:
            seen.add(clean)
            terms.append(clean)
    return terms


def normalized_text(text: str) -> str:
    return normalize_whitespace(remove_accents((text or "").lower()))


def _compact_document_id(doc: dict) -> str:
    unit_type = doc.get("unit_type", "unknown") or "unknown"
    unit_id = doc.get("unit_id", "") or doc.get("frame_id", "") or "unknown"
    return f"compact:{unit_type}:{unit_id}"


def _event_step_document_id(doc: dict) -> str:
    event_id = doc.get("event_id", "") or doc.get("shot_id", "") or "unknown"
    return f"event_step:{event_id}"


def _boosts_for_unit(unit_type: str) -> dict[str, float]:
    if unit_type == "frame":
        return {
            "caption_text": 2.0,
            "ocr_text": 2.5,
            "audio_text": 1.2,
            "object_text": 1.8,
            "scene_text": 1.0,
            "all_text": 0.8,
        }
    if unit_type == "shot":
        return {
            "caption_text": 2.2,
            "trake_text": 2.0,
            "ocr_text": 2.0,
            "audio_text": 1.6,
            "object_text": 1.5,
            "scene_text": 1.0,
            "all_text": 0.8,
        }
    return {
        "trake_text": 2.8,
        "current_observation": 2.2,
        "event_caption": 2.0,
        "before_context": 0.8,
        "after_context": 0.8,
        "scene": 0.8,
        "caption_text": 1.4,
        "temporal_caption": 1.4,
        "ocr_text": 1.2,
        "audio_text": 1.2,
        "object_text": 1.2,
    }
