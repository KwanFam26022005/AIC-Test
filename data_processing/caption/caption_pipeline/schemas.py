"""Schema version constants used across the caption pipeline outputs."""

from __future__ import annotations

# Evidence schemas
FRAME_EVIDENCE_SCHEMA = "caption_frame_evidence_v1"
SHOT_EVIDENCE_SCHEMA = "caption_shot_evidence_v1"

# Search index schema
COMPACT_SEARCH_SCHEMA = "caption_compact_search_v1"

# Expected input schema versions (informational; loaders log warnings if mismatched)
EXPECTED_OCR_SCHEMA = "ocr_vlm_pipeline_v2_es_2"
EXPECTED_OBJECT_SCHEMA = "ram_gdino_object_detection_v1_1"
EXPECTED_AUDIO_SCHEMA = "audio_feature_v1"
