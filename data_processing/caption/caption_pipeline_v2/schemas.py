"""Schema version constants and enum contracts for caption pipeline v2.

All schema strings and valid enum values are defined here so that every
module can import them from a single source.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema versions
# ---------------------------------------------------------------------------

# Frame selection (Phase 1)
FRAME_SELECTION_SCHEMA = "caption_frame_selection_v1"

# Frame captions (Phase 3)
FRAME_CAPTION_SCHEMA = "caption_frame_v2"

# Shot audio context (Phase 4)
SHOT_AUDIO_CONTEXT_SCHEMA = "caption_shot_audio_context_v1"

# Shot captions (Phase 5)
SHOT_CAPTION_SCHEMA = "caption_shot_v2"

# Recap memory (Phase 5)
RECAP_MEMORY_SCHEMA = "caption_recap_memory_v1"

# Escalation (Phase 6)
ESCALATION_SCHEMA = "caption_escalation_v1"

# Run manifest (Phase 7)
RUN_MANIFEST_SCHEMA = "caption_run_manifest_v1"

# Pipeline identifier
PIPELINE_VERSION = "caption_audio_context_v1"

# Expected upstream audio schema
EXPECTED_AUDIO_SCHEMA = "audio_feature_v1"

# ---------------------------------------------------------------------------
# Enum contracts  (§11.6)
# ---------------------------------------------------------------------------

VALID_SELECTION_ROLES = frozenset({"anchor", "novel", "propagated", "forced"})

VALID_CAPTION_MODES = frozenset({"vlm_generated", "propagated", "fallback"})

VALID_ACTION_STATES = frozenset({
    "start", "middle", "end", "transition", "result", "unknown",
})

VALID_TEMPORAL_ROLES = frozenset({
    "beginning", "continuation", "change", "completion", "unknown",
})

VALID_GENERATION_MODES = frozenset({
    "text_recap", "vlm_escalated", "fallback",
})

# Known aliases that should be normalized instead of regenerated.
TEMPORAL_ROLE_ALIASES: dict[str, str] = {
    "transition": "change",
    "intro": "beginning",
    "ending": "completion",
    "continues": "continuation",
    "ongoing": "continuation",
}

ACTION_STATE_ALIASES: dict[str, str] = {
    "beginning": "start",
    "ending": "end",
    "concluded": "end",
    "midway": "middle",
    "ongoing": "middle",
}


# ---------------------------------------------------------------------------
# Enum validation helpers
# ---------------------------------------------------------------------------

def normalize_action_state(value: str) -> str:
    """Normalize an action_state value, applying aliases."""
    v = value.strip().lower()
    v = ACTION_STATE_ALIASES.get(v, v)
    return v if v in VALID_ACTION_STATES else "unknown"


def normalize_temporal_role(value: str) -> str:
    """Normalize a temporal_role value, applying aliases."""
    v = value.strip().lower()
    v = TEMPORAL_ROLE_ALIASES.get(v, v)
    return v if v in VALID_TEMPORAL_ROLES else "unknown"


def normalize_generation_mode(value: str) -> str:
    """Normalize a generation mode value."""
    v = value.strip().lower()
    return v if v in VALID_GENERATION_MODES else "text_recap"


def validate_selection_role(value: str) -> bool:
    """Check whether *value* is a valid selection role."""
    return value.strip().lower() in VALID_SELECTION_ROLES


def validate_caption_mode(value: str) -> bool:
    """Check whether *value* is a valid caption mode."""
    return value.strip().lower() in VALID_CAPTION_MODES
