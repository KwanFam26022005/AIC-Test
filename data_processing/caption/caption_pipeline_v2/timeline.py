"""Timeline — sort, validate, and compute signatures for keyframes and shots.

Phase 0 responsibilities (continued):
  - Sort frames by timestamp.
  - Validate shot membership and time ranges.
  - Compute immutable frame and shot signatures for resume.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sort and validate
# ---------------------------------------------------------------------------

def sort_frames(frames: list[dict]) -> list[dict]:
    """Sort frames by timestamp_sec (monotonic), then canonical_frame_id."""
    return sorted(
        frames,
        key=lambda f: (f["timestamp_sec"], f["canonical_frame_id"]),
    )


def validate_timeline(frames: list[dict]) -> list[str]:
    """Validate that the frame timeline is monotonically non-decreasing.

    Returns a list of warnings (empty when everything is fine).
    """
    warnings: list[str] = []
    prev_ts = -1.0
    for f in frames:
        ts = f["timestamp_sec"]
        if ts < prev_ts:
            warnings.append(
                f"Non-monotonic timestamp: {f['canonical_frame_id']} "
                f"ts={ts} < prev={prev_ts}"
            )
        prev_ts = ts
    return warnings


def validate_shot_membership(
    frames: list[dict],
    shots: list[dict],
) -> list[str]:
    """Validate that every frame belongs to exactly one shot.

    Returns a list of warnings.
    """
    warnings: list[str] = []
    frame_to_shot: dict[str, str] = {}

    for shot in shots:
        shot_id = shot["shot_id"]
        for fid in shot["frame_ids"]:
            if fid in frame_to_shot:
                warnings.append(
                    f"Frame {fid} in multiple shots: "
                    f"{frame_to_shot[fid]} and {shot_id}"
                )
            frame_to_shot[fid] = shot_id

    frame_ids = {f["canonical_frame_id"] for f in frames}
    unassigned = frame_ids - set(frame_to_shot.keys())
    if unassigned:
        warnings.append(
            f"{len(unassigned)} frames not assigned to any shot: "
            f"{sorted(unassigned)[:5]}..."
        )

    return warnings


# ---------------------------------------------------------------------------
# Signature computation  (§6)
# ---------------------------------------------------------------------------

def _stable_hash(value: Any) -> str:
    """Compute a deterministic SHA-256 hash of a JSON-serializable value."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_frame_signature(
    frame: dict,
    model_id: str,
    model_revision: str,
    prompt_version: str,
    prompt_hash: str,
    generation_settings: dict,
    schema_version: str,
    image_hash: str = "",
) -> str:
    """Compute an immutable frame result signature for resume.

    Includes:
      - image content hash
      - canonical_frame_id
      - model ID and revision
      - prompt version and hash
      - generation settings
      - schema version
    """
    sig_input = {
        "image_hash": image_hash,
        "canonical_frame_id": frame["canonical_frame_id"],
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "generation_settings": generation_settings,
        "schema_version": schema_version,
    }
    return _stable_hash(sig_input)


def compute_shot_signature(
    shot: dict,
    frame_caption_signatures: list[str],
    audio_feature_ids: list[str],
    memory_before: dict,
    model_id: str,
    prompt_version: str,
    schema_version: str,
) -> str:
    """Compute an immutable shot result signature for resume.

    Includes:
      - ordered frame-caption signatures
      - ASR feature IDs
      - shot bounds
      - memory_before
      - model and prompt identifiers
      - schema version
    """
    sig_input = {
        "shot_id": shot["shot_id"],
        "start_sec": shot["start_sec"],
        "end_sec": shot["end_sec"],
        "frame_caption_signatures": frame_caption_signatures,
        "audio_feature_ids": audio_feature_ids,
        "memory_before": memory_before,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
    }
    return _stable_hash(sig_input)


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA-256 of a prompt template for signature stability."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
