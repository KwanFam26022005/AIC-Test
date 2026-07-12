"""Frame selector — shot-aware selection of anchor, novel, and propagated frames.

Phase 1 (§7): reduce VLM calls while guaranteeing at least one directly
observed frame per shot.

Selection roles:
  anchor      — primary frame representing the shot (temporal midpoint)
  novel       — additional frame with meaningful visual change
  propagated  — redundant frame assigned to an anchor or novel source
  forced      — selected by safety rule despite low novelty
"""

from __future__ import annotations

import logging
from typing import Any

from .config import FrameSelectionConfig
from .schemas import FRAME_SELECTION_SCHEMA

logger = logging.getLogger(__name__)

# Lazy import — pHash is only used when similarity_method == "phash"
_imagehash = None


def _get_imagehash():
    """Lazy-load imagehash to avoid import errors when not installed."""
    global _imagehash
    if _imagehash is None:
        try:
            import imagehash
            _imagehash = imagehash
        except ImportError:
            _imagehash = None
    return _imagehash


def _phash_distance(path_a: str, path_b: str) -> float:
    """Compute normalized pHash distance between two images.

    Returns a similarity score in [0, 1] where 1 = identical.
    """
    imagehash = _get_imagehash()
    if imagehash is None:
        logger.warning("imagehash not installed — falling back to 0.5 similarity")
        return 0.5

    try:
        from PIL import Image
        hash_a = imagehash.phash(Image.open(path_a))
        hash_b = imagehash.phash(Image.open(path_b))
        # pHash returns Hamming distance; max distance for 64-bit hash = 64
        distance = hash_a - hash_b
        similarity = 1.0 - (distance / 64.0)
        return max(0.0, min(1.0, similarity))
    except Exception as exc:
        logger.warning("pHash comparison failed: %s", exc)
        return 0.5


def select_frames(
    shot: dict,
    cfg: FrameSelectionConfig,
    run_id: str = "",
    video_id: str = "",
) -> list[dict[str, Any]]:
    """Select frames for VLM inference within a single shot.

    Returns a list of frame_selection records for every frame in the shot.
    """
    frames = shot["frames"]
    shot_id = shot["shot_id"]
    n = len(frames)

    if n == 0:
        return []

    # ---- Step 1: Choose anchor (temporal midpoint) ----
    mid_ts = (frames[0]["timestamp_sec"] + frames[-1]["timestamp_sec"]) / 2.0
    anchor_idx = min(
        range(n),
        key=lambda i: abs(frames[i]["timestamp_sec"] - mid_ts),
    )

    # Track selected indices and their roles
    selected: dict[int, str] = {anchor_idx: "anchor"}

    # ---- Step 2: Novelty detection for remaining frames ----
    if n > 1 and cfg.max_selected_per_shot > 1:
        # Compare each non-anchor frame against the most recent selected frame
        selected_indices = sorted(selected.keys())

        for i in range(n):
            if i in selected:
                continue
            if len(selected) >= cfg.max_selected_per_shot:
                break

            # Find the closest selected frame for comparison
            closest_selected = min(
                selected_indices,
                key=lambda si: abs(i - si),
            )

            if cfg.similarity_method == "phash":
                sim = _phash_distance(
                    frames[i]["image_path"],
                    frames[closest_selected]["image_path"],
                )
            else:
                sim = 0.5  # Default for unknown methods

            novelty = 1.0 - sim
            if novelty >= cfg.novelty_threshold:
                selected[i] = "novel"
                selected_indices = sorted(selected.keys())

    # ---- Step 3: Safety — force first/last for long shots ----
    if (
        cfg.force_first_last_for_long_shots
        and n >= cfg.long_shot_min_frames
        and len(selected) < cfg.max_selected_per_shot
    ):
        if 0 not in selected:
            selected[0] = "forced"
        if n - 1 not in selected and len(selected) < cfg.max_selected_per_shot:
            selected[n - 1] = "forced"

    # ---- Step 4: Build selection records for ALL frames ----
    results: list[dict[str, Any]] = []

    for i, frame in enumerate(frames):
        if i in selected:
            role = selected[i]
            record = {
                "schema_version": FRAME_SELECTION_SCHEMA,
                "run_id": run_id,
                "video_id": video_id or frame.get("video_id", ""),
                "canonical_frame_id": frame["canonical_frame_id"],
                "keyframe_idx": frame.get("keyframe_idx", 0),
                "timestamp_sec": frame["timestamp_sec"],
                "shot_id": shot_id,
                "selection_role": role,
                "selected_for_vlm": True,
                "selection_reasons": [f"{role}_selected"],
                "source_frame_id": None,
                "similarity_method": cfg.similarity_method,
                "similarity_score": None,
                "novelty_score": 1.0 if role == "anchor" else None,
                "warnings": [],
            }
        else:
            # Propagated — find nearest selected frame in the same shot
            nearest_selected_idx = min(
                selected.keys(),
                key=lambda si: abs(i - si),
            )
            source_frame = frames[nearest_selected_idx]

            # Compute similarity to source
            if cfg.similarity_method == "phash":
                sim = _phash_distance(
                    frame["image_path"],
                    source_frame["image_path"],
                )
            else:
                sim = 0.5

            record = {
                "schema_version": FRAME_SELECTION_SCHEMA,
                "run_id": run_id,
                "video_id": video_id or frame.get("video_id", ""),
                "canonical_frame_id": frame["canonical_frame_id"],
                "keyframe_idx": frame.get("keyframe_idx", 0),
                "timestamp_sec": frame["timestamp_sec"],
                "shot_id": shot_id,
                "selection_role": "propagated",
                "selected_for_vlm": False,
                "selection_reasons": ["similar_to_anchor"],
                "source_frame_id": source_frame["canonical_frame_id"],
                "similarity_method": cfg.similarity_method,
                "similarity_score": round(sim, 4),
                "novelty_score": round(1.0 - sim, 4),
                "warnings": [],
            }

        results.append(record)

    selected_count = sum(1 for r in results if r["selected_for_vlm"])
    logger.debug(
        "Shot %s: %d/%d frames selected for VLM (anchor + %d others)",
        shot_id, selected_count, n, selected_count - 1,
    )

    return results


def select_all_shots(
    shots: list[dict],
    cfg: FrameSelectionConfig,
    run_id: str = "",
    video_id: str = "",
) -> list[dict[str, Any]]:
    """Run frame selection for all shots. Returns a flat list of selection records."""
    all_selections: list[dict[str, Any]] = []
    for shot in shots:
        selections = select_frames(shot, cfg, run_id, video_id)
        all_selections.extend(selections)

    total = len(all_selections)
    selected = sum(1 for s in all_selections if s["selected_for_vlm"])
    propagated = total - selected

    logger.info(
        "Frame selector: %d total frames, %d selected for VLM (%.1f%%), %d propagated",
        total, selected,
        (selected / total * 100) if total else 0,
        propagated,
    )
    return all_selections
