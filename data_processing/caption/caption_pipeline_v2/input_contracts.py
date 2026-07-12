"""Input contracts — load and validate keyframes and audio features.

Phase 0 responsibilities:
  - Resolve paths and validate existence.
  - Load keyframe map (JSONL or directory scan).
  - Load audio features (read-only, filter usable_for_caption).
  - Reject duplicate canonical frame IDs.
  - Validate monotonic timestamps.
  - Compute image content hashes for resume signatures.
"""

from __future__ import annotations

import hashlib
import csv
import json
import logging
from pathlib import Path
from typing import Any

from .schemas import EXPECTED_AUDIO_SCHEMA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL helpers (self-contained — no dependency on old caption_pipeline)
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _round_sec(value: float | int | None, digits: int = 3) -> float:
    if value is None:
        return 0.0
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Keyframe loader
# ---------------------------------------------------------------------------

def load_keyframes(
    keyframe_map: str | Path,
    keyframes_root: str | Path,
    video_id: str,
) -> list[dict[str, Any]]:
    """Load keyframes from a JSONL map or by scanning a directory.

    Each returned dict has at minimum:
        video_id, canonical_frame_id, keyframe_idx, source_frame_idx,
        timestamp_sec, image_path, image_relpath
    """
    map_path = Path(keyframe_map)
    root_path = Path(keyframes_root)

    frames: list[dict[str, Any]]
    if map_path.is_file() and map_path.suffix in (".jsonl", ".json"):
        frames = _load_from_jsonl(map_path, root_path, video_id)
    elif map_path.is_dir():
        frames = _load_from_directory(map_path, video_id)
    elif root_path.is_dir():
        frames = _load_from_directory(root_path, video_id)
    else:
        raise FileNotFoundError(
            f"Cannot find keyframe source: map={keyframe_map}, root={keyframes_root}"
        )

    # Validate
    _validate_keyframes(frames, video_id)

    logger.info(
        "Input contracts: loaded %d keyframes for %s", len(frames), video_id,
    )
    return frames


def _load_from_jsonl(
    path: Path, root: Path, video_id: str,
) -> list[dict[str, Any]]:
    """Load keyframe metadata from a JSONL file."""
    rows = _iter_jsonl(path)
    frames: list[dict[str, Any]] = []

    for row in rows:
        vid = row.get("video_id", video_id)
        if vid != video_id:
            continue

        canonical = row.get("canonical_frame_id", "")
        if not canonical:
            kf_idx = row.get("keyframe_idx")
            if kf_idx is not None:
                canonical = f"{video_id}_{kf_idx}"
            else:
                continue

        image_relpath = row.get("image_relpath", "")
        image_path = row.get("image_path", "")
        if not image_path and image_relpath:
            image_path = str(root / image_relpath)
        elif not image_path:
            # Try to reconstruct from keyframe_idx
            kf_idx = row.get("keyframe_idx")
            if kf_idx is not None:
                for ext in (".jpg", ".png", ".jpeg"):
                    candidate = root / video_id / f"{kf_idx}{ext}"
                    if candidate.exists():
                        image_path = str(candidate)
                        image_relpath = f"{video_id}/{kf_idx}{ext}"
                        break

        frames.append({
            "video_id": vid,
            "canonical_frame_id": canonical,
            "keyframe_idx": row.get("keyframe_idx", 0),
            "source_frame_idx": row.get("source_frame_idx"),
            "timestamp_sec": _round_sec(row.get("timestamp_sec")),
            "image_path": image_path,
            "image_relpath": image_relpath,
        })

    return frames


def _load_from_directory(
    root: Path, video_id: str,
) -> list[dict[str, Any]]:
    """Scan a directory for keyframe images and build frame records.

    Expects structure:  root/video_id/*.jpg  or  root/*.jpg
    File names are expected to be integer keyframe indices.
    """
    video_dir = root / video_id
    if video_dir.is_dir():
        scan_dir = video_dir
    else:
        scan_dir = root

    csv_rows = _load_keyframe_csv(scan_dir / f"{video_id}.csv")
    csv_by_stem: dict[str, dict[str, Any]] = {}
    for row in csv_rows:
        stem = f"{int(row['n']):03d}"
        csv_by_stem[stem] = row

    image_exts = {".jpg", ".jpeg", ".png"}
    files = sorted(
        f for f in scan_dir.iterdir()
        if f.suffix.lower() in image_exts and f.stem.isdigit()
    )

    frames: list[dict[str, Any]] = []
    for f in files:
        kf_idx = int(f.stem)
        canonical = f"{video_id}_{f.stem}"
        csv_row = csv_by_stem.get(f.stem, {})

        if video_dir.is_dir():
            relpath = f"{video_id}/{f.name}"
        else:
            relpath = f.name

        frames.append({
            "video_id": video_id,
            "canonical_frame_id": canonical,
            "keyframe_idx": kf_idx,
            "source_frame_idx": csv_row.get("frame_idx"),
            "timestamp_sec": _round_sec(csv_row.get("pts_time"), 3),
            "image_path": str(f),
            "image_relpath": relpath,
        })

    return frames


def _load_keyframe_csv(path: Path) -> list[dict[str, Any]]:
    """Load AIC keyframe CSV map with n, pts_time, fps, frame_idx columns."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append({
                    "n": int(row.get("n") or 0),
                    "pts_time": float(row.get("pts_time") or 0.0),
                    "fps": float(row.get("fps") or 0.0),
                    "frame_idx": int(float(row.get("frame_idx") or 0)),
                })
            except ValueError:
                continue
    return rows


def _validate_keyframes(frames: list[dict], video_id: str) -> None:
    """Validate keyframe list — reject duplicates, check required fields."""
    if not frames:
        raise ValueError(f"No keyframes found for {video_id}")

    seen_ids: set[str] = set()
    for f in frames:
        cfid = f["canonical_frame_id"]
        if cfid in seen_ids:
            raise ValueError(f"Duplicate canonical_frame_id: {cfid}")
        seen_ids.add(cfid)

        if not f.get("image_path"):
            logger.warning("Frame %s has no image_path", cfid)


# ---------------------------------------------------------------------------
# Audio features loader
# ---------------------------------------------------------------------------

def load_audio_features(
    path: str | Path,
    video_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load audio features, returning usable segments sorted by time.

    Only rows with ``usable_for_caption=True`` are kept.
    Text is ``caption_text`` if available, else ``clean_transcript``.

    Returns ``(usable_segments, stats)``.
    """
    p = Path(path)
    stats = {"num_audio_features_total": 0, "num_audio_features_usable": 0}

    if not p.exists():
        logger.warning("Audio features not found: %s", p)
        return [], stats

    rows = _iter_jsonl(p)
    usable: list[dict[str, Any]] = []

    for row in rows:
        stats["num_audio_features_total"] += 1

        # Optional schema check
        sv = row.get("schema_version", "")
        if sv and sv != EXPECTED_AUDIO_SCHEMA:
            logger.debug("Audio schema %s (expected %s)", sv, EXPECTED_AUDIO_SCHEMA)

        vid = row.get("video_id", "")
        if vid and vid != video_id:
            continue

        if not row.get("usable_for_caption", False):
            continue

        caption_text = (
            row.get("caption_text")
            or row.get("text")
            or row.get("transcript")
            or row.get("asr_text")
            or ""
        ).strip()
        if not caption_text:
            caption_text = (row.get("clean_transcript") or "").strip()

        start_sec = _first_number(row, ("start_sec", "start_time_sec", "start", "segment_start_sec"))
        end_sec = _first_number(row, ("end_sec", "end_time_sec", "end", "segment_end_sec"))

        usable.append({
            "feature_id": row.get("feature_id", ""),
            "video_id": row.get("video_id", video_id),
            "start_sec": _round_sec(start_sec),
            "end_sec": _round_sec(end_sec),
            "caption_text": caption_text,
            "clean_transcript": (row.get("clean_transcript") or "").strip(),
            "quality_level": row.get("quality_level", ""),
            "usable_for_caption": True,
        })

    # Sort by timeline
    usable.sort(key=lambda s: (s["start_sec"], s["end_sec"], s["feature_id"]))
    stats["num_audio_features_usable"] = len(usable)

    logger.info(
        "Audio loader: %d usable / %d total from %s",
        len(usable), stats["num_audio_features_total"], p.name,
    )
    return usable, stats


def _first_number(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row or row.get(key) is None:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Image hashing for signatures
# ---------------------------------------------------------------------------

def compute_image_hash(image_path: str | Path) -> str:
    """Compute SHA-256 hash of an image file for provenance."""
    p = Path(image_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Preflight image resolution
# ---------------------------------------------------------------------------

def resolve_images(
    frames: list[dict], keyframes_root: str | Path,
) -> tuple[list[dict], list[str]]:
    """Resolve and verify all image paths before GPU model load.

    Returns ``(resolved_frames, warnings)``.
    """
    root = Path(keyframes_root)
    warnings: list[str] = []

    for f in frames:
        p = Path(f["image_path"])
        if p.exists():
            continue

        # Try resolving against root
        relpath = f.get("image_relpath", "")
        if relpath:
            candidate = root / relpath
            if candidate.exists():
                f["image_path"] = str(candidate)
                continue

        warnings.append(f"Image not found: {f['canonical_frame_id']} -> {f['image_path']}")

    return frames, warnings
