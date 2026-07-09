# -*- coding: utf-8 -*-
"""Enrich legacy object-detection JSONL without rerunning RAM/GroundingDINO."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from tag_canonicalization import build_search_fields, canonicalize_detections


SCHEMA_VERSION = "ram_gdino_object_detection_v1_1"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(doc, dict):
                yield doc


def write_jsonl(path: str | Path, docs: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")


def extract_frame_idx(value: str | Path | None) -> int:
    stem = Path(str(value or "")).stem
    digits = ""
    for char in reversed(stem):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return int(digits) if digits else 0


def frame_name_with_extension(doc: dict[str, Any]) -> str:
    image_path = doc.get("image_path")
    if image_path:
        name = Path(str(image_path)).name
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
            return name

    frame_name = str(doc.get("frame_name") or "")
    if Path(frame_name).suffix.lower() in IMAGE_EXTENSIONS:
        return frame_name
    if frame_name:
        return f"{frame_name}.jpg"

    frame_id = str(doc.get("frame_id") or "")
    suffix = frame_id.rsplit("_", 1)[-1] if "_" in frame_id else frame_id
    return f"{suffix or '000'}.jpg"


def load_keyframe_map_csv(csv_path: str | Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    path = Path(csv_path)
    if not path.is_file():
        return result

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                keyframe_idx = int(row["n"])
                result[keyframe_idx] = {
                    "pts_time": float(row["pts_time"]),
                    "fps": float(row.get("fps", 0) or 0),
                    "frame_idx": int(float(row.get("frame_idx", 0) or 0)),
                }
            except (KeyError, ValueError):
                continue
    return result


def find_keyframe_map(keyframe_map_path: str | Path | None, video_id: str) -> dict[int, dict[str, Any]]:
    if not keyframe_map_path:
        return {}

    path = Path(keyframe_map_path)
    if path.is_file():
        return load_keyframe_map_csv(path)
    if not path.is_dir():
        return {}

    for candidate in (path / video_id / f"{video_id}.csv", path / f"{video_id}.csv"):
        if candidate.is_file():
            return load_keyframe_map_csv(candidate)
    return {}


def load_video_manifest(manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not manifest_path or not Path(manifest_path).is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for doc in iter_jsonl(manifest_path):
        video_id = doc.get("video_id")
        if video_id:
            result[str(video_id)] = doc
    return result


def resolve_timestamp(
    keyframe_idx: int,
    keyframe_map: dict[int, dict[str, Any]],
    num_keyframes: int,
    manifest_doc: dict[str, Any] | None,
    strategy: str,
) -> tuple[float | None, str]:
    if strategy == "none":
        return None, "none"

    if strategy in ("map_or_uniform", "map_only") and keyframe_idx in keyframe_map:
        return float(keyframe_map[keyframe_idx]["pts_time"]), "keyframe_map_csv"

    if strategy == "map_only":
        return None, "map_missing"

    if strategy in ("map_or_uniform", "uniform") and manifest_doc and num_keyframes > 1:
        duration = manifest_doc.get("duration_sec")
        if duration and float(duration) > 0:
            value = (keyframe_idx - 1) * float(duration) / max(1, num_keyframes - 1)
            return round(value, 6), "uniform_interpolation"

    return None, "unavailable"


def coerce_detection(obj: dict[str, Any]) -> dict[str, Any] | None:
    label = obj.get("label") or obj.get("label_lower")
    box = obj.get("box") or obj.get("bbox_xyxy")
    if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        score = float(obj.get("score", obj.get("confidence", 0.0)) or 0.0)
        clean_box = [float(value) for value in box]
    except (TypeError, ValueError):
        return None

    det = dict(obj)
    det["label"] = str(label)
    det["score"] = round(score, 4)
    det["box"] = clean_box
    det.setdefault("source", obj.get("source") or "legacy_jsonl")
    return det


def enrich_doc(
    doc: dict[str, Any],
    *,
    video_id: str,
    keyframe_map: dict[int, dict[str, Any]],
    num_keyframes: int,
    manifest_doc: dict[str, Any] | None,
    timestamp_strategy: str,
    scene_area_threshold: float,
) -> dict[str, Any]:
    out = dict(doc)
    frame_name = str(out.get("frame_name") or Path(frame_name_with_extension(out)).stem)
    keyframe_idx = int(out.get("keyframe_idx") or extract_frame_idx(frame_name))
    map_row = keyframe_map.get(keyframe_idx)

    out["schema_version"] = SCHEMA_VERSION
    out["video_id"] = str(out.get("video_id") or video_id)
    out["frame_id"] = str(out.get("frame_id") or f"{out['video_id']}_{frame_name}")
    out["canonical_frame_id"] = str(out.get("canonical_frame_id") or out["frame_id"])
    out["frame_name"] = frame_name
    out["frame_idx"] = int(out.get("frame_idx") or keyframe_idx)
    out["keyframe_idx"] = keyframe_idx
    out["source_frame_idx"] = out.get("source_frame_idx")
    if out["source_frame_idx"] is None and map_row:
        out["source_frame_idx"] = map_row.get("frame_idx")

    if out.get("timestamp_sec") is None:
        timestamp_sec, source = resolve_timestamp(
            keyframe_idx,
            keyframe_map,
            num_keyframes,
            manifest_doc,
            timestamp_strategy,
        )
        out["timestamp_sec"] = timestamp_sec
        out["timestamp_source"] = source
    else:
        out.setdefault("timestamp_source", "existing")

    out.setdefault("image_relpath", f"{out['video_id']}/{frame_name_with_extension(out)}")

    image_size = out.get("image_size") or [0, 0]
    try:
        img_w, img_h = int(image_size[0]), int(image_size[1])
    except (TypeError, ValueError, IndexError):
        img_w, img_h = 0, 0

    raw_objects = [coerce_detection(obj) for obj in (out.get("objects") or []) if isinstance(obj, dict)]
    detections = [obj for obj in raw_objects if obj is not None]
    if img_w > 0 and img_h > 0:
        objects, scene_labels, quality_stats = canonicalize_detections(
            detections,
            img_w,
            img_h,
            nms_iou_threshold=1.01,
            scene_area_threshold=scene_area_threshold,
        )
    else:
        objects, scene_labels, quality_stats = detections, [], {}

    raw_tags = out.get("raw_tags") or out.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [tag.strip() for tag in raw_tags.split("|") if tag.strip()]
    existing_scene_tags = [str(tag).lower() for tag in (out.get("scene_tags") or [])]
    scene_labels = sorted(set(existing_scene_tags + [str(tag).lower() for tag in scene_labels]))
    search_fields = build_search_fields(objects, scene_labels, raw_tags)

    out["objects"] = objects
    out["ram_tags"] = search_fields["ram_tags"]
    out["scene_tags"] = search_fields["scene_tags"]
    out["object_tags"] = search_fields["object_tags"]
    out["object_summary"] = sorted(search_fields["object_counts"].keys())
    out["object_counts"] = search_fields["object_counts"]
    out["object_counts_normalized"] = search_fields["object_counts_normalized"]
    out["object_count_items"] = search_fields["object_count_items"]
    out["important_objects"] = search_fields["important_objects"]
    out["object_text"] = search_fields["object_text"]
    out["scene_text"] = search_fields["scene_text"]
    out["ram_tag_text"] = search_fields["ram_tag_text"]
    out["all_object_text"] = search_fields["all_object_text"]

    quality = dict(out.get("quality") or {})
    quality.update(
        {
            "adapter_enriched": True,
            "num_final_boxes": len(objects),
            **quality_stats,
        }
    )
    out["quality"] = quality
    return out


def build_summary(docs: list[dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    top_counts: Counter[str] = Counter()
    missing_timestamp = 0
    missing_area = 0
    missing_position = 0
    scene_labels_in_counts = 0
    total_objects = 0

    for doc in docs:
        if doc.get("timestamp_sec") is None:
            missing_timestamp += 1
        scene_tag_set = set(doc.get("scene_tags") or [])
        counts = doc.get("object_counts_normalized") or {}
        for label, count in counts.items():
            count_int = int(count or 0)
            top_counts[str(label)] += count_int
            if str(label) in scene_tag_set:
                scene_labels_in_counts += count_int
        for obj in doc.get("objects") or []:
            total_objects += 1
            if obj.get("area_ratio") is None:
                missing_area += 1
            if not obj.get("position"):
                missing_position += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output": str(output_path),
        "num_frames": len(docs),
        "num_objects": total_objects,
        "num_frames_missing_timestamp": missing_timestamp,
        "num_objects_missing_area_ratio": missing_area,
        "num_objects_missing_position": missing_position,
        "num_scene_labels_in_counts": scene_labels_in_counts,
        "top_object_counts": dict(top_counts.most_common(20)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich legacy RAM+GDINO JSONL for caption/search.")
    parser.add_argument("--input", required=True, help="Input object JSONL.")
    parser.add_argument("--output", required=True, help="Output enriched JSONL.")
    parser.add_argument("--summary-output", help="Optional summary JSON path.")
    parser.add_argument("--video-id", help="Video ID fallback if records do not contain video_id.")
    parser.add_argument("--keyframe-map", help="CSV path, keyframe map root, or directory containing <video_id>/<video_id>.csv.")
    parser.add_argument("--frames-root", help="Compatibility alias for keyframe map root when --keyframe-map is omitted.")
    parser.add_argument("--video-manifest", help="Optional video_manifest.jsonl for uniform timestamp fallback.")
    parser.add_argument(
        "--timestamp-strategy",
        default="map_or_uniform",
        choices=["map_or_uniform", "map_only", "uniform", "none"],
    )
    parser.add_argument("--scene-area-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs_in = list(iter_jsonl(args.input))
    if not docs_in:
        write_jsonl(args.output, [])
        return

    video_id = args.video_id or str(docs_in[0].get("video_id") or Path(args.input).stem.replace("_objects", ""))
    keyframe_map = find_keyframe_map(args.keyframe_map or args.frames_root, video_id)
    manifest = load_video_manifest(args.video_manifest)
    manifest_doc = manifest.get(video_id)

    docs_out = [
        enrich_doc(
            doc,
            video_id=video_id,
            keyframe_map=keyframe_map,
            num_keyframes=len(docs_in),
            manifest_doc=manifest_doc,
            timestamp_strategy=args.timestamp_strategy,
            scene_area_threshold=args.scene_area_threshold,
        )
        for doc in docs_in
    ]

    write_jsonl(args.output, docs_out)
    summary = build_summary(docs_out, args.output)

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
