"""Static HTML visualization for caption pipeline v2 outputs."""

from __future__ import annotations

import base64
import html
import json
import mimetypes
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_text


def write_caption_v2_visualization(
    *,
    caption_dir: str | Path,
    video_id: str,
    output_html: str | Path | None = None,
    keyframes_root: str | Path | None = None,
    image_mode: str = "copy",
) -> Path:
    """Write a single-page review HTML for ``caption_v2/<video_id>``."""
    if image_mode not in {"copy", "embed", "link"}:
        raise ValueError("image_mode must be one of: copy, embed, link")

    base_dir = Path(caption_dir) / video_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Caption v2 video directory not found: {base_dir}")

    paths = {
        "frames": base_dir / "captions" / "frame_captions.jsonl",
        "shots": base_dir / "captions" / "shot_captions.jsonl",
        "audio": base_dir / "evidence" / "shot_audio_context.jsonl",
        "selection": base_dir / "selection" / "frame_selection.jsonl",
        "manifest": base_dir / "manifest" / "run_manifest.json",
    }
    required = ("frames", "shots")
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError("Missing required caption v2 files: " + ", ".join(missing))

    frames = read_jsonl(paths["frames"])
    shots = read_jsonl(paths["shots"])
    audio_contexts = read_jsonl(paths["audio"]) if paths["audio"].exists() else []
    selections = read_jsonl(paths["selection"]) if paths["selection"].exists() else []
    manifest = _read_json(paths["manifest"]) if paths["manifest"].exists() else {}

    if output_html is None:
        output_path = base_dir / "reports" / "visualization_v2" / "index.html"
    else:
        output_path = Path(output_html)

    assets_dir = output_path.parent / "assets" if image_mode == "copy" else None
    keyframes = Path(keyframes_root) if keyframes_root else None
    context = _build_context(
        video_id=video_id,
        base_dir=base_dir,
        frames=frames,
        shots=shots,
        audio_contexts=audio_contexts,
        selections=selections,
        manifest=manifest,
        output_path=output_path,
        keyframes_root=keyframes,
        assets_dir=assets_dir,
        image_mode=image_mode,
    )
    write_text(output_path, _render_page(context))
    return output_path


def _build_context(
    *,
    video_id: str,
    base_dir: Path,
    frames: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    audio_contexts: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    assets_dir: Path | None,
    image_mode: str,
) -> dict[str, Any]:
    frame_map = {str(row.get("canonical_frame_id")): row for row in frames}
    shot_map = {str(row.get("shot_id")): row for row in shots}
    audio_map = {str(row.get("shot_id")): row for row in audio_contexts}
    selection_map = {str(row.get("canonical_frame_id")): row for row in selections}

    frames_by_shot: dict[str, list[dict[str, Any]]] = {sid: [] for sid in shot_map}
    for frame in frames:
        shot_id = str(frame.get("shot_id") or "")
        frames_by_shot.setdefault(shot_id, []).append(frame)
    for rows in frames_by_shot.values():
        rows.sort(key=lambda row: (float(row.get("timestamp_sec") or 0.0), str(row.get("canonical_frame_id"))))

    summary = _build_summary(frames, shots, audio_contexts, manifest)
    return {
        "video_id": video_id,
        "base_dir": base_dir,
        "frames": sorted(frames, key=lambda row: (float(row.get("timestamp_sec") or 0.0), str(row.get("canonical_frame_id")))),
        "shots": sorted(shots, key=lambda row: (float(row.get("start_sec") or 0.0), str(row.get("shot_id")))),
        "frame_map": frame_map,
        "shot_map": shot_map,
        "audio_map": audio_map,
        "selection_map": selection_map,
        "frames_by_shot": frames_by_shot,
        "manifest": manifest,
        "summary": summary,
        "output_path": output_path,
        "keyframes_root": keyframes_root,
        "assets_dir": assets_dir,
        "image_mode": image_mode,
    }


def _build_summary(
    frames: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    audio_contexts: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    modes = Counter(str((row.get("caption") or {}).get("mode") or "unknown") for row in frames)
    roles = Counter(str((row.get("selection") or {}).get("role") or "unknown") for row in frames)
    action_states = Counter(str((row.get("structure") or {}).get("action_state") or "unknown") for row in shots)
    temporal_roles = Counter(str((row.get("structure") or {}).get("temporal_role") or "unknown") for row in shots)
    audio_count = sum(1 for row in audio_contexts if row.get("has_usable_audio"))
    selected = sum(1 for row in frames if (row.get("caption") or {}).get("mode") == "vlm_generated")
    propagated = sum(1 for row in frames if (row.get("caption") or {}).get("mode") == "propagated")
    fallback = sum(1 for row in frames if (row.get("caption") or {}).get("mode") == "fallback")
    return {
        "cards": [
            ("Frames", len(frames), "canonical keyframes"),
            ("Shots", len(shots), "shot-level ReCap rows"),
            ("VLM frames", selected, "direct visual captions"),
            ("Propagated", propagated, "frames copied within shot"),
            ("Fallback", fallback, "frames needing fallback"),
            ("Shots w/ audio", audio_count, "aligned ASR context"),
        ],
        "frame_modes": modes,
        "selection_roles": roles,
        "action_states": action_states,
        "temporal_roles": temporal_roles,
        "manifest_status": manifest.get("status", "unknown"),
    }


def _render_page(context: dict[str, Any]) -> str:
    video_id = context["video_id"]
    cards = "\n".join(_summary_card(label, value, hint) for label, value, hint in context["summary"]["cards"])
    frames = "\n".join(_render_frame_card(row, context) for row in context["frames"])
    shots = "\n".join(_render_shot_card(row, context) for row in context["shots"])
    manifest = _escape(json.dumps(context["manifest"], ensure_ascii=False, indent=2))
    frame_modes = _counter_text(context["summary"]["frame_modes"])
    selection_roles = _counter_text(context["summary"]["selection_roles"])
    action_states = _counter_text(context["summary"]["action_states"])
    temporal_roles = _counter_text(context["summary"]["temporal_roles"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Caption V2 Review - {_escape(video_id)}</title>
  <style>{_css()}</style>
</head>
<body>
  <header class="app-header">
    <div>
      <h1>Caption V2 Review: {_escape(video_id)}</h1>
      <p class="subtle">Visual frame captions + audio-context shot ReCap. OCR/object detector outputs are not used by this caption branch.</p>
    </div>
    <div class="status">Status: <b>{_escape(context["summary"]["manifest_status"])}</b></div>
  </header>
  <main>
    <section class="cards">{cards}</section>
    <section class="metrics">
      <div><b>Frame modes</b><span>{_escape(frame_modes)}</span></div>
      <div><b>Selection roles</b><span>{_escape(selection_roles)}</span></div>
      <div><b>Action states</b><span>{_escape(action_states)}</span></div>
      <div><b>Temporal roles</b><span>{_escape(temporal_roles)}</span></div>
    </section>
    <section class="tabs" aria-label="Review tabs">
      <button class="tab active" type="button" data-tab-target="frames">Frames</button>
      <button class="tab" type="button" data-tab-target="shots">Shots</button>
      <button class="tab" type="button" data-tab-target="run">Run</button>
    </section>
    <section class="toolbar">
      <input id="search" placeholder="Search id, caption, audio, entities, TRAKE...">
      <select id="filter">
        <option value="">All rows</option>
        <option value="vlm_generated">VLM generated</option>
        <option value="propagated">Propagated</option>
        <option value="fallback">Fallback</option>
        <option value="has_audio">Has audio</option>
        <option value="no_audio">No audio</option>
        <option value="warning">Warning</option>
      </select>
      <span id="visibleCount" class="subtle"></span>
    </section>
    <section id="frames" class="tab-panel active">
      <div class="frame-grid">{frames}</div>
    </section>
    <section id="shots" class="tab-panel">
      <div class="shot-list">{shots}</div>
    </section>
    <section id="run" class="tab-panel">
      <article class="panel">
        <h2>Run Manifest</h2>
        <pre>{manifest}</pre>
      </article>
    </section>
  </main>
  <script>{_js()}</script>
</body>
</html>
"""


def _render_frame_card(frame: dict[str, Any], context: dict[str, Any]) -> str:
    frame_id = str(frame.get("canonical_frame_id") or "")
    shot_id = str(frame.get("shot_id") or "")
    caption = frame.get("caption") or {}
    selection = frame.get("selection") or {}
    quality = frame.get("quality") or {}
    shot = context["shot_map"].get(shot_id, {})
    audio = context["audio_map"].get(shot_id, {})
    image = _render_image(frame, context, alt=frame_id)
    chips = [
        _chip(str(caption.get("mode") or "unknown"), _mode_class(str(caption.get("mode") or ""))),
        _chip(str(selection.get("role") or "unknown"), "info"),
        _chip("audio", "good" if audio.get("has_usable_audio") else "warn"),
    ]
    if quality.get("warnings"):
        chips.append(_chip("warning", "warn"))
    if caption.get("mode") == "fallback":
        chips.append(_chip("fallback", "bad"))
    search = _join_search(
        frame_id,
        shot_id,
        caption.get("text"),
        audio.get("text"),
        (shot.get("caption") or {}).get("shot_caption"),
        (shot.get("caption") or {}).get("trake_text"),
    )
    features = _feature_tokens(caption.get("mode"), audio.get("has_usable_audio"), quality.get("warnings"))
    mapping_rows = [
        ("Shot", shot_id),
        ("Timestamp", _fmt_time(frame.get("timestamp_sec"))),
        ("Keyframe", frame.get("keyframe_idx")),
        ("Mode", caption.get("mode")),
        ("Selection", selection.get("role")),
        ("Source frame", selection.get("source_frame_id")),
        ("Similarity", selection.get("similarity_score")),
        ("Warnings", "; ".join(map(str, quality.get("warnings") or []))),
    ]
    raw = _escape(json.dumps(frame, ensure_ascii=False, indent=2))
    return f"""<article class="frame-card row" data-search="{_escape_attr(search)}" data-features="{_escape_attr(features)}">
  {image}
  <div class="content">
    <div class="title-row"><b>{_escape(frame_id)}</b><span>{_fmt_time(frame.get("timestamp_sec"))}</span></div>
    <div class="chips">{''.join(chips)}</div>
    <p class="caption">{_value(caption.get("text"))}</p>
    <details>
      <summary>Frame details</summary>
      <div class="detail-grid">
        <section class="box"><h3>Mapping</h3>{_table(mapping_rows)}</section>
        <section class="box audio"><h3>Audio context from shot</h3><p>{_value(audio.get("text"))}</p></section>
        <section class="box"><h3>Shot caption</h3><p>{_value((shot.get("caption") or {}).get("shot_caption"))}</p></section>
        <section class="box"><h3>TRAKE text</h3><p>{_value((shot.get("caption") or {}).get("trake_text"))}</p></section>
      </div>
      <details class="raw"><summary>Raw frame JSON</summary><pre>{raw}</pre></details>
    </details>
  </div>
</article>"""


def _render_shot_card(shot: dict[str, Any], context: dict[str, Any]) -> str:
    shot_id = str(shot.get("shot_id") or "")
    frames = context["frames_by_shot"].get(shot_id, [])
    audio = context["audio_map"].get(shot_id, (shot.get("audio_context") or {}))
    caption = shot.get("caption") or {}
    structure = shot.get("structure") or {}
    visual = shot.get("visual_evidence") or {}
    strip = _render_image_strip(frames[:3], context)
    chips = [
        _chip(f"frames:{len(frames)}", "info"),
        _chip("audio", "good" if audio.get("has_usable_audio") or audio.get("text") else "warn"),
        _chip(str(structure.get("action_state") or "unknown"), "state"),
        _chip(str(structure.get("temporal_role") or "unknown"), "role"),
    ]
    search = _join_search(
        shot_id,
        caption.get("shot_caption"),
        caption.get("event_caption"),
        caption.get("trake_text"),
        audio.get("text"),
        " ".join(map(str, structure.get("actors") or [])),
        " ".join(map(str, structure.get("actions") or [])),
        " ".join(map(str, structure.get("objects_involved") or [])),
        structure.get("scene"),
    )
    features = _feature_tokens(caption.get("mode"), audio.get("has_usable_audio") or bool(audio.get("text")), (shot.get("quality") or {}).get("warnings"))
    event_rows = [
        ("Shot caption", caption.get("shot_caption")),
        ("Temporal caption", caption.get("temporal_caption")),
        ("Event caption", caption.get("event_caption")),
        ("TRAKE text", caption.get("trake_text")),
        ("Action state", structure.get("action_state")),
        ("Temporal role", structure.get("temporal_role")),
        ("Scene", structure.get("scene")),
        ("Actors", ", ".join(map(str, structure.get("actors") or []))),
        ("Actions", ", ".join(map(str, structure.get("actions") or []))),
        ("Objects involved", ", ".join(map(str, structure.get("objects_involved") or []))),
    ]
    visual_rows = [
        (item.get("canonical_frame_id"), item.get("caption_text"))
        for item in visual.get("captions") or []
    ]
    raw = _escape(json.dumps(shot, ensure_ascii=False, indent=2))
    return f"""<article class="shot-card row" data-search="{_escape_attr(search)}" data-features="{_escape_attr(features)}">
  <div class="shot-head">
    <div>{strip}</div>
    <div>
      <div class="title-row"><b>{_escape(shot_id)}</b><span>{_fmt_time(shot.get("start_sec"))} - {_fmt_time(shot.get("end_sec"))}</span></div>
      <div class="chips">{''.join(chips)}</div>
      <p class="text-block"><b>Shot caption</b>{_value(caption.get("shot_caption"))}</p>
      <p class="text-block"><b>Event caption</b>{_value(caption.get("event_caption"))}</p>
    </div>
    <div class="trake"><b>TRAKE text</b><p>{_value(caption.get("trake_text"))}</p></div>
  </div>
  <details>
    <summary>Shot details</summary>
    <div class="shot-detail-grid">
      <section class="box event"><h3>Event / ReCap</h3>{_table(event_rows)}</section>
      <section class="box audio"><h3>Audio / ASR context</h3><p>{_value(audio.get("text"))}</p></section>
      <section class="box"><h3>Visual frame evidence</h3>{_table(visual_rows)}</section>
      <section class="box"><h3>Memory</h3><pre>{_escape(json.dumps(shot.get("memory") or {}, ensure_ascii=False, indent=2))}</pre></section>
    </div>
    <details class="raw"><summary>Raw shot JSON</summary><pre>{raw}</pre></details>
  </details>
</article>"""


def _render_image_strip(frames: list[dict[str, Any]], context: dict[str, Any]) -> str:
    if not frames:
        return '<div class="image-missing">No frames</div>'
    parts = "".join(_render_image(frame, context, alt=str(frame.get("canonical_frame_id") or "")) for frame in frames)
    return f'<div class="thumb-strip">{parts}</div>'


def _render_image(frame: dict[str, Any], context: dict[str, Any], alt: str) -> str:
    src, status = _image_src(
        frame=frame,
        output_path=context["output_path"],
        keyframes_root=context["keyframes_root"],
        image_mode=context["image_mode"],
        assets_dir=context["assets_dir"],
    )
    if not src:
        return f'<div class="image-missing">Image unavailable<br>{_escape(status)}</div>'
    return f'<div class="image-wrap"><img src="{_escape_attr(src)}" alt="{_escape_attr(alt)}" loading="lazy"></div>'


def _image_src(
    *,
    frame: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    image_mode: str,
    assets_dir: Path | None,
) -> tuple[str, str]:
    path, candidates = _resolve_image_path(frame, keyframes_root)
    if path is None:
        sample = ", ".join(str(item) for item in candidates[:4])
        return "", f"checked: {sample}" if sample else "no image metadata"
    if image_mode == "embed":
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}", str(path)
    if image_mode == "copy":
        target_dir = assets_dir or output_path.parent / "assets"
        target_dir.mkdir(parents=True, exist_ok=True)
        frame_id = str(frame.get("canonical_frame_id") or path.stem)
        target = target_dir / f"{_safe_name(frame_id)}{path.suffix.lower()}"
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
        return target.relative_to(output_path.parent).as_posix(), str(path)
    try:
        return path.resolve().relative_to(output_path.parent.resolve()).as_posix(), str(path)
    except ValueError:
        return path.resolve().as_uri(), str(path)


def _resolve_image_path(frame: dict[str, Any], keyframes_root: Path | None) -> tuple[Path | None, list[Path]]:
    video_id = str(frame.get("video_id") or "")
    raw = str(frame.get("image_path") or "")
    rel = str(frame.get("image_relpath") or "")
    frame_id = str(frame.get("canonical_frame_id") or "")
    keyframe_idx = frame.get("keyframe_idx")
    names: list[str] = []
    if rel:
        names.append(Path(rel).name)
    if frame_id:
        suffix = frame_id.rsplit("_", 1)[-1]
        names.extend([f"{suffix}.jpg", f"{suffix}.png", f"{suffix}.jpeg"])
    if isinstance(keyframe_idx, int):
        names.extend([f"{keyframe_idx:03d}.jpg", f"{keyframe_idx:06d}.jpg", f"{keyframe_idx}.jpg"])

    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
    if keyframes_root:
        if rel:
            candidates.extend([keyframes_root / rel, keyframes_root / video_id / Path(rel).name])
        for name in names:
            candidates.extend([keyframes_root / video_id / name, keyframes_root / name])
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if candidate.is_file():
            return candidate, unique
    return None, unique


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _summary_card(label: str, value: Any, hint: str) -> str:
    return f"""<article class="card">
  <span>{_escape(label)}</span>
  <b>{_escape(value)}</b>
  <small>{_escape(hint)}</small>
</article>"""


def _table(rows: list[tuple[Any, Any]]) -> str:
    body = "".join(f"<tr><th>{_escape(k)}</th><td>{_value(v)}</td></tr>" for k, v in rows)
    return f'<table class="kv"><tbody>{body}</tbody></table>'


def _chip(label: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{_escape(label)}</span>'


def _mode_class(mode: str) -> str:
    if mode == "vlm_generated":
        return "good"
    if mode == "propagated":
        return "info"
    if mode == "fallback":
        return "bad"
    return ""


def _feature_tokens(mode: Any, has_audio: Any, warnings: Any) -> str:
    tokens = []
    if mode:
        tokens.append(str(mode))
    if has_audio:
        tokens.append("has_audio")
    else:
        tokens.append("no_audio")
    if warnings:
        tokens.append("warning")
    return " ".join(tokens)


def _value(value: Any) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return '<span class="empty">empty</span>'
    return _escape(value)


def _counter_text(counter: Counter) -> str:
    if not counter:
        return "empty"
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common())


def _join_search(*values: Any) -> str:
    return " ".join(str(value) for value in values if value).lower()


def _fmt_time(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = 0.0
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=False)


def _escape_attr(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _css() -> str:
    return """
:root {
  --bg: #f5f7f9;
  --panel: #ffffff;
  --ink: #12202e;
  --muted: #617386;
  --line: #d9e1e8;
  --soft: #eef3f6;
  --green: #087f76;
  --green-soft: #def4f1;
  --blue: #1769aa;
  --blue-soft: #e5f1fb;
  --amber: #a45f08;
  --amber-soft: #fff3d5;
  --red: #b91c1c;
  --red-soft: #fee2e2;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.app-header {
  max-width: 1580px;
  margin: 0 auto;
  padding: 22px 24px 12px;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-end;
}
h1 { margin: 0 0 4px; font-size: 25px; letter-spacing: 0; }
h2 { margin: 0 0 12px; font-size: 18px; }
h3 { margin: 0 0 8px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0; }
main { max-width: 1580px; margin: 0 auto; padding: 0 24px 36px; }
.subtle, .empty { color: var(--muted); }
.empty { font-style: italic; }
.status {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 9px 12px;
  background: white;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
  margin: 12px 0;
}
.card {
  background: white;
  border: 1px solid var(--line);
  border-top: 3px solid var(--green);
  border-radius: 8px;
  padding: 11px 12px;
}
.card:nth-child(3n+2) { border-top-color: var(--blue); }
.card:nth-child(3n) { border-top-color: var(--amber); }
.card span, .card small { display: block; color: var(--muted); }
.card b { display: block; font-size: 22px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}
.metrics div {
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 9px 10px;
}
.metrics b, .metrics span { display: block; }
.metrics span { color: var(--muted); }
.tabs {
  display: flex;
  gap: 8px;
  margin: 12px 0;
}
.tab {
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 14px;
  background: white;
  color: var(--ink);
  font: inherit;
  font-weight: 800;
  cursor: pointer;
}
.tab.active {
  border-color: var(--green);
  background: var(--green);
  color: white;
}
.toolbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 190px auto;
  gap: 10px;
  align-items: center;
  background: rgba(245, 247, 249, 0.96);
  border-bottom: 1px solid var(--line);
  padding: 10px 0 12px;
}
input, select {
  width: 100%;
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 8px 10px;
  background: white;
  color: var(--ink);
  font: inherit;
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.frame-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(295px, 1fr));
  gap: 12px;
}
.shot-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.frame-card, .shot-card, .panel {
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.content { padding: 11px 12px; }
.title-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: flex-start;
}
.title-row b { word-break: break-word; }
.title-row span { color: var(--green); font-weight: 800; white-space: nowrap; }
.image-wrap { background: #e6edf2; }
.image-wrap img {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: cover;
}
.image-missing {
  display: grid;
  place-items: center;
  min-height: 150px;
  aspect-ratio: 16 / 9;
  padding: 14px;
  background: #e6edf2;
  color: var(--muted);
  text-align: center;
  word-break: break-word;
}
.thumb-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 6px;
}
.thumb-strip .image-wrap img, .thumb-strip .image-missing {
  border-radius: 6px;
  min-height: 80px;
}
.chips { display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0; }
.chip {
  display: inline-flex;
  border-radius: 999px;
  background: var(--soft);
  padding: 3px 8px;
  color: #34465a;
  font-size: 12px;
  white-space: nowrap;
}
.chip.good { background: var(--green-soft); color: #075e58; }
.chip.info, .chip.role { background: var(--blue-soft); color: #174f7a; }
.chip.warn, .chip.state { background: var(--amber-soft); color: var(--amber); }
.chip.bad { background: var(--red-soft); color: var(--red); }
.caption {
  margin: 8px 0 10px;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
details { margin-top: 8px; }
summary {
  cursor: pointer;
  color: var(--green);
  font-weight: 800;
}
.detail-grid, .shot-detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.shot-head {
  display: grid;
  grid-template-columns: minmax(260px, 0.8fr) minmax(0, 1.4fr) minmax(260px, 0.9fr);
  gap: 14px;
  padding: 12px;
  align-items: start;
}
.box {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fbfdfe;
  min-width: 0;
}
.box.audio { border-top: 3px solid var(--amber); }
.box.event { border-top: 3px solid var(--blue); }
.text-block { margin: 8px 0; }
.text-block b, .trake b { display: block; margin-bottom: 3px; }
.kv {
  width: 100%;
  border-collapse: collapse;
}
.kv th, .kv td {
  text-align: left;
  vertical-align: top;
  border-bottom: 1px solid #e8eef3;
  padding: 7px 8px;
}
.kv th {
  width: 150px;
  color: var(--muted);
}
pre {
  max-height: 460px;
  overflow: auto;
  border-radius: 8px;
  background: #0f172a;
  color: #e2e8f0;
  padding: 12px;
}
.panel { padding: 14px; }
.hidden { display: none !important; }
@media (max-width: 1050px) {
  .app-header, .toolbar, .shot-head, .detail-grid, .shot-detail-grid { grid-template-columns: 1fr; }
  .app-header { align-items: stretch; }
}
"""


def _js() -> str:
    return """
const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.tab-panel');
const search = document.getElementById('search');
const filter = document.getElementById('filter');
const visibleCount = document.getElementById('visibleCount');

function activePanel() {
  return document.querySelector('.tab-panel.active');
}

function rows() {
  const panel = activePanel();
  if (!panel) return [];
  return Array.from(panel.querySelectorAll('.row'));
}

function applyFilter() {
  const q = (search.value || '').trim().toLowerCase();
  const f = filter.value || '';
  let shown = 0;
  for (const row of rows()) {
    const haystack = row.dataset.search || '';
    const features = row.dataset.features || '';
    const okSearch = !q || haystack.includes(q);
    const okFilter = !f || features.split(/\\s+/).includes(f);
    row.classList.toggle('hidden', !(okSearch && okFilter));
    if (okSearch && okFilter) shown += 1;
  }
  visibleCount.textContent = `${shown} visible`;
}

for (const tab of tabs) {
  tab.addEventListener('click', () => {
    for (const item of tabs) item.classList.remove('active');
    for (const panel of panels) panel.classList.remove('active');
    tab.classList.add('active');
    document.getElementById(tab.dataset.tabTarget).classList.add('active');
    search.value = '';
    filter.value = '';
    applyFilter();
  });
}
search.addEventListener('input', applyFilter);
filter.addEventListener('change', applyFilter);
applyFilter();
"""
