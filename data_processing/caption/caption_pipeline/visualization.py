"""HTML visualization for caption pipeline outputs."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl, write_text


def write_caption_visualization(
    *,
    caption_dir: str | Path,
    video_id: str,
    output_html: str | Path | None = None,
    keyframes_root: str | Path | None = None,
    max_frames_per_shot: int = 3,
) -> Path:
    """Write an interactive HTML visualization for one caption output folder."""
    base_dir = Path(caption_dir) / video_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Caption video directory not found: {base_dir}")

    paths = _default_paths(base_dir)
    required = ["frame_index", "shot_index", "event_step_index", "compact_index"]
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError("Missing required caption output files: " + ", ".join(missing))

    frame_index = read_jsonl(paths["frame_index"])
    shot_index = read_jsonl(paths["shot_index"])
    event_index = read_jsonl(paths["event_step_index"])
    compact_docs = read_jsonl(paths["compact_index"])
    reports = _read_reports(paths)

    if output_html is None:
        output_path = base_dir / "reports" / "caption_pipeline_visualization.html"
    else:
        output_path = Path(output_html)

    html_text = render_caption_visualization_html(
        video_id=video_id,
        base_dir=base_dir,
        frame_index=frame_index,
        shot_index=shot_index,
        event_index=event_index,
        compact_docs=compact_docs,
        reports=reports,
        output_path=output_path,
        keyframes_root=Path(keyframes_root) if keyframes_root else None,
        max_frames_per_shot=max_frames_per_shot,
    )
    write_text(output_path, html_text)
    return output_path


def render_caption_visualization_html(
    *,
    video_id: str,
    base_dir: Path,
    frame_index: list[dict],
    shot_index: list[dict],
    event_index: list[dict],
    compact_docs: list[dict],
    reports: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    max_frames_per_shot: int,
) -> str:
    """Render a self-contained HTML page."""
    frame_map = {
        row.get("canonical_frame_id") or row.get("frame_id"): row
        for row in frame_index
        if row.get("canonical_frame_id") or row.get("frame_id")
    }
    event_map = {
        row.get("shot_id"): row
        for row in event_index
        if row.get("shot_id")
    }
    summary = _build_summary(frame_index, shot_index, event_index, compact_docs, reports)
    cards = "\n".join(
        _summary_card(label, value, hint)
        for label, value, hint in summary["cards"]
    )
    warnings = _render_warnings(summary["warnings"])
    mode_blocks = _render_mode_blocks(summary)
    shot_cards = "\n".join(
        _render_shot_card(
            shot=shot,
            event=event_map.get(shot.get("shot_id")),
            frame_map=frame_map,
            output_path=output_path,
            keyframes_root=keyframes_root,
            max_frames=max_frames_per_shot,
        )
        for shot in shot_index
    )
    raw_report_json = _escape(json.dumps(summary["report_metrics"], ensure_ascii=False, indent=2))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Caption Pipeline Visualization - {_escape(video_id)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d8dee9;
      --accent: #0f766e;
      --accent-soft: #dff5f1;
      --warn: #b45309;
      --warn-soft: #fff4d6;
      --bad: #b91c1c;
      --bad-soft: #fee2e2;
      --chip: #eef2f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(246, 248, 251, 0.94);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--line);
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
    h1 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 12px;
      margin-top: 14px;
      align-items: center;
    }}
    input, select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .card, .shot {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .card {{ padding: 14px; }}
    .card .value {{ font-size: 24px; font-weight: 700; }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .grid-2 {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .mode-row {{ display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid #edf0f5; padding: 6px 0; }}
    .mode-row:last-child {{ border-bottom: 0; }}
    .timeline {{ display: flex; flex-direction: column; gap: 12px; }}
    .shot {{ overflow: hidden; }}
    .shot-head {{
      display: grid;
      grid-template-columns: 190px 1fr auto;
      gap: 14px;
      align-items: start;
      padding: 14px;
      border-left: 5px solid var(--accent);
      border-bottom: 1px solid var(--line);
    }}
    .shot-id {{ font-weight: 700; }}
    .time {{ color: var(--accent); font-weight: 700; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      max-width: 100%;
      border-radius: 999px;
      background: var(--chip);
      padding: 3px 8px;
      font-size: 12px;
      color: #334155;
      white-space: nowrap;
    }}
    .chip.warn {{ background: var(--warn-soft); color: var(--warn); }}
    .chip.bad {{ background: var(--bad-soft); color: var(--bad); }}
    .body {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 14px; padding: 14px; }}
    .text-block {{ margin: 0 0 10px; }}
    .text-block b {{ display: block; margin-bottom: 3px; color: #0f172a; }}
    .evidence {{
      display: grid;
      gap: 10px;
      color: #334155;
    }}
    .thumbs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-top: 10px; }}
    .thumb {{
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #f8fafc;
    }}
    .thumb img {{ width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; background: #e2e8f0; }}
    .thumb .cap {{ padding: 7px; font-size: 12px; color: var(--muted); word-break: break-word; }}
    details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 600; }}
    pre {{
      overflow: auto;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 8px;
      padding: 12px;
      max-height: 360px;
    }}
    .hidden {{ display: none; }}
    mark {{ background: #fef08a; padding: 0 2px; }}
    @media (max-width: 900px) {{
      .toolbar, .shot-head, .body, .grid-2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>Caption Pipeline Visualization: {_escape(video_id)}</h1>
    <div class="subtle">Source: {_escape(str(base_dir))}</div>
    <div class="toolbar">
      <input id="filter" placeholder="Filter by caption, OCR, audio, object, shot id...">
      <select id="modeFilter" aria-label="Mode filter">
        <option value="">All modes</option>
        <option value="fallback">Fallback only</option>
        <option value="llm">LLM only</option>
        <option value="template">Template only</option>
      </select>
      <select id="eventFilter" aria-label="Event filter">
        <option value="">All events</option>
        <option value="change">Temporal change</option>
        <option value="completion">Completion</option>
        <option value="beginning">Beginning</option>
        <option value="unknown">Unknown role</option>
      </select>
    </div>
  </div>
</header>
<main class="wrap">
  <section class="cards">{cards}</section>
  {warnings}
  <section class="grid-2">{mode_blocks}</section>
  <section>
    <h2>Shot Timeline</h2>
    <div id="timeline" class="timeline">{shot_cards}</div>
  </section>
  <section>
    <h2>Report Metrics</h2>
    <pre>{raw_report_json}</pre>
  </section>
</main>
<script>
const filter = document.getElementById('filter');
const modeFilter = document.getElementById('modeFilter');
const eventFilter = document.getElementById('eventFilter');
const cards = Array.from(document.querySelectorAll('.shot'));

function applyFilter() {{
  const q = filter.value.trim().toLowerCase();
  const mode = modeFilter.value;
  const role = eventFilter.value;
  let visible = 0;
  for (const card of cards) {{
    const textOk = !q || card.dataset.search.includes(q);
    const modeOk = !mode || card.dataset.modes.includes(mode);
    const roleOk = !role || card.dataset.role === role;
    const show = textOk && modeOk && roleOk;
    card.classList.toggle('hidden', !show);
    if (show) visible += 1;
  }}
  document.title = `Caption Pipeline Visualization - ${{visible}}/${{cards.length}} shots`;
}}
filter.addEventListener('input', applyFilter);
modeFilter.addEventListener('change', applyFilter);
eventFilter.addEventListener('change', applyFilter);
</script>
</body>
</html>
"""


def _default_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "frame_index": base_dir / "captions" / "frame_index.jsonl",
        "shot_index": base_dir / "captions" / "shot_index.jsonl",
        "event_step_index": base_dir / "captions" / "event_step_index.jsonl",
        "compact_index": base_dir / "indexes" / "compact_search_index.jsonl",
        "frame_report": base_dir / "reports" / "frame_caption_report.json",
        "shot_report": base_dir / "reports" / "shot_caption_report.json",
        "event_report": base_dir / "reports" / "event_step_report.json",
        "retrieval_report": base_dir / "eval" / "retrieval_eval_report.json",
    }


def _read_reports(paths: dict[str, Path]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name in ["frame_report", "shot_report", "event_report", "retrieval_report"]:
        path = paths[name]
        if path.exists():
            reports[name] = read_json(path)
    return reports


def _build_summary(
    frame_index: list[dict],
    shot_index: list[dict],
    event_index: list[dict],
    compact_docs: list[dict],
    reports: dict[str, Any],
) -> dict[str, Any]:
    frame_modes = Counter((row.get("quality") or {}).get("caption_mode", "") for row in frame_index)
    shot_modes = Counter((row.get("quality") or {}).get("caption_mode", "") for row in shot_index)
    event_modes = Counter((row.get("quality") or {}).get("event_mode", "") for row in event_index)
    action_states = Counter(row.get("action_state", "unknown") for row in event_index)
    temporal_roles = Counter(row.get("temporal_role", "unknown") for row in event_index)
    frame_fallback = sum(1 for row in frame_index if (row.get("quality") or {}).get("fallback_used"))
    shot_fallback = sum(1 for row in shot_index if (row.get("quality") or {}).get("fallback_used"))
    event_fallback = sum(1 for row in event_index if (row.get("quality") or {}).get("fallback_used"))
    warning_rows = [
        (row.get("shot_id") or row.get("canonical_frame_id") or row.get("event_id"), (row.get("quality") or {}).get("warnings") or [])
        for row in frame_index + shot_index + event_index
        if (row.get("quality") or {}).get("warnings")
    ]
    report_metrics = {
        name: _compact_report_metrics(report)
        for name, report in reports.items()
    }
    return {
        "cards": [
            ("Frames", len(frame_index), f"{frame_fallback} fallback"),
            ("Shots", len(shot_index), f"{shot_fallback} fallback"),
            ("Event steps", len(event_index), f"{event_fallback} fallback"),
            ("Search docs", len(compact_docs), "compact index"),
            ("Frame modes", _counter_text(frame_modes), "caption_mode"),
            ("Shot modes", _counter_text(shot_modes), "caption_mode"),
            ("Event modes", _counter_text(event_modes), "event_mode"),
            ("Warnings", len(warning_rows), "quality warnings"),
        ],
        "frame_modes": frame_modes,
        "shot_modes": shot_modes,
        "event_modes": event_modes,
        "action_states": action_states,
        "temporal_roles": temporal_roles,
        "warnings": warning_rows[:80],
        "report_metrics": report_metrics,
    }


def _compact_report_metrics(report: Any) -> Any:
    if not isinstance(report, dict):
        return report
    keep = {}
    for key, value in report.items():
        if key.endswith("_warnings") or key in {"frame_warnings", "summary_warnings", "warnings"}:
            keep[key] = value[:20] if isinstance(value, list) else value
        elif isinstance(value, (int, float, str, bool)) or key in {"coverage", "caption_mode_counts", "event_mode_counts"}:
            keep[key] = value
    return keep


def _render_mode_blocks(summary: dict[str, Any]) -> str:
    blocks = [
        ("Caption Modes", summary["frame_modes"] + summary["shot_modes"]),
        ("Event Modes", summary["event_modes"]),
        ("Action States", summary["action_states"]),
        ("Temporal Roles", summary["temporal_roles"]),
    ]
    return "\n".join(
        f"""<div class="panel"><h2>{_escape(title)}</h2>{_render_counter(counter)}</div>"""
        for title, counter in blocks
    )


def _render_counter(counter: Counter) -> str:
    if not counter:
        return '<div class="subtle">No data</div>'
    return "\n".join(
        f"""<div class="mode-row"><span>{_escape(str(key or "empty"))}</span><b>{count}</b></div>"""
        for key, count in counter.most_common()
    )


def _summary_card(label: str, value: Any, hint: str) -> str:
    return f"""<article class="card">
  <div class="label">{_escape(label)}</div>
  <div class="value">{_escape(str(value))}</div>
  <div class="subtle">{_escape(hint)}</div>
</article>"""


def _render_warnings(warnings: list[tuple[str, list[str]]]) -> str:
    if not warnings:
        return ""
    rows = "\n".join(
        f"""<div class="mode-row"><span>{_escape(str(item_id))}</span><span>{_escape("; ".join(map(str, values)))}</span></div>"""
        for item_id, values in warnings
    )
    return f"""<section class="panel" style="margin-top: 14px">
  <h2>Warnings</h2>
  {rows}
</section>"""


def _render_shot_card(
    *,
    shot: dict,
    event: dict | None,
    frame_map: dict[str, dict],
    output_path: Path,
    keyframes_root: Path | None,
    max_frames: int,
) -> str:
    event = event or {}
    shot_id = shot.get("shot_id", "")
    quality = shot.get("quality") or {}
    event_quality = event.get("quality") or {}
    evidence = shot.get("evidence_text") or {}
    source = event.get("source_text") or {}
    role = event.get("temporal_role", "unknown")
    modes = " ".join(
        str(v)
        for v in [
            quality.get("caption_mode", ""),
            event_quality.get("event_mode", ""),
            "fallback" if quality.get("fallback_used") or event_quality.get("fallback_used") else "",
        ]
        if v
    )
    search_text = " ".join(
        str(v)
        for v in [
            shot_id,
            shot.get("caption_text", ""),
            shot.get("temporal_caption", ""),
            event.get("event_caption", ""),
            event.get("trake_text", ""),
            evidence.get("merged_ocr_text", ""),
            evidence.get("merged_audio_text", ""),
            evidence.get("object_text", ""),
            evidence.get("scene_text", ""),
        ]
        if v
    ).lower()
    chips = [
        ("caption:" + str(quality.get("caption_mode", "unknown")), _chip_class(quality)),
        ("event:" + str(event_quality.get("event_mode", "unknown")), _chip_class(event_quality)),
        ("state:" + str(event.get("action_state", "unknown")), ""),
        ("role:" + str(role), ""),
        (f"frames:{shot.get('frame_count', 0)}", ""),
    ]
    thumb_html = _render_thumbnails(
        shot=shot,
        frame_map=frame_map,
        output_path=output_path,
        keyframes_root=keyframes_root,
        max_frames=max_frames,
    )
    objects = evidence.get("object_text", "") or source.get("object_text", "")
    scene = evidence.get("scene_text", "") or source.get("scene_text", "")
    ocr = evidence.get("merged_ocr_text", "") or source.get("merged_ocr_text", "")
    audio = evidence.get("merged_audio_text", "") or source.get("merged_audio_text", "")
    warnings = (quality.get("warnings") or []) + (event_quality.get("warnings") or [])
    warn_html = ""
    if warnings:
        warn_html = f"""<p class="text-block"><b>Warnings</b>{_escape("; ".join(map(str, warnings)))}</p>"""
    return f"""<article class="shot" data-search="{_escape_attr(search_text)}" data-modes="{_escape_attr(modes)}" data-role="{_escape_attr(str(role))}">
  <div class="shot-head">
    <div>
      <div class="shot-id">{_escape(str(shot_id))}</div>
      <div class="time">{_fmt_time(shot.get("start_sec"))} - {_fmt_time(shot.get("end_sec"))}</div>
      <div class="chips">{''.join(_chip(label, cls) for label, cls in chips)}</div>
    </div>
    <div>
      <p class="text-block"><b>Shot caption</b>{_escape(shot.get("caption_text", ""))}</p>
      <p class="text-block"><b>Temporal caption</b>{_escape(shot.get("temporal_caption", ""))}</p>
    </div>
    <div class="subtle">step {_escape(str(event.get("step_order", "")))}</div>
  </div>
  <div class="body">
    <div>
      <p class="text-block"><b>Event caption</b>{_escape(event.get("event_caption", ""))}</p>
      <p class="text-block"><b>TRAKE text</b>{_escape(event.get("trake_text", ""))}</p>
      {warn_html}
      {thumb_html}
    </div>
    <div class="evidence">
      {_evidence_block("Objects", objects)}
      {_evidence_block("Scene", scene)}
      {_evidence_block("OCR", ocr)}
      {_evidence_block("Audio", audio)}
      <details>
        <summary>Raw event JSON</summary>
        <pre>{_escape(json.dumps(event, ensure_ascii=False, indent=2))}</pre>
      </details>
    </div>
  </div>
</article>"""


def _render_thumbnails(
    *,
    shot: dict,
    frame_map: dict[str, dict],
    output_path: Path,
    keyframes_root: Path | None,
    max_frames: int,
) -> str:
    frame_ids = (shot.get("representative_frame_ids") or [])[:max(max_frames, 0)]
    if not frame_ids:
        return ""
    thumbs = []
    for frame_id in frame_ids:
        frame = frame_map.get(frame_id, {})
        caption = frame.get("caption_text", "")
        src = _image_src(frame, output_path, keyframes_root)
        image = f'<img src="{_escape_attr(src)}" alt="{_escape_attr(frame_id)}">' if src else ""
        thumbs.append(
            f"""<div class="thumb">{image}<div class="cap"><b>{_escape(frame_id)}</b><br>{_escape(caption)}</div></div>"""
        )
    return f"""<div class="thumbs">{''.join(thumbs)}</div>"""


def _image_src(frame: dict, output_path: Path, keyframes_root: Path | None) -> str:
    raw = frame.get("image_path") or ""
    rel = frame.get("image_relpath") or ""
    video_id = frame.get("video_id", "")
    candidates = []
    if raw:
        candidates.append(Path(raw))
    if keyframes_root and rel:
        rel_path = Path(rel)
        candidates.append(keyframes_root / rel_path)
        if video_id and (not rel_path.parts or rel_path.parts[0] != video_id):
            candidates.append(keyframes_root / str(video_id) / rel_path)
    if keyframes_root and frame.get("frame_id"):
        candidates.append(keyframes_root / str(video_id) / str(frame["frame_id"]))
    for path in candidates:
        if path.exists():
            try:
                return path.resolve().relative_to(output_path.parent.resolve()).as_posix()
            except ValueError:
                return path.resolve().as_uri()
    return ""


def _evidence_block(label: str, text: Any) -> str:
    text = str(text or "")
    if not text:
        return f"""<p class="text-block subtle"><b>{_escape(label)}</b>empty</p>"""
    return f"""<p class="text-block"><b>{_escape(label)}</b>{_escape(text)}</p>"""


def _chip(label: str, class_name: str = "") -> str:
    class_attr = f" {class_name}" if class_name else ""
    return f"""<span class="chip{class_attr}">{_escape(label)}</span>"""


def _chip_class(quality: dict) -> str:
    if quality.get("fallback_used"):
        return "bad"
    if quality.get("warnings"):
        return "warn"
    return ""


def _counter_text(counter: Counter) -> str:
    if not counter:
        return "0"
    return ", ".join(f"{key or 'empty'}:{value}" for key, value in counter.most_common())


def _fmt_time(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = 0.0
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=False)


def _escape_attr(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)
