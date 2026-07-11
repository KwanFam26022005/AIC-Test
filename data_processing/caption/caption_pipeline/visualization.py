"""HTML visualization for caption pipeline outputs."""

from __future__ import annotations

import base64
import html
import json
import mimetypes
import shutil
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
    image_mode: str = "embed",
) -> Path:
    """Write an interactive HTML visualization for one caption output folder."""
    if image_mode not in {"embed", "copy", "link"}:
        raise ValueError("image_mode must be one of: embed, copy, link")
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
    frame_evidence = read_jsonl(paths["frame_evidence"]) if paths["frame_evidence"].exists() else []
    shot_evidence = read_jsonl(paths["shot_evidence"]) if paths["shot_evidence"].exists() else []
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
        frame_evidence=frame_evidence,
        shot_evidence=shot_evidence,
        reports=reports,
        output_path=output_path,
        keyframes_root=Path(keyframes_root) if keyframes_root else None,
        max_frames_per_shot=max_frames_per_shot,
        image_mode=image_mode,
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
    frame_evidence: list[dict],
    shot_evidence: list[dict],
    reports: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    max_frames_per_shot: int,
    image_mode: str,
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
    frame_evidence_map = {
        row.get("canonical_frame_id") or row.get("frame_id"): row
        for row in frame_evidence
        if row.get("canonical_frame_id") or row.get("frame_id")
    }
    shot_evidence_map = {
        row.get("shot_id"): row for row in shot_evidence if row.get("shot_id")
    }
    shot_frames = _frames_by_shot(shot_index, frame_index)
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
            frame_evidence_map=frame_evidence_map,
            shot_evidence=shot_evidence_map.get(shot.get("shot_id"), {}),
            shot_frames=shot_frames.get(shot.get("shot_id"), []),
            output_path=output_path,
            keyframes_root=keyframes_root,
            max_frames=max_frames_per_shot,
            image_mode=image_mode,
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
      --bg: #f3f6f8;
      --panel: #ffffff;
      --ink: #15212b;
      --muted: #627181;
      --line: #d5dde3;
      --accent: #087f76;
      --accent-soft: #def4f1;
      --blue: #1769aa;
      --blue-soft: #e5f1fb;
      --violet: #7051a6;
      --violet-soft: #f0ebf8;
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
      z-index: 20;
      background: rgba(243, 246, 248, 0.96);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--line);
    }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 18px 22px; }}
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
    .card {{ padding: 14px; border-top: 3px solid var(--accent); }}
    .card:nth-child(3n+2) {{ border-top-color: var(--blue); }}
    .card:nth-child(3n) {{ border-top-color: var(--warn); }}
    .card .value {{ font-size: 24px; font-weight: 700; }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .grid-2 {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .mode-row {{ display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid #edf0f5; padding: 6px 0; }}
    .mode-row:last-child {{ border-bottom: 0; }}
    .timeline {{ display: flex; flex-direction: column; gap: 12px; }}
    .shot {{ overflow: hidden; scroll-margin-top: 150px; }}
    .shot-head {{
      display: grid;
      grid-template-columns: minmax(210px, .65fr) minmax(0, 1.7fr) auto;
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
    .shot-main {{ padding: 0 16px 16px; }}
    .tabbar {{
      display: flex;
      gap: 4px;
      overflow-x: auto;
      padding: 10px 16px 0;
      border-bottom: 1px solid var(--line);
      background: #f8fafb;
    }}
    .tab-button {{
      flex: 0 0 auto;
      border: 0;
      border-bottom: 3px solid transparent;
      background: transparent;
      color: var(--muted);
      padding: 9px 12px 8px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .tab-button.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
    .tab-pane {{ display: none; padding-top: 14px; }}
    .tab-pane.active {{ display: block; }}
    .overview-grid {{ display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, .8fr); gap: 18px; }}
    .caption-stack {{ border-left: 3px solid var(--blue); padding-left: 12px; }}
    .text-block {{ margin: 0 0 10px; }}
    .text-block b {{ display: block; margin-bottom: 3px; color: #0f172a; }}
    .feature-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }}
    .feature {{ background: white; padding: 13px; min-width: 0; }}
    .feature h3 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; }}
    .feature.ocr {{ border-top: 3px solid var(--violet); }}
    .feature.audio {{ border-top: 3px solid var(--warn); }}
    .feature.objects {{ border-top: 3px solid var(--accent); }}
    .feature.scene {{ border-top: 3px solid var(--blue); }}
    .thumbs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
    .thumb {{
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #f8fafc;
    }}
    .thumb img {{ width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; background: #e2e8f0; }}
    .thumb .cap {{ padding: 8px; font-size: 12px; color: var(--muted); word-break: break-word; }}
    .image-missing {{ aspect-ratio: 16/9; display: grid; place-items: center; padding: 14px; text-align: center; color: var(--muted); background: #e8edf1; }}
    .frame-list {{ border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }}
    .frame-row {{ display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 14px; padding: 12px; border-bottom: 1px solid var(--line); }}
    .frame-row:last-child {{ border-bottom: 0; }}
    .frame-meta {{ color: var(--muted); font-size: 12px; }}
    .frame-features {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 16px; margin-top: 8px; }}
    .metric-table {{ width: 100%; border-collapse: collapse; }}
    .metric-table th, .metric-table td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid #e9eef2; vertical-align: top; }}
    .metric-table th {{ width: 190px; color: var(--muted); font-weight: 600; }}
    .tag-list {{ display: flex; flex-wrap: wrap; gap: 5px; }}
    .tag {{ background: var(--blue-soft); color: #174f7a; border-radius: 4px; padding: 2px 6px; font-size: 12px; }}
    .tag.object {{ background: var(--accent-soft); color: #075e58; }}
    .tag.ocr {{ background: var(--violet-soft); color: #563d82; }}
    .empty {{ color: var(--muted); font-style: italic; }}
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
      .toolbar, .shot-head, .overview-grid, .grid-2, .feature-grid, .frame-row, .frame-features {{ grid-template-columns: 1fr; }}
      .frame-row {{ gap: 8px; }}
    }}
  </style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>Caption Pipeline Visualization: {_escape(video_id)}</h1>
    <div class="subtle">Source: {_escape(str(base_dir))} | images: {_escape(image_mode)}</div>
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
    <h2>Shot Timeline <span id="visibleCount" class="subtle"></span></h2>
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
const visibleCount = document.getElementById('visibleCount');

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
  visibleCount.textContent = `(${{visible}}/${{cards.length}})`;
}}
filter.addEventListener('input', applyFilter);
modeFilter.addEventListener('change', applyFilter);
eventFilter.addEventListener('change', applyFilter);
document.addEventListener('click', (event) => {{
  const button = event.target.closest('.tab-button');
  if (!button) return;
  const shot = button.closest('.shot');
  const tabName = button.dataset.tab;
  shot.querySelectorAll('.tab-button').forEach(item => item.classList.toggle('active', item === button));
  shot.querySelectorAll('.tab-pane').forEach(pane => pane.classList.toggle('active', pane.dataset.pane === tabName));
}});
applyFilter();
</script>
</body>
</html>
"""


def _default_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "frame_index": base_dir / "captions" / "frame_index.jsonl",
        "frame_evidence": base_dir / "evidence" / "frame_evidence.jsonl",
        "shot_evidence": base_dir / "evidence" / "shot_evidence.jsonl",
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
    frame_evidence_map: dict[str, dict],
    shot_evidence: dict,
    shot_frames: list[dict],
    output_path: Path,
    keyframes_root: Path | None,
    max_frames: int,
    image_mode: str,
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
    representative_frames = []
    for frame_id in shot.get("representative_frame_ids") or []:
        if frame_id in frame_map:
            representative_frames.append(frame_map[frame_id])
    if not representative_frames:
        representative_frames = shot_frames
    thumb_html = _render_thumbnails(
        frames=representative_frames[:max(max_frames, 0)],
        frame_evidence_map=frame_evidence_map,
        output_path=output_path,
        keyframes_root=keyframes_root,
        image_mode=image_mode,
    )
    warnings = (quality.get("warnings") or []) + (event_quality.get("warnings") or [])
    warn_html = ""
    if warnings:
        warn_html = f"""<p class="text-block"><b>Warnings</b>{_escape("; ".join(map(str, warnings)))}</p>"""
    frame_rows = _render_frame_rows(shot_frames, frame_evidence_map)
    feature_html = _render_feature_overview(shot_evidence, evidence, source)
    event_html = _render_event_detail(event)
    diagnostics_html = _render_diagnostics(shot, event, shot_frames)
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
  <div class="tabbar" role="tablist">
    <button class="tab-button active" data-tab="overview" type="button">Overview</button>
    <button class="tab-button" data-tab="frames" type="button">Frames ({len(shot_frames)})</button>
    <button class="tab-button" data-tab="features" type="button">Features</button>
    <button class="tab-button" data-tab="event" type="button">Event / TRAKE</button>
    <button class="tab-button" data-tab="diagnostics" type="button">Search &amp; Quality</button>
  </div>
  <div class="shot-main">
    <section class="tab-pane active" data-pane="overview">
      <div class="overview-grid">
        <div>{thumb_html}</div>
        <div class="caption-stack">
          <p class="text-block"><b>Event caption</b>{_value(event.get("event_caption"))}</p>
          <p class="text-block"><b>TRAKE text</b>{_value(event.get("trake_text"))}</p>
          {warn_html}
        </div>
      </div>
    </section>
    <section class="tab-pane" data-pane="frames">{frame_rows}</section>
    <section class="tab-pane" data-pane="features">{feature_html}</section>
    <section class="tab-pane" data-pane="event">{event_html}</section>
    <section class="tab-pane" data-pane="diagnostics">{diagnostics_html}</section>
  </div>
</article>"""


def _render_thumbnails(
    *,
    frames: list[dict],
    frame_evidence_map: dict[str, dict],
    output_path: Path,
    keyframes_root: Path | None,
    image_mode: str,
) -> str:
    if not frames:
        return '<div class="image-missing">No representative frames</div>'
    thumbs = []
    for frame in frames:
        frame_id = frame.get("canonical_frame_id") or frame.get("frame_id") or "unknown"
        resolved_frame = dict(frame)
        frame_evidence = frame_evidence_map.get(frame_id, {})
        for key in (
            "image_path",
            "image_relpath",
            "frame_name",
            "keyframe_idx",
            "source_frame_idx",
        ):
            if resolved_frame.get(key) in (None, "") and frame_evidence.get(key) not in (None, ""):
                resolved_frame[key] = frame_evidence[key]
        caption = frame.get("caption_text", "")
        src, image_status = _image_src(
            resolved_frame, output_path, keyframes_root, image_mode,
        )
        if src:
            image = f'<img src="{_escape_attr(src)}" alt="{_escape_attr(frame_id)}" loading="lazy">'
        else:
            image = f'<div class="image-missing">Image unavailable<br>{_escape(image_status)}</div>'
        thumbs.append(
            f"""<div class="thumb">{image}<div class="cap"><b>{_escape(frame_id)}</b> &middot; {_fmt_time(frame.get("timestamp_sec"))}<br>{_value(caption)}</div></div>"""
        )
    return f"""<div class="thumbs">{''.join(thumbs)}</div>"""


def _image_src(
    frame: dict,
    output_path: Path,
    keyframes_root: Path | None,
    image_mode: str,
) -> tuple[str, str]:
    path, candidates = _resolve_image_path(frame, keyframes_root)
    if path is None:
        sample = ", ".join(str(item) for item in candidates[:3])
        return "", f"checked: {sample}" if sample else "no image metadata"

    if image_mode == "embed":
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}", str(path)

    if image_mode == "copy":
        assets_dir = output_path.parent / f"{output_path.stem}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        frame_id = str(frame.get("canonical_frame_id") or frame.get("frame_id") or path.stem)
        target = assets_dir / f"{frame_id}{path.suffix.lower()}"
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
        return target.relative_to(output_path.parent).as_posix(), str(path)

    try:
        return path.resolve().relative_to(output_path.parent.resolve()).as_posix(), str(path)
    except ValueError:
        return path.resolve().as_uri(), str(path)


def _resolve_image_path(frame: dict, keyframes_root: Path | None) -> tuple[Path | None, list[Path]]:
    video_id = str(frame.get("video_id") or "")
    raw = str(frame.get("image_path") or "")
    rel = str(frame.get("image_relpath") or "")
    frame_id = str(frame.get("frame_id") or frame.get("canonical_frame_id") or "")
    canonical_id = str(frame.get("canonical_frame_id") or frame_id)
    frame_name = str(frame.get("frame_name") or "")
    names = [frame_name, Path(rel).name if rel else ""]
    if frame_id:
        names.extend([frame_id, f"{frame_id}.jpg", f"{frame_id}.png"])
    suffix = canonical_id.rsplit("_", 1)[-1] if canonical_id else ""
    if suffix:
        names.extend([f"{suffix}.jpg", f"{suffix}.png"])
    keyframe_idx = frame.get("keyframe_idx")
    if isinstance(keyframe_idx, int):
        names.extend([f"{keyframe_idx:03d}.jpg", f"{keyframe_idx:06d}.jpg"])

    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
    if keyframes_root:
        if rel:
            rel_path = Path(rel)
            candidates.extend([keyframes_root / rel_path, keyframes_root / video_id / rel_path.name])
        for name in names:
            if name:
                candidates.append(keyframes_root / video_id / name)
                candidates.append(keyframes_root / name)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
            if candidate.is_file():
                return candidate, unique
    return None, unique


def _frames_by_shot(shot_index: list[dict], frame_index: list[dict]) -> dict[str, list[dict]]:
    frames = sorted(
        frame_index,
        key=lambda row: (
            float(row.get("timestamp_sec") or 0),
            str(row.get("canonical_frame_id") or ""),
        ),
    )
    result: dict[str, list[dict]] = {}
    for shot in shot_index:
        start = float(shot.get("start_sec") or 0)
        end = float(shot.get("end_sec") or start)
        result[str(shot.get("shot_id") or "")] = [
            frame for frame in frames
            if start - 0.001 <= float(frame.get("timestamp_sec") or 0) <= end + 0.001
        ]
    return result


def _render_frame_rows(frames: list[dict], frame_evidence_map: dict[str, dict]) -> str:
    if not frames:
        return '<div class="empty">No frame records mapped to this shot.</div>'
    rows = []
    for frame in frames:
        frame_id = str(frame.get("canonical_frame_id") or frame.get("frame_id") or "unknown")
        evidence = frame_evidence_map.get(frame_id, {})
        obj = evidence.get("object_evidence") or {}
        ocr = evidence.get("ocr_evidence") or {}
        audio = evidence.get("audio_evidence") or {}
        quality = frame.get("quality") or {}
        rows.append(f"""<div class="frame-row">
  <div>
    <b>{_escape(frame_id)}</b>
    <div class="frame-meta">{_fmt_time(frame.get("timestamp_sec"))} | mode: {_escape(quality.get("caption_mode", "unknown"))}</div>
    <div class="chips">{_boolean_chips(evidence.get("quality") or {})}</div>
  </div>
  <div>
    <p class="text-block"><b>Frame caption</b>{_value(frame.get("caption_text"))}</p>
    <div class="frame-features">
      <div><b>Objects</b><br>{_tags(_object_items(obj), "object")}</div>
      <div><b>Scene / RAM</b><br>{_tags((obj.get("scene_tags") or []) + (obj.get("ram_tags") or []))}</div>
      <div><b>OCR</b><br>{_value(ocr.get("ocr_text"))}</div>
      <div><b>Audio ({_escape(audio.get("num_segments", 0))} segments)</b><br>{_value(audio.get("audio_text"))}</div>
    </div>
    <details><summary>Frame feature details</summary><pre>{_escape(json.dumps(evidence, ensure_ascii=False, indent=2))}</pre></details>
  </div>
</div>""")
    return f'<div class="frame-list">{"".join(rows)}</div>'


def _render_feature_overview(shot_evidence: dict, evidence: dict, source: dict) -> str:
    counts = shot_evidence.get("merged_object_counts") or {}
    object_items = [
        f"{name} x{count}"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    scene_tags = shot_evidence.get("merged_scene_tags") or []
    objects = evidence.get("object_text") or source.get("object_text") or ""
    scene = evidence.get("scene_text") or source.get("scene_text") or ""
    ocr = evidence.get("merged_ocr_text") or source.get("merged_ocr_text") or ""
    audio = evidence.get("merged_audio_text") or source.get("merged_audio_text") or ""
    return f"""<div class="feature-grid">
  <section class="feature objects"><h3>Object Detection</h3>{_tags(object_items, "object")}<p>{_value(objects)}</p></section>
  <section class="feature scene"><h3>Scene and RAM tags</h3>{_tags(scene_tags)}<p>{_value(scene)}</p></section>
  <section class="feature ocr"><h3>OCR</h3><p>{_value(ocr)}</p></section>
  <section class="feature audio"><h3>Audio / ASR</h3><p>{_value(audio)}</p></section>
</div>"""


def _render_event_detail(event: dict) -> str:
    rows = [
        ("Event caption", event.get("event_caption")),
        ("Current observation", event.get("current_observation")),
        ("Before context", event.get("before_context")),
        ("After context", event.get("after_context")),
        ("Action state", event.get("action_state")),
        ("Temporal role", event.get("temporal_role")),
        ("Actors", ", ".join(map(str, event.get("actors") or []))),
        ("Actions", ", ".join(map(str, event.get("actions") or []))),
        ("Objects involved", ", ".join(map(str, event.get("objects_involved") or []))),
        ("Scene", event.get("scene")),
        ("TRAKE text", event.get("trake_text")),
    ]
    return _table(rows)


def _render_diagnostics(shot: dict, event: dict, frames: list[dict]) -> str:
    quality_rows = [
        ("Shot quality", json.dumps(shot.get("quality") or {}, ensure_ascii=False)),
        ("Event quality", json.dumps(event.get("quality") or {}, ensure_ascii=False)),
    ]
    search_rows = [(name, value) for name, value in (shot.get("search_fields") or {}).items()]
    raw = {"shot": shot, "event": event, "frames": frames}
    return f"""<div class="grid-2">
  <div><h3>Quality and provenance</h3>{_table(quality_rows)}</div>
  <div><h3>Search fields</h3>{_table(search_rows)}</div>
</div>
<details><summary>Raw shot, event and frame JSON</summary><pre>{_escape(json.dumps(raw, ensure_ascii=False, indent=2))}</pre></details>"""


def _table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(f"<tr><th>{_escape(label)}</th><td>{_value(value)}</td></tr>" for label, value in rows)
    return f'<table class="metric-table"><tbody>{body}</tbody></table>'


def _object_items(obj: dict) -> list[str]:
    counts = obj.get("object_counts") or {}
    if counts:
        return [
            f"{name} x{count}"
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    return list(obj.get("important_objects") or obj.get("object_tags") or [])


def _tags(values: list[Any], kind: str = "") -> str:
    clean = [str(value) for value in values if str(value).strip()]
    if not clean:
        return '<span class="empty">none</span>'
    class_name = f"tag {kind}".strip()
    items = "".join(
        '<span class="{}">{}</span>'.format(class_name, _escape(value))
        for value in clean
    )
    return f'<span class="tag-list">{items}</span>'


def _boolean_chips(quality: dict) -> str:
    labels = []
    fields = [
        ("has_object", "objects"),
        ("has_scene_tags", "scene"),
        ("has_ocr", "ocr"),
        ("has_audio", "audio"),
    ]
    for key, label in fields:
        labels.append(_chip(label, "" if quality.get(key) else "warn"))
    return "".join(labels)


def _value(value: Any) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return '<span class="empty">empty</span>'
    return _escape(value)


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
