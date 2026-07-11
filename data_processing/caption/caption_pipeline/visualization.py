"""HTML review pages for caption pipeline outputs."""

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
    image_mode: str = "copy",
) -> Path:
    """Write frame and shot review HTML pages for one caption output folder."""
    if image_mode not in {"embed", "copy", "link"}:
        raise ValueError("image_mode must be one of: embed, copy, link")

    base_dir = Path(caption_dir) / video_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Caption video directory not found: {base_dir}")

    paths = _default_paths(base_dir)
    required = ["frame_index", "shot_index", "event_step_index", "compact_index"]
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required caption output files: " + ", ".join(missing)
        )

    frame_index = read_jsonl(paths["frame_index"])
    shot_index = read_jsonl(paths["shot_index"])
    event_index = read_jsonl(paths["event_step_index"])
    compact_docs = read_jsonl(paths["compact_index"])
    frame_evidence = (
        read_jsonl(paths["frame_evidence"]) if paths["frame_evidence"].exists() else []
    )
    shot_evidence = (
        read_jsonl(paths["shot_evidence"]) if paths["shot_evidence"].exists() else []
    )
    reports = _read_reports(paths)

    if output_html is None:
        output_path = base_dir / "reports" / "visualization" / "index.html"
    else:
        output_path = Path(output_html)

    review_dir = output_path.parent
    frame_path = review_dir / "frame_review.html"
    shot_path = review_dir / "shot_review.html"
    assets_dir = review_dir / "assets" if image_mode == "copy" else None
    keyframes = Path(keyframes_root) if keyframes_root else None

    context = _build_context(
        video_id=video_id,
        base_dir=base_dir,
        frame_index=frame_index,
        shot_index=shot_index,
        event_index=event_index,
        compact_docs=compact_docs,
        frame_evidence=frame_evidence,
        shot_evidence=shot_evidence,
        reports=reports,
    )

    write_text(
        output_path,
        render_visualization_index_html(
            context=context,
            output_path=output_path,
            image_mode=image_mode,
        ),
    )
    write_text(
        frame_path,
        render_frame_review_html(
            context=context,
            output_path=frame_path,
            keyframes_root=keyframes,
            image_mode=image_mode,
            assets_dir=assets_dir,
        ),
    )
    write_text(
        shot_path,
        render_shot_review_html(
            context=context,
            output_path=shot_path,
            keyframes_root=keyframes,
            max_frames_per_shot=max_frames_per_shot,
            image_mode=image_mode,
            assets_dir=assets_dir,
        ),
    )
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
    """Backward-compatible renderer returning the shot review page."""
    context = _build_context(
        video_id=video_id,
        base_dir=base_dir,
        frame_index=frame_index,
        shot_index=shot_index,
        event_index=event_index,
        compact_docs=compact_docs,
        frame_evidence=frame_evidence,
        shot_evidence=shot_evidence,
        reports=reports,
    )
    return render_shot_review_html(
        context=context,
        output_path=output_path,
        keyframes_root=keyframes_root,
        max_frames_per_shot=max_frames_per_shot,
        image_mode=image_mode,
        assets_dir=None,
    )


def render_visualization_index_html(
    *,
    context: dict[str, Any],
    output_path: Path,
    image_mode: str,
) -> str:
    """Render the lightweight landing page for the visualization workspace."""
    summary = context["summary"]
    cards = "\n".join(_summary_card(label, value, hint) for label, value, hint in summary["cards"])
    warnings = _render_warnings(summary["warnings"])
    report_metrics = _escape(json.dumps(summary["report_metrics"], ensure_ascii=False, indent=2))
    video_id = context["video_id"]
    base_dir = context["base_dir"]
    return _page(
        title=f"Caption Review - {video_id}",
        active="index",
        body=f"""
<section class="hero">
  <div>
    <h1>Caption Review: {_escape(video_id)}</h1>
    <p class="subtle">Source: {_escape(str(base_dir))} | images: {_escape(image_mode)}</p>
  </div>
  <nav class="top-links">
    <a href="frame_review.html">Frame Review</a>
    <a href="shot_review.html">Shot Review</a>
  </nav>
</section>
<section class="cards">{cards}</section>
{warnings}
<section class="review-choice">
  <a class="choice frame" href="frame_review.html">
    <span>Frame Review</span>
    <b>Inspect each keyframe image, frame caption, shot mapping, OCR, objects, scene tags and audio evidence.</b>
  </a>
  <a class="choice shot" href="shot_review.html">
    <span>Shot Review</span>
    <b>Inspect the merged shot caption, event caption, TRAKE text, search fields and shot-level evidence.</b>
  </a>
</section>
<section>
  <h2>Report Metrics</h2>
  <pre>{report_metrics}</pre>
</section>""",
    )


def render_frame_review_html(
    *,
    context: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    image_mode: str,
    assets_dir: Path | None,
) -> str:
    """Render the frame-first review page."""
    frames = context["frame_index"]
    frame_cards = "\n".join(
        _render_frame_card(
            frame=frame,
            context=context,
            output_path=output_path,
            keyframes_root=keyframes_root,
            image_mode=image_mode,
            assets_dir=assets_dir,
        )
        for frame in frames
    )
    summary = context["summary"]
    cards = "\n".join(_summary_card(label, value, hint) for label, value, hint in summary["frame_cards"])
    return _page(
        title=f"Frame Review - {context['video_id']}",
        active="frames",
        body=f"""
<section class="hero">
  <div>
    <h1>Frame Review: {_escape(context['video_id'])}</h1>
    <p class="subtle">Click a frame to inspect caption, shot mapping and feature evidence.</p>
  </div>
  <nav class="top-links">
    <a href="index.html">Summary</a>
    <a href="shot_review.html">Shot Review</a>
  </nav>
</section>
<section class="cards compact">{cards}</section>
<section class="toolbar">
  <input id="filter" placeholder="Search frame id, shot id, caption, OCR, audio, object...">
  <select id="featureFilter" aria-label="Feature filter">
    <option value="">All frames</option>
    <option value="missing_ocr">Missing OCR</option>
    <option value="missing_audio">Missing audio</option>
    <option value="missing_object">Missing objects</option>
    <option value="warning">Warnings</option>
  </select>
  <span id="visibleCount" class="subtle"></span>
</section>
<section id="frameGrid" class="frame-grid">{frame_cards}</section>
{_filter_script(".frame-card")}
""",
    )


def render_shot_review_html(
    *,
    context: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    max_frames_per_shot: int,
    image_mode: str,
    assets_dir: Path | None,
) -> str:
    """Render the shot-first review page."""
    shot_cards = "\n".join(
        _render_shot_card(
            shot=shot,
            context=context,
            output_path=output_path,
            keyframes_root=keyframes_root,
            max_frames=max_frames_per_shot,
            image_mode=image_mode,
            assets_dir=assets_dir,
        )
        for shot in context["shot_index"]
    )
    summary = context["summary"]
    cards = "\n".join(_summary_card(label, value, hint) for label, value, hint in summary["shot_cards"])
    return _page(
        title=f"Shot Review - {context['video_id']}",
        active="shots",
        body=f"""
<section class="hero">
  <div>
    <h1>Shot Review: {_escape(context['video_id'])}</h1>
    <p class="subtle">A shot is a short time segment made from nearby frames and merged evidence.</p>
  </div>
  <nav class="top-links">
    <a href="index.html">Summary</a>
    <a href="frame_review.html">Frame Review</a>
  </nav>
</section>
<section class="cards compact">{cards}</section>
<section class="toolbar">
  <input id="filter" placeholder="Search shot id, caption, TRAKE, OCR, audio, object...">
  <select id="featureFilter" aria-label="Feature filter">
    <option value="">All shots</option>
    <option value="missing_ocr">Missing OCR</option>
    <option value="missing_audio">Missing audio</option>
    <option value="missing_object">Missing objects</option>
    <option value="warning">Warnings</option>
  </select>
  <span id="visibleCount" class="subtle"></span>
</section>
<section id="shotList" class="shot-list">{shot_cards}</section>
{_filter_script(".shot-card")}
""",
    )


def _page(*, title: str, active: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7f9;
      --panel: #ffffff;
      --ink: #14212d;
      --muted: #647487;
      --line: #d8e0e7;
      --soft: #edf2f6;
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
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color: inherit; }}
    .wrap {{ max-width: 1540px; margin: 0 auto; padding: 20px 24px 34px; }}
    .hero {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      margin-bottom: 16px;
    }}
    h1 {{ margin: 0; font-size: 25px; letter-spacing: 0; }}
    h2 {{ margin: 24px 0 10px; font-size: 18px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 13px; letter-spacing: 0; color: var(--muted); text-transform: uppercase; }}
    .subtle {{ color: var(--muted); }}
    .top-links {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .top-links a, .choice {{
      text-decoration: none;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 11px;
      font-weight: 700;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 14px 0 16px;
    }}
    .cards.compact {{ grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 3px solid var(--accent);
      border-radius: 8px;
      padding: 12px;
    }}
    .card:nth-child(3n+2) {{ border-top-color: var(--blue); }}
    .card:nth-child(3n) {{ border-top-color: var(--warn); }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .card .value {{ font-size: 22px; font-weight: 800; }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 15;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 190px auto;
      gap: 10px;
      align-items: center;
      background: rgba(245, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 10px 0 12px;
      backdrop-filter: blur(8px);
    }}
    input, select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    .review-choice {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 16px; }}
    .choice {{ display: block; min-height: 130px; padding: 16px; border-top: 4px solid var(--accent); }}
    .choice.shot {{ border-top-color: var(--blue); }}
    .choice span {{ display: block; font-size: 22px; font-weight: 800; margin-bottom: 10px; }}
    .choice b {{ display: block; color: var(--muted); font-weight: 500; }}
    .mode-row {{ display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid #e9eef2; padding: 7px 0; }}
    .mode-row:last-child {{ border-bottom: 0; }}
    .frame-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
      align-items: start;
      margin-top: 14px;
    }}
    .frame-card, .shot-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      overflow: hidden;
    }}
    .frame-card summary, .shot-card summary {{ cursor: pointer; list-style: none; }}
    .frame-card summary::-webkit-details-marker, .shot-card summary::-webkit-details-marker {{ display: none; }}
    .frame-card[open], .shot-card[open] {{ border-color: #9bb8ca; }}
    .image-wrap {{ position: relative; background: #e5ebf0; }}
    .image-wrap img {{
      width: 100%;
      aspect-ratio: 16/9;
      object-fit: cover;
      display: block;
    }}
    .image-missing {{
      aspect-ratio: 16/9;
      display: grid;
      place-items: center;
      padding: 14px;
      text-align: center;
      color: var(--muted);
      background: #e8edf1;
      word-break: break-word;
    }}
    .frame-summary {{ padding: 10px 12px 12px; }}
    .frame-title, .shot-title {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; }}
    .frame-title b, .shot-title b {{ word-break: break-word; }}
    .time {{ color: var(--accent); font-weight: 800; white-space: nowrap; }}
    .caption {{ margin: 8px 0 0; color: #26394a; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      background: var(--soft);
      padding: 3px 8px;
      color: #334155;
      font-size: 12px;
      white-space: nowrap;
    }}
    .chip.good {{ background: var(--accent-soft); color: #075e58; }}
    .chip.info {{ background: var(--blue-soft); color: #174f7a; }}
    .chip.ocr {{ background: var(--violet-soft); color: #563d82; }}
    .chip.warn {{ background: var(--warn-soft); color: var(--warn); }}
    .chip.bad {{ background: var(--bad-soft); color: var(--bad); }}
    .detail {{ border-top: 1px solid var(--line); padding: 12px; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .detail-grid.three {{ grid-template-columns: 1fr 1fr 1fr; }}
    .box {{ border: 1px solid var(--line); border-radius: 8px; padding: 11px; min-width: 0; background: #fbfdfe; }}
    .box.objects {{ border-top: 3px solid var(--accent); }}
    .box.scene {{ border-top: 3px solid var(--blue); }}
    .box.ocr {{ border-top: 3px solid var(--violet); }}
    .box.audio {{ border-top: 3px solid var(--warn); }}
    .box.event {{ border-top: 3px solid var(--blue); }}
    .text-block {{ margin: 0 0 10px; }}
    .text-block b {{ display: block; margin-bottom: 3px; color: #0f172a; }}
    .tag-list {{ display: flex; flex-wrap: wrap; gap: 5px; }}
    .tag {{ background: var(--blue-soft); color: #174f7a; border-radius: 4px; padding: 2px 6px; font-size: 12px; }}
    .tag.object {{ background: var(--accent-soft); color: #075e58; }}
    .empty {{ color: var(--muted); font-style: italic; }}
    .shot-list {{ display: flex; flex-direction: column; gap: 12px; margin-top: 14px; }}
    .shot-preview {{
      display: grid;
      grid-template-columns: minmax(260px, 0.8fr) minmax(0, 1.6fr) minmax(280px, 0.9fr);
      gap: 14px;
      padding: 12px;
    }}
    .thumb-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 6px;
    }}
    .thumb-strip .image-wrap img, .thumb-strip .image-missing {{ border-radius: 6px; }}
    .metric-table {{ width: 100%; border-collapse: collapse; }}
    .metric-table th, .metric-table td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid #e9eef2; vertical-align: top; }}
    .metric-table th {{ width: 170px; color: var(--muted); font-weight: 700; }}
    details.raw {{ margin-top: 12px; }}
    details.raw summary {{ color: var(--accent); font-weight: 800; cursor: pointer; }}
    pre {{
      overflow: auto;
      max-height: 380px;
      border-radius: 8px;
      background: #0f172a;
      color: #e2e8f0;
      padding: 12px;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 1050px) {{
      .hero, .toolbar, .review-choice, .shot-preview, .detail-grid, .detail-grid.three {{ grid-template-columns: 1fr; display: grid; }}
      .toolbar {{ position: static; }}
    }}
  </style>
</head>
<body data-active="{_escape_attr(active)}">
  <main class="wrap">{body}</main>
</body>
</html>"""


def _filter_script(card_selector: str) -> str:
    return f"""<script>
const filter = document.getElementById('filter');
const featureFilter = document.getElementById('featureFilter');
const cards = Array.from(document.querySelectorAll('{card_selector}'));
const visibleCount = document.getElementById('visibleCount');
function applyFilter() {{
  const q = filter.value.trim().toLowerCase();
  const feature = featureFilter.value;
  let visible = 0;
  for (const card of cards) {{
    const textOk = !q || card.dataset.search.includes(q);
    const featureOk = !feature || card.dataset.features.includes(feature);
    const show = textOk && featureOk;
    card.classList.toggle('hidden', !show);
    if (show) visible += 1;
  }}
  visibleCount.textContent = `${{visible}}/${{cards.length}}`;
}}
filter.addEventListener('input', applyFilter);
featureFilter.addEventListener('change', applyFilter);
applyFilter();
</script>"""


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


def _build_context(
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
) -> dict[str, Any]:
    frame_map = {
        row.get("canonical_frame_id") or row.get("frame_id"): row
        for row in frame_index
        if row.get("canonical_frame_id") or row.get("frame_id")
    }
    event_map = {row.get("shot_id"): row for row in event_index if row.get("shot_id")}
    frame_evidence_map = {
        row.get("canonical_frame_id") or row.get("frame_id"): row
        for row in frame_evidence
        if row.get("canonical_frame_id") or row.get("frame_id")
    }
    shot_evidence_map = {
        row.get("shot_id"): row for row in shot_evidence if row.get("shot_id")
    }
    shot_frames = _frames_by_shot(shot_index, frame_index)
    frame_shot_map = _frame_to_shot_map(shot_index, shot_frames)
    summary = _build_summary(
        frame_index=frame_index,
        shot_index=shot_index,
        event_index=event_index,
        compact_docs=compact_docs,
        reports=reports,
        frame_evidence_map=frame_evidence_map,
        shot_evidence_map=shot_evidence_map,
    )
    return {
        "video_id": video_id,
        "base_dir": base_dir,
        "frame_index": frame_index,
        "shot_index": shot_index,
        "event_index": event_index,
        "compact_docs": compact_docs,
        "frame_map": frame_map,
        "event_map": event_map,
        "frame_evidence_map": frame_evidence_map,
        "shot_evidence_map": shot_evidence_map,
        "shot_frames": shot_frames,
        "frame_shot_map": frame_shot_map,
        "summary": summary,
    }


def _build_summary(
    *,
    frame_index: list[dict],
    shot_index: list[dict],
    event_index: list[dict],
    compact_docs: list[dict],
    reports: dict[str, Any],
    frame_evidence_map: dict[str, dict],
    shot_evidence_map: dict[str, dict],
) -> dict[str, Any]:
    frame_modes = Counter((row.get("quality") or {}).get("caption_mode", "") for row in frame_index)
    shot_modes = Counter((row.get("quality") or {}).get("caption_mode", "") for row in shot_index)
    event_modes = Counter((row.get("quality") or {}).get("event_mode", "") for row in event_index)
    frame_fallback = sum(1 for row in frame_index if (row.get("quality") or {}).get("fallback_used"))
    shot_fallback = sum(1 for row in shot_index if (row.get("quality") or {}).get("fallback_used"))
    event_fallback = sum(1 for row in event_index if (row.get("quality") or {}).get("fallback_used"))
    warning_rows = [
        (
            row.get("shot_id") or row.get("canonical_frame_id") or row.get("event_id"),
            (row.get("quality") or {}).get("warnings") or [],
        )
        for row in frame_index + shot_index + event_index
        if (row.get("quality") or {}).get("warnings")
    ]
    missing_frame_ocr = sum(
        1 for row in frame_index
        if not _frame_feature_flags(row, frame_evidence_map.get(_frame_id(row), {}))["has_ocr"]
    )
    missing_frame_audio = sum(
        1 for row in frame_index
        if not _frame_feature_flags(row, frame_evidence_map.get(_frame_id(row), {}))["has_audio"]
    )
    missing_frame_object = sum(
        1 for row in frame_index
        if not _frame_feature_flags(row, frame_evidence_map.get(_frame_id(row), {}))["has_object"]
    )
    missing_shot_ocr = sum(
        1 for row in shot_index
        if not _shot_feature_flags(row, shot_evidence_map.get(str(row.get("shot_id") or ""), {}))["has_ocr"]
    )
    missing_shot_audio = sum(
        1 for row in shot_index
        if not _shot_feature_flags(row, shot_evidence_map.get(str(row.get("shot_id") or ""), {}))["has_audio"]
    )
    missing_shot_object = sum(
        1 for row in shot_index
        if not _shot_feature_flags(row, shot_evidence_map.get(str(row.get("shot_id") or ""), {}))["has_object"]
    )
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
        "frame_cards": [
            ("Frames", len(frame_index), f"{frame_fallback} fallback"),
            ("Missing OCR", missing_frame_ocr, "frame evidence"),
            ("Missing audio", missing_frame_audio, "frame evidence"),
            ("Missing objects", missing_frame_object, "frame evidence"),
        ],
        "shot_cards": [
            ("Shots", len(shot_index), f"{shot_fallback} fallback"),
            ("Event steps", len(event_index), f"{event_fallback} fallback"),
            ("Missing OCR", missing_shot_ocr, "shot evidence"),
            ("Missing audio", missing_shot_audio, "shot evidence"),
            ("Missing objects", missing_shot_object, "shot evidence"),
        ],
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
        elif isinstance(value, (int, float, str, bool)) or key in {
            "coverage",
            "caption_mode_counts",
            "event_mode_counts",
        }:
            keep[key] = value
    return keep


def _render_frame_card(
    *,
    frame: dict,
    context: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    image_mode: str,
    assets_dir: Path | None,
) -> str:
    frame_id = _frame_id(frame)
    evidence = context["frame_evidence_map"].get(frame_id, {})
    shot = context["frame_shot_map"].get(frame_id, {})
    obj = evidence.get("object_evidence") or {}
    ocr = evidence.get("ocr_evidence") or {}
    audio = evidence.get("audio_evidence") or {}
    quality = frame.get("quality") or {}
    flags = _frame_feature_flags(frame, evidence)
    image = _render_image(
        frame=_merge_image_metadata(frame, evidence),
        output_path=output_path,
        keyframes_root=keyframes_root,
        image_mode=image_mode,
        assets_dir=assets_dir,
        alt=frame_id,
    )
    chips = [
        (str(shot.get("shot_id") or "no-shot"), "info"),
        ("OCR", "good" if flags["has_ocr"] else "warn"),
        ("Audio", "good" if flags["has_audio"] else "warn"),
        ("Objects", "good" if flags["has_object"] else "warn"),
        ("Scene", "good" if flags["has_scene"] else "warn"),
    ]
    if quality.get("fallback_used"):
        chips.append(("fallback", "bad"))
    if quality.get("warnings"):
        chips.append(("warning", "warn"))
    features = _feature_tokens(flags, quality)
    search_text = _join_search(
        frame_id,
        shot.get("shot_id"),
        frame.get("caption_text"),
        ocr.get("ocr_text"),
        audio.get("audio_text"),
        obj.get("object_text"),
        " ".join(_object_items(obj)),
        " ".join((obj.get("scene_tags") or []) + (obj.get("ram_tags") or [])),
    )
    raw = {"frame": frame, "frame_evidence": evidence, "shot": shot}
    mapping_table = _table([
        ("Shot", shot.get("shot_id")),
        ("Timestamp", _fmt_time(frame.get("timestamp_sec"))),
        ("Caption mode", quality.get("caption_mode")),
        ("Warnings", "; ".join(map(str, quality.get("warnings") or []))),
    ])
    return f"""<details class="frame-card" data-search="{_escape_attr(search_text)}" data-features="{_escape_attr(features)}">
  <summary>
    {image}
    <div class="frame-summary">
      <div class="frame-title">
        <b>{_escape(frame_id)}</b>
        <span class="time">{_fmt_time(frame.get("timestamp_sec"))}</span>
      </div>
      <div class="chips">{''.join(_chip(label, cls) for label, cls in chips)}</div>
      <p class="caption">{_value(frame.get("caption_text"))}</p>
    </div>
  </summary>
  <div class="detail">
    <div class="detail-grid">
      <section class="box"><h3>Frame Mapping</h3>{mapping_table}</section>
      <section class="box"><h3>Caption</h3><p>{_value(frame.get("caption_text"))}</p></section>
      <section class="box objects"><h3>Objects</h3>{_tags(_object_items(obj), "object")}</section>
      <section class="box scene"><h3>Scene / RAM</h3>{_tags((obj.get("scene_tags") or []) + (obj.get("ram_tags") or []))}</section>
      <section class="box ocr"><h3>OCR</h3><p>{_value(ocr.get("ocr_text"))}</p></section>
      <section class="box audio"><h3>Audio / ASR</h3><p>{_value(audio.get("audio_text"))}</p></section>
    </div>
    <details class="raw"><summary>Raw frame evidence</summary><pre>{_escape(json.dumps(raw, ensure_ascii=False, indent=2))}</pre></details>
  </div>
</details>"""


def _render_shot_card(
    *,
    shot: dict,
    context: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    max_frames: int,
    image_mode: str,
    assets_dir: Path | None,
) -> str:
    shot_id = str(shot.get("shot_id") or "")
    event = context["event_map"].get(shot_id, {})
    shot_evidence = context["shot_evidence_map"].get(shot_id, {})
    frames = context["shot_frames"].get(shot_id, [])
    evidence = shot.get("evidence_text") or {}
    source = event.get("source_text") or {}
    quality = shot.get("quality") or {}
    event_quality = event.get("quality") or {}
    flags = _shot_feature_flags(shot, shot_evidence)
    representative_frames = _representative_frames(shot, frames, context["frame_map"])
    image_strip = _render_image_strip(
        frames=representative_frames[:max(max_frames, 1)],
        context=context,
        output_path=output_path,
        keyframes_root=keyframes_root,
        image_mode=image_mode,
        assets_dir=assets_dir,
    )
    chips = [
        (f"frames:{len(frames)}", "info"),
        ("OCR", "good" if flags["has_ocr"] else "warn"),
        ("Audio", "good" if flags["has_audio"] else "warn"),
        ("Objects", "good" if flags["has_object"] else "warn"),
        ("Scene", "good" if flags["has_scene"] else "warn"),
        ("event:" + str(event_quality.get("event_mode", "unknown")), "info"),
    ]
    if quality.get("fallback_used") or event_quality.get("fallback_used"):
        chips.append(("fallback", "bad"))
    if quality.get("warnings") or event_quality.get("warnings"):
        chips.append(("warning", "warn"))
    search_text = _join_search(
        shot_id,
        shot.get("caption_text"),
        shot.get("temporal_caption"),
        event.get("event_caption"),
        event.get("trake_text"),
        evidence.get("merged_ocr_text"),
        evidence.get("merged_audio_text"),
        evidence.get("object_text"),
        evidence.get("scene_text"),
        source.get("merged_ocr_text"),
        source.get("merged_audio_text"),
    )
    features = _feature_tokens(flags, quality, event_quality)
    object_items = _shot_object_items(shot_evidence, evidence, source)
    scene_tags = shot_evidence.get("merged_scene_tags") or []
    ocr_text = evidence.get("merged_ocr_text") or source.get("merged_ocr_text") or ""
    audio_text = evidence.get("merged_audio_text") or source.get("merged_audio_text") or ""
    raw = {
        "shot": shot,
        "event": event,
        "shot_evidence": shot_evidence,
        "frame_ids": [_frame_id(frame) for frame in frames],
    }
    quality_table = _table([
        ("Caption mode", quality.get("caption_mode")),
        ("Event mode", event_quality.get("event_mode")),
        ("Action state", event.get("action_state")),
        ("Temporal role", event.get("temporal_role")),
        (
            "Warnings",
            "; ".join(map(str, (quality.get("warnings") or []) + (event_quality.get("warnings") or []))),
        ),
    ])
    return f"""<details class="shot-card" data-search="{_escape_attr(search_text)}" data-features="{_escape_attr(features)}">
  <summary>
    <div class="shot-preview">
      <div>{image_strip}</div>
      <div>
        <div class="shot-title">
          <b>{_escape(shot_id)}</b>
          <span class="time">{_fmt_time(shot.get("start_sec"))} - {_fmt_time(shot.get("end_sec"))}</span>
        </div>
        <div class="chips">{''.join(_chip(label, cls) for label, cls in chips)}</div>
        <p class="text-block"><b>Shot caption</b>{_value(shot.get("caption_text"))}</p>
        <p class="text-block"><b>Event caption</b>{_value(event.get("event_caption"))}</p>
      </div>
      <div>
        <p class="text-block"><b>TRAKE text</b>{_value(event.get("trake_text"))}</p>
      </div>
    </div>
  </summary>
  <div class="detail">
    <div class="detail-grid three">
      <section class="box event"><h3>Event / TRAKE</h3>{_render_event_detail(event)}</section>
      <section class="box objects"><h3>Objects</h3>{_tags(object_items, "object")}</section>
      <section class="box scene"><h3>Scene / RAM</h3>{_tags(scene_tags)}</section>
      <section class="box ocr"><h3>OCR</h3><p>{_value(ocr_text)}</p></section>
      <section class="box audio"><h3>Audio / ASR</h3><p>{_value(audio_text)}</p></section>
      <section class="box"><h3>Search / Quality</h3>{quality_table}</section>
    </div>
    <details class="raw"><summary>Raw shot and event JSON</summary><pre>{_escape(json.dumps(raw, ensure_ascii=False, indent=2))}</pre></details>
  </div>
</details>"""


def _render_image_strip(
    *,
    frames: list[dict],
    context: dict[str, Any],
    output_path: Path,
    keyframes_root: Path | None,
    image_mode: str,
    assets_dir: Path | None,
) -> str:
    if not frames:
        return '<div class="image-missing">No representative frames</div>'
    parts = []
    for frame in frames:
        frame_id = _frame_id(frame)
        evidence = context["frame_evidence_map"].get(frame_id, {})
        parts.append(
            _render_image(
                frame=_merge_image_metadata(frame, evidence),
                output_path=output_path,
                keyframes_root=keyframes_root,
                image_mode=image_mode,
                assets_dir=assets_dir,
                alt=frame_id,
            )
        )
    return f'<div class="thumb-strip">{"".join(parts)}</div>'


def _render_image(
    *,
    frame: dict,
    output_path: Path,
    keyframes_root: Path | None,
    image_mode: str,
    assets_dir: Path | None,
    alt: str,
) -> str:
    src, image_status = _image_src(frame, output_path, keyframes_root, image_mode, assets_dir)
    if src:
        return (
            f'<div class="image-wrap"><img src="{_escape_attr(src)}" '
            f'alt="{_escape_attr(alt)}" loading="lazy"></div>'
        )
    return f'<div class="image-missing">Image unavailable<br>{_escape(image_status)}</div>'


def _image_src(
    frame: dict,
    output_path: Path,
    keyframes_root: Path | None,
    image_mode: str,
    assets_dir: Path | None = None,
) -> tuple[str, str]:
    path, candidates = _resolve_image_path(frame, keyframes_root)
    if path is None:
        sample = ", ".join(str(item) for item in candidates[:4])
        return "", f"checked: {sample}" if sample else "no image metadata"

    if image_mode == "embed":
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}", str(path)

    if image_mode == "copy":
        target_dir = assets_dir or output_path.parent / f"{output_path.stem}_assets"
        target_dir.mkdir(parents=True, exist_ok=True)
        frame_id = str(frame.get("canonical_frame_id") or frame.get("frame_id") or path.stem)
        target = target_dir / f"{_safe_asset_name(frame_id)}{path.suffix.lower()}"
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
    keyframe_idx = frame.get("keyframe_idx") or frame.get("source_frame_idx")
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
            frame
            for frame in frames
            if start - 0.001 <= float(frame.get("timestamp_sec") or 0) <= end + 0.001
        ]
    return result


def _frame_to_shot_map(shot_index: list[dict], shot_frames: dict[str, list[dict]]) -> dict[str, dict]:
    shot_map = {str(row.get("shot_id") or ""): row for row in shot_index}
    result: dict[str, dict] = {}
    for shot_id, frames in shot_frames.items():
        shot = shot_map.get(shot_id, {})
        for frame in frames:
            result[_frame_id(frame)] = shot
    return result


def _representative_frames(
    shot: dict,
    frames: list[dict],
    frame_map: dict[str, dict],
) -> list[dict]:
    result = []
    for frame_id in shot.get("representative_frame_ids") or []:
        if frame_id in frame_map:
            result.append(frame_map[frame_id])
    return result or frames


def _merge_image_metadata(frame: dict, evidence: dict) -> dict:
    merged = dict(frame)
    for key in ("image_path", "image_relpath", "frame_name", "keyframe_idx", "source_frame_idx"):
        if merged.get(key) in (None, "") and evidence.get(key) not in (None, ""):
            merged[key] = evidence[key]
    return merged


def _frame_feature_flags(frame: dict, evidence: dict) -> dict[str, bool]:
    obj = evidence.get("object_evidence") or {}
    ocr = evidence.get("ocr_evidence") or {}
    audio = evidence.get("audio_evidence") or {}
    quality = evidence.get("quality") or {}
    return {
        "has_ocr": bool(quality.get("has_ocr") or ocr.get("ocr_text")),
        "has_audio": bool(quality.get("has_audio") or audio.get("audio_text")),
        "has_object": bool(quality.get("has_object") or _object_items(obj)),
        "has_scene": bool(quality.get("has_scene_tags") or obj.get("scene_tags") or obj.get("ram_tags")),
        "warning": bool((frame.get("quality") or {}).get("warnings")),
    }


def _shot_feature_flags(shot: dict, shot_evidence: dict) -> dict[str, bool]:
    evidence = shot.get("evidence_text") or {}
    counts = shot_evidence.get("merged_object_counts") or {}
    return {
        "has_ocr": bool(evidence.get("merged_ocr_text") or shot_evidence.get("merged_ocr_text")),
        "has_audio": bool(evidence.get("merged_audio_text") or shot_evidence.get("merged_audio_text")),
        "has_object": bool(counts or evidence.get("object_text")),
        "has_scene": bool(shot_evidence.get("merged_scene_tags") or evidence.get("scene_text")),
        "warning": bool((shot.get("quality") or {}).get("warnings")),
    }


def _feature_tokens(*items: dict[str, Any]) -> str:
    tokens = []
    flags: dict[str, Any] = {}
    for item in items:
        flags.update(item)
    if not flags.get("has_ocr"):
        tokens.append("missing_ocr")
    if not flags.get("has_audio"):
        tokens.append("missing_audio")
    if not flags.get("has_object"):
        tokens.append("missing_object")
    if flags.get("warning") or flags.get("warnings"):
        tokens.append("warning")
    if flags.get("fallback_used"):
        tokens.append("warning")
    return " ".join(tokens)


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


def _render_warnings(warnings: list[tuple[str, list[str]]]) -> str:
    if not warnings:
        return ""
    rows = "\n".join(
        f"""<div class="mode-row"><span>{_escape(str(item_id))}</span><span>{_escape("; ".join(map(str, values)))}</span></div>"""
        for item_id, values in warnings
    )
    return f"""<section class="box" style="margin-top: 14px">
  <h2>Warnings</h2>
  {rows}
</section>"""


def _summary_card(label: str, value: Any, hint: str) -> str:
    return f"""<article class="card">
  <div class="label">{_escape(label)}</div>
  <div class="value">{_escape(str(value))}</div>
  <div class="subtle">{_escape(hint)}</div>
</article>"""


def _table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{_escape(label)}</th><td>{_value(value)}</td></tr>"
        for label, value in rows
    )
    return f'<table class="metric-table"><tbody>{body}</tbody></table>'


def _shot_object_items(shot_evidence: dict, evidence: dict, source: dict) -> list[str]:
    counts = shot_evidence.get("merged_object_counts") or {}
    if counts:
        return [
            f"{name} x{count}"
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    text = evidence.get("object_text") or source.get("object_text") or ""
    return [item.strip() for item in str(text).split(",") if item.strip()]


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


def _value(value: Any) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return '<span class="empty">empty</span>'
    return _escape(value)


def _chip(label: str, class_name: str = "") -> str:
    class_attr = f" {class_name}" if class_name else ""
    return f"""<span class="chip{class_attr}">{_escape(label)}</span>"""


def _counter_text(counter: Counter) -> str:
    if not counter:
        return "0"
    return ", ".join(f"{key or 'empty'}:{value}" for key, value in counter.most_common())


def _join_search(*values: Any) -> str:
    return " ".join(str(value) for value in values if value).lower()


def _frame_id(frame: dict) -> str:
    return str(frame.get("canonical_frame_id") or frame.get("frame_id") or "unknown")


def _safe_asset_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


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
