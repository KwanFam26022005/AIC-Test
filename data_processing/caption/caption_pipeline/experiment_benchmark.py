"""Experiment benchmark utilities for Phase 6 caption retrieval comparisons."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl, utc_now_iso
from .text_utils import truncate

def load_experiment(label: str, root: str | Path, video_id: str, eval_name: str = "") -> dict[str, Any]:
    """Load Phase 5 retrieval outputs for one benchmark candidate."""
    root_path = Path(root)
    video_dir = _resolve_video_dir(root_path, video_id)
    eval_dir = _named_artifact_dir(video_dir / "eval", eval_name)
    exports_dir = _named_artifact_dir(video_dir / "exports", eval_name)

    report_path = eval_dir / "retrieval_eval_report.json"
    results_path = eval_dir / "retrieval_results.jsonl"
    queries_path = eval_dir / "retrieval_queries.jsonl"
    corpus_path = exports_dir / "retrieval_corpus.jsonl"

    missing = [
        str(path)
        for path in [report_path, results_path, queries_path, corpus_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Experiment '{label}' is missing required Phase 5 files: {missing}"
        )

    report = read_json(report_path)
    results = read_jsonl(results_path)
    queries = read_jsonl(queries_path)
    corpus = read_jsonl(corpus_path)
    return {
        "label": label,
        "root": str(root_path),
        "video_dir": str(video_dir),
        "report_path": str(report_path),
        "results_path": str(results_path),
        "queries_path": str(queries_path),
        "corpus_path": str(corpus_path),
        "report": report,
        "results": results,
        "queries": queries,
        "corpus": corpus,
        "summary": summarize_experiment(label, report, results, queries, corpus),
    }

def summarize_experiment(
    label: str,
    report: dict[str, Any],
    results: list[dict],
    queries: list[dict],
    corpus: list[dict],
) -> dict[str, Any]:
    """Compute stable comparable metrics from Phase 5 outputs."""
    top1_by_query = _top_result_by_query(results, rank=1)
    query_ids = [row.get("query_id", "") for row in queries]
    top1_scores = [
        float(top1_by_query[qid].get("score", 0.0))
        for qid in query_ids
        if qid in top1_by_query
    ]
    hit_queries = set(top1_by_query)
    warnings = report.get("warnings") or []
    unit_at_1: dict[str, int] = defaultdict(int)
    matched_field_counts: dict[str, int] = defaultdict(int)
    for qid, row in top1_by_query.items():
        if qid not in hit_queries:
            continue
        unit_at_1[row.get("unit_type", "unknown")] += 1
        for field in row.get("matched_fields") or []:
            matched_field_counts[field] += 1

    return {
        "label": label,
        "num_corpus_docs": len(corpus),
        "num_queries": len(queries),
        "num_queries_with_hits": report.get("num_queries_with_hits", len(hit_queries)),
        "num_queries_without_hits": report.get("num_queries_without_hits", 0),
        "hit_rate": _safe_div(report.get("num_queries_with_hits", len(hit_queries)), len(queries)),
        "num_results": len(results),
        "avg_top1_score": _mean(top1_scores),
        "min_top1_score": min(top1_scores) if top1_scores else 0.0,
        "max_top1_score": max(top1_scores) if top1_scores else 0.0,
        "warnings": warnings,
        "num_warnings": len(warnings),
        "unit_type_hits_at_1": dict(sorted(unit_at_1.items())),
        "matched_field_counts_at_1": dict(sorted(matched_field_counts.items())),
    }

def build_benchmark_report(
    video_id: str,
    baseline: dict[str, Any],
    experiments: list[dict[str, Any]],
    top_n_samples: int,
) -> dict[str, Any]:
    """Build comparison report across baseline and one or more experiments."""
    candidates = [baseline] + experiments
    baseline_top1 = _top_result_by_query(baseline["results"], rank=1)
    comparisons: list[dict[str, Any]] = []
    for exp in experiments:
        comparisons.append(_compare_to_baseline(baseline, exp, baseline_top1))

    ranking = sorted(
        [item["summary"] for item in candidates],
        key=lambda row: (
            -float(row.get("hit_rate", 0.0)),
            -float(row.get("avg_top1_score", 0.0)),
            int(row.get("num_warnings", 0)),
            row.get("label", ""),
        ),
    )

    warnings: list[str] = []
    for item in candidates:
        label = item["label"]
        summary = item["summary"]
        if summary["num_queries_without_hits"] > 0:
            warnings.append(
                f"{label}: {summary['num_queries_without_hits']} queries without hits"
            )
        if summary["num_warnings"] > 0:
            warnings.append(f"{label}: {summary['num_warnings']} retrieval warnings")

    return {
        "schema_version": "caption_phase6_experiment_benchmark_v1",
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "baseline_label": baseline["label"],
        "num_candidates": len(candidates),
        "top_n_samples": top_n_samples,
        "candidate_summaries": [item["summary"] for item in candidates],
        "ranking": ranking,
        "comparisons_to_baseline": comparisons,
        "warnings": warnings,
    }

def build_manual_review_samples(
    baseline: dict[str, Any],
    experiments: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """Create per-query top result samples for manual inspection."""
    candidates = [baseline] + experiments
    query_map = {
        row.get("query_id", ""): row
        for row in baseline.get("queries", [])
        if row.get("query_id")
    }
    query_ids = sorted(query_map)
    samples: list[dict[str, Any]] = []
    for query_id in query_ids:
        item = {
            "query_id": query_id,
            "query": query_map[query_id].get("query", ""),
            "route": query_map[query_id].get("route", ""),
            "candidates": [],
        }
        for candidate in candidates:
            top_rows = [
                row for row in candidate["results"]
                if row.get("query_id") == query_id and int(row.get("rank", 0)) <= top_n
            ]
            item["candidates"].append({
                "label": candidate["label"],
                "top_results": [_sample_result(row) for row in top_rows],
            })
        samples.append(item)
    return samples

def render_benchmark_report_markdown(report: dict[str, Any]) -> str:
    """Render Phase 6 benchmark report as Markdown."""
    lines: list[str] = []
    lines.append(f"# Caption Experiment Benchmark - {report['video_id']}")
    lines.append("")
    lines.append(f"Created: {report.get('created_at', 'N/A')}")
    lines.append("")

    lines.append("## Candidate Summary")
    lines.append("")
    lines.append(
        "| Label | Corpus | Queries | Hits | Hit Rate | Avg Top1 | Warnings |"
    )
    lines.append("|-------|-------:|--------:|-----:|---------:|---------:|---------:|")
    for row in report.get("candidate_summaries") or []:
        lines.append(
            f"| {row['label']} | {row['num_corpus_docs']} | {row['num_queries']} | "
            f"{row['num_queries_with_hits']} | {row['hit_rate']:.3f} | "
            f"{row['avg_top1_score']:.3f} | {row['num_warnings']} |"
        )
    lines.append("")

    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | Label | Hit Rate | Avg Top1 | Warnings |")
    lines.append("|-----:|-------|---------:|---------:|---------:|")
    for idx, row in enumerate(report.get("ranking") or [], start=1):
        lines.append(
            f"| {idx} | {row['label']} | {row['hit_rate']:.3f} | "
            f"{row['avg_top1_score']:.3f} | {row['num_warnings']} |"
        )
    lines.append("")

    comparisons = report.get("comparisons_to_baseline") or []
    if comparisons:
        lines.append("## Comparison To Baseline")
        lines.append("")
        lines.append(
            "| Experiment | Shared Queries | Improved | Regressed | Same Top1 | Avg Top1 Delta |"
        )
        lines.append("|------------|---------------:|---------:|----------:|----------:|---------------:|")
        for row in comparisons:
            lines.append(
                f"| {row['experiment_label']} | {row['shared_queries']} | "
                f"{row['num_improved_top1_score']} | {row['num_regressed_top1_score']} | "
                f"{row['num_same_top1_document']} | {row['avg_top1_score_delta']:.3f} |"
            )
        lines.append("")

    warnings = report.get("warnings") or []
    lines.append("## Status")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- [WARN] {warning}")
    else:
        lines.append("OK: No benchmark warnings.")
    lines.append("")
    return "\n".join(lines)

def _compare_to_baseline(
    baseline: dict[str, Any],
    experiment: dict[str, Any],
    baseline_top1: dict[str, dict],
) -> dict[str, Any]:
    exp_top1 = _top_result_by_query(experiment["results"], rank=1)
    shared = sorted(set(baseline_top1) & set(exp_top1))
    improved = 0
    regressed = 0
    same_doc = 0
    deltas: list[float] = []
    changed: list[dict[str, Any]] = []
    for query_id in shared:
        base_row = baseline_top1[query_id]
        exp_row = exp_top1[query_id]
        base_score = float(base_row.get("score", 0.0))
        exp_score = float(exp_row.get("score", 0.0))
        delta = exp_score - base_score
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta < 0:
            regressed += 1
        if base_row.get("document_id") == exp_row.get("document_id"):
            same_doc += 1
        else:
            changed.append({
                "query_id": query_id,
                "baseline_document_id": base_row.get("document_id", ""),
                "experiment_document_id": exp_row.get("document_id", ""),
                "baseline_score": base_score,
                "experiment_score": exp_score,
                "score_delta": round(delta, 6),
            })

    return {
        "baseline_label": baseline["label"],
        "experiment_label": experiment["label"],
        "shared_queries": len(shared),
        "num_improved_top1_score": improved,
        "num_regressed_top1_score": regressed,
        "num_same_top1_document": same_doc,
        "num_changed_top1_document": len(changed),
        "avg_top1_score_delta": _mean(deltas),
        "changed_top1_documents": changed[:50],
    }

def _named_artifact_dir(base: Path, eval_name: str) -> Path:
    name = (eval_name or "").strip().strip("/\\")
    if not name:
        return base
    if "/" in name or "\\" in name:
        raise ValueError(f"Invalid eval_name '{eval_name}'. Use a simple folder name.")
    return base / name


def _resolve_video_dir(root: Path, video_id: str) -> Path:
    if (root / "eval").exists() and (root / "exports").exists():
        return root
    return root / video_id

def _top_result_by_query(results: list[dict], rank: int) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in results:
        if int(row.get("rank", 0)) != rank:
            continue
        query_id = row.get("query_id", "")
        if query_id and query_id not in rows:
            rows[query_id] = row
    return rows

def _sample_result(row: dict) -> dict[str, Any]:
    return {
        "rank": row.get("rank", 0),
        "score": row.get("score", 0.0),
        "document_id": row.get("document_id", ""),
        "unit_type": row.get("unit_type", ""),
        "unit_id": row.get("unit_id", ""),
        "shot_id": row.get("shot_id", ""),
        "event_id": row.get("event_id", ""),
        "start_sec": row.get("start_sec", 0.0),
        "end_sec": row.get("end_sec", 0.0),
        "matched_fields": row.get("matched_fields") or [],
        "matched_terms": row.get("matched_terms") or [],
        "snippet": truncate(row.get("snippet", "") or "", 360),
    }

def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


