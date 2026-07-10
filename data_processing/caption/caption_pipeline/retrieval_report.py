"""Report builder for Phase 5 retrieval export/evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .io_utils import utc_now_iso


def build_retrieval_report(
    video_id: str,
    corpus: list[dict],
    queries: list[dict],
    results: list[dict],
    top_k: int,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact validation report for retrieval export/eval."""
    warnings = list(warnings or [])
    query_ids = [row.get("query_id", "") for row in queries]
    result_query_ids = {row.get("query_id", "") for row in results}

    route_counts: dict[str, int] = defaultdict(int)
    for query in queries:
        route_counts[query.get("route", "general")] += 1

    unit_type_counts: dict[str, int] = defaultdict(int)
    for doc in corpus:
        unit_type_counts[doc.get("unit_type", "unknown")] += 1

    unit_type_hits_at_1: dict[str, int] = defaultdict(int)
    top_results_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        if row.get("rank") == 1:
            unit_type_hits_at_1[row.get("unit_type", "unknown")] += 1
        if len(top_results_by_query[row.get("query_id", "")]) < 5:
            top_results_by_query[row.get("query_id", "")].append(row)

    duplicate_document_ids = _count_duplicates(corpus, "document_id")
    empty_search_text = sum(1 for row in corpus if not row.get("search_text"))
    queries_without_hits = [
        query_id for query_id in query_ids if query_id not in result_query_ids
    ]

    if duplicate_document_ids:
        warnings.append(f"{duplicate_document_ids} duplicate corpus document_ids")
    if empty_search_text:
        warnings.append(f"{empty_search_text} corpus docs with empty search_text")
    for query_id in queries_without_hits:
        warnings.append(f"query without hits: {query_id}")

    return {
        "video_id": video_id,
        "created_at": utc_now_iso(),
        "top_k": top_k,
        "num_corpus_docs": len(corpus),
        "corpus_unit_type_counts": dict(sorted(unit_type_counts.items())),
        "num_queries": len(queries),
        "num_queries_with_hits": len(result_query_ids),
        "num_queries_without_hits": len(queries_without_hits),
        "queries_without_hits": queries_without_hits,
        "route_counts": dict(sorted(route_counts.items())),
        "unit_type_hits_at_1": dict(sorted(unit_type_hits_at_1.items())),
        "duplicate_document_ids": duplicate_document_ids,
        "empty_search_text_docs": empty_search_text,
        "num_results": len(results),
        "warnings": warnings,
        "top_results_by_query": dict(top_results_by_query),
    }


def render_retrieval_report_markdown(report: dict[str, Any]) -> str:
    """Render retrieval report as Markdown."""
    lines: list[str] = []
    lines.append(f"# Retrieval Eval Report - {report['video_id']}")
    lines.append("")
    lines.append(f"Created: {report.get('created_at', 'N/A')}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    metrics = [
        ("Eval name", "eval_name"),
        ("Scoring profile", "scoring_profile"),
        ("Corpus docs", "num_corpus_docs"),
        ("Queries", "num_queries"),
        ("Queries with hits", "num_queries_with_hits"),
        ("Queries without hits", "num_queries_without_hits"),
        ("Top K", "top_k"),
        ("Results", "num_results"),
        ("Duplicate document IDs", "duplicate_document_ids"),
        ("Empty search_text docs", "empty_search_text_docs"),
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {report.get(key, 0)} |")
    lines.append("")

    _append_counts(lines, "Corpus Unit Types", report.get("corpus_unit_type_counts") or {})
    _append_counts(lines, "Route Counts", report.get("route_counts") or {})
    _append_counts(lines, "Unit Type Hits At 1", report.get("unit_type_hits_at_1") or {})

    ground_truth = report.get("ground_truth") or {}
    if ground_truth:
        _append_ground_truth_metrics(lines, ground_truth)

    warnings = report.get("warnings") or []
    lines.append("## Status")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- [WARN] {warning}")
    else:
        lines.append("OK: No retrieval export/eval warnings.")
    lines.append("")

    lines.append("## Top Results")
    lines.append("")
    top_results = report.get("top_results_by_query") or {}
    for query_id, rows in sorted(top_results.items()):
        lines.append(f"### {query_id}")
        lines.append("")
        lines.append("| Rank | Score | Unit | ID | Fields | Snippet |")
        lines.append("|-----:|------:|------|----|--------|---------|")
        for row in rows:
            fields = ", ".join(row.get("matched_fields") or [])
            snippet = (row.get("snippet", "") or "").replace("|", " ")
            lines.append(
                f"| {row.get('rank')} | {row.get('score')} | "
                f"{row.get('unit_type')} | {row.get('unit_id') or row.get('event_id')} | "
                f"{fields} | {snippet} |"
            )
        lines.append("")

    lines.append("## Acceptance Checks")
    lines.append("")
    checks = [
        (
            report.get("num_corpus_docs", 0) > 0,
            f"Corpus docs: {report.get('num_corpus_docs', 0)}",
        ),
        (
            report.get("duplicate_document_ids", 0) == 0,
            f"Duplicate document IDs: {report.get('duplicate_document_ids', 0)}",
        ),
        (
            report.get("empty_search_text_docs", 0) == 0,
            f"Empty search_text docs: {report.get('empty_search_text_docs', 0)}",
        ),
        (
            report.get("num_queries_without_hits", 0) == 0,
            f"Queries without hits: {report.get('num_queries_without_hits', 0)}",
        ),
    ]
    if ground_truth.get("num_labeled_queries", 0) > 0:
        checks.append((
            ground_truth.get("num_unlabeled_queries", 0) == 0,
            "Ground-truth coverage: "
            f"{ground_truth.get('num_labeled_queries', 0)}/{ground_truth.get('num_queries', 0)}",
        ))
    for passed, desc in checks:
        status = "[PASS]" if passed else "[FAIL]"
        lines.append(f"- {status} {desc}")
    lines.append("")
    return "\n".join(lines)


def _append_counts(lines: list[str], title: str, counts: dict[str, int]) -> None:
    if not counts:
        return
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Value | Count |")
    lines.append("|-------|------:|")
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.append("")


def _append_ground_truth_metrics(lines: list[str], ground_truth: dict[str, Any]) -> None:
    lines.append("## Ground Truth Metrics")
    lines.append("")
    lines.append(
        f"Labeled queries: {ground_truth.get('num_labeled_queries', 0)}/"
        f"{ground_truth.get('num_queries', 0)}"
    )
    lines.append("")
    metrics = ground_truth.get("metrics") or {}
    lines.append("| Scope | Hit@1 | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |")
    lines.append("|-------|------:|---------:|---------:|----------:|----:|--------:|")
    _append_metric_row(lines, "all", metrics)
    for route, row in sorted((ground_truth.get("per_route") or {}).items()):
        _append_metric_row(lines, route, row.get("metrics") or {})
    lines.append("")

    unlabeled = ground_truth.get("unlabeled_query_ids") or []
    if unlabeled:
        lines.append("Unlabeled queries: " + ", ".join(unlabeled))
        lines.append("")


def _append_metric_row(lines: list[str], scope: str, metrics: dict[str, float]) -> None:
    lines.append(
        f"| {scope} | {metrics.get('hit_at_1', 0.0):.3f} | "
        f"{metrics.get('recall_at_1', 0.0):.3f} | "
        f"{metrics.get('recall_at_5', 0.0):.3f} | "
        f"{metrics.get('recall_at_10', 0.0):.3f} | "
        f"{metrics.get('reciprocal_rank', 0.0):.3f} | "
        f"{metrics.get('ndcg_at_10', 0.0):.3f} |"
    )


def _count_duplicates(rows: list[dict], key: str) -> int:
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        value = row.get(key, "")
        if value in seen:
            duplicates += 1
        seen.add(value)
    return duplicates

