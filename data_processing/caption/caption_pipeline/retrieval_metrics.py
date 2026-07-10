"""Ground-truth metrics for caption retrieval experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


DEFAULT_CUTOFFS = (1, 5, 10)


def merge_ground_truth(
    queries: list[dict],
    ground_truth_rows: list[dict],
) -> tuple[list[dict], list[str]]:
    """Merge ground-truth fields into queries by query_id."""
    labels_by_id: dict[str, dict] = {}
    warnings: list[str] = []
    for row in ground_truth_rows:
        query_id = row.get("query_id", "")
        if not query_id:
            warnings.append("ground-truth row without query_id")
            continue
        if query_id in labels_by_id:
            warnings.append(f"duplicate ground-truth query_id: {query_id}")
        labels_by_id[query_id] = row

    query_ids = {row.get("query_id", "") for row in queries}
    for query_id in sorted(set(labels_by_id) - query_ids):
        warnings.append(f"ground-truth query_id not found in queries: {query_id}")

    merged: list[dict] = []
    for query in queries:
        item = dict(query)
        label = labels_by_id.get(item.get("query_id", ""), {})
        for key in _ground_truth_keys():
            if key in label:
                item[key] = label[key]
        merged.append(item)
    return merged, warnings


def build_ground_truth_template(queries: list[dict]) -> list[dict]:
    """Create review rows without inventing relevance labels."""
    return [
        {
            "query_id": row.get("query_id", ""),
            "query": row.get("query", ""),
            "route": row.get("route", "general"),
            "relevance_judgments": [],
            "review_notes": "",
        }
        for row in queries
    ]


def evaluate_ground_truth(
    queries: list[dict],
    results: list[dict],
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Evaluate ranked results using document, unit, or time-range judgments."""
    normalized_cutoffs = tuple(sorted({int(k) for k in cutoffs if int(k) > 0}))
    results_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        query_id = row.get("query_id", "")
        if query_id:
            results_by_query[query_id].append(row)
    for rows in results_by_query.values():
        rows.sort(key=lambda row: int(row.get("rank", 0) or 0))

    per_query: list[dict[str, Any]] = []
    unlabeled_query_ids: list[str] = []
    for query in queries:
        query_id = query.get("query_id", "")
        judgments = normalize_relevance_judgments(query)
        if not judgments:
            unlabeled_query_ids.append(query_id)
            continue
        per_query.append(
            _evaluate_query(
                query,
                results_by_query.get(query_id, []),
                judgments,
                normalized_cutoffs,
            )
        )

    per_route_rows: dict[str, list[dict]] = defaultdict(list)
    for row in per_query:
        per_route_rows[row["route"]].append(row)

    return {
        "num_queries": len(queries),
        "num_labeled_queries": len(per_query),
        "num_unlabeled_queries": len(unlabeled_query_ids),
        "unlabeled_query_ids": unlabeled_query_ids,
        "cutoffs": list(normalized_cutoffs),
        "metrics": _aggregate_metrics(per_query, normalized_cutoffs),
        "per_route": {
            route: {
                "num_labeled_queries": len(rows),
                "metrics": _aggregate_metrics(rows, normalized_cutoffs),
            }
            for route, rows in sorted(per_route_rows.items())
        },
        "per_query": per_query,
    }


def normalize_relevance_judgments(query: dict) -> list[dict[str, Any]]:
    """Normalize preferred and legacy ground-truth fields into judgments."""
    judgments: list[dict[str, Any]] = []
    for index, row in enumerate(query.get("relevance_judgments") or [], start=1):
        if not isinstance(row, dict):
            continue
        judgment = dict(row)
        judgment["grade"] = max(int(judgment.get("grade", 1) or 1), 1)
        judgment["target_key"] = _judgment_key(judgment, f"judgment:{index}")
        if _has_target(judgment):
            judgments.append(judgment)

    legacy_fields = (
        ("relevant_document_ids", "document_id"),
        ("relevant_shot_ids", "shot_id"),
        ("relevant_frame_ids", "frame_id"),
        ("relevant_event_ids", "event_id"),
    )
    for source_key, target_key in legacy_fields:
        for value in query.get(source_key) or []:
            if value:
                judgments.append({
                    target_key: str(value),
                    "grade": 1,
                    "target_key": f"{target_key}:{value}",
                })

    for index, value in enumerate(query.get("relevant_time_ranges") or [], start=1):
        if isinstance(value, dict):
            start_sec = value.get("start_sec")
            end_sec = value.get("end_sec", start_sec)
            grade = value.get("grade", 1)
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            start_sec, end_sec = value[0], value[1]
            grade = 1
        else:
            continue
        if start_sec is None or end_sec is None:
            continue
        judgments.append({
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "grade": max(int(grade or 1), 1),
            "target_key": f"time:{index}:{float(start_sec):.3f}-{float(end_sec):.3f}",
        })

    unique: dict[str, dict[str, Any]] = {}
    for row in judgments:
        key = row["target_key"]
        if key not in unique or row["grade"] > unique[key]["grade"]:
            unique[key] = row
    return list(unique.values())


def match_result_relevance(
    result: dict,
    judgments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return all relevance judgments matched by one result row."""
    return [row for row in judgments if _matches_judgment(result, row)]


def _evaluate_query(
    query: dict,
    results: list[dict],
    judgments: list[dict[str, Any]],
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    seen_targets: set[str] = set()
    ranked_gains: list[float] = []
    matched_at_rank: dict[int, list[str]] = {}
    first_relevant_rank = 0

    max_cutoff = max(cutoffs, default=0)
    for result in results[:max_cutoff]:
        rank = int(result.get("rank", 0) or 0)
        matched = match_result_relevance(result, judgments)
        new_matches = [row for row in matched if row["target_key"] not in seen_targets]
        if matched and not first_relevant_rank:
            first_relevant_rank = rank
        if new_matches:
            matched_at_rank[rank] = [row["target_key"] for row in new_matches]
            seen_targets.update(matched_at_rank[rank])
            grade = max(row["grade"] for row in new_matches)
            ranked_gains.append(float((2 ** grade) - 1))
        else:
            ranked_gains.append(0.0)

    ideal_gains = sorted(
        (float((2 ** row["grade"]) - 1) for row in judgments),
        reverse=True,
    )
    metrics: dict[str, float] = {
        "reciprocal_rank": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
    }
    for cutoff in cutoffs:
        matched_keys = {
            key
            for rank, keys in matched_at_rank.items()
            if rank <= cutoff
            for key in keys
        }
        metrics[f"hit_at_{cutoff}"] = 1.0 if matched_keys else 0.0
        metrics[f"recall_at_{cutoff}"] = len(matched_keys) / len(judgments)
        dcg = _dcg(ranked_gains[:cutoff])
        idcg = _dcg(ideal_gains[:cutoff])
        metrics[f"ndcg_at_{cutoff}"] = dcg / idcg if idcg else 0.0

    return {
        "query_id": query.get("query_id", ""),
        "route": query.get("route", "general"),
        "num_relevance_targets": len(judgments),
        "first_relevant_rank": first_relevant_rank or None,
        "matched_targets_at_rank": {
            str(rank): keys for rank, keys in sorted(matched_at_rank.items())
        },
        "metrics": metrics,
    }


def _aggregate_metrics(rows: list[dict], cutoffs: tuple[int, ...]) -> dict[str, float]:
    keys = ["reciprocal_rank"]
    for cutoff in cutoffs:
        keys.extend((f"hit_at_{cutoff}", f"recall_at_{cutoff}", f"ndcg_at_{cutoff}"))
    if not rows:
        return {key: 0.0 for key in keys}
    return {
        key: sum(float(row["metrics"].get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
    }


def _matches_judgment(result: dict, judgment: dict[str, Any]) -> bool:
    id_fields = ("document_id", "shot_id", "frame_id", "event_id")
    for field in id_fields:
        expected = judgment.get(field)
        if not expected:
            continue
        actual_values = {
            str(result.get(field, "")),
            str(result.get("unit_id", "")),
            str(result.get("document_id", "")),
        }
        if str(expected) in actual_values:
            return True

    if judgment.get("start_sec") is None:
        return False
    expected_start = float(judgment["start_sec"])
    expected_end = float(judgment.get("end_sec", expected_start))
    result_start = float(result.get("start_sec", 0.0) or 0.0)
    result_end = float(result.get("end_sec", result_start) or result_start)
    return result_start <= expected_end and result_end >= expected_start


def _judgment_key(judgment: dict[str, Any], fallback: str) -> str:
    for field in ("document_id", "shot_id", "frame_id", "event_id"):
        if judgment.get(field):
            return f"{field}:{judgment[field]}"
    if judgment.get("start_sec") is not None:
        start_sec = float(judgment["start_sec"])
        end_sec = float(judgment.get("end_sec", start_sec))
        return f"time:{start_sec:.3f}-{end_sec:.3f}"
    return fallback


def _has_target(judgment: dict[str, Any]) -> bool:
    return any(judgment.get(key) for key in ("document_id", "shot_id", "frame_id", "event_id")) or (
        judgment.get("start_sec") is not None
    )


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _ground_truth_keys() -> tuple[str, ...]:
    return (
        "relevance_judgments",
        "relevant_document_ids",
        "relevant_shot_ids",
        "relevant_frame_ids",
        "relevant_event_ids",
        "relevant_time_ranges",
        "review_notes",
    )
