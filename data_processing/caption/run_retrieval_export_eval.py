#!/usr/bin/env python3
"""Phase 5 retrieval export and offline evaluation CLI.

This script consumes caption pipeline outputs and produces:

- retrieval_corpus.jsonl for local lexical search checks
- Elasticsearch mapping JSON files
- optional Elasticsearch bulk JSONL files
- retrieval query/result/report files
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from caption_pipeline.io_utils import read_jsonl, write_json, write_jsonl, write_text
from caption_pipeline.retrieval_eval import run_local_retrieval
from caption_pipeline.retrieval_export import (
    build_compact_es_bulk,
    build_event_step_es_bulk,
    build_retrieval_corpus,
    compact_es_mapping,
    event_step_es_mapping,
)
from caption_pipeline.retrieval_queries import build_default_queries, normalize_queries
from caption_pipeline.retrieval_report import (
    build_retrieval_report,
    render_retrieval_report_markdown,
)

logger = logging.getLogger("caption_retrieval")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export caption outputs for retrieval and run offline eval.",
    )
    parser.add_argument("--video-id", required=True, help="Video identifier, e.g. L22_V012")
    parser.add_argument(
        "--caption-dir",
        required=True,
        help="Base caption output directory containing <video_id>/",
    )
    parser.add_argument(
        "--compact-index",
        default="",
        help="Optional compact_search_index.jsonl override",
    )
    parser.add_argument(
        "--event-step-index",
        default="",
        help="Optional event_step_index.jsonl override",
    )
    parser.add_argument(
        "--queries-jsonl",
        default="",
        help="Optional retrieval_queries.jsonl override",
    )
    parser.add_argument(
        "--eval-name",
        default="",
        help=(
            "Optional named eval/export subdirectory. Example: --eval-name visual "
            "writes eval/visual/* and exports/visual/* without overwriting default eval."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-K results per query")
    parser.add_argument(
        "--scoring-profile",
        default="default",
        choices=["default", "route_aware"],
        help="Local lexical scoring profile. default preserves old behavior.",
    )
    parser.add_argument(
        "--write-es-bulk",
        action="store_true",
        help="Write Elasticsearch bulk JSONL files",
    )
    parser.add_argument(
        "--write-default-queries",
        action="store_true",
        help="Write default query set to eval/retrieval_queries.jsonl",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if acceptance checks fail",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.monotonic()
    base_dir = Path(args.caption_dir) / args.video_id
    exports_dir = _named_output_dir(base_dir / "exports", args.eval_name)
    eval_dir = _named_output_dir(base_dir / "eval", args.eval_name)
    compact_path = Path(args.compact_index) if args.compact_index else (
        base_dir / "indexes" / "compact_search_index.jsonl"
    )
    event_step_path = Path(args.event_step_index) if args.event_step_index else (
        base_dir / "captions" / "event_step_index.jsonl"
    )
    input_queries_path = Path(args.queries_jsonl) if args.queries_jsonl else (
        eval_dir / "retrieval_queries.jsonl"
    )
    output_queries_path = eval_dir / "retrieval_queries.jsonl"

    logger.info("=== Caption Retrieval Export/Eval - %s ===", args.video_id)
    logger.info("Compact:    %s", compact_path)
    logger.info("EventStep:  %s", event_step_path)
    logger.info("Queries:    %s", input_queries_path)
    logger.info("Exports:    %s", exports_dir)
    logger.info("Eval:       %s", eval_dir)
    logger.info("Scoring:    %s", args.scoring_profile)

    missing = [str(path) for path in [compact_path, event_step_path] if not path.exists()]
    if missing:
        for path in missing:
            logger.error("Missing required input: %s", path)
        return 2

    compact_docs = read_jsonl(compact_path)
    event_step_docs = read_jsonl(event_step_path)
    logger.info(
        "Loaded %d compact docs and %d event-step docs",
        len(compact_docs), len(event_step_docs),
    )

    corpus = build_retrieval_corpus(compact_docs, event_step_docs)
    retrieval_corpus_path = exports_dir / "retrieval_corpus.jsonl"
    write_jsonl(retrieval_corpus_path, corpus)
    logger.info("Wrote retrieval corpus -> %s", retrieval_corpus_path)

    compact_mapping_path = exports_dir / "es_mapping_compact_search.json"
    event_mapping_path = exports_dir / "es_mapping_event_steps.json"
    write_json(compact_mapping_path, compact_es_mapping())
    write_json(event_mapping_path, event_step_es_mapping())
    logger.info("Wrote ES compact mapping -> %s", compact_mapping_path)
    logger.info("Wrote ES event-step mapping -> %s", event_mapping_path)

    warnings: list[str] = []
    if len(corpus) != len(compact_docs) + len(event_step_docs):
        warnings.append(
            f"corpus rows {len(corpus)} != compact + event_step "
            f"{len(compact_docs) + len(event_step_docs)}"
        )

    if args.write_es_bulk:
        compact_bulk = build_compact_es_bulk(compact_docs)
        event_bulk = build_event_step_es_bulk(event_step_docs)
        compact_bulk_path = exports_dir / "es_bulk_compact_search.jsonl"
        event_bulk_path = exports_dir / "es_bulk_event_steps.jsonl"
        write_jsonl(compact_bulk_path, compact_bulk)
        write_jsonl(event_bulk_path, event_bulk)
        logger.info(
            "Wrote ES compact bulk -> %s (%d lines)",
            compact_bulk_path, len(compact_bulk),
        )
        logger.info(
            "Wrote ES event-step bulk -> %s (%d lines)",
            event_bulk_path, len(event_bulk),
        )
        if len(compact_bulk) != len(compact_docs) * 2:
            warnings.append("compact ES bulk line count mismatch")
        if len(event_bulk) != len(event_step_docs) * 2:
            warnings.append("event-step ES bulk line count mismatch")

    queries = _load_or_build_queries(
        queries_path=input_queries_path,
        video_id=args.video_id,
        write_default=args.write_default_queries,
    )
    queries = normalize_queries(queries, args.video_id)
    write_jsonl(output_queries_path, queries)
    logger.info("Wrote/loaded %d retrieval queries -> %s", len(queries), output_queries_path)

    results = run_local_retrieval(
        corpus,
        queries,
        top_k=args.top_k,
        scoring_profile=args.scoring_profile,
    )
    results_path = eval_dir / "retrieval_results.jsonl"
    write_jsonl(results_path, results)
    logger.info("Wrote %d retrieval results -> %s", len(results), results_path)

    report = build_retrieval_report(
        video_id=args.video_id,
        corpus=corpus,
        queries=queries,
        results=results,
        top_k=args.top_k,
        warnings=warnings,
    )
    report["scoring_profile"] = args.scoring_profile
    report["eval_name"] = args.eval_name or "default"
    report_json_path = eval_dir / "retrieval_eval_report.json"
    report_md_path = eval_dir / "retrieval_eval_report.md"
    write_json(report_json_path, report)
    write_text(report_md_path, render_retrieval_report_markdown(report))
    logger.info("Wrote retrieval report JSON -> %s", report_json_path)
    logger.info("Wrote retrieval report MD -> %s", report_md_path)

    elapsed = time.monotonic() - t0
    logger.info("=== Done in %.1fs ===", elapsed)
    logger.info("  Corpus:  %d docs", len(corpus))
    logger.info("  Queries: %d", len(queries))
    logger.info("  Results: %d", len(results))
    logger.info("  Hits:    %d/%d", report["num_queries_with_hits"], report["num_queries"])
    logger.info("  Warnings: %d", len(report.get("warnings", [])))

    problems = _acceptance_problems(report, strict=args.strict)
    if problems:
        logger.warning("Acceptance issues: %s", "; ".join(problems))
        return 1 if args.strict else 0
    logger.info("All retrieval export/eval acceptance checks passed.")
    return 0

def _load_or_build_queries(
    queries_path: Path,
    video_id: str,
    write_default: bool,
) -> list[dict]:
    if queries_path.exists() and not write_default:
        rows = read_jsonl(queries_path)
        if rows:
            return rows
    rows = build_default_queries(video_id)
    return rows

def _named_output_dir(base: Path, eval_name: str) -> Path:
    name = (eval_name or "").strip().strip("/\\")
    if not name:
        return base
    if "/" in name or "\\" in name:
        raise ValueError(f"Invalid --eval-name '{eval_name}'. Use a simple folder name.")
    return base / name


def _acceptance_problems(report: dict, strict: bool) -> list[str]:
    problems: list[str] = []
    if report.get("num_corpus_docs", 0) <= 0:
        problems.append("empty retrieval corpus")
    if report.get("duplicate_document_ids", 0) > 0:
        problems.append(f"{report['duplicate_document_ids']} duplicate document_ids")
    if report.get("empty_search_text_docs", 0) > 0:
        problems.append(f"{report['empty_search_text_docs']} empty search_text docs")
    if report.get("num_queries_without_hits", 0) > 0:
        problems.append(f"{report['num_queries_without_hits']} queries without hits")
    if strict and report.get("warnings"):
        for warning in report.get("warnings") or []:
            if warning not in problems:
                problems.append(warning)
    return problems


if __name__ == "__main__":
    sys.exit(main())



