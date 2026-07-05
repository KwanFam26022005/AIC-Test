"""
Offline search for OCR VLM Pipeline v2 JSONL results.

Use this before indexing into Elasticsearch to quickly inspect whether a query
can find the expected frames.

Examples:
    python search_ocr_results.py \
      --jsonl outputs_video/L22_V012_full_w4/L22_V012_full_w4_ocr_es_docs.jsonl \
      --query "container lật ngang"

    python search_ocr_results.py --jsonl outputs_video/L22_V012_full_w4/*.jsonl \
      --query "my phuoc tan van" --include_review --show_lines
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return normalize_space(without_marks.casefold())


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(strip_accents(text)) if len(tok) >= 2]


def iter_jsonl(paths: list[Path]) -> Iterable[tuple[Path, int, dict]]:
    for path in paths:
        if path.suffix.lower() == ".json":
            try:
                with path.open("r", encoding="utf-8") as f:
                    yield path, 1, json.load(f)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip invalid JSON {path}: {exc}", file=sys.stderr)
            continue

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] skip invalid JSON {path}:{line_no}: {exc}", file=sys.stderr)


def resolve_jsonl_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [candidate]
        for path in matches:
            if path.is_dir():
                paths.extend(path.glob("*_ocr_es_docs.jsonl"))
                paths.extend(path.glob("**/*_ocr_es_docs.jsonl"))
                continue
            if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}:
                paths.append(path)
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError(f"No JSONL/JSON files matched: {patterns}")
    return unique


def doc_text(doc: dict, include_review: bool) -> str:
    parts = [
        doc.get("ocr_text_search") or "",
        doc.get("ocr_text_clean") or "",
        " ".join(doc.get("ocr_group_texts_clean") or []),
    ]
    if include_review:
        parts.extend([
            doc.get("ocr_text_review") or "",
            " ".join(doc.get("ocr_group_texts_review") or []),
        ])
    return normalize_space("\n".join(str(part) for part in parts if part))


def best_token_similarity(token: str, candidates: set[str]) -> float:
    if not token or not candidates:
        return 0.0
    best = 0.0
    for candidate in candidates:
        if not candidate:
            continue
        # Cheap guard before SequenceMatcher on large term sets.
        if abs(len(token) - len(candidate)) > max(3, len(token) // 2):
            continue
        ratio = SequenceMatcher(None, token, candidate).ratio()
        if ratio > best:
            best = ratio
    return best


def score_doc(
    doc: dict,
    query: str,
    query_norm: str,
    query_tokens: list[str],
    include_review: bool,
    fuzzy_threshold: float,
) -> tuple[float, dict]:
    text = doc_text(doc, include_review)
    text_norm = strip_accents(text)
    doc_terms = set(doc.get("ocr_terms") or [])
    if not doc_terms:
        doc_terms = set(tokenize(text))

    score = 0.0
    exact_terms = []
    fuzzy_terms = []
    missing_terms = []

    if query_norm and query_norm in text_norm:
        score += 5.0 + min(len(query_tokens), 8) * 0.5

    for token in query_tokens:
        if token in doc_terms or re.search(rf"\b{re.escape(token)}\b", text_norm):
            score += 1.5
            exact_terms.append(token)
            continue

        best = best_token_similarity(token, doc_terms)
        if best >= fuzzy_threshold:
            score += best
            fuzzy_terms.append((token, round(best, 3)))
        else:
            missing_terms.append(token)

    coverage = (len(exact_terms) + len(fuzzy_terms)) / max(1, len(query_tokens))
    score += coverage * 2.0

    # Prefer documents with more clean text when scores are tied.
    clean_len = len(doc.get("ocr_text_clean") or "")
    score += min(clean_len, 500) / 5000.0

    detail = {
        "coverage": round(coverage, 3),
        "exact_terms": exact_terms,
        "fuzzy_terms": fuzzy_terms,
        "missing_terms": missing_terms,
    }
    return score, detail


def line_matches(doc: dict, query_tokens: list[str], include_review: bool, max_lines: int) -> list[str]:
    matches = []
    for line in doc.get("ocr_lines") or []:
        texts = [
            line.get("final_text") or "",
            line.get("vintern_text") or "",
            line.get("vietocr_text") or "",
        ]
        if include_review:
            texts.append(line.get("raw_text") or "")
        combined = normalize_space(" | ".join(text for text in texts if text))
        combined_norm = strip_accents(combined)
        if any(tok in combined_norm for tok in query_tokens):
            line_id = line.get("line_id")
            source = line.get("final_source") or line.get("filter_status") or ""
            matches.append(f"line={line_id} source={source}: {combined}")
        if len(matches) >= max_lines:
            break
    return matches


def snippet(doc: dict, query_norm: str, include_review: bool, width: int = 220) -> str:
    text = doc_text(doc, include_review)
    text_norm = strip_accents(text)
    if query_norm:
        pos = text_norm.find(query_norm)
    else:
        pos = -1
    if pos < 0:
        return normalize_space(text)[:width]
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + normalize_space(text[start:end]) + suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search OCR JSONL results locally")
    parser.add_argument("--jsonl", nargs="+", required=True, help="JSONL path or glob pattern")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to show")
    parser.add_argument("--min_score", type=float, default=1.0, help="Minimum score to keep")
    parser.add_argument("--fuzzy_threshold", type=float, default=0.78, help="Token fuzzy threshold")
    parser.add_argument("--include_review", action="store_true", help="Also search review text")
    parser.add_argument("--show_lines", action="store_true", help="Show matching OCR lines")
    parser.add_argument("--frame_from", type=int, default=None, help="Minimum frame_number")
    parser.add_argument("--frame_to", type=int, default=None, help="Maximum frame_number")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_jsonl_patterns(args.jsonl)
    query_norm = strip_accents(args.query)
    query_tokens = tokenize(args.query)
    if not query_tokens:
        raise ValueError("Query must contain at least one searchable token")

    heap: list[tuple[float, int, dict]] = []
    scanned = 0
    matched = 0

    for path, line_no, doc in iter_jsonl(paths):
        frame_number = doc.get("frame_number")
        if args.frame_from is not None and frame_number is not None and frame_number < args.frame_from:
            continue
        if args.frame_to is not None and frame_number is not None and frame_number > args.frame_to:
            continue

        scanned += 1
        score, detail = score_doc(
            doc,
            args.query,
            query_norm,
            query_tokens,
            args.include_review,
            args.fuzzy_threshold,
        )
        if score < args.min_score:
            continue

        matched += 1
        result = {
            "score": round(score, 3),
            "document_id": doc.get("document_id"),
            "video_id": doc.get("video_id"),
            "frame_id": doc.get("frame_id"),
            "frame_number": frame_number,
            "image_path": doc.get("image_path"),
            "snippet": snippet(doc, query_norm, args.include_review),
            "quality": doc.get("quality") or {},
            "match": detail,
            "source": f"{path}:{line_no}",
        }
        if args.show_lines:
            result["line_matches"] = line_matches(doc, query_tokens, args.include_review, max_lines=8)

        item = (score, scanned, result)
        if len(heap) < args.top_k:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    results = [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[1]))]
    if args.json:
        print(json.dumps({
            "query": args.query,
            "paths": [str(p) for p in paths],
            "scanned": scanned,
            "matched": matched,
            "results": results,
        }, ensure_ascii=False, indent=2))
        return

    print(f"Query: {args.query}")
    print(f"Files: {len(paths)} | scanned={scanned} matched={matched} shown={len(results)}")
    print("-" * 88)
    for rank, result in enumerate(results, start=1):
        quality = result.get("quality") or {}
        print(
            f"{rank:02d}. score={result['score']:.3f} "
            f"doc={result.get('document_id')} frame={result.get('frame_id')} "
            f"clean={quality.get('clean_chars', 0)} review={quality.get('review_chars', 0)}"
        )
        print(f"    image: {result.get('image_path')}")
        print(f"    text:  {result.get('snippet')}")
        match = result.get("match") or {}
        print(
            f"    match: coverage={match.get('coverage')} "
            f"exact={match.get('exact_terms')} fuzzy={match.get('fuzzy_terms')} "
            f"missing={match.get('missing_terms')}"
        )
        for line in result.get("line_matches") or []:
            print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
