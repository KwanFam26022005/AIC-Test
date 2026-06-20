# -*- coding: utf-8 -*-
"""
search_simulator.py — Giả lập Multi-Channel Search trên metadata Object Detection

Mô phỏng kiến trúc tìm kiếm MMRS thực tế (FAISS + Elasticsearch) nhưng chạy
hoàn toàn offline trên file JSONL từ ram_gdino_pipeline.py.

3 kênh tìm kiếm (channels):
    Channel 1 — TAGS:    Khớp từ khóa trong danh sách tags từ RAM++
    Channel 2 — OBJECTS: Khớp label + confidence score từ GroundingDINO
    Channel 3 — COUNTS:  So khớp điều kiện số lượng object_counts

Kết hợp bằng Reciprocal Rank Fusion (RRF) — cùng công thức mà hệ thống
production sẽ dùng với FAISS + Elasticsearch.

Cách sử dụng:
    # CLI trực tiếp
    python search_simulator.py --metadata /path/to/output.jsonl --query "person fish"

    # Interactive shell
    python search_simulator.py --metadata /path/to/output.jsonl

    # Chạy demo với các query mẫu
    python search_simulator.py --metadata /path/to/output.jsonl --demo

Query syntax:
    - Từ khóa đơn:       "person"
    - Nhiều từ khóa:     "person fish market"
    - Số lượng:          "2 person"    (frame phải có >= 2 PERSON)
    - Kết hợp:           "2 person fish"  (>= 2 PERSON + có fish)
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


# =============================================================================
# DATA LOADING
# =============================================================================

def load_metadata(metadata_path):
    """Đọc tất cả frame metadata từ JSONL file hoặc thư mục chứa JSONL.

    Args:
        metadata_path: str — path tới file .jsonl hoặc thư mục

    Returns:
        list[dict]: danh sách frame metadata
    """
    all_data = []
    paths = []

    if os.path.isdir(metadata_path):
        paths = sorted(glob.glob(os.path.join(metadata_path, "*.jsonl")))
    elif os.path.isfile(metadata_path):
        paths = [metadata_path]

    if not paths:
        print(f"[ERROR] Không tìm thấy file JSONL: {metadata_path}", file=sys.stderr)
        return []

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    print(f"Loaded {len(all_data)} frames from {len(paths)} file(s)")
    return all_data


# =============================================================================
# QUERY PARSING
# =============================================================================

STOP_WORDS = frozenset({
    "and", "or", "a", "an", "the", "in", "on", "at", "with", "of", "to",
    "is", "are", "was", "were", "có", "và", "những", "một", "các", "đang",
    "near", "next", "beside", "by", "for",
})


def parse_query(query_str):
    """Phân tích query → (keywords, count_constraints).

    Ví dụ:
        "2 person fish market"
        → keywords = ["person", "fish", "market"]
        → count_constraints = {"person": 2}

        "3 cow and 1 dog"
        → keywords = ["cow", "dog"]
        → count_constraints = {"cow": 3, "dog": 1}

    Args:
        query_str: str — câu truy vấn

    Returns:
        tuple: (keywords: list[str], count_constraints: dict[str, int])
    """
    query_str = query_str.lower().strip()

    # Tìm cụm: <số> <từ> — ví dụ "2 person", "3 cows"
    count_constraints = {}
    for match in re.finditer(r"(\d+)\s+([a-z_]+)", query_str):
        count = int(match.group(1))
        name = match.group(2)
        # Chuẩn hóa số nhiều: "persons" → "person", "cows" → "cow"
        if name.endswith("s") and len(name) > 3:
            name = name[:-1]
        if count > 0:
            count_constraints[name] = count

    # Extract tất cả từ khóa (bỏ stop words và số)
    words = re.findall(r"[a-z_]+", query_str)
    keywords = []
    seen = set()
    for w in words:
        if w.endswith("s") and len(w) > 3:
            w = w[:-1]
        if w not in STOP_WORDS and w not in seen and not w.isdigit():
            keywords.append(w)
            seen.add(w)

    return keywords, count_constraints


# =============================================================================
# MULTI-CHANNEL SCORING
# =============================================================================

def score_channel_tags(frame, keywords):
    """Channel 1 — TAG SEARCH: Khớp từ khóa trong danh sách tags.

    Cơ chế: Mỗi keyword khớp với 1 tag → +1 điểm.
    Tags là output thô từ RAM++ (scene-level + object-level).

    Args:
        frame: dict — frame metadata
        keywords: list[str]

    Returns:
        tuple: (score, details_list)
    """
    tags = [t.lower() for t in frame.get("tags", [])]
    score = 0.0
    details = []

    for kw in keywords:
        for tag in tags:
            if kw == tag or kw in tag or tag in kw:
                score += 1.0
                details.append(f"tag:\"{tag}\"")
                break  # Mỗi keyword chỉ khớp 1 lần trong channel này

    return score, details


def score_channel_objects(frame, keywords):
    """Channel 2 — OBJECT SEARCH: Khớp label + tận dụng confidence score.

    Cơ chế:
        - Mỗi keyword khớp object label → +1.0 điểm base
        - Cộng thêm confidence score của detection (0.0–1.0)
        - Cộng thêm 0.5 cho mỗi instance bổ sung (cùng label)

    GDINO cho kết quả tin cậy hơn tags vì đã xác nhận bằng bounding box.

    Args:
        frame: dict
        keywords: list[str]

    Returns:
        tuple: (score, details_list)
    """
    objects = frame.get("objects", [])
    score = 0.0
    details = []

    for kw in keywords:
        matched_objs = []
        for obj in objects:
            label = obj["label"].lower().replace("_", " ")
            if kw == label or kw in label or label in kw:
                matched_objs.append(obj)

        if matched_objs:
            # Base score cho keyword match
            score += 1.0
            # Cộng confidence của detection tốt nhất
            best = max(matched_objs, key=lambda o: o["score"])
            score += best["score"]
            # Bonus cho multi-instance
            if len(matched_objs) > 1:
                score += 0.5 * (len(matched_objs) - 1)

            details.append(
                f"obj:{best['label']}×{len(matched_objs)} "
                f"(best={best['score']:.2f})"
            )

    return score, details


def score_channel_counts(frame, count_constraints):
    """Channel 3 — COUNT SEARCH: So khớp yêu cầu số lượng.

    Cơ chế:
        - Đúng hoặc nhiều hơn yêu cầu → +3.0 điểm
        - Có nhưng không đủ → điểm tỷ lệ (actual/required) × 1.5
        - Không có → 0 điểm

    Args:
        frame: dict
        count_constraints: dict[str, int]

    Returns:
        tuple: (score, details_list)
    """
    if not count_constraints:
        return 0.0, []

    counts = {k.lower(): v for k, v in frame.get("object_counts", {}).items()}
    score = 0.0
    details = []

    for obj_name, required in count_constraints.items():
        # Tìm key phù hợp trong object_counts
        actual = 0
        matched_key = obj_name
        for k, v in counts.items():
            if obj_name == k or obj_name in k or k in obj_name:
                actual = v
                matched_key = k
                break

        if actual >= required:
            score += 3.0
            details.append(f"count:{matched_key.upper()}={actual}≥{required} ✓")
        elif actual > 0:
            ratio = actual / required
            score += 1.5 * ratio
            details.append(f"count:{matched_key.upper()}={actual}<{required} ({ratio:.0%})")

    return score, details


# =============================================================================
# RRF FUSION
# =============================================================================

def rrf_fusion(rankings, k=60):
    """Reciprocal Rank Fusion — kết hợp rankings từ nhiều channels.

    Công thức: score(d) = Σ  1 / (k + rank(d))  cho mỗi channel

    Đây là công thức chuẩn mà hệ thống production (FAISS + ES) sẽ dùng.

    Args:
        rankings: list[list[str]] — mỗi list là danh sách frame_ids
                  đã sort theo score giảm dần của channel đó
        k: int — RRF constant (default 60, chuẩn industry)

    Returns:
        dict[str, float]: frame_id → RRF score
    """
    rrf_scores = defaultdict(float)
    for ranking in rankings:
        for rank, frame_id in enumerate(ranking):
            rrf_scores[frame_id] += 1.0 / (k + rank)
    return dict(rrf_scores)


# =============================================================================
# SEARCH ENGINE
# =============================================================================

def search(all_data, query_str, top_k=10):
    """Multi-channel search với RRF fusion.

    Pipeline:
        1. Parse query → keywords + count_constraints
        2. Score mỗi frame trên 3 channels riêng biệt
        3. Sort mỗi channel → ranking list
        4. RRF fusion → final ranking
        5. Return top_k kết quả kèm chi tiết scoring

    Args:
        all_data: list[dict] — tất cả frame metadata
        query_str: str — câu truy vấn
        top_k: int — số kết quả trả về

    Returns:
        list[dict]: top_k kết quả, mỗi dict chứa:
            - frame: dict (metadata gốc)
            - rrf_score: float
            - channel_scores: dict (điểm từng channel)
            - channel_details: dict (chi tiết khớp từng channel)
    """
    keywords, count_constraints = parse_query(query_str)

    if not keywords and not count_constraints:
        return [], keywords, count_constraints

    # Score từng frame trên 3 channels
    frame_scores = {}  # frame_id → {channel: (score, details)}

    for frame in all_data:
        fid = frame["frame_id"]

        s_tags, d_tags = score_channel_tags(frame, keywords)
        s_objs, d_objs = score_channel_objects(frame, keywords)
        s_cnts, d_cnts = score_channel_counts(frame, count_constraints)

        total = s_tags + s_objs + s_cnts
        if total > 0:
            frame_scores[fid] = {
                "frame": frame,
                "scores": {"tags": s_tags, "objects": s_objs, "counts": s_cnts},
                "details": {"tags": d_tags, "objects": d_objs, "counts": d_cnts},
            }

    if not frame_scores:
        return [], keywords, count_constraints

    # Tạo ranking cho từng channel (sort theo score giảm dần)
    ranking_tags = sorted(
        frame_scores.keys(),
        key=lambda fid: frame_scores[fid]["scores"]["tags"],
        reverse=True,
    )
    ranking_objs = sorted(
        frame_scores.keys(),
        key=lambda fid: frame_scores[fid]["scores"]["objects"],
        reverse=True,
    )
    ranking_cnts = sorted(
        frame_scores.keys(),
        key=lambda fid: frame_scores[fid]["scores"]["counts"],
        reverse=True,
    )

    # RRF fusion
    rrf = rrf_fusion([ranking_tags, ranking_objs, ranking_cnts])

    # Sort theo RRF score giảm dần
    ranked_ids = sorted(rrf.keys(), key=lambda fid: rrf[fid], reverse=True)

    # Build results
    results = []
    for fid in ranked_ids[:top_k]:
        info = frame_scores[fid]
        results.append({
            "frame": info["frame"],
            "rrf_score": rrf[fid],
            "channel_scores": info["scores"],
            "channel_details": info["details"],
        })

    return results, keywords, count_constraints


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_results(results, query_str, keywords, count_constraints):
    """In kết quả với chi tiết từng channel scoring."""
    if not results:
        print(f"\n  Không tìm thấy kết quả cho: \"{query_str}\"\n")
        return

    print(f"\n{'='*90}")
    print(f"  QUERY: \"{query_str}\"")
    print(f"  Parsed → keywords={keywords}, counts={count_constraints}")
    print(f"{'='*90}")

    for rank, res in enumerate(results, 1):
        frame = res["frame"]
        rrf = res["rrf_score"]
        cs = res["channel_scores"]
        cd = res["channel_details"]

        # Header
        print(f"\n  #{rank}  {frame['frame_id']:<22}  RRF={rrf:.4f}")
        print(f"  {'─'*60}")

        # Channel breakdown
        # Tags
        tags_detail = ", ".join(cd["tags"]) if cd["tags"] else "—"
        print(f"  Ch.1 TAGS    │ score={cs['tags']:<6.2f} │ {tags_detail}")

        # Objects
        objs_detail = ", ".join(cd["objects"]) if cd["objects"] else "—"
        print(f"  Ch.2 OBJECTS │ score={cs['objects']:<6.2f} │ {objs_detail}")

        # Counts
        cnts_detail = ", ".join(cd["counts"]) if cd["counts"] else "—"
        print(f"  Ch.3 COUNTS  │ score={cs['counts']:<6.2f} │ {cnts_detail}")

        # Frame summary
        n_tags = len(frame.get("tags", []))
        n_objs = len(frame.get("objects", []))
        obj_labels = frame.get("object_summary", [])
        top_tags = frame.get("tags", [])[:8]

        print(f"  {'─'*60}")
        print(f"  Summary: {n_objs} objects {obj_labels}, {n_tags} tags")
        print(f"  Tags:    {', '.join(top_tags)}{'...' if n_tags > 8 else ''}")

        # Object counts
        oc = frame.get("object_counts", {})
        if oc:
            counts_str = ", ".join(f"{k}={v}" for k, v in sorted(oc.items()))
            print(f"  Counts:  {counts_str}")

    print(f"\n{'='*90}")
    print(f"  {len(results)} kết quả (RRF fusion từ 3 channels: TAGS + OBJECTS + COUNTS)")
    print(f"{'='*90}\n")


# =============================================================================
# DEMO MODE
# =============================================================================

DEMO_QUERIES = [
    ("person",            "Tìm tất cả frame có người"),
    ("fish market",       "Tìm frame có cá + chợ (multi-keyword)"),
    ("2 person",          "Tìm frame có ít nhất 2 người (count constraint)"),
    ("cow",               "Tìm frame có bò"),
    ("car truck",         "Tìm frame có xe hơi hoặc xe tải"),
    ("3 person fish",     "Kết hợp: ≥3 người + có cá"),
    ("tree house",        "Tìm frame có cây + nhà"),
    ("boat water",        "Tìm frame có thuyền + nước"),
]


def run_demo(all_data):
    """Chạy demo với các query mẫu — minh họa sức mạnh của 3 channels."""
    print("\n" + "=" * 90)
    print("  DEMO MODE — Chạy các query mẫu để minh họa multi-channel search")
    print("=" * 90)

    for query, description in DEMO_QUERIES:
        print(f"\n{'─'*90}")
        print(f"  Demo: {description}")
        results, keywords, count_constraints = search(all_data, query, top_k=3)
        print_results(results, query, keywords, count_constraints)


# =============================================================================
# CLI & MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Channel Search Simulator (Tags + Objects + Counts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python search_simulator.py --metadata output/L21_V001.jsonl --query "person fish"
  python search_simulator.py --metadata output/ --demo
  python search_simulator.py --metadata output/L21_V001.jsonl  (interactive)
        """,
    )
    parser.add_argument(
        "--metadata", required=True,
        help="File JSONL hoặc thư mục chứa các file JSONL",
    )
    parser.add_argument(
        "--query", default=None,
        help="Câu truy vấn (chạy 1 lần rồi thoát)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Chạy demo với các query mẫu",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Số kết quả trả về (default: 10)",
    )

    args = parser.parse_args()

    # Load data
    all_data = load_metadata(args.metadata)
    if not all_data:
        sys.exit(1)

    # Tóm tắt dữ liệu
    all_tags = set()
    all_labels = set()
    for frame in all_data:
        all_tags.update(frame.get("tags", []))
        all_labels.update(frame.get("object_summary", []))
    print(f"  Unique tags: {len(all_tags)}, Unique object labels: {len(all_labels)}")

    # Mode: demo
    if args.demo:
        run_demo(all_data)
        sys.exit(0)

    # Mode: single query
    if args.query:
        results, keywords, count_constraints = search(all_data, args.query, top_k=args.top_k)
        print_results(results, args.query, keywords, count_constraints)
        sys.exit(0)

    # Mode: interactive
    print("\n" + "─" * 70)
    print("  MMRS Search Simulator — Interactive Mode")
    print("  Channels: [TAGS] + [OBJECTS] + [COUNTS] → RRF Fusion")
    print("─" * 70)
    print("  Syntax:")
    print("    person fish         → tìm frame có person + fish")
    print("    2 person            → tìm frame có ≥ 2 person")
    print("    3 person fish       → ≥ 3 person + có fish")
    print("    demo                → chạy demo queries")
    print("    exit                → thoát")
    print("─" * 70 + "\n")

    while True:
        try:
            query = input("query> ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                break
            if query.lower() == "demo":
                run_demo(all_data)
                continue

            results, keywords, count_constraints = search(all_data, query, top_k=args.top_k)
            print_results(results, query, keywords, count_constraints)

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
