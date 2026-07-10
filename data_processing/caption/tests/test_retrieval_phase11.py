"""Focused tests for Phase 11 route gating and relevance metrics."""

from __future__ import annotations

import unittest

from caption_pipeline.retrieval_eval import run_local_retrieval
from caption_pipeline.retrieval_metrics import evaluate_ground_truth, merge_ground_truth


def _doc(document_id: str, fields: dict, unit_type: str = "shot", **metadata) -> dict:
    row = {
        "document_id": document_id,
        "unit_type": unit_type,
        "video_id": "L22_V012",
        "unit_id": metadata.pop("unit_id", document_id),
        "fields": fields,
        "boosts": {},
        "search_text": " ".join(fields.values()),
        "start_sec": metadata.pop("start_sec", 0.0),
        "end_sec": metadata.pop("end_sec", 0.0),
    }
    row.update(metadata)
    return row


class RouteGatedRrfTests(unittest.TestCase):
    def test_audio_route_excludes_visual_only_distractor(self) -> None:
        corpus = [
            _doc(
                "compact:shot:true",
                {"audio_text": "chuong trinh chuyen doi so", "caption_text": "anchor"},
                shot_id="shot:true",
            ),
            _doc(
                "compact:shot:distractor",
                {
                    "audio_text": "",
                    "caption_text": "spoken content chuyen doi so on a screen",
                },
                shot_id="shot:distractor",
            ),
        ]
        query = {
            "query_id": "q_audio",
            "query": "spoken content chuyen doi so",
            "route": "audio",
            "target_unit_types": ["shot"],
        }

        results = run_local_retrieval(
            corpus,
            [query],
            top_k=10,
            scoring_profile="route_gated_rrf",
        )

        self.assertEqual([row["document_id"] for row in results], ["compact:shot:true"])
        self.assertEqual(results[0]["ranking_channels"], ["audio:1"])

    def test_unit_bonus_does_not_create_a_match(self) -> None:
        corpus = [_doc("compact:shot:none", {"audio_text": "unrelated words"})]
        query = {
            "query_id": "q_audio",
            "query": "spoken content chuyen doi so",
            "route": "audio",
        }
        self.assertEqual(
            run_local_retrieval(
                corpus,
                [query],
                top_k=10,
                scoring_profile="route_gated_rrf",
            ),
            [],
        )


class GroundTruthMetricTests(unittest.TestCase):
    def test_shot_relevance_matches_frame_from_same_shot(self) -> None:
        queries = [{
            "query_id": "q_visual",
            "route": "object_visual",
            "relevant_shot_ids": ["L22_V012_shot_0007"],
        }]
        results = [{
            "query_id": "q_visual",
            "rank": 1,
            "document_id": "compact:frame:L22_V012_020",
            "unit_id": "L22_V012_020",
            "frame_id": "L22_V012_020",
            "shot_id": "L22_V012_shot_0007",
            "start_sec": 80.0,
            "end_sec": 80.0,
        }]

        report = evaluate_ground_truth(queries, results)
        metrics = report["metrics"]
        self.assertEqual(report["num_labeled_queries"], 1)
        self.assertEqual(metrics["hit_at_1"], 1.0)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 1.0)
        self.assertEqual(metrics["ndcg_at_10"], 1.0)

    def test_time_range_relevance_uses_first_relevant_rank(self) -> None:
        queries = [{
            "query_id": "q_time",
            "route": "trake",
            "relevant_time_ranges": [[100.0, 110.0]],
        }]
        results = [
            {
                "query_id": "q_time",
                "rank": 1,
                "document_id": "wrong",
                "start_sec": 10.0,
                "end_sec": 20.0,
            },
            {
                "query_id": "q_time",
                "rank": 2,
                "document_id": "right",
                "start_sec": 105.0,
                "end_sec": 105.0,
            },
        ]

        metrics = evaluate_ground_truth(queries, results)["metrics"]
        self.assertEqual(metrics["hit_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)

    def test_ground_truth_merge_is_query_id_based(self) -> None:
        queries = [{"query_id": "q1", "query": "one"}, {"query_id": "q2", "query": "two"}]
        labels = [{"query_id": "q2", "relevant_document_ids": ["doc:2"]}]
        merged, warnings = merge_ground_truth(queries, labels)
        self.assertEqual(warnings, [])
        self.assertNotIn("relevant_document_ids", merged[0])
        self.assertEqual(merged[1]["relevant_document_ids"], ["doc:2"])


if __name__ == "__main__":
    unittest.main()
