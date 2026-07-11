"""Unit tests for the prompt-backed official caption runtime."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from caption_pipeline.config import PipelineConfig
from caption_pipeline.frame_caption_fuser import fuse_frame_captions
from caption_pipeline.runtime import JsonlCheckpoint, TextLLMRuntime
from caption_pipeline.runtime.json_output import parse_json_object
from caption_pipeline.shot_captioner import fuse_shot_captions
from caption_pipeline.trake_event_builder import build_trake_event_steps


class OfficialRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PipelineConfig(video_id="L22_V012")
        self.cfg.frame_caption.caption_mode = "llm"
        self.cfg.shot_caption.caption_mode = "llm"
        self.cfg.trake_event.event_mode = "llm"
        self.cfg.llm.provider = "mock"
        self.cfg.llm.model_name = "mock-caption-model"
        self.cfg.llm.prompt_dir = str(
            Path(__file__).resolve().parents[1] / "caption_pipeline" / "prompts"
        )
        self.runtime = TextLLMRuntime(self.cfg.llm)

    def test_json_parser_accepts_fenced_object(self) -> None:
        result = parse_json_object(
            '```json\n{"caption_text": "A presenter."}\n```',
            required_fields=("caption_text",),
        )
        self.assertEqual(result["caption_text"], "A presenter.")

    def test_mock_runtime_runs_all_three_caption_levels(self) -> None:
        frame_evidence = [{
            "canonical_frame_id": "L22_V012_001",
            "object_evidence": {
                "scene_tags": ["studio"],
                "important_objects": ["1 person", "1 screen"],
            },
            "ocr_evidence": {"ocr_text": "HTV7"},
            "audio_evidence": {"audio_text": "Tin tức"},
            "quality": {},
        }]
        frame_records = fuse_frame_captions(
            frame_evidence,
            self.cfg,
            llm_runtime=self.runtime,
            initial_captions={"L22_V012_001": "A presenter stands in a studio."},
        )
        self.assertEqual(frame_records[0]["caption_mode"], "llm")
        self.assertFalse(frame_records[0]["fallback_used"])

        frame_index = [{
            "canonical_frame_id": "L22_V012_001",
            "caption_text": frame_records[0]["caption_text"],
        }]
        shot_evidence = [{
            "video_id": "L22_V012",
            "shot_id": "L22_V012_shot_0001",
            "start_sec": 0.0,
            "end_sec": 3.0,
            "representative_frame_ids": ["L22_V012_001"],
            "merged_object_counts": {"person": 1, "screen": 1},
            "merged_scene_tags": ["studio"],
            "merged_ocr_text": "HTV7",
            "merged_audio_text": "Tin tức",
        }]
        shot_records = fuse_shot_captions(
            shot_evidence,
            frame_index,
            self.cfg,
            llm_runtime=self.runtime,
        )
        self.assertEqual(shot_records[0]["caption_mode"], "llm")

        shot_index = [{
            **shot_evidence[0],
            "caption_text": shot_records[0]["caption_text"],
            "temporal_caption": shot_records[0]["temporal_caption"],
            "evidence_text": {
                "merged_ocr_text": "HTV7",
                "merged_audio_text": "Tin tức",
                "object_text": "1 person, 1 screen",
                "scene_text": "studio",
            },
        }]
        events = build_trake_event_steps(
            shot_index,
            shot_evidence,
            self.cfg,
            llm_runtime=self.runtime,
        )
        self.assertEqual(events[0]["event_mode"], "llm")
        self.assertTrue(events[0]["trake_text"])

    def test_checkpoint_reuses_matching_non_fallback_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            checkpoint = JsonlCheckpoint(path, "frame_id", "sig", write_every=1)
            checkpoint.put({"frame_id": "f1", "fallback_used": False})
            resumed = JsonlCheckpoint(path, "frame_id", "sig", resume=True)
            self.assertEqual(resumed.get("f1")["frame_id"], "f1")


if __name__ == "__main__":
    unittest.main()
