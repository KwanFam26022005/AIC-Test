from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from caption_pipeline.visualization import (
    _image_src,
    _resolve_image_path,
    render_caption_visualization_html,
    write_caption_visualization,
)


class CaptionVisualizationTests(unittest.TestCase):
    def test_resolves_canonical_id_to_numeric_keyframe_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "keyframes"
            image = root / "L22_V012" / "001.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpeg-test")

            frame = {
                "video_id": "L22_V012",
                "frame_id": "L22_V012_001",
                "canonical_frame_id": "L22_V012_001",
            }
            resolved, _ = _resolve_image_path(frame, root)

            self.assertEqual(resolved, image)

    def test_embed_mode_produces_portable_data_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "keyframes"
            image = root / "L22_V012" / "001.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpeg-test")
            output = Path(tmp) / "reports" / "visualization.html"
            frame = {
                "video_id": "L22_V012",
                "canonical_frame_id": "L22_V012_001",
            }

            src, _ = _image_src(frame, output, root, "embed")

            self.assertTrue(src.startswith("data:image/jpeg;base64,"))

    def test_render_includes_feature_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "visualization.html"
            frame = {
                "video_id": "L22_V012",
                "frame_id": "L22_V012_001",
                "canonical_frame_id": "L22_V012_001",
                "timestamp_sec": 0.0,
                "caption_text": "A news studio.",
                "quality": {"caption_mode": "llm"},
            }
            shot = {
                "video_id": "L22_V012",
                "shot_id": "L22_V012_shot_0001",
                "start_sec": 0.0,
                "end_sec": 0.0,
                "frame_count": 1,
                "representative_frame_ids": ["L22_V012_001"],
                "caption_text": "A presenter appears in a studio.",
                "temporal_caption": "The program begins.",
                "evidence_text": {"merged_ocr_text": "HTV7"},
                "search_fields": {"onscreen_text": "HTV7"},
                "quality": {"caption_mode": "llm"},
            }
            event = {
                "shot_id": "L22_V012_shot_0001",
                "event_id": "L22_V012_event_0001",
                "step_order": 1,
                "event_caption": "The broadcast opens.",
                "trake_text": "The presenter opens the broadcast.",
                "temporal_role": "beginning",
                "quality": {"event_mode": "llm"},
            }
            frame_evidence = {
                "canonical_frame_id": "L22_V012_001",
                "object_evidence": {"object_counts": {"person": 1}},
                "ocr_evidence": {"ocr_text": "HTV7"},
                "audio_evidence": {"audio_text": "Welcome", "num_segments": 1},
                "quality": {"has_object": True, "has_ocr": True, "has_audio": True},
            }
            shot_evidence = {
                "shot_id": "L22_V012_shot_0001",
                "merged_object_counts": {"person": 1},
                "merged_scene_tags": ["studio"],
            }

            rendered = render_caption_visualization_html(
                video_id="L22_V012",
                base_dir=Path(tmp),
                frame_index=[frame],
                shot_index=[shot],
                event_index=[event],
                compact_docs=[],
                frame_evidence=[frame_evidence],
                shot_evidence=[shot_evidence],
                reports={},
                output_path=output,
                keyframes_root=None,
                max_frames_per_shot=3,
                image_mode="embed",
            )

            self.assertIn("Shot Review", rendered)
            self.assertIn("Event / TRAKE", rendered)
            self.assertIn("Objects", rendered)
            self.assertIn("Audio / ASR", rendered)

    def test_write_visualization_creates_frame_and_shot_pages_with_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            caption_video = root / "caption" / "L22_V012"
            keyframes = root / "keyframes" / "L22_V012"
            keyframes.mkdir(parents=True)
            (keyframes / "001.jpg").write_bytes(b"jpeg-test")
            for dirname in ("captions", "evidence", "indexes", "reports"):
                (caption_video / dirname).mkdir(parents=True)

            _write_jsonl(
                caption_video / "captions" / "frame_index.jsonl",
                [{
                    "video_id": "L22_V012",
                    "frame_id": "L22_V012_001",
                    "canonical_frame_id": "L22_V012_001",
                    "timestamp_sec": 0.0,
                    "caption_text": "A news studio.",
                    "quality": {"caption_mode": "llm"},
                }],
            )
            _write_jsonl(
                caption_video / "captions" / "shot_index.jsonl",
                [{
                    "video_id": "L22_V012",
                    "shot_id": "L22_V012_shot_0001",
                    "start_sec": 0.0,
                    "end_sec": 0.1,
                    "frame_count": 1,
                    "representative_frame_ids": ["L22_V012_001"],
                    "caption_text": "A presenter appears in a studio.",
                    "evidence_text": {"merged_ocr_text": "HTV7"},
                    "quality": {"caption_mode": "llm"},
                }],
            )
            _write_jsonl(
                caption_video / "captions" / "event_step_index.jsonl",
                [{
                    "shot_id": "L22_V012_shot_0001",
                    "event_id": "L22_V012_event_0001",
                    "event_caption": "The broadcast opens.",
                    "trake_text": "The presenter opens the broadcast.",
                    "quality": {"event_mode": "llm"},
                }],
            )
            _write_jsonl(
                caption_video / "indexes" / "compact_search_index.jsonl",
                [{"document_id": "doc-1"}],
            )
            _write_jsonl(
                caption_video / "evidence" / "frame_evidence.jsonl",
                [{
                    "canonical_frame_id": "L22_V012_001",
                    "object_evidence": {"object_counts": {"person": 1}},
                    "ocr_evidence": {"ocr_text": "HTV7"},
                    "audio_evidence": {"audio_text": "Welcome", "num_segments": 1},
                    "quality": {"has_object": True, "has_ocr": True, "has_audio": True},
                }],
            )
            _write_jsonl(
                caption_video / "evidence" / "shot_evidence.jsonl",
                [{
                    "shot_id": "L22_V012_shot_0001",
                    "merged_object_counts": {"person": 1},
                    "merged_scene_tags": ["studio"],
                }],
            )

            index_path = write_caption_visualization(
                caption_dir=root / "caption",
                video_id="L22_V012",
                keyframes_root=root / "keyframes",
                image_mode="copy",
            )

            frame_html = (index_path.parent / "frame_review.html").read_text(encoding="utf-8")
            shot_html = (index_path.parent / "shot_review.html").read_text(encoding="utf-8")

            self.assertTrue(index_path.exists())
            self.assertIn('<img src="assets/L22_V012_001.jpg"', frame_html)
            self.assertIn('class="feature-dialog"', frame_html)
            self.assertIn("Open frame detail", frame_html)
            self.assertIn('<img src="assets/L22_V012_001.jpg"', shot_html)
            self.assertIn('class="shot-detail"', shot_html)
            self.assertTrue((index_path.parent / "assets" / "L22_V012_001.jpg").exists())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
