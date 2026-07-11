from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from caption_pipeline.visualization import (
    _image_src,
    _resolve_image_path,
    render_caption_visualization_html,
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

            self.assertIn("Frames (1)", rendered)
            self.assertIn("Object Detection", rendered)
            self.assertIn("Audio / ASR", rendered)
            self.assertIn("Search &amp; Quality", rendered)


if __name__ == "__main__":
    unittest.main()
