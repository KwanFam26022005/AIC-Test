from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_caption_audio_context_pipeline import main


class CaptionAudioContextPipelineV2Tests(unittest.TestCase):
    def test_mock_pipeline_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_id = "L22_V012"
            frame_dir = root / "keyframe_test" / video_id
            frame_dir.mkdir(parents=True)
            (frame_dir / "001.jpg").write_bytes(b"fake-jpg-1")
            (frame_dir / "002.jpg").write_bytes(b"fake-jpg-2")
            (frame_dir / "003.jpg").write_bytes(b"fake-jpg-3")
            (frame_dir / f"{video_id}.csv").write_text(
                "n,pts_time,fps,frame_idx\n"
                "1,0.0,30.0,0\n"
                "2,3.0,30.0,90\n"
                "3,12.0,30.0,360\n",
                encoding="utf-8",
            )

            audio_dir = root / "outputs" / "audio" / video_id / "features"
            audio_dir.mkdir(parents=True)
            audio_path = audio_dir / "audio_features.jsonl"
            audio_path.write_text(
                json.dumps(
                    {
                        "schema_version": "audio_feature_v1",
                        "feature_id": "aud_1",
                        "video_id": video_id,
                        "start_sec": 0.0,
                        "end_sec": 5.0,
                        "caption_text": "Nguoi dan dang noi ve chuong trinh.",
                        "clean_transcript": "Nguoi dan dang noi ve chuong trinh.",
                        "usable_for_caption": True,
                        "quality_level": "high",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            config_path = root / "config.yaml"
            output_root = root / "caption_v2_mock"
            config_path.write_text(
                f"""
pipeline:
  video_id: {video_id}
inputs:
  keyframe_map: {root / 'keyframe_test'}
  keyframes_root: {root / 'keyframe_test'}
  audio_features: {audio_path}
paths:
  output_root: {output_root}
  prompt_dir: ""
models:
  vlm:
    provider: mock
    model_name: mock-vlm
  text_llm:
    provider: mock
    model_name: mock-llm
frame_selection:
  similarity_method: none
  max_selected_per_shot: 1
  force_first_last_for_long_shots: false
generation:
  shot_max_gap_sec: 5.0
  shot_max_duration_sec: 20.0
checkpoint:
  resume: false
strict: true
""",
                encoding="utf-8",
            )

            code = main(["--config", str(config_path), "--strict"])
            self.assertEqual(code, 0)

            out_dir = output_root / video_id
            frame_rows = _read_jsonl(out_dir / "captions" / "frame_captions.jsonl")
            shot_rows = _read_jsonl(out_dir / "captions" / "shot_captions.jsonl")
            manifest = json.loads((out_dir / "reports" / "caption_v2_report.json").read_text(encoding="utf-8"))

            self.assertEqual([row["canonical_frame_id"] for row in frame_rows], [
                "L22_V012_001",
                "L22_V012_002",
                "L22_V012_003",
            ])
            self.assertEqual(len(shot_rows), 2)
            self.assertEqual(manifest["status"], "pass")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
