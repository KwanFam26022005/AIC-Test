#!/usr/bin/env python3
"""Run the official caption workflow in separate GPU-safe subprocess stages."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from caption_pipeline.io_utils import write_text
from caption_pipeline.runtime import expand_env_values, load_env_file

logger = logging.getLogger("caption_official_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official caption stages end to end.")
    parser.add_argument("--config", required=True, help="Official pipeline YAML config")
    parser.add_argument("--env-file", action="append", default=[])
    parser.add_argument("--env-override", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for env_file in args.env_file:
        load_env_file(env_file, override=args.env_override)
    cfg = _load_config(args.config)
    _validate_config(cfg)

    root = Path(__file__).resolve().parent
    pipeline = cfg["pipeline"]
    inputs = cfg["inputs"]
    paths = cfg["paths"]
    models = cfg["models"]
    generation = cfg.get("generation") or {}
    video_id = pipeline["video_id"]
    output_root = Path(paths["output_root"])
    work_root = Path(paths["work_root"])
    env = dict(os.environ)

    common = [
        "--video-id", video_id,
        "--ocr-jsonl", inputs["ocr_jsonl"],
        "--object-jsonl", inputs["object_jsonl"],
        "--audio-features", inputs["audio_features"],
        "--keyframe-map", inputs.get("keyframe_map", ""),
        "--output-dir", str(output_root),
        "--enable-frame-captions",
        "--enable-shot-captions",
        "--enable-trake-events",
        "--rebuild-compact-with-captions",
    ]
    if args.verbose:
        common.append("--verbose")

    if not args.skip_bootstrap:
        command = [
            sys.executable,
            str(root / "run_caption_pipeline.py"),
            *common,
            "--caption-mode", "template",
            "--shot-caption-mode", "template",
            "--trake-event-mode", "template",
        ]
        if _run(command, "template bootstrap", root, env):
            return 1

    vlm_output = work_root / "vlm" / video_id / "frame_caption_overrides.jsonl"
    if not args.skip_vlm:
        runtime_dir = work_root / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        vlm_cfg_path = runtime_dir / f"{video_id}_vlm.yaml"
        vlm_cfg = _build_vlm_config(cfg, output_root, work_root)
        write_text(vlm_cfg_path, _dump_yaml(vlm_cfg))
        command = [
            sys.executable,
            str(root / "run_vlm_frame_caption_experiment.py"),
            "--config", str(vlm_cfg_path),
        ]
        if args.no_resume:
            command.append("--no-resume")
        if args.verbose:
            command.append("--verbose")
        if _run(command, "VLM initial captions", root, env):
            return 1

    if not args.skip_llm:
        llm = models["text_llm"]
        command = [
            sys.executable,
            str(root / "run_caption_pipeline.py"),
            *common,
            "--caption-mode", "llm",
            "--shot-caption-mode", "llm",
            "--trake-event-mode", "llm",
            "--llm-provider", llm.get("provider", "transformers"),
            "--llm-model", llm["model_name"],
            "--llm-dtype", llm.get("dtype", "bfloat16"),
            "--llm-device-map", llm.get("device_map", "auto"),
            "--llm-attention", llm.get("attn_implementation", "auto"),
            "--prompt-dir", paths["prompt_dir"],
            "--prompt-version", generation.get(
                "prompt_version", "caption_qwen25_official_v1",
            ),
            "--max-fallback-rate", str(generation.get("max_fallback_rate", 0.15)),
        ]
        if vlm_output.exists():
            command.extend(("--initial-caption-overrides", str(vlm_output)))
        elif not args.skip_vlm:
            logger.error("Expected VLM overrides are missing: %s", vlm_output)
            return 1
        else:
            logger.warning("Running text pipeline without initial VLM captions")
        if args.no_resume:
            command.append("--no-resume")
        if _run(command, "prompt-backed caption pipeline", root, env):
            return 1

    if not args.skip_retrieval:
        retrieval = cfg.get("retrieval") or {}
        command = [
            sys.executable,
            str(root / "run_retrieval_export_eval.py"),
            "--video-id", video_id,
            "--caption-dir", str(output_root),
            "--eval-name", retrieval.get("eval_name", "official_route_gated"),
            "--scoring-profile", "route_gated_rrf",
            "--top-k", str(retrieval.get("top_k", 20)),
            "--strict",
        ]
        if args.verbose:
            command.append("--verbose")
        if _run(command, "route-gated retrieval export", root, env):
            return 1

    logger.info("Official caption pipeline completed for %s", video_id)
    return 0


def _build_vlm_config(cfg: dict[str, Any], output_root: Path, work_root: Path) -> dict[str, Any]:
    pipeline = cfg["pipeline"]
    paths = cfg["paths"]
    model = cfg["models"]["vlm"]
    generation = cfg.get("generation") or {}
    return {
        "experiment": {
            "name": "vlm",
            "video_id": pipeline["video_id"],
        },
        "paths": {
            "baseline_dir": str(output_root),
            "keyframes_root": cfg["inputs"]["keyframes_root"],
            "raw_output_root": str(work_root),
        },
        "model": model,
        "batch": {
            "frame_selection": "representative",
            "sample_stride": 5,
            "max_total_frames": 0,
            "max_frames": 0,
            "shard_index": 0,
            "num_shards": 1,
            "write_every": generation.get("vlm_write_every", 20),
            "resume": True,
            "output_name": "frame_caption_overrides.jsonl",
        },
        "generation": {
            "prompt_version": generation.get(
                "vlm_prompt_version", "qwen25vl_initial_caption_v1",
            ),
            "prompt_file": str(Path(paths["prompt_dir"]) / "vlm_initial_caption_prompt.txt"),
            "max_new_tokens": generation.get("vlm_max_new_tokens", 96),
            "max_caption_chars": 320,
            "do_sample": False,
            "temperature": 0.0,
        },
    }


def _run(command: list[str], label: str, cwd: Path, env: dict[str, str]) -> int:
    logger.info("=== %s ===", label)
    logger.debug("Command: %s", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    if completed.returncode:
        logger.error("Stage '%s' failed with exit code %d", label, completed.returncode)
    return completed.returncode


def _load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for the official pipeline config") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Official pipeline config must be a YAML mapping")
    return expand_env_values(value)


def _dump_yaml(value: dict[str, Any]) -> str:
    import yaml
    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)


def _validate_config(cfg: dict[str, Any]) -> None:
    required_sections = ("pipeline", "inputs", "paths", "models")
    missing = [section for section in required_sections if not isinstance(cfg.get(section), dict)]
    if missing:
        raise ValueError(f"Config is missing sections: {', '.join(missing)}")
    if not cfg["pipeline"].get("video_id"):
        raise ValueError("pipeline.video_id is required")
    required_inputs = ("ocr_jsonl", "object_jsonl", "audio_features", "keyframes_root")
    missing_inputs = [key for key in required_inputs if not cfg["inputs"].get(key)]
    if missing_inputs:
        raise ValueError(f"Config is missing inputs: {', '.join(missing_inputs)}")
    for path_key in ("output_root", "work_root", "prompt_dir"):
        if not cfg["paths"].get(path_key):
            raise ValueError(f"paths.{path_key} is required")
    for model_key in ("vlm", "text_llm"):
        if not isinstance(cfg["models"].get(model_key), dict):
            raise ValueError(f"models.{model_key} is required")


if __name__ == "__main__":
    sys.exit(main())
