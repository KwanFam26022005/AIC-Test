from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import add_config_arg, load_config
from src.es_document_builder import run_build_es_documents
from src.frame_registry import run_frame_registry
from src.grouping import group_ocr_lines
from src.io_utils import read_table, write_table
from src.merge_results import run_merge
from src.ppocr_runner import run_ppocr
from src.risk_scoring import score_ocr_dataframe
from src.cropper import crop_group_images


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--skip-vlm", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_frame_registry(cfg)
    run_ppocr(cfg)
    out_dir = Path(cfg.project.output_dir)
    risk = score_ocr_dataframe(read_table(out_dir / "ppocr_raw.parquet"), cfg)
    write_table(risk, out_dir / "ocr_risk.parquet")
    groups = crop_group_images(group_ocr_lines(risk, cfg), cfg)
    write_table(groups, out_dir / "ocr_groups.parquet")
    write_table(groups[groups["need_vlm_group"]], out_dir / "vlm_jobs.parquet")
    if not args.skip_vlm:
        raise SystemExit("Run scripts/04_run_vlm_correction.py as a separate GPU/VLM stage, then rerun with --skip-vlm for merge/docs.")
    run_merge(cfg)
    print(run_build_es_documents(cfg))


if __name__ == "__main__":
    main()

