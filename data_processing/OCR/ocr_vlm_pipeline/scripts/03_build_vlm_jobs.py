from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import add_config_arg, load_config
from src.cropper import crop_group_images
from src.grouping import group_ocr_lines
from src.io_utils import read_table, write_table
from src.risk_scoring import score_ocr_dataframe


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.project.output_dir)
    raw = read_table(out_dir / "ppocr_raw.parquet")
    risk = score_ocr_dataframe(raw, cfg)
    write_table(risk, out_dir / "ocr_risk.parquet")
    groups = group_ocr_lines(risk, cfg)
    groups = crop_group_images(groups, cfg)
    write_table(groups, out_dir / "ocr_groups.parquet")
    jobs = groups[groups["need_vlm_group"]].copy()
    print(write_table(jobs, out_dir / "vlm_jobs.parquet"))


if __name__ == "__main__":
    main()

