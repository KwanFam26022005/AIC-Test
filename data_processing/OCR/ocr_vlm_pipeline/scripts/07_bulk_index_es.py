from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import add_config_arg, load_config
from src.es_bulk_indexer import bulk_index_documents


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(bulk_index_documents(cfg, Path(cfg.project.output_dir) / "es_documents.jsonl"))


if __name__ == "__main__":
    main()

