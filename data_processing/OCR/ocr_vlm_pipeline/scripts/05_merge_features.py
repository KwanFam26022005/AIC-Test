from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.config import add_config_arg, load_config
from src.merge_results import run_merge


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    args = parser.parse_args()
    print(run_merge(load_config(args.config)))


if __name__ == "__main__":
    main()

