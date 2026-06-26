from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from src.config import add_config_arg, load_config
from src.es_search_examples import search


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--query", required=True)
    parser.add_argument("--size", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(search(load_config(args.config), args.query, args.size).body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

