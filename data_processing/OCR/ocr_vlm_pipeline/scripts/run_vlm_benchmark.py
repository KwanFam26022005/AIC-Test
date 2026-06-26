from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.benchmark import DEFAULT_BENCHMARK_MODELS, benchmark_models
from src.config import add_config_arg, load_config


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--sample_groups", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    models = args.models or list(cfg.vlm.get("benchmark_model_ids", DEFAULT_BENCHMARK_MODELS))
    output = args.output or str(Path(cfg.project.output_dir) / "model_benchmark.csv")
    print(benchmark_models(cfg, models, args.sample_groups, output))


if __name__ == "__main__":
    main()

