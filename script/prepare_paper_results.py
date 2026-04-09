import sys
import os
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.paper_results import build_default_paper_tables


def parse_args():
    parser = ArgumentParser(description="Build paper-facing result tables.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_tables"),
        help="Directory where normalized paper-facing tables will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = build_default_paper_tables(output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
