import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plot_functions import (
    plot_experiment1_combined,
    plot_experiment2,
    plot_heads_heatmap,
)
from src.paper_results import build_default_paper_tables


def parse_args():
    parser = ArgumentParser(description="Generate paper plots from paper-facing result tables.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_figures"),
        help="Directory where the generated figures will be written.",
    )
    return parser.parse_args()


def default_table_paths():
    return {
        "figure3_heads_llava": Path("results/paper_tables/figure3_heads_llava-next.csv"),
        "figure3_heads_gemma": Path("results/paper_tables/figure3_heads_gemma3.csv"),
        "figure3_attention_summary_llava": Path(
            "results/paper_tables/figure3_attention_summary_llava-next.csv"
        ),
        "figure3_attention_summary_gemma": Path(
            "results/paper_tables/figure3_attention_summary_gemma3.csv"
        ),
        "figure4_intervention": Path("results/paper_tables/figure4_intervention.csv"),
        "figure5_localization": Path("results/paper_tables/figure5_localization.csv"),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_paths = default_table_paths()
    if not all(path.exists() for path in table_paths.values()):
        table_paths = build_default_paper_tables()

    llava_heads = pd.read_csv(table_paths["figure3_heads_llava"])
    gemma_heads = pd.read_csv(table_paths["figure3_heads_gemma"])
    llava_attention = pd.read_csv(table_paths["figure3_attention_summary_llava"])
    gemma_attention = pd.read_csv(table_paths["figure3_attention_summary_gemma"])
    intervention = pd.read_csv(table_paths["figure4_intervention"])
    localization = pd.read_csv(table_paths["figure5_localization"])

    plot_heads_heatmap(
        llava_heads,
        stats=llava_attention.rename(
            columns={"group": "metric", "attention_to_image_pct": "mean"}
        ).assign(se=0.0),
        save_path=args.output_dir / "figure3_llava.png",
    )
    plot_heads_heatmap(
        gemma_heads,
        stats=gemma_attention.rename(
            columns={"group": "metric", "attention_to_image_pct": "mean"}
        ).assign(se=0.0),
        save_path=args.output_dir / "figure3_gemma.png",
    )

    plot_experiment1_combined(
        intervention[intervention["model"] == "LLaVA-NeXT"].copy(),
        intervention[intervention["model"] == "Gemma3"].copy(),
        save_path=args.output_dir / "figure4_intervention.png",
    )

    plot_experiment2(
        localization[localization["model"] == "LLaVA-NeXT"].copy(),
        model_name="LLaVA-NeXT",
        save_path=args.output_dir / "figure5_localization_llava.png",
    )
    plot_experiment2(
        localization[localization["model"] == "Gemma3"].copy(),
        model_name="Gemma3",
        save_path=args.output_dir / "figure5_localization_gemma.png",
    )


if __name__ == "__main__":
    main()
