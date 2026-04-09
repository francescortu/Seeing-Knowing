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
from src.paper_results import default_table_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = ArgumentParser(description="Generate plots from the canonical result tables.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures",
        help="Directory where the generated figures will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_paths = default_table_paths()
    missing = [str(path.relative_to(REPO_ROOT)) for path in table_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing results tables. Run the experiment scripts first: "
            + ", ".join(missing)
        )

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
