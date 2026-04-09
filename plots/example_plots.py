from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from plot_functions import (
    plot_experiment1_combined,
    plot_experiment2,
    plot_heads_heatmap,
)


def parse_args():
    parser = ArgumentParser(description="Generate paper plots from checked-in artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_figures"),
        help="Directory where the generated figures will be written.",
    )
    return parser.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_intervention_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    if "Fact Acc" not in plot_df.columns and "Image Cfact>Fact" in plot_df.columns:
        plot_df["Fact Acc"] = plot_df["Image Cfact>Fact"]
    return plot_df


def prepare_localization_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    if "Fact Acc" not in plot_df.columns and "Image Cfact>Fact" in plot_df.columns:
        plot_df["Fact Acc"] = plot_df["Image Cfact>Fact"]
    return plot_df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    llava_heads = load_csv(
        "results/0_heads_selection/v16_arXiv/llava_2025-07-07_16-26-26/selected_heads.csv"
    )
    gemma_heads = load_csv(
        "results/0_heads_selection/v16_arXiv/gemma_2025-07-03_18-20-56/selected_heads.csv"
    )

    llava_intervention = load_csv(
        "results/1_heads_ablation/v16_arXiv/llava-hf-llava-v1.6-mistral-7b-hf_2025-07-12_19-05-07/v16_arXiv.csv"
    )
    gemma_intervention = load_csv(
        "results/1_heads_ablation/v16_arXiv/google-gemma-3-12b-it_2025-07-14_17-28-15/v16_arXiv.csv"
    )

    llava_localization = load_csv(
        "results/2_ImgCfactLocalization/v16_arXiv/llava-hf-llava-v1.6-mistral-7b-hf_2025-07-07_17-25-14/results.csv"
    )
    gemma_localization = load_csv(
        "results/2_ImgCfactLocalization/v16_arXiv/google-gemma-3-12b-it_2025-07-03_18-56-49/results.csv"
    )

    plot_heads_heatmap(
        llava_heads,
        save_path=args.output_dir / "heads_heatmap_llava.png",
    )
    plot_heads_heatmap(
        gemma_heads,
        save_path=args.output_dir / "heads_heatmap_gemma.png",
    )
    plot_experiment1_combined(
        prepare_intervention_plot_df(llava_intervention),
        prepare_intervention_plot_df(gemma_intervention),
        save_path=args.output_dir / "paired_intervention.png",
    )
    plot_experiment2(
        prepare_localization_plot_df(llava_localization),
        model_name="LLaVA-NeXT",
        save_path=args.output_dir / "pixel_localization_llava.png",
    )
    plot_experiment2(
        prepare_localization_plot_df(gemma_localization),
        model_name="Gemma3",
        save_path=args.output_dir / "pixel_localization_gemma.png",
    )


if __name__ == "__main__":
    main()
