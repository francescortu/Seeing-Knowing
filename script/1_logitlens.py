import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass, field
import logging
import datetime
from pathlib import Path
import pandas as pd
from easyroutine.interpretability import (
    ExtractionConfig,
)
import json
from argparse import ArgumentParser

from src.experiment_manager import ExperimentManager, BaseConfig, DebugConfig

OUTPUT_DIR = Path("results/0_heads_selection")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="Model name or path to the model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
        default=False,
    )
    parser.add_argument(
        "--k_heads",
        type=int,
        default=20,
        help="Number of heads to select based on logit lens",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="full",
        help="Subset of the dataset to use for logit lens analysis. Options: 'full', 'small1', 'small2'. 'full' uses the entire dataset, 'small1' uses a small subset of 100 samples, and 'small2' uses another small subset of 100 samples.",
    )
    parser.add_argument("--tag", type=str, default="")

    args = parser.parse_args()

    config = BaseConfig(
        model_name=args.model,
        # dataset_name="francescortu/manual_visual_counterfactual_02-04-2025-15",
        experiment_tag=args.tag,
        debug=DebugConfig(debug=args.debug),
        dataset_subset=args.dataset_subset,
    )
    experiment = ExperimentManager(config)
    experiment.launch_std_setup_routine()

    # experiment.setup_model(device_map="auto").create_dataloader().filter_dataloader()

    df = experiment.select_heads(
        k_heads=args.k_heads, return_df=True, metric="logit_diff"
    )

    # save the dataframe to a csv file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "llava" in config.model_name:
        model_tag = "llava"
    elif "gemma" in config.model_name:
        model_tag = "gemma"
    else:
        model_tag = "other"
    # Create folder structure: results/0_heads_selection/experiment_tag/model_timestamp
    tag_folder = OUTPUT_DIR / config.experiment_tag
    output_dir = tag_folder / f"{model_tag}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir / "selected_heads.csv", index=False)

    df_attn, df_mlp = experiment.logit_lens_attn_and_mlp()
    df_attn.to_csv(output_dir / "logit_lens_attn.csv", index=False)
    df_mlp.to_csv(output_dir / "logit_lens_mlp.csv", index=False)

    cache = experiment.model.extract_cache(
        [d["text_image_inputs"] for d in experiment.dataloader],
        target_token_positions=["all-image", "all-text"],
        extraction_config=ExtractionConfig(
            extract_attn_pattern=True,
            attn_pattern_avg="sum",
            attn_pattern_row_positions=["last"],
        ),
    )
    import torch

    for key in cache.keys():
        if "pattern" in key:
            cache[key] = cache[key].squeeze()

    mean_cache = {
        k: v.mean(0).detach().cpu().to(torch.float32)
        for k, v in cache.items()
        if "pattern" in k
    }

    matrix = torch.zeros(
        (
            experiment.model.model_config.num_hidden_layers,
            experiment.model.model_config.num_attention_heads,
        )
    )
    for layer in range(experiment.model.model_config.num_hidden_layers):
        for head in range(experiment.model.model_config.num_attention_heads):
            matrix[layer, head] = mean_cache[f"pattern_L{layer}H{head}"][0]

    full_attn_to_img = pd.DataFrame(columns=["layer", "head", "value"])
    for layer in range(experiment.model.model_config.num_hidden_layers):
        for head in range(experiment.model.model_config.num_attention_heads):
            attn_to_img = matrix[layer, head].cpu().numpy()
            full_attn_to_img = pd.concat(
                [
                    full_attn_to_img,
                    pd.DataFrame(
                        {"layer": [layer], "head": [head], "value": [attn_to_img]}
                    ),
                ],
                ignore_index=True,
            )
    full_attn_to_img.to_csv(output_dir / "full_attn_to_img.csv", index=False)

    import numpy as np
    import matplotlib.pyplot as plt

    # Example matrix conversion to ensure it's a numpy array
    matrix = matrix.detach().cpu().numpy()

    # Compute values for counterfactual heads (assuming cfact_heads is defined correctly)
    values_cfact = {}
    for layer, head in experiment.cfact_heads:
        values_cfact[(layer, head)] = matrix[layer, head]
    values_cfact_array = np.array(list(values_cfact.values()))
    mean_values_cfact = values_cfact_array.mean()
    std_cfact = values_cfact_array.std()
    std_err_cfact = std_cfact / np.sqrt(len(values_cfact_array))

    # Compute values for factual heads
    values_fact = {}
    for layer, head in experiment.fact_heads:
        values_fact[(layer, head)] = matrix[layer, head]
    values_fact_array = np.array(list(values_fact.values()))
    mean_values_fact = values_fact_array.mean()
    std_fact = values_fact_array.std()
    std_err_fact = std_fact / np.sqrt(len(values_fact_array))

    # Overall matrix statistics
    mean_all_matrix = matrix.mean()
    std_all_matrix = matrix.std()
    std_err_all_matrix = std_all_matrix / np.sqrt(matrix.size)

    # Check types
    print("Type of len(values_cfact_array):", type(len(values_cfact_array)))
    print("Type of matrix.size:", type(matrix.size))

    labels = ["Counterfactual Heads", "Factual Heads", "All Heads"]
    means = [1 - mean_values_cfact, 1 - mean_values_fact, 1 - mean_all_matrix]
    errors = [std_err_cfact, std_err_fact, std_err_all_matrix]

    stats = {
        "mean_values_cfact": mean_values_cfact,
        "std_err_cfact": std_err_cfact,
        "mean_values_fact": mean_values_fact,
        "std_err_fact": std_err_fact,
        "mean_all_matrix": mean_all_matrix,
        "std_all_matrix": std_all_matrix,
        "std_err_all_matrix": std_err_all_matrix,
        "model_name": config.model_name,
        "num_heads": len(experiment.cfact_heads),
        "factual_heads": experiment.fact_heads,
        "counterfactual_heads": experiment.cfact_heads,
    }

    stats = {k: str(v) for k, v in stats.items()}
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
