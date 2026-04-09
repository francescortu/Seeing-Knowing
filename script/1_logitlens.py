import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argparse import ArgumentParser

import pandas as pd
from easyroutine.interpretability import ExtractionConfig

from src.experiment_manager import ExperimentManager, BaseConfig, DebugConfig
from src.paper_results import save_head_selection_results


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
    save_head_selection_results(df, full_attn_to_img, config.model_name)


if __name__ == "__main__":
    main()
