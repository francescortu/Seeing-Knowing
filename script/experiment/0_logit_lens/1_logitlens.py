from dataclasses import dataclass, field
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import logging
import datetime
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any, Union
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import torch
from easyroutine.interpretability import (
    HookedModel,
    ExtractionConfig,
)
from easyroutine.interpretability.tools import LogitLens
from src.datastatistics import statistics_computer
import json
from argparse import ArgumentParser



from src.experiment_manager import ExperimentManager, BaseConfig

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
        "--tag",
        type=str,
        default=""
    )

    args = parser.parse_args()
    
    
    
    
    config = BaseConfig(
        model_name=args.model,
        experiment_tag=args.tag,
        debug=args.debug,
    )
    experiment = ExperimentManager(config)

    experiment.setup_model(device_map="auto").create_dataloader().filter_dataloader()

    df = experiment.select_heads(return_df=True)

    # save the dataframe to a csv file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "llava" in config.model_name:
        model_tag = "llava"
    elif "gemma" in config.model_name:
        model_tag = "gemma"
    else:
        model_tag = "other"
    output_dir = OUTPUT_DIR / f"{model_tag}_{config.experiment_tag}_{timestamp}"
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
        attn_pattern_row_positions=["last"]
        )
    )
    import torch

    for key in cache.keys():
        if "pattern" in key:
            cache[key] = cache[key].squeeze()
            
    mean_cache = {k: v.mean(0).detach().cpu().to(torch.float32) for k, v in cache.items() if "pattern" in k}

    matrix = torch.zeros((experiment.model.model_config.num_hidden_layers,experiment.model.model_config.num_attention_heads))
    for layer in range(experiment.model.model_config.num_hidden_layers):
        for head in range(experiment.model.model_config.num_attention_heads):
            matrix[layer,head] = mean_cache[f"pattern_L{layer}H{head}"][0] 
    
    full_attn_to_img = pd.DataFrame(columns=["layer", "head", "value"])
    for layer in range(experiment.model.model_config.num_hidden_layers):
        for head in range(experiment.model.model_config.num_attention_heads):
            attn_to_img = matrix[layer, head].cpu().numpy()
            full_attn_to_img = pd.concat([full_attn_to_img, pd.DataFrame({"layer": [layer], "head": [head], "value": [attn_to_img]})], ignore_index=True)
    full_attn_to_img.to_csv(output_dir / "full_attn_to_img.csv", index=False)
    
      
    import numpy as np
    import matplotlib.pyplot as plt

    # Example matrix conversion to ensure it's a numpy array
    matrix = matrix.detach().cpu().numpy()

    # Compute values for counterfactual heads (assuming cfact_heads is defined correctly)
    values_cfact = {}
    for layer,head in experiment.cfact_heads:
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
    means = [1-mean_values_cfact, 1-mean_values_fact, 1-mean_all_matrix]
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
        "num_heads": len(experiment.cfact_heads)
    }
    
    stats = {k: str(v) for k, v in stats.items()}
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
if __name__ == "__main__":
    main()