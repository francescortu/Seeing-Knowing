from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from easyroutine.interpretability import HookedModel
from rich import print
import io
import base64
from PIL import Image
import json
import re
import datetime
from tqdm import tqdm
import ollama
from typing import Literal, Optional
from src.utils import get_whoops_element_by_id, start_ollama, ollama_model_map
from pathlib import Path
import torch
from src.datastatistics import statistics_computer
from easyroutine.interpretability import HookedModel, ExtractionConfig, ActivationSaver, ActivationLoader, Intervention
from easyroutine.interpretability.tools import LogitLens
from easyroutine.logger import logger, enable_debug_logging, enable_info_logging
from tqdm import tqdm
import torch
import json
from datasets import load_dataset
from pathlib import Path
import pandas as pd

CFACT_HEADS = [(18, 12),
 (19, 9),
 (20, 14),
 (20, 17),
 (21, 7),
 (24, 21),
 (25, 9),
 (28, 13),
 (28, 15),
 (29, 2),
 (29, 12),
 (29, 14),
 (30, 9),
 (30, 29),
 (31, 18),
 (31, 22),
 (31, 25),
 (31, 27)]

# FACT_HEADS = [(0, 22), (1, 5), (1, 23), (1, 27), (1, 31), (2, 28), (3, 8), (3, 17), (3, 18), (4, 4), (4, 26), (5, 5), (5, 10), (5, 13), (5, 17), (5, 23), (6, 8), (6, 12), (6, 20), (6, 23), (7, 11), (7, 17), (7, 20), (7, 27), (7, 28), (8, 12), (8, 14), (8, 17), (9, 10), (9, 23), (9, 27), (10, 2), (10, 8), (10, 14), (10, 30), (11, 9), (11, 23), (12, 8), (12, 10), (12, 16), (12, 17), (12, 22), (12, 25), (12, 26), (12, 31), (13, 1), (13, 8), (13, 9), (13, 12), (13, 15), (13, 16), (14, 3), (14, 18), (14, 23), (14, 30), (14, 31), (15, 5), (15, 9), (15, 17), (15, 18), (15, 28), (16, 2), (16, 9), (16, 13), (16, 20), (16, 23), (16, 24), (17, 1), (17, 2), (17, 5), (17, 6), (17, 18), (17, 19), (17, 20), (17, 21), (17, 26), (18, 11), (18, 18), (19, 0), (19, 3), (19, 10), (19, 15), (19, 19), (19, 28), (20, 0), (20, 2), (20, 13), (20, 19), (20, 20), (20, 22), (20, 25), (20, 29), (21, 6), (21, 9), (21, 10), (21, 13), (21, 15), (21, 21), (22, 16), (22, 21), (22, 23), (22, 27), (22, 29), (23, 2), (23, 9), (23, 14), (23, 15), (23, 18), (23, 25), (24, 3), (24, 10), (24, 20), (25, 8), (25, 11), (25, 13), (25, 26), (25, 28), (26, 3), (26, 7), (26, 13), (26, 17), (27, 2), (27, 4), (27, 5), (27, 12), (27, 20), (27, 31), (28, 0), (28, 2), (28, 14), (28, 16), (28, 17), (28, 22), (28, 24), (29, 0), (29, 1), (29, 9), (29, 13), (29, 15), (30, 3), (30, 10), (30, 19), (30, 24), (30, 28), (31, 5), (31, 11), (31, 12), (31, 14), (31, 17), (31, 19), (31, 20), (31, 26)]

FACT_HEADS = None

TAG = "_v2"


def set_interventions(model:HookedModel, heads, ablation_type):
    """
    Set the interventions for the model based on the ablation type.
    """
    if ablation_type == "last-row":
        model.register_interventions(
                interventions=[
                    Intervention(
                        type = "grid",
                        activation = f"pattern_L{layer}H{head}",
                        token_positions=(["last"],["all"]),
                        # token_positions = ["all"],
                        patching_values = "ablation",
                        multiplication_value=weight
                        # ablation_values = 10
                    ) for layer, head, weight in heads
                ]
        )
    elif ablation_type == "full":
        model.register_interventions(
                interventions=[
                    Intervention(
                        type = "grid",
                        activation = f"pattern_L{layer}H{head}",
                        token_positions=(["all"],["all"]),
                        # token_positions = ["all"],
                        patching_values = "ablation",
                        multiplication_value=weight
                        # ablation_values = 10
                    ) for layer, head, weight in heads
                ]
        )
    elif ablation_type == "last-row-img":
        model.register_interventions(
                interventions=[
                    Intervention(
                        type = "grid",
                        activation = f"pattern_L{layer}H{head}",
                        token_positions=(["last"],["all-image"]),
                        # token_positions = ["all"],
                        patching_values = "ablation",
                        multiplication_value=weight
                        # ablation_values = 10
                    ) for layer, head, weight in heads
                ]
        )
    elif ablation_type == "last-row-text":
        model.register_interventions(
                interventions=[
                    Intervention(
                        type = "grid",
                        activation = f"pattern_L{layer}H{head}",
                        token_positions=(["last"],["all-text"]),
                        # token_positions = ["all"],
                        patching_values = "ablation",
                        multiplication_value=weight
                        # ablation_values = 10
                    ) for layer, head, weight in heads
                ]
        )
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
def get_multiplication_weights(heads, gamma:Optional[int] = None):
    if gamma is None:
        return [(layer, head, 0) for layer, head in heads]
    else:
        layer_head_and_values = {}
        for layer,head in heads:
            if layer not in layer_head_and_values:
                layer_head_and_values[layer] = set()
            layer_head_and_values[layer].add(head)
            
        intervention_heads = []
        for layer,head in heads:
            if layer in layer_head_and_values:
                #retrive the list of heads in the set
                target_heads = list(layer_head_and_values[layer])
                other_heads = [h for h in range(32) if h not in target_heads]
                N = len(target_heads)
                M = len(other_heads)  # Note: N + M should equal 32.
                assert N + M == 32, f"Layer {layer} has {N} target heads and {M} other heads, but they should sum to 32."
                    
                # # Compute scaling weights using piecewise definitions
                # if gamma >= 0:
                #     weight_target = 1 + gamma * ((32 / N) - 1)
                #     weight_other = 1 - gamma
                # else:
                #     weight_target = 1 + gamma
                #     weight_other = 1 - gamma * ((32 / M) - 1)

                weight_target = 1 + gamma
                weight_other = 1 + gamma * (1-(32 / M))
                    
                # print("sum", N*weight_target + M*weight_other)
                
                for head in target_heads:
                    # assign the weights to the heads
                    intervention_heads.append((layer, head, weight_target))
                
                for head in other_heads:
                    # assign the weights to the heads
                    intervention_heads.append((layer, head, weight_other))
        return intervention_heads

def main():
    dataset_path=Path("data/openai/manual_visual_counterfactual_2025-03-27_16-30-11.json")
    dataloader_path= Path("data/manual/dataloader_llava-hf-llava-v1.6-mistral-7b-hf_visual_counterfactual_02-04-2025-15-16.pt" )
    model_name="llava-hf/llava-v1.6-mistral-7b-hf"

    # dataset_path=Path("data/openai/manual_visual_counterfactual_2025-03-27_16-30-11.json")
    # dataloader_path= Path("data/manual/dataloader_google-gemma-3-12b-it_visual_counterfactual_02-04-2025-15-16.pt" )
    # model_name="google/gemma-3-12b-it"

    model = HookedModel.from_pretrained(model_name, device_map="auto")


    dataloader = torch.load(
        # "data/gemma/llava-hf-llava-v1.6-mistral-7b-hf_dataloader_2025-03-25_13-32-42.pt"
        dataloader_path
        )
    dataset = json.load(open(
        dataset_path
        
        ))
    
    token_pair = [t["token_pair"] for t in dataloader]
    print("Model Loaded. Dataset Len:",len(dataset))
    
    result_df = pd.DataFrame(columns=["ExperimentDesc", "AblationType", "Gamma", "Image Cfact logit", "Image Fact Logit", "Text Cfact Logit", "Text Fact Logit", "Image Cfact>Fact", "Text Cfact>Fact", "Image Valid Examples", "Text Valid Examples", "Image Higer Pos", "Text Higer Pos"])
    
    
    gamma = [-5,-4,-3,-2,-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    experiment_desc = [
        "Zero-Ablation Cfact Heads",
        # "Zero-Ablation Fact Heads",
        "Gamma-intervention For Cfact Heads",
        # "Gamma-intervention For Fact Heads",
    ]
    ablation_type = [
        "last-row",
        "last-row-img",
        "last-row-text",
        "full"
    ]
    total_experiments = (
         len(ablation_type) +       # Zero-Ablation (no gamma)
        len(ablation_type) * len(gamma)  # Gamma-intervention
        +1
    )
    file_name = f"results/1_Ablation/full_experiment_{model_name.replace('/','-')}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{TAG}.csv"
    # create the directory if it does not exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # write the header to the file
    result_df.to_csv(file_name, index=False)

    with tqdm(total=total_experiments, desc="Running Experiments") as pbar:
        # baseline
        model.clean_interventions()
        data= statistics_computer(
            model = model,
            dataloader = dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=token_pair,
            return_essential_data=True
        )
        # add to result_df: data is a dict with keys "Image Cfact logit", "Image Fact Logit", "Text Cfact Logit", "Text Fact Logit", "Image Cfact>Fact", "Text Cfact>Fact"
        result_df = pd.concat([result_df, pd.DataFrame([{
            "ExperimentDesc": "Baseline",
            "AblationType": None,
            "Gamma": None,
            "Image Cfact logit": data["Image Cfact logit"],
            "Image Fact Logit": data["Image Fact Logit"],
            "Text Cfact Logit": data["Text Cfact Logit"],
            "Text Fact Logit": data["Text Fact Logit"],
            "Image Cfact>Fact": data["Image Cfact>Fact"],
            "Text Cfact>Fact": data["Text Cfact>Fact"],
            "Image Valid Examples": data["Image Valid Examples"],
            "Text Valid Examples": data["Text Valid Examples"],
            "Image Higer Pos": data["Image Pos Higher"],
            "Text Higer Pos": data["Text Pos Higher"]
        }])], ignore_index=True)
        # write to file
        result_df.to_csv(file_name, index=False)
        pbar.update(1)
        
        # Zero-Ablation Cfact Heads and Zero-Ablation Fact Heads
        for i in range(1):
            if i == 0:
                heads = get_multiplication_weights(CFACT_HEADS, None)
            elif i == 1:
                heads = get_multiplication_weights(FACT_HEADS, None)
            else:
                raise ValueError("Unknown index")
            
            for ablation in ablation_type:
                model.clean_interventions()
                set_interventions(model, heads, ablation)
                data= statistics_computer(
                    model = model,
                    dataloader = dataloader,
                    write_to_file=False,
                    filename=None,
                    dataset_path=Path(""),
                    given_token_pair=token_pair,
                    return_essential_data=True,
                )
                # add to result_df: data is a dict with keys "Image Cfact logit", "Image Fact Logit", "Text Cfact Logit", "Text Fact Logit", "Image Cfact>Fact", "Text Cfact>Fact"
                result_df = pd.concat([result_df, pd.DataFrame([{
                    "ExperimentDesc": experiment_desc[i],
                    "AblationType": ablation,
                    "Gamma": None,
                    "Image Cfact logit": data["Image Cfact logit"],
                    "Image Fact Logit": data["Image Fact Logit"],
                    "Text Cfact Logit": data["Text Cfact Logit"],
                    "Text Fact Logit": data["Text Fact Logit"],
                    "Image Cfact>Fact": data["Image Cfact>Fact"],
                    "Text Cfact>Fact": data["Text Cfact>Fact"],
                    "Image Valid Examples": data["Image Valid Examples"],
                    "Text Valid Examples": data["Text Valid Examples"],
                    "Image Higer Pos": data["Image Pos Higher"],
                    "Text Higer Pos": data["Text Pos Higher"]
                }])], ignore_index=True)       
                
                # write to file
                result_df.to_csv(file_name, index=False)
                pbar.update(1)
        # Gamma-intervention For Cfact Heads and Gamma-intervention For Fact Heads
        for i in range(1):
            
            for ablation in ablation_type:
                for g in gamma:
                    if i == 0:
                        heads = get_multiplication_weights(CFACT_HEADS, g)
                    elif i == 1:
                        heads = get_multiplication_weights(FACT_HEADS, g)
                    else:
                        raise ValueError("Unknown index")
                    model.clean_interventions()
                    set_interventions(model, heads, ablation)
                    data= statistics_computer(
                        model = model,
                        dataloader = dataloader,
                        write_to_file=False,
                        filename=None,
                        dataset_path=Path(""),
                        given_token_pair=token_pair,
                        return_essential_data=True,
                    )
                    # add to result_df: data is a dict with keys "Image Cfact logit", "Image Fact Logit", "Text Cfact Logit", "Text Fact Logit", "Image Cfact>Fact", "Text Cfact>Fact"
                    result_df = pd.concat([result_df, pd.DataFrame([{
                        "ExperimentDesc": experiment_desc[i],
                        "AblationType": ablation,
                        "Gamma": g,
                        "Image Cfact logit": data["Image Cfact logit"],
                        "Image Fact Logit": data["Image Fact Logit"],
                        "Text Cfact Logit": data["Text Cfact Logit"],
                        "Text Fact Logit": data["Text Fact Logit"],
                        "Image Cfact>Fact": data["Image Cfact>Fact"],
                        "Text Cfact>Fact": data["Text Cfact>Fact"],
                        "Image Valid Examples": data["Image Valid Examples"],
                        "Text Valid Examples": data["Text Valid Examples"],
                        "Image Higer Pos": data["Image Pos Higher"],
                        "Text Higer Pos": data["Text Pos Higher"]
                    }])], ignore_index=True)       
                    
                    # write to file
                    result_df.to_csv(file_name, index=False)
                    pbar.update(1)
    # write to file
    result_df.to_csv(file_name, index=False)


if __name__ == "__main__":
    # enable_debug_logging()
    # enable_info_logging()
    main()