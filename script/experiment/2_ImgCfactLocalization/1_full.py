import sys
import os
import pandas as pd

# import numpy as np # Removed unused import
import json
import datetime
import argparse
import torch

# import copy  # Removed unused import
from pathlib import Path
# Removed unused rich imports
# from rich.progress import (
#     Progress,
#     TextColumn,
#     BarColumn,
#     TaskProgressColumn,
#     TimeRemainingColumn,
#     SpinnerColumn,
#     TimeElapsedColumn,
# )
# from rich.console import Console

from typing import List, Tuple, Optional, Dict, Any, Literal  # Removed unused Union
from dataclasses import dataclass  # Removed unused field, asdict

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from easyroutine.interpretability import Intervention
from src.datastatistics import statistics_computer
from src.experiment_manager import ExperimentManager, BaseConfig

from easyroutine.logger import logger, setup_logging
from easyroutine.interpretability import ExtractionConfig
from easyroutine.interpretability.activation_cache import ActivationCache, sublist

# import and set .env variables
from dotenv import load_dotenv
import random

random.seed(42)

# load ../../../.env file
dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")


setup_logging(level="INFO")  # Set up logging for the experiment


@dataclass
class ImgCfactLocalizationConfig(BaseConfig):
    """
    Configuration class for ImgCfactLocalization experiment.
    """

    experiment_description: Optional[str] = None
    # result_filename: Optional[str] = None # Removed, path handled differently now
    k_heads: int = 20
    run_experiments: Optional[List[str]] = None  # Added metadata
    commit_hash: Optional[str] = None  # Added metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        """
        # Use a temporary dict to handle potential None values gracefully
        data = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "experiment_tag": self.experiment_tag,
            "extra_metadata": self.extra_metadata,
            "debug": self.debug,
            "debug_samples": self.debug_samples,
            "prompt_templates": self.prompt_templates,
            # "result_filename": self.result_filename, # Removed
            "experiment_description": self.experiment_description,
            "k_heads": self.k_heads,  # Added k_heads
            "run_experiments": self.run_experiments,  # Added metadata
            "commit_hash": self.commit_hash,  # Added metadata
        }
        # Filter out None values if desired, or keep them
        return {k: v for k, v in data.items() if v is not None}

    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ImgCfactLocalizationExperiment(ExperimentManager):
    def __init__(self, config: ImgCfactLocalizationConfig):
        """
        Initialize the ImgCfactLocalizationExperiment with the given configuration.
        """
        super().__init__(config)
        self.config = config
        # self.result_dir = Path("results/2_ImgCfactLocalization") # Base directory
        # self.result_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir: Optional[Path] = None  # Specific output dir for this run
        self.cfact_heads: Optional[List[Tuple[int, int]]] = None
        self.fact_heads: Optional[List[Tuple[int, int]]] = None

    def get_multiplication_weights(
        self,
        cfact_heads: List[Tuple[int, int]],
        fact_heads: List[Tuple[int, int]],
        gamma: Optional[float] = None,
        lambda_param: Optional[float] = None,
        rebalanced_weight=True,
    ) -> List[Tuple[int, int, float, str]]:
        """Calculate weights for head intervention with balanced weights"""
        if not cfact_heads or not fact_heads:
            raise ValueError("Both cfact_heads and fact_heads must be provided")

        if gamma is None and lambda_param is None:
            return [(layer, head, 0, "cfact") for layer, head in cfact_heads]

        layer_head_map = {}
        # Group heads by layer
        for layer, head in cfact_heads:
            if layer not in layer_head_map:
                layer_head_map[layer] = {
                    "cfact": set(),
                    "fact": set(),
                    "other": set(range(self.model.model_config.num_attention_heads)),
                }
            layer_head_map[layer]["cfact"].add(head)
            layer_head_map[layer]["other"].remove(head)

        for layer, head in fact_heads:
            if layer not in layer_head_map:
                layer_head_map[layer] = {
                    "cfact": set(),
                    "fact": set(),
                    "other": set(range(self.model.model_config.num_attention_heads)),
                }
            layer_head_map[layer]["fact"].add(head)
            if head in layer_head_map[layer]["other"]:
                layer_head_map[layer]["other"].remove(head)

        intervention_heads = []
        for layer in layer_head_map:
            N = len(layer_head_map[layer]["cfact"])  # num cfact heads
            F = len(layer_head_map[layer]["fact"])  # num fact heads
            M = len(layer_head_map[layer]["other"])  # num other heads

            # Calculate weights to maintain sum = self.model.model_config.num_attention_heads (N + F + M)
            w_cfact = 1 + gamma if gamma is not None else 1
            w_fact = 1 + lambda_param if lambda_param is not None else 1

            # Calculate other weight based on rebalanced_weight config
            if rebalanced_weight:
                # Calculate other weight to maintain sum = N + F + M
                w_other = (
                    (
                        self.model.model_config.num_attention_heads
                        - N * w_cfact
                        - F * w_fact
                    )
                    / M
                    if M > 0
                    else 0
                )
            else:
                # If not rebalancing, other heads keep their original weight (1)
                w_other = 1

            # Add cfact heads
            for head in layer_head_map[layer]["cfact"]:
                intervention_heads.append((layer, head, w_cfact, "cfact"))

            # Add fact heads
            for head in layer_head_map[layer]["fact"]:
                intervention_heads.append((layer, head, w_fact, "fact"))

            # Add other heads
            for head in layer_head_map[layer]["other"]:
                intervention_heads.append((layer, head, w_other, "other"))

        return intervention_heads

    def extract_gradients(self):
        self.logger.debug("Identifying top pixel using gradients...")

        if self.cfact_heads is None:
            raise ValueError("cfact_heads must be set before calling ablate_top_pixel.")
        if self.dataloader is None:
            raise ValueError("dataloader must be set before calling ablate_top_pixel.")
        if self.token_pair is None:
            raise ValueError("token_pair must be set before calling ablate_top_pixel.")

        tokenizer = self.model.get_tokenizer()
        for i in range(len(self.dataloader)):
            paired_token = self.token_pair[i]
            cfact_idx = tokenizer(text=paired_token[0])["input_ids"][0][1]

            self.dataloader[i]["text_image_inputs"]["vocabulary_index"] = cfact_idx

        cache = self.model.extract_cache(
            [d["text_image_inputs"] for d in self.dataloader],
            target_token_positions=["all-image"],
            extraction_config=ExtractionConfig(
                extract_embed=True,
                keep_gradient=True,
                save_logits=False,
            ),
            register_aggregation=("input_embeddings_gradients", sublist),
        )
        return cache

    def extract_patterns(self):
        self.logger.debug("Identifying top pixel...")

        if self.cfact_heads is None:
            raise ValueError("cfact_heads must be set before calling ablate_top_pixel.")
        if self.dataloader is None:
            raise ValueError("dataloader must be set before calling ablate_top_pixel.")

        def pattern_agr(old, new):
            if old is None:
                return [new]
            return old + [new]

        # extract the top pixel from the cfact_heads
        pattern_cache = self.model.extract_cache(
            [d["text_image_inputs"] for d in self.dataloader],
            target_token_positions=["all-image"],
            extraction_config=ExtractionConfig(
                extract_resid_out=False,
                extract_attn_pattern=True,
                attn_pattern_avg="none",
                attn_pattern_row_positions=["last"],
                attn_heads=[
                    {"layer": head[0], "head": head[1]} for head in self.cfact_heads
                ],
                save_logits=False,
            ),
            register_aggregation=("pattern_", pattern_agr),
        )
        return pattern_cache

    def select_top_pixels(
        self,
        pattern_cache: ActivationCache,
        frac_top_img_attn=0.2,
        random_control=False,
        threshold: Optional[float] = None,  # value in [0,1]
        mode: Literal[
            "individual",
            "top_sum",
            "top_union",
            "union_threshold",
            "threshold_on_max",
            "union_threshold_on_max",
        ] = "top_union",
        saliency: Optional[ActivationCache] = None,
    ):
        """
        For each example and for each head find the top pixels and zerout it in the pattern of the head
        """
        if self.dataloader is None:
            raise ValueError("dataloader must be set before calling ablate_top_pixel.")

        # get the keys (pattern_LiHj) from the cache
        patter_keys = [k for k in pattern_cache.keys() if k.startswith("pattern_")]

        # Changed type hint to List[Any] to handle different return types
        top_pixels: Dict[int, Dict] = {}

        if mode == "individual":
            len_dataset = len(pattern_cache[patter_keys[0]])
            for idx in range(len_dataset):
                top_pixels[idx] = {}  # Appending dict
                for key in patter_keys:
                    if random_control:
                        top_indices = torch.randint(
                            0,
                            len(pattern_cache[key][idx][0, 0]),
                            (
                                int(
                                    len(pattern_cache[key][idx][0, 0])
                                    * frac_top_img_attn
                                ),
                            ),
                        )
                    else:
                        top_percent = int(
                            len(pattern_cache[key][idx][0, 0]) * frac_top_img_attn
                        )
                        # get the index of the top 20% highest values
                        top_indices = torch.argsort(pattern_cache[key][idx][0, 0])[
                            -top_percent:
                        ]
                    top_pixels[idx][key] = top_indices.tolist()
            return top_pixels

        elif mode == "threshold":
            assert threshold is not None, "threshold must be set for mode 'threshold'"
            assert 0 <= threshold <= 1, "threshold must be in [0,1]"
            len_dataset = len(pattern_cache[patter_keys[0]])
            for idx in range(len_dataset):
                top_pixels[idx] = {}  # Appending dict
                for key in patter_keys:
                    if threshold == 1:
                        top_pixels[idx][key] = list(
                            range(len(pattern_cache[key][idx][0, 0]))
                        )
                    elif threshold == 0:
                        top_pixels[idx][key] = []
                    else:
                        pattern = pattern_cache[key][idx][0, 0]  # (seq_len,)

                        # sort tokens by weight, high → low
                        vals, idxs = torch.sort(pattern, descending=True)

                        # cumulative sum of sorted weights
                        cum = torch.cumsum(vals, dim=0)

                        adjusted_threshold = cum[-1] * threshold
                        # ───────────────────────────────────────────────────────────────
                        # 1) using torch.searchsorted  (fastest, no Boolean tensor at all)
                        cutoff = torch.searchsorted(
                            cum, adjusted_threshold, right=False
                        ).item()

                        # ───────────────────────────────────────────────────────────────
                        # 2) or, if you prefer the Boolean-mask route:
                        # mask   = cum >= prob_threshold          # Bool tensor
                        # cutoff = mask.nonzero(as_tuple=False)[0, 0].item()

                        # take every token up to (and including) that position
                        top_index = idxs[
                            : cutoff + 1
                        ].tolist()  # list of original column indices
                        top_pixels[idx][key] = top_index

            return top_pixels

        elif mode == "threshold_on_max":
            assert threshold is not None, "threshold must be set for mode 'threshold'"
            assert 0 <= threshold <= 1, "threshold must be in [0,1]"
            len_dataset = len(pattern_cache[patter_keys[0]])
            for idx in range(len_dataset):
                top_pixels[idx] = {}  # Appending dict
                for key in patter_keys:
                    pattern = pattern_cache[key][idx][0, 0]
                    max_value = torch.max(pattern)
                    threshold_value = max_value * threshold
                    if threshold_value == 1:
                        top_pixels[idx][key] = []
                    else:
                        # get the index of the top 20% highest values
                        top_indices = torch.argsort(pattern, descending=True)
                        top_indices = top_indices[
                            pattern[top_indices] >= threshold_value
                        ]
                        top_pixels[idx][key] = top_indices.tolist()

            return top_pixels

        elif (
            mode == "union_threshold"
            or mode == "union_threshold_on_max"
            or mode == "union_threshold_on_max"
        ):
            if saliency is not None:
                union_pixels_index = {}
                for idx in range(len(pattern_cache[patter_keys[0]])):
                    gradients = (
                        saliency["input_embeddings_gradients"][idx].sum(-1).abs()[0]
                    )
                    max_value = torch.max(gradients)
                    threshold_value = max_value * threshold
                    if threshold_value == 1:
                        union_pixels_index[idx] = list(
                            range(len(pattern_cache[patter_keys[0]][idx][0, 0]))
                        )
                    else:
                        if mode == "union_threshold_on_max":
                            top_indices = torch.argsort(gradients, descending=True)
                            top_indices = top_indices[
                                gradients[top_indices] >= threshold_value
                            ]
                            union_pixels_index[idx] = top_indices.tolist()
                        else:
                            raise NotImplementedError(
                                "mode must be 'union_threshold_on_max' for saliency"
                            )
                    if random_control:
                        k = len(union_pixels_index[idx])
                        assert k <= len(pattern_cache[patter_keys[0]][idx][0, 0]), (
                            "k must not exceed the population size"
                        )
                        random_indices = torch.randperm(
                            len(pattern_cache[patter_keys[0]][idx][0, 0])
                        )[:k]
                        union_pixels_index[idx] = random_indices.tolist()
                return union_pixels_index

            else:
                top_pixels_threshold = self.select_top_pixels(
                    pattern_cache,
                    frac_top_img_attn=frac_top_img_attn,
                    random_control=random_control,
                    threshold=threshold,
                    mode="threshold"
                    if mode == "union_threshold"
                    else "threshold_on_max",
                )

                union_pixels_index = {}
                for idx, top_pixels_dict in top_pixels_threshold.items():
                    union_pixels_index[idx] = []
                    for key, top_pixel in top_pixels_dict.items():
                        union_pixels_index[idx].append(
                            (
                                top_pixel,
                                pattern_cache[key][idx][0, 0][top_pixel].tolist(),
                            )
                        )

                    all_pixels = {}
                    for top_pixel_list, top_values_list in union_pixels_index[idx]:
                        for top_pixel, top_value in zip(
                            top_pixel_list, top_values_list
                        ):
                            if top_pixel not in all_pixels:
                                all_pixels[top_pixel] = top_value
                            else:
                                all_pixels[top_pixel] += top_value

                    union_pixels_index[idx] = [
                        x[0]
                        for x in sorted(
                            all_pixels.items(), reverse=True, key=lambda x: x[1]
                        )
                    ]
                    # get the index of the top 20% highest values
                    input_len = int(
                        len(pattern_cache[list(pattern_cache.keys())[0]][idx][0, 0])
                    )
                    if len(union_pixels_index[idx]) > input_len * frac_top_img_attn:
                        union_pixels_index[idx] = union_pixels_index[idx][
                            : int(input_len * frac_top_img_attn)
                        ]
                    else:
                        union_pixels_index[idx] = union_pixels_index[idx]

                    if random_control:
                        k = len(
                            union_pixels_index[idx]
                        )  # how many unique indices you need
                        assert k <= input_len, "k must not exceed the population size"

                        random_indices = torch.randperm(input_len)[:k]
                        # random_indices = torch.randint(low=0, high=input_len, size=(len(union_pixels_index[idx]),) )
                        union_pixels_index[idx] = random_indices.tolist()

            return union_pixels_index

        else:
            raise ValueError(
                "mode must be one of ['individual', 'top_sum', 'top_union'] got {mode}"
            )

    def ablate_heads(
        self,
        pattern_cache: ActivationCache,
        frac_top_img_attn=0.2,
        random_control=False,
        mode: Literal["individual", "top_sum", "top_union"] = "top_sum",
        ablation_mode: Literal["plain", "weighted"] = "plain",
        gamma_lambda: Optional[Tuple[float, float]] = None,
        threshold=None,
    ):
        """
        For each example and for each head find the top pixels and zerout it in the pattern of the head
        """
        if self.dataloader is None:
            raise ValueError("dataloader must be set before calling ablate_top_pixel.")

        top_pixels = self.select_top_pixels(
            pattern_cache,
            frac_top_img_attn=frac_top_img_attn,
            random_control=random_control,
            mode=mode,
            threshold=threshold,
        )

        interventions = {}
        if ablation_mode == "plain":
            # define intervention
            for index, top_pixels_dict in top_pixels.items():
                interventions[index] = []
                for key, top_pixel in top_pixels_dict.items():
                    interventions[index].append(
                        Intervention(
                            type="grid",
                            activation=key,
                            token_positions=(["last"], list(top_pixel)),
                            patching_values="ablation",
                            multiplication_value=0.0,
                        )
                    )

        elif ablation_mode == "weighted":
            assert self.cfact_heads is not None and self.fact_heads is not None, (
                "cfact_heads and fact_heads must be set before calling ablate_top_pixel."
            )
            assert gamma_lambda is not None, (
                "gamma_lambda must be set before calling ablate_top_pixel."
            )

            weights = self.get_multiplication_weights(
                cfact_heads=self.cfact_heads,
                fact_heads=self.fact_heads,
                gamma=gamma_lambda[0],
                lambda_param=gamma_lambda[1],
                rebalanced_weight=False,
            )

            # top_pixels = [{"pattern_L1H2": [0, 1, 2], "pattern_L2H3": [3, 4, 5]}, {"pattern_L1H2": [6, 7, 8], "pattern_L2H3": [9, 10, 11]}]

            # weights = [(1,2,0.3,"cfact"), (1,3,0.5,"fact"), (2,2,0.7,"other")]
            weights = {
                f"pattern_L{layer}H{head}": weight for layer, head, weight, _ in weights
            }

            # define intervention
            for index, top_pixels_dict in top_pixels.items():
                interventions[index] = []
                for key in weights.keys():
                    if key in top_pixels_dict:
                        top_pixel = top_pixels_dict[key]
                        interventions[index].append(
                            Intervention(
                                type="grid",
                                activation=key,
                                token_positions=(["last"], list(top_pixel)),
                                patching_values="ablation",
                                multiplication_value=weights[key],
                            )
                        )
                    else:
                        # If the key is not in top_pixels_dict, we can still add an intervention with a weight of 0
                        interventions[index].append(
                            Intervention(
                                type="grid",
                                activation=key,
                                token_positions=(
                                    ["last"],
                                    ["all-img"],
                                ),  # No tokens to ablate
                                patching_values="ablation",
                                multiplication_value=weights[
                                    key
                                ],  # Use the weight from the weights list
                            )
                        )

        data = statistics_computer(
            model=self.model,
            dataloader=self.dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=self.token_pair,
            return_essential_data=True,
            interventions=interventions,
            disable_text_interventions=True,
            compute=["image"],
        )
        return data

    def ablate_resid(
        self,
        pattern_cache: ActivationCache,
        frac_top_img_attn=0.2,
        random_control=False,
        mode: Literal["union_threshold", "union_threshold_on_max"] = "union_threshold",
        threshold: Optional[float] = None,  # value in [0,1]
        saliency: Optional[ActivationCache] = None,
    ):
        """
        From each example, find the best top pixels across all the relevant heads and ablate that position in the residual stream
        """
        if self.dataloader is None:
            raise ValueError("dataloader must be set before calling ablate_top_pixel.")

        assert mode == "union_threshold" or "union_threshold_on_max", (
            "mode must be 'union_threshold' for ablation"
        )

        top_pixels = self.select_top_pixels(
            pattern_cache,
            frac_top_img_attn=frac_top_img_attn,
            random_control=random_control,
            mode=mode,
            threshold=threshold,
            saliency=saliency,
        )
        stored_top_pixels = {}
        interventions = {}
        for index, top_pixels_list in top_pixels.items():
            start_image_position = pattern_cache["token_dict"][index]["all-image"][0]
            top_pixels_list = [x + start_image_position for x in top_pixels_list]

            sum_of_all_attn_values = None
            top_pixels_tensor = torch.tensor(top_pixels_list, dtype=torch.int)
            sum_of_all_attn_values = torch.zeros_like(
                top_pixels_tensor, dtype=torch.bfloat16
            )
            start_image = min(top_pixels_list)
            for key in pattern_cache.keys():
                if key.startswith("pattern_"):
                    attn_values = pattern_cache[key][index][0, 0][
                        top_pixels_tensor - start_image
                    ]
                    sum_of_all_attn_values += attn_values

            stored_top_pixels[index] = list(
                zip(top_pixels_list, sum_of_all_attn_values.tolist())
            )

            interventions[index] = [
                Intervention(
                    type="full",
                    activation="resid_in_0",
                    token_positions=list(top_pixels_list),
                    patching_values="ablation",
                )
            ]

        data = statistics_computer(
            model=self.model,
            dataloader=self.dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=self.token_pair,
            return_essential_data=True,
            interventions=interventions,
            compute=["image"],
        )
        all_top_pixels_lenght = [len(x) for x in top_pixels.values()]
        avg_num_pixel = sum(all_top_pixels_lenght) / len(all_top_pixels_lenght)

        data["avg_num_pixel"] = avg_num_pixel
        data["threshold"] = threshold
        data["LogitDiff"] = data["Image Cfact logit"] - data["Image Fact Logit"]
        return data, stored_top_pixels


def add_to_json(top_pixels, threshold, filename):
    # top pixels is a dict with index as key and list of top pixels as value
    # i want to save to a json with index the threshold and as a value top pixels
    # check if the file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}
    # add the top pixels to the data
    data[threshold] = top_pixels
    # save the data to the file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    # filename = "top_pixels.json"
    # with
    # open(filename, "w") as f


# Helper function to add experiment results to the DataFrame
def add_to_df(df, desc, top_pixel_pct, data, results_csv_path=None):
    """Helper function to add experiment results to the DataFrame."""
    new_row = {
        "ExperimentDesc": desc,
        "% top pixel": top_pixel_pct,
        **data,
    }
    # Ensure the new row only contains columns present in the DataFrame
    new_row_filtered = {k: v for k, v in new_row.items() if k in df.columns}
    # Use pd.concat instead of append for newer pandas versions
    df = pd.concat([df, pd.DataFrame([new_row_filtered])], ignore_index=True)
    if results_csv_path:
        df.to_csv(results_csv_path, index=False)
    return df


# --- Experiment Runner Functions ---


def run_baseline_experiment(
    experiment: "ImgCfactLocalizationExperiment", df: pd.DataFrame
):
    """Runs baseline statistics and adds them to the DataFrame."""
    logger.info("Running baseline experiment...")
    baseline_data = experiment.run_baseline_stats(
        filename="tmp" if not experiment.config.debug else None
    )
    baseline_data["avg_num_pixel"] = 0.0  # Set avg_num_pixel to 0 for baseline
    baseline_data["LogitDiff"] = (
        baseline_data["Image Cfact logit"] - baseline_data["Image Fact logit"]
    )  # Calculate LogitDiff for baseline
    baseline_data["threshold"] = None  # Set threshold to None for baseline
    df = add_to_df(df, "baseline", "-", baseline_data)
    logger.info("Baseline experiment complete.")
    # Return baseline_data separately for initial column setup if needed
    return df, baseline_data


def run_multiple_resid_ablation_with_control(
    experiment: "ImgCfactLocalizationExperiment",
    cache: ActivationCache,
    df: pd.DataFrame,
    frac_top_img_attn: float,
    mode: Literal["union_threshold"] = "union_threshold_on_max",
    # threshold=[0,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009],#0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009, 0.01,0.02,0.03,0.04,0.05,0.1,0.15, 0.2, 0.4, 0.6, 0.8, 1],
    threshold=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1],
    top_pixels_filename: str = "top_pixels.json",
    saliency: Optional[ActivationCache] = None,
    results_csv_path: Optional[Path] = None,
):
    base_desc = "resid_ablation"
    for t in threshold:
        logger.info(
            f"Running experiment: {base_desc} with frac_top_img_attn={frac_top_img_attn} and threshold={t}"
        )
        data, top_pixels = experiment.ablate_resid(
            cache,
            frac_top_img_attn=1,
            random_control=False,
            mode="union_threshold_on_max",
            threshold=t,
            saliency=None,
        )
        df = add_to_df(df, f"{base_desc}", frac_top_img_attn, data, results_csv_path)
        data, _ = experiment.ablate_resid(
            cache,
            frac_top_img_attn=1,
            random_control=True,
            mode="union_threshold_on_max",
            threshold=t,
            saliency=None,
        )
        df = add_to_df(
            df, f"{base_desc}_control", frac_top_img_attn, data, results_csv_path
        )
        data, top_pixels_grad = experiment.ablate_resid(
            cache,
            frac_top_img_attn=1,
            random_control=False,
            mode="union_threshold_on_max",
            threshold=t,
            saliency=saliency,
        )
        df = add_to_df(
            df, f"{base_desc}_grad", frac_top_img_attn, data, results_csv_path
        )
        add_to_json(
            {"top_pixels": top_pixels, "top_pixels_grad": top_pixels_grad},
            t,
            top_pixels_filename,
        )
        logger.info(f"Finished experiment: {base_desc} with threshold={t}")
    return df


def run_resid_ablation(
    experiment: "ImgCfactLocalizationExperiment",
    cache: ActivationCache,
    df: pd.DataFrame,
    frac_top_img_attn: float,
    random_control=False,
    mode: Literal["union_threshold"] = "union_threshold",
    threshold: Optional[float] = None,  # value in [0,1]
):
    """Runs ablation on common top pixels across heads (residual stream ablation)."""
    desc = "resid_ablation" + ("_control" if random_control else "")
    logger.info(
        f"Running experiment: {desc} with frac_top_img_attn={frac_top_img_attn}"
    )
    data = experiment.ablate_resid(
        cache,
        frac_top_img_attn=frac_top_img_attn,
        random_control=random_control,
        mode=mode,  # Assuming top_union is the desired mode
        threshold=threshold,
    )
    df = add_to_df(df, desc, frac_top_img_attn, data)
    logger.info(f"Finished experiment: {desc}")
    return df


def run_ablation_pattern_plain(
    experiment: "ImgCfactLocalizationExperiment",
    cache: ActivationCache,
    df: pd.DataFrame,
    frac_top_img_attn: float,
    random_control=False,
    mode: Literal["individual", "top_sum", "top_union"] = "top_union",
    threshold: Optional[float] = None,  # value in [0,1]
):
    """Runs plain ablation on top pixels for individual heads."""
    desc = "ablation_pattern_plain" + ("_control" if random_control else "")
    logger.info(
        f"Running experiment: {desc} with frac_top_img_attn={frac_top_img_attn}"
    )
    data = experiment.ablate_heads(
        cache,
        frac_top_img_attn=frac_top_img_attn,
        ablation_mode="plain",
        random_control=random_control,
        mode=mode,  # Assuming top_union is the desired mode,
        threshold=threshold,
    )
    df = add_to_df(df, desc, frac_top_img_attn, data)
    logger.info(f"Finished experiment: {desc} \n {df}")
    return df


def run_ablation_pattern_weighted(
    experiment: "ImgCfactLocalizationExperiment",
    cache: ActivationCache,
    df: pd.DataFrame,
    frac_top_img_attn: float,
    random_control=False,
    mode: Literal["individual", "top_sum", "top_union"] = "top_union",
    threshold: Optional[float] = None,  # value in [0,1]
):
    """Runs weighted ablation on top pixels for individual heads across gamma/lambda pairs."""
    desc_base = "ablation_pattern_weighted" + ("_control" if random_control else "")
    logger.info(
        f"Running experiment set: {desc_base} with frac_top_img_attn={frac_top_img_attn}. Mode: {mode} and threshold: {'' if mode != 'threshold' else threshold}"
    )

    # Define gamma_lambda pairs (can be moved to config or args if needed)
    gamma_lambda_pairs = [
        (0, 0),
        (-0.5, 0.5),
        (-1, 1),
        (-1.5, 1.5),
        (-2, 2),
        (-2.5, 2.5),
        (-3, 3),
        (-3.5, 3.5),
        (-4, 4),
    ]
    # gamma_lambda_pairs = [
    #     (0, 0),
    #     (-1,0),
    #     (-2,0),
    #     (-3,0),
    #     (-4,0)
    # ]

    # Renamed loop variable 'l' to 'lambda_val' for clarity
    for g, lambda_val in gamma_lambda_pairs:
        logger.info(f"Running weighted ablation with gamma={g}, lambda={lambda_val}")
        data = experiment.ablate_heads(
            cache,
            frac_top_img_attn=frac_top_img_attn,
            ablation_mode="weighted",
            gamma_lambda=(g, lambda_val),
            random_control=random_control,
            mode=mode,  # Assuming top_union is the desired mode
            threshold=threshold,
        )
        desc = f"{desc_base} g={g}, l={lambda_val}"  # Updated description format
        df = add_to_df(df, desc, frac_top_img_attn, data)
        # Save incrementally after each gamma/lambda pair for long runs
        # assert experiment.config.result_filename is not None # Removed assertion
        assert experiment.output_dir is not None  # Added assertion for output_dir
        results_csv_path = experiment.output_dir / "results.csv"
        df.to_csv(results_csv_path, index=False)
        logger.info(f"Results updated and saved for gamma={g}, lambda={lambda_val}")

    logger.info(f"Finished experiment set: {desc_base}")
    return df


# --- Main Execution Logic ---


def parse_args():
    parser = argparse.ArgumentParser(description="ImgCfactLocalization Experiment")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with a limited number of samples.",
    )
    parser.add_argument(
        "--k_heads",
        type=int,
        default=20,
        help="Number of counterfactual heads to analyze/intervene on.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        choices=[
            "baseline",
            "resid_ablation",
            "resid_ablation_control",
            "ablation_pattern_plain",
            "ablation_pattern_plain_control",
            "ablation_pattern_weighted",
            "ablation_pattern_weighted_control",
            "multiple_resid_ablation_with_control",
        ],
        help="List of experiments to run.",
    )
    parser.add_argument(
        "--frac_top_img_attn",
        type=float,
        default=0.2,
        help="Fraction of top image attention pixels to consider/ablate.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="_no_tag_",
        help="Tag for the experiment run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="top_union",
        choices=[
            "individual",
            "top_sum",
            "top_union",
            "threshold",
            "union_threshold",
            "union_threshold_on_max",
        ],
        help="Mode for selecting top pixels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Threshold for selecting top pixels (only used in 'threshold' mode).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="Model name to use for the experiment.",
    )
    parser.add_argument(
        "--saliency",
        action="store_true",
        help="Use saliency maps for the experiment.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Get Commit Hash ---
    try:
        # Use run_in_terminal to get the commit hash
        # Note: This requires the 'run_in_terminal' tool to be available and configured.
        # The actual call is made outside this function before calling main() or passed in.
        # For now, we assume it's fetched and stored in a variable `commit_hash_output`
        # commit_hash = commit_hash_output.strip() if commit_hash_output else "N/A"
        # Placeholder - replace with actual tool call result handling
        commit_hash = "FETCHED_HASH_PLACEHOLDER"  # Replace with actual hash later
        logger.info(f"Current Git commit hash: {commit_hash}")
    except Exception as e:
        logger.warning(f"Could not get git commit hash: {e}")
        commit_hash = "N/A"

    # --- Configuration Setup ---
    config = ImgCfactLocalizationConfig(
        model_name=args.model,
        debug=args.debug,
        experiment_tag=args.tag,  # Updated tag
        k_heads=args.k_heads,
        # Auto-generate description based on selected experiments
        experiment_description=f"Run: {', '.join(args.experiments)}; k={args.k_heads}; frac_attn={args.frac_top_img_attn}",
        run_experiments=args.experiments,  # Store executed experiments
        commit_hash=commit_hash,  # Store commit hash
    )
    if config.debug:
        config.experiment_tag += "_debug"
        assert config.experiment_description is not None  # Added assertion
        config.experiment_description += " (Debug)"

    # --- Output Directory Setup ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_name = f"ImgCfactLoc_{config.model_name.replace('/', '-')}_{timestamp}{config.experiment_tag}"
    output_dir = Path("results/2_ImgCfactLocalization") / base_output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_json_path = output_dir / "config.json"
    results_csv_path = output_dir / "results.csv"

    # Save config to the new location
    config.to_json(config_json_path)

    # --- Experiment Setup ---
    experiment = ImgCfactLocalizationExperiment(config)
    experiment.output_dir = (
        output_dir  # Assign the output directory to the experiment object
    )

    logger.info(f"Configuration saved to {config_json_path}")
    logger.info(f"Results will be saved to: {results_csv_path}")
    logger.info(f"Experiment description: {config.experiment_description}")

    # ... rest of the setup (model, dataloader, heads, cache) remains the same ...
    logger.info("Setting up model and dataloader...")
    experiment.setup_model().create_dataloader(None).filter_dataloader()
    logger.info("Selecting heads...")
    # Selects and stores cfact_heads/fact_heads in the experiment object
    experiment.select_heads(k_heads=config.k_heads)
    logger.info("Extracting activation patterns...")
    saliency_cache = experiment.extract_gradients()

    cache = experiment.extract_patterns()
    logger.info("Setup complete.")

    # --- Result Initialization ---
    # ... baseline run remains the same ...
    logger.info("Running baseline to determine result columns...")
    baseline_data = experiment.run_baseline_stats(
        filename="tmp" if not config.debug else None
    )
    # Assuming baseline_data is Dict[str, Any] despite potential type checker issue
    df_columns = [
        "ExperimentDesc",
        "% top pixel",
        "threshold",
        "LogitDiff",
        "avg_num_pixel",
    ] + list(baseline_data.keys())
    results_df = pd.DataFrame(columns=df_columns)
    logger.info("Result DataFrame initialized.")

    # --- Experiment Execution ---
    # ... experiment_runners map remains the same ...
    experiment_runners = {
        "baseline": (run_baseline_experiment, {}),
        "resid_ablation": (run_resid_ablation, {"random_control": False}),
        "resid_ablation_control": (
            run_resid_ablation,
            {"random_control": True},
        ),
        "ablation_pattern_plain": (
            run_ablation_pattern_plain,
            {"random_control": False},
        ),
        "ablation_pattern_plain_control": (
            run_ablation_pattern_plain,
            {"random_control": True},
        ),
        "ablation_pattern_weighted": (
            run_ablation_pattern_weighted,
            {"random_control": False},
        ),
        "ablation_pattern_weighted_control": (
            run_ablation_pattern_weighted,
            {"random_control": True},
        ),
        "multiple_resid_ablation_with_control": (
            run_multiple_resid_ablation_with_control,
            {
                "top_pixels_filename": str(output_dir / "top_pixels.json"),
                "saliency": saliency_cache,
            },
        ),
    }

    # Add baseline results to DF only if requested
    if "baseline" in args.experiments:
        results_df = add_to_df(results_df, "baseline", "-", baseline_data)
        logger.info("Baseline results added to DataFrame.")
        # Save initial baseline result to the new path
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Results saved to {results_csv_path}")

    # Run selected experiments (excluding baseline if already handled)
    for exp_name in args.experiments:
        if exp_name == "baseline":
            continue  # Already handled above

        if exp_name in experiment_runners:
            runner_func, runner_params = experiment_runners[exp_name]

            # Prepare arguments for the runner function
            call_args = {
                "experiment": experiment,
                "cache": cache,
                "df": results_df,
                "frac_top_img_attn": args.frac_top_img_attn,
                "mode": args.mode,
                "results_csv_path": str(results_csv_path),
                **runner_params,  # Add specific params like random_control
            }

            # Call the runner function
            results_df = runner_func(**call_args)

            # Save after each experiment type completes (weighted saves internally per pair)
            if "weighted" not in exp_name:
                # Save to the new path
                results_df.to_csv(results_csv_path, index=False)
                logger.info(f"Results updated and saved to {results_csv_path}")
        else:
            # This case should not happen due to 'choices' in argparse, but good practice
            logger.warning(
                f"Experiment '{exp_name}' is selected but not recognized. Skipping."
            )

    # --- Finalization ---
    logger.info("All selected experiments completed.")
    # Final save (can be redundant but ensures the latest state is saved)
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Final results saved to {results_csv_path}")

    print(
        f"\nExperiment run complete. Results and config saved in {experiment.output_dir}"
    )


if __name__ == "__main__":
    main()
