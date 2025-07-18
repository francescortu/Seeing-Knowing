import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

# import numpy as np # Removed unused import
import json
import datetime
import argparse
import torch

# import copy  # Removed unused import
from pathlib import Path


from typing import (
    List,
    Tuple,
    Optional,
    Dict,
    Any,
    Literal,
    Union,
)  # Removed unused Union
from dataclasses import dataclass  # Removed unused field, asdict

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
class ImgCfactLocalizationConfig:
    """
    Configuration class for ImgCfactLocalization experiment.
    """

    model_name: str
    experiment_tag: str
    debug: bool = False
    experiment_description: Optional[str] = None
    k_heads: int = 20
    run_experiments: Optional[List[str]] = None  # Added metadata
    commit_hash: Optional[str] = None  # Added metadata


class ImgCfactLocalizationExperiment:
    def __init__(self, config: ImgCfactLocalizationConfig):
        """
        Initialize the ImgCfactLocalizationExperiment with the given configuration.
        """

        self.config = config
        self.manager = ExperimentManager.init(
            model_name=config.model_name,
            tag=config.experiment_tag,
        )
        if self.config.debug:
            self.manager.config.debug.debug = True
            self.manager.config.debug.debug_samples = 20
        self.output_dir: Optional[Path] = None  # Specific output dir for this run
        # self.cfact_heads: Optional[List[Tuple[int, int]]] = None
        # self.fact_heads: Optional[List[Tuple[int, int]]] = None
        self.manager.load_dataset_from_hf()
        self.manager.setup_model()
        self.manager.setup_dataloader()
        self.manager.setup_model_specific_variables(filter_dataloader=True)

        self.logger = logger.getChild(
            f"ImgCfactLocalizationExperiment-{self.config.experiment_tag}"
        )

        self.cfact_heads, self.fact_heads = self.manager.select_heads(
            k_heads=self.config.k_heads,
        )

    def extract_gradients(self):
        self.logger.debug("Identifying top pixel using gradients...")

        if self.cfact_heads is None:
            raise ValueError("cfact_heads must be set before calling ablate_top_pixel.")

        for i in range(len(self.manager.dataloader)):
            paired_token = self.manager.token_pair[i]
            cfact_idx = self.manager.tokenizer(text=paired_token[0])["input_ids"][0][1]

            self.manager.dataloader[i]["text_image_inputs"]["vocabulary_index"] = (
                cfact_idx
            )

        cache = self.manager.model.extract_cache(
            [d["text_image_inputs"] for d in self.manager.dataloader],
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
        if self.manager.dataloader is None:
            raise ValueError("dataloader must be set before calling ablate_top_pixel.")

        def pattern_agr(old, new):
            if old is None:
                return [new]
            return old + [new]

        # extract the top pixel from the cfact_heads
        pattern_cache = self.manager.model.extract_cache(
            [d["text_image_inputs"] for d in self.manager.dataloader],
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
            "union_percentage_on_max",
        ] = "top_union",
        saliency: Optional[ActivationCache] = None,
    ):
        """
        For each example and for each head find the top pixels and zerout it in the pattern of the head
        """
        if self.manager.dataloader is None:
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

        elif mode == "union_percentage_on_max":
            if threshold is not None:
                self.logger.warning(
                    f"Mode is '{mode}', but a 'threshold' value was provided. "
                    "The 'threshold' parameter will be ignored in favor of 'frac_top_img_attn'."
                )
            if saliency:  # we use gradients
                union_pixels_index = {}
                for idx in range(len(pattern_cache[patter_keys[0]])):
                    gradients = (
                        saliency["input_embeddings_gradients"][idx].sum(-1).abs()[0]
                    )
                    top_indices = torch.argsort(gradients, descending=True)
                    top_indices = top_indices[
                        : int(
                            len(pattern_cache[patter_keys[0]][idx][0, 0])
                            * frac_top_img_attn
                        )
                    ]
                    union_pixels_index[idx] = top_indices.tolist()

            else:
                # get top pixels for all the patterns
                len_dataset = len(pattern_cache[patter_keys[0]])
                for idx in range(len_dataset):
                    top_pixels[idx] = {}
                    for key in patter_keys:
                        pattern = pattern_cache[key][idx][0, 0]
                        top_indices = torch.argsort(pattern, descending=True)
                        top_pixels[idx][key] = top_indices.tolist()

                union_pixels_index = {}
                for idx, top_pixels_dict in top_pixels.items():
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

                    union_pixels_index[idx] = union_pixels_index[idx][
                        : int(input_len * frac_top_img_attn)
                    ]
            if random_control:
                random_union_pixels_index = {}
                for idx, union_pixels in union_pixels_index.items():
                    k = len(union_pixels)
                    random_indices = torch.randperm(
                        len(pattern_cache[patter_keys[0]][idx][0, 0])
                    )[:k]
                    random_union_pixels_index[idx] = random_indices.tolist()
                union_pixels_index = random_union_pixels_index
            return union_pixels_index
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
                    frac_top_img_attn=1,
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
                    if len(union_pixels_index[idx]) > input_len * 1:
                        union_pixels_index[idx] = union_pixels_index[idx][
                            : int(input_len * 1)
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

    def ablate_resid(
        self,
        pattern_cache: ActivationCache,
        frac_top_img_attn=0.2,
        random_control=False,
        mode: Literal[
            "union_threshold", "union_threshold_on_max", "union_percentage_on_max"
        ] = "union_threshold",
        threshold: Optional[float] = None,  # value in [0,1]
        saliency: Optional[ActivationCache] = None,
    ):
        """
        From each example, find the best top pixels across all the relevant heads and ablate that position in the residual stream
        """
        if self.manager.dataloader is None:
            raise ValueError("dataloader must be set before calling ablate_top_pixel.")

        assert mode in [
            "union_threshold",
            "union_threshold_on_max",
            "union_percentage_on_max",
        ], "mode must be one of the union modes for ablation"

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

            # stored_top_pixels[index] = list(
            #     zip(top_pixels_list, sum_of_all_attn_values.tolist())
            # )

            stored_top_pixels[
                f"original_index_{self.manager.map_filtered_to_original_index[index]}"
            ] = list(zip(top_pixels_list, sum_of_all_attn_values.tolist()))
            # )

            interventions[index] = [
                Intervention(
                    type="full",
                    activation="resid_in_0",
                    token_positions=list(top_pixels_list),
                    patching_values="ablation",
                )
            ]

        _, data = statistics_computer(
            model=self.manager.model,
            dataloader=self.manager.dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=self.manager.token_pair,
            # return_essential_data=True,
            interventions=interventions,
            compute=["image"],
        )
        all_top_pixels_lenght = [len(x) for x in top_pixels.values()]
        avg_num_pixel = sum(all_top_pixels_lenght) / len(all_top_pixels_lenght)

        data["avg_num_pixel"] = avg_num_pixel
        data["threshold"] = threshold if threshold is not None else frac_top_img_attn
        data["LogitDiff"] = data["Image Cfact logit"] - data["Image Fact Logit"]
        return data, stored_top_pixels

    def config_to_json(self, path: Union[Path, str]):
        """
        Convert the experiment configuration to a JSON-compatible dictionary.
        """
        base_config_dict = self.manager.config.to_dict()
        config_dict = {
            "model_name": self.config.model_name,
            "experiment_tag": self.config.experiment_tag,
            "debug": self.config.debug,
            "experiment_description": self.config.experiment_description,
            "k_heads": self.config.k_heads,
            "commit_hash": self.config.commit_hash,
            **base_config_dict,
        }
        return config_dict

    def format_baseline_data(self):
        baseline = self.manager.get_baseline()

        return baseline


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


def run_multiple_resid_ablation_with_control(
    experiment: "ImgCfactLocalizationExperiment",
    cache: ActivationCache,
    df: pd.DataFrame,
    mode: Literal[
        "union_threshold", "union_threshold_on_max", "union_percentage_on_max"
    ],
    iteration_values: List[float] = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.07,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
    ],
    top_pixels_filename: str = "top_pixels.json",
    saliency: Optional[ActivationCache] = None,
    results_csv_path: Optional[Path] = None,
):
    base_desc = "resid_ablation"

    # Determine which parameter to use based on the mode
    is_percentage_mode = "percentage" in mode
    iteration_param_name = "frac_top_img_attn" if is_percentage_mode else "threshold"

    for val in iteration_values:
        logger.info(
            f"Running experiment: {base_desc} with {iteration_param_name}={val}. Mode: {mode}"
        )

        # Prepare base arguments for ablate_resid
        ablate_args: Dict[str, Any] = {
            "mode": mode,
        }
        if is_percentage_mode:
            ablate_args["frac_top_img_attn"] = val
        else:
            ablate_args["threshold"] = val

        # 1. Main experiment (no control, no saliency)
        data, top_pixels = experiment.ablate_resid(
            pattern_cache=cache, random_control=False, saliency=None, **ablate_args
        )
        df = add_to_df(df, f"{base_desc}", "N/A", data, results_csv_path)

        # 2. Control experiment (random control, no saliency)
        data_control, _ = experiment.ablate_resid(
            pattern_cache=cache, random_control=True, saliency=None, **ablate_args
        )
        df = add_to_df(
            df, f"{base_desc}_control", "N/A", data_control, results_csv_path
        )

        # 3. Gradient-based experiment (no control, with saliency)
        if saliency:
            data_grad, top_pixels_grad = experiment.ablate_resid(
                pattern_cache=cache,
                random_control=False,
                saliency=saliency,
                **ablate_args,
            )
            df = add_to_df(df, f"{base_desc}_grad", "N/A", data_grad, results_csv_path)
            add_to_json(
                {"top_pixels": top_pixels, "top_pixels_grad": top_pixels_grad},
                val,
                top_pixels_filename,
            )
        else:
            add_to_json(
                {"top_pixels": top_pixels},
                val,
                top_pixels_filename,
            )

        logger.info(
            f"Finished experiment: {base_desc} with {iteration_param_name}={val}"
        )
    return df


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

    # parser.add_argument(
    #     "--frac_top_img_attn",
    #     type=float,
    #     default=0.2,
    #     help="Fraction of top image attention pixels to consider/ablate.",
    # )
    parser.add_argument(
        "--tag",
        type=str,
        default="_no_tag_",
        help="Tag for the experiment run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="union_threshold_on_max",
        choices=[
            "individual",
            "top_sum",
            "top_union",
            "threshold",
            "union_threshold",
            "union_threshold_on_max",
            "union_percentage_on_max",
        ],
        help="Mode for selecting top pixels.",
    )
    # parser.add_argument(
    #     "--threshold",
    #     type=float,
    #     default=0.8,
    #     help="Threshold for selecting top pixels (only used in 'threshold' mode).",
    # )
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

    # --- Configuration Setup ---
    config = ImgCfactLocalizationConfig(
        model_name=args.model,
        debug=args.debug,
        experiment_tag=args.tag,  # Updated tag
        k_heads=args.k_heads,
        # Auto-generate description based on selected experiments
        experiment_description=f"Run: k={args.k_heads}",
    )
    if config.debug:
        config.experiment_tag += "_debug"
        assert config.experiment_description is not None  # Added assertion
        config.experiment_description += " (Debug)"

    # --- Output Directory Setup ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create folder structure: results/2_ImgCfactLocalization/tag_name/model_timestamp
    tag_folder = Path("results/2_ImgCfactLocalization") / config.experiment_tag
    output_dir = tag_folder / f"{config.model_name.replace('/', '-')}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_json_path = output_dir / "config.json"
    results_csv_path = output_dir / "results.csv"

    # Save config to the new location

    # --- Experiment Setup ---
    experiment = ImgCfactLocalizationExperiment(config)
    experiment.output_dir = (
        output_dir  # Assign the output directory to the experiment object
    )
    experiment.config_to_json(config_json_path)

    logger.info(f"Configuration saved to {config_json_path}")
    logger.info(f"Results will be saved to: {results_csv_path}")
    logger.info(f"Experiment description: {config.experiment_description}")

    # ... rest of the setup (model, dataloader, heads, cache) remains the same ...

    saliency_cache = experiment.extract_gradients()
    cache = experiment.extract_patterns()
    logger.info("Setup complete.")

    # --- Result Initialization ---
    # ... baseline run remains the same ...
    logger.info("Running baseline to determine result columns...")
    baseline_data = experiment.format_baseline_data()
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

    results_df = add_to_df(results_df, "baseline", "-", baseline_data)
    logger.info("Baseline results added to DataFrame.")
    # Save initial baseline result to the new path
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to {results_csv_path}")

    results_df = run_multiple_resid_ablation_with_control(
        experiment,
        cache,
        results_df,
        mode=args.mode,
        top_pixels_filename=str(output_dir / "top_pixels.json"),
        saliency=saliency_cache if args.saliency else None,
        results_csv_path=results_csv_path,
    )
    logger.info("Multiple resid ablation experiments completed.")

    # Final save (can be redundant but ensures the latest state is saved)
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Final results saved to {results_csv_path}")

    print(
        f"\nExperiment run complete. Results and config saved in {experiment.output_dir}"
    )


if __name__ == "__main__":
    main()
