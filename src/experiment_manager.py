from dataclasses import dataclass, field
import os
import logging
import datetime
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import GenerationConfig
from tqdm import tqdm
import torch
from easyroutine.interpretability import (
    HookedModel,
    ExtractionConfig,
)
from easyroutine.interpretability.tools import LogitLens
from easyroutine.interpretability.utils import selective_log_softmax
from src.datastatistics import statistics_computer
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from PIL import Image
import json
import urllib.request
import io


@dataclass
class DebugConfig:
    debug: bool = False
    debug_samples: int = 10


@dataclass
class SelectHeadsConfig:
    k_heads: int = 20
    prec: float = 0.02
    sampling_mode: str = "fixed_number"  # "fixed_number" or "percentile"
    metric: str = "accuracy"  # "accuracy" or "loss"


@dataclass
class BaseConfig:
    """Base configuration class for experiments"""

    model_name: str
    experiment_tag: str
    debug: DebugConfig = field(
        default_factory=lambda: DebugConfig(debug=False, debug_samples=10)
    )
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "llava-hf/llava-v1.6-mistral-7b-hf": "[INST] <image>\n [/INST] {text}",
            "llava-hf/llava-v1.6-mistral-7b-hf-instruction": "[INST] <image>\n {text} [/INST]",
            "google/gemma-3-12b-it": "<bos><start_of_turn>user\n<start_of_image><end_of_turn>\n<start_of_turn>model\n{text}",
            "google/gemma-3-12b-it-instruction": "<bos><start_of_turn>user\n<start_of_image><end_of_turn>\n<start_of_turn>model\n{text}",
            "facebook/chameleon-7b": "<image>\n {text}",
        }
    )
    dataset_subset: str = "full"  # "full", "small1", or "small2"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary"""
        return {
            "model_name": self.model_name,
            # "tag": self.tag,
            "experiment_tag": self.experiment_tag,
            "extra_metadata": self.extra_metadata,
            "debug": self.debug,
            "prompt_templates": self.prompt_templates,
        }

    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


class ExperimentManager:
    """Base class for running experiments with vision-language models"""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.model: HookedModel
        self.tokenizer = None
        self.text_tokenizer = None

        self.dataloader: Optional[List[Dict[str, Any]]] = None
        self.dataset: Optional[Union[Dataset, DatasetDict]] = None
        self.token_pair = None

        self.baseline = None
        self.cfact_heads: Optional[List[Tuple[int, int]]] = None
        self.fact_heads: Optional[List[Tuple[int, int]]] = None

        # Setup logging
        self.setup_logging()

        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def init(
        cls,
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        tag="default",
    ):
        """Initialize the experiment manager with a configuration"""
        config = BaseConfig(
            model_name=model_name,
            experiment_tag=tag,
        )
        return cls(config)

    def load_dataset_from_hf(
        self, dataset_name_or_object: Union[str, Dataset] = "francescortu/whoops-aha"
    ):
        """Load a dataset from Hugging Face"""
        if isinstance(dataset_name_or_object, Dataset):
            self.dataset = dataset_name_or_object
        else:
            self.dataset = load_dataset(dataset_name_or_object)

        self.dataset = (
            self.dataset["train"] if "train" in self.dataset else self.dataset
        )
        return self

    def load_dataset_from_local(
        self, dataset_path: Path = Path("data/emnlp_submission/dataset")
    ):
        csv_path = dataset_path / "whoops-aha!.csv"
        images_path = dataset_path / "images"

        # Load CSV data
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise

        # Create a list to hold dataset items
        dataset_items = []

        for _, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Loading dataset items"
        ):
            image_filename = row["image_filename"]
            image_file_path = images_path / image_filename

            if not image_file_path.exists():
                print(f"Image file not found: {image_file_path}, skipping this item.")
                continue

            try:
                # Load image using PIL
                from PIL import Image

                img = Image.open(image_file_path)

                dataset_item = {
                    "image": img,
                    "text": row["text"],  # Assuming 'theme' is the text column
                    "image_id": image_filename,
                    # Explicitly get factual/counterfactual tokens, as they might be used later.
                    # Default to None if not present in the CSV.
                    "factual_tokens": ast.literal_eval(row.get("factual_tokens")),
                    "counterfactual_tokens": ast.literal_eval(
                        row.get("counterfactual_tokens")
                    ),
                }
                # Add all other columns from the CSV to the dataset_item,
                # skipping those already handled or special.
                for col in df.columns:
                    if col not in [
                        "image_filename",
                        "theme",
                        "factual_tokens",
                        "counterfactual_tokens",
                    ]:
                        dataset_item[col] = row[col]

                dataset_items.append(dataset_item)
            except Exception as e:
                print(f"Error processing image {image_file_path}: {e}")
                continue

        if not dataset_items:
            print("No items were successfully loaded into the dataset.")
            raise ValueError("Dataset could not be created as no items were loaded.")

        # Create Hugging Face Dataset from pandas DataFrame
        # This requires specifying features if there are complex types like PIL.Image.Image
        # For images, datasets library handles PIL Image objects directly.
        self.dataset = Dataset.from_list(dataset_items)

    def setup_model(self, device_map="auto"):
        """Initialize model and tokenizers"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        self.model = HookedModel.from_pretrained(
            self.config.model_name, device_map=device_map
        )
        self.tokenizer = self.model.get_processor()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.logger.info("Model loaded successfully")

    # def get_model_and_dataloader(self):
    #     """Initialize model and dataloader"""
    #     self.setup_model()
    #     self.create_dataloader()
    #     return self.model, self.dataloader

    def setup_logging(self):
        """Configure logging with both file and console output"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Setup file handler
        file_handler = logging.FileHandler(log_dir / f"experiment_{timestamp}.log")
        file_handler.setLevel(logging.INFO)

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Configure format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        # Remove any existing handlers to avoid duplicates
        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("Logger initialized")
        return self

    def setup_dataloader(self):
        """Create and save the dataloader with both image and text inputs"""
        assert self.dataset is not None, (
            "Dataset must be loaded before creating dataloader"
        )
        assert self.model is not None, (
            "Model must be initialized before creating dataloader"
        )
        assert self.tokenizer is not None, (
            "Tokenizer must be initialized before creating dataloader"
        )

        if not self.model or not self.tokenizer or not self.text_tokenizer:
            raise ValueError("Model must be initialized before creating dataloader")

        self.logger.info("Creating new dataloader...")

        self.logger.info(f"Dataset loaded. Length: {len(self.dataset)}")

        # In debug mode, take only the first few examples
        dataset_iter = self.dataset
        if self.config.debug.debug:
            self.logger.info(
                f"Debug mode: using only first {self.config.debug.debug_samples} examples"
            )
            dataset_iter = list(dataset_iter)[: self.config.debug.debug_samples]
        if self.config.dataset_subset == "small1":
            self.logger.info("Using small1 subset of the dataset")
            dataset_iter = list(dataset_iter)[:250]
        elif self.config.dataset_subset == "small2":
            self.logger.info("Using small2 subset of the dataset")
            dataset_iter = list(dataset_iter)[250]
        else:
            self.logger.info("Using full dataset")

        self.dataloader = []
        for item in tqdm(dataset_iter, desc="Processing dataset"):
            # Convert item to dict if it's not already
            item_dict = dict(item)

            # Process image and text with processor
            inputs = self.tokenizer(
                text=self.config.prompt_templates[self.config.model_name].format(
                    text=item_dict["text"]
                ),
                images=[item_dict["image"].convert("RGB")],
                return_tensors="pt",
            )

            # Process text with text tokenizer
            if hasattr(self.text_tokenizer, "tokenizer"):
                # Handle processor with tokenizer attribute
                text_inputs = self.text_tokenizer.tokenizer(
                    item_dict["text"], return_tensors="pt"
                )
            elif hasattr(self.text_tokenizer, "__call__"):
                # Handle direct tokenizer
                text_inputs = self.text_tokenizer(
                    item_dict["text"], return_tensors="pt"
                )
            else:
                raise ValueError("Invalid text tokenizer configuration")

            self.dataloader.append(
                {
                    "text_image_inputs": {**inputs},
                    "text_inputs": text_inputs,
                    "text": item_dict["text"],
                    "image": item_dict["image"],
                    "image_id": item_dict.get("image_id"),
                    "counterfactual_tokens": item_dict["counterfactual_tokens"],
                    "factual_tokens": item_dict["factual_tokens"],
                }
            )

        # if filter:
        #     self.logger.info("Filtering dataloader...")
        #     self.filter_dataloader()

        # return self

    def setup_model_specific_variables(
        self,
        filter_dataloader: bool = True,
        # compute_baseline: bool = True,
    ):
        assert self.model is not None, (
            "Model must be initialized before setting up variables"
        )
        assert self.dataloader is not None, (
            "Dataloader must be initialized before setting up variables"
        )
        assert self.tokenizer is not None, (
            "Tokenizer must be initialized before setting up variables"
        )
        assert self.text_tokenizer is not None, (
            "Text tokenizer must be initialized before setting up variables"
        )

        self.logger.info("Setting up model-specific variables...")

        self.logger.info("Getting token pairs...")
        self.token_pair, self.baseline = self.compute_token_pair()
        self.logger.info(f"Got {len(self.token_pair)} token pairs")

        if filter_dataloader:
            self.logger.info("Filtering dataloader...")
            self.token_pair, self.dataloader, self.baseline = self.filter_dataloader()

        # baseline_deprecated = self.deprecated_run_baseline_stats()
        # assert baseline_deprecated["Fact Acc"] == self.baseline["Fact Acc"]

    def compute_token_pair(self):
        """Compute the token pairs for counterfactual and factual tokens"""
        assert self.dataloader is not None, (
            "Dataloader must be initialized before computing token pairs"
        )
        assert self.model is not None, (
            "Model must be initialized before computing token pairs"
        )
        assert self.text_tokenizer is not None, (
            "Text tokenizer must be initialized before computing token pairs"
        )

        token_pair, baseline = statistics_computer(
            model=self.model,
            dataloader=self.dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=None,
            # return_essential_data=True,
        )
        self.logger.info("compute_token_pair: Token pairs computed successfully")
        return token_pair, baseline

    def filter_dataloader(self):
        assert self.model is not None, (
            "Model must be initialized before filtering dataloader"
        )
        assert self.dataloader is not None, (
            "Dataloader must be initialized before filtering"
        )
        assert self.token_pair is not None, (
            "Token pair must be initialized before filtering dataloader"
        )
        assert self.baseline is not None, (
            "Baseline must be computed before filtering dataloader"
        )

        # data, token_pair = statistics_computer(
        #     model=self.model,
        #     dataloader=self.dataloader,
        #     write_to_file=False,
        #     filename=None,
        #     dataset_path=Path(""),
        #     given_token_pair=self.token_pair,
        #     # return_essential_data=True,
        # )

        indexes_cfact_gt_fact_text = self.baseline["indexes_cfact_gt_fact_text"]
        self.logger.info(
            f"Filtering dataloader: {len(self.dataloader)} -> {len(self.dataloader) - len(indexes_cfact_gt_fact_text)}. Removing {len(indexes_cfact_gt_fact_text)} samples"
        )

        dataloader = []
        token_pair = []
        self.map_filtered_to_original_index = {}
        for i in range(len(self.dataloader)):
            if i not in indexes_cfact_gt_fact_text:
                # add to dataloader
                dataloader.append(self.dataloader[i])
                token_pair.append(self.token_pair[i])
                self.map_filtered_to_original_index[len(dataloader) - 1] = i

        # dataloader = [
        #     item
        #     for i, item in enumerate(self.dataloader)
        #     if i not in indexes_cfact_gt_fact_text
        # ]
        # token_pair = [
        #     item
        #     for i, item in enumerate(self.token_pair)
        #     if i not in indexes_cfact_gt_fact_text
        # ]
        self.logger.info(f"Filtered dataloader length: {len(self.dataloader)}")
        self.logger.info(f"Index removed: {indexes_cfact_gt_fact_text}")
        # for i, pair in enumerate(self.token_pair):
        #     self.logger.info(f"Token pair {i}: {pair}")

        self.logger.info("Baseline found. Updating with filtered data")
        _, new_baseline = statistics_computer(
            model=self.model,
            dataloader=dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=token_pair,
            # return_essential_data=True,
        )
        self.logger.info(
            f"New baseline computed after filtering dataloader: {self.baseline['Fact Acc']} -> {new_baseline['Fact Acc']}"
        )
        return token_pair, dataloader, new_baseline

    def launch_std_setup_routine(self, device_map="auto"):
        """Run the setup routine for the experiment"""
        self.logger.info("Running setup routine...")
        self.load_dataset_from_hf()
        self.logger.info("Dataset loaded successfully")
        self.setup_model(device_map=device_map)
        self.logger.info("Model loaded successfully")
        self.setup_dataloader()
        self.logger.info("Dataloader created successfully")
        self.setup_model_specific_variables(filter_dataloader=True)
        self.logger.info("Model-specific variables set up successfully")
        self.logger.info("Setup routine completed successfully")

    def select_heads(
        self,
        k_heads=20,
        return_df: bool = False,
        perc=0.02,
        sampling_mode="fixed_number",
        metric="accuracy",
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Use LogitLens to identify counterfactual and factual heads
        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: Lists of (layer, head) tuples for cfact and fact heads
        """
        assert self.model is not None, (
            "Model must be initialized before selecting heads"
        )
        assert self.dataloader is not None, (
            "Dataloader must be initialized before selecting heads"
        )
        assert self.token_pair is not None, (
            "Token pair must be initialized before selecting heads"
        )
        assert self.text_tokenizer is not None, (
            "Text tokenizer must be initialized before selecting heads"
        )
        assert k_heads > 0, "k_heads must be a positive integer"

        self.logger.info("Starting head selection with LogitLens analysis...")
        logit_lens = LogitLens.from_model(self.model)

        # Extract cache for analysis
        self.logger.info("Extracting model cache...")
        self.model.use_full_model()
        cache = self.model.extract_cache(
            [d["text_image_inputs"] for d in self.dataloader],
            target_token_positions=["last"],
            extraction_config=ExtractionConfig(
                extract_resid_out=True,
                extract_mlp_out=True,
                extract_attn_out=True,
                extract_head_out=True,
                extract_last_layernorm=True,
            ),
        )

        # Process tokens using tokenizer with proper handling
        token_directions = []
        for t0, t1 in self.token_pair:
            if hasattr(self.text_tokenizer, "tokenizer"):
                # Handle processor with tokenizer attribute
                result0 = self.text_tokenizer.tokenizer(t0, return_tensors="pt")
                result1 = self.text_tokenizer.tokenizer(t1, return_tensors="pt")
            elif hasattr(self.text_tokenizer, "__call__"):
                # Handle direct tokenizer
                result0 = self.text_tokenizer(t0, return_tensors="pt")
                result1 = self.text_tokenizer(t1, return_tensors="pt")
            else:
                raise ValueError("Invalid text tokenizer configuration")

            token_directions.append(
                (result0["input_ids"][0][1], result1["input_ids"][0][1])
            )

        # Analyze head contributions using the correct token pairs
        self.logger.info("Computing head contributions...")
        out = logit_lens.compute(
            activations=cache,
            target_key="head_out_L{i}H{j}",
            token_directions=token_directions,
            metric="accuracy",
            # metric=metric,
        )

        # Calculate mean impact of each head
        # mean_out = {
        #     k: float(2 * (v.mean().detach().cpu() - 0.5)) for k, v in out.items()
        # }
        mean_out = {k: float((v.mean().detach().cpu() - 0.5)) for k, v in out.items()}

        # mean_out = {
        #     k: float((v.mean().detach().cpu() )) for k, v in out.items()
        # }
        # Convert to dataframe for analysis
        df = pd.DataFrame(mean_out.items(), columns=["Head", "Value"])

        if sampling_mode == "fixed_number":
            full_cfact_heads = df[df["Value"] > 0]
            full_fact_heads = df[df["Value"] < 0]

            # Get top 20 counterfactual and factual heads
            cfact_heads_df = full_cfact_heads.nlargest(k_heads, "Value")
            fact_heads_df = full_fact_heads.nsmallest(k_heads, "Value")

        elif sampling_mode == "percentile":
            # get the top %perc and bottom %perc heads
            qs = df["Value"].quantile([1 - perc, perc])
            cfact_heads_df = df[df["Value"] > qs[0]]
            fact_heads_df = df[df["Value"] < qs[1]]

        # Extract (layer, head) tuples
        import re

        cfact_heads = []
        fact_heads = []

        for head in cfact_heads_df["Head"]:
            layer_match = re.search(r"L(\d+)", head)
            head_match = re.search(r"H(\d+)", head)
            if layer_match and head_match:
                cfact_heads.append(
                    (
                        int(layer_match.group(1)),
                        int(head_match.group(1)),
                    )
                )

        for head in fact_heads_df["Head"]:
            layer_match = re.search(r"L(\d+)", head)
            head_match = re.search(r"H(\d+)", head)
            if layer_match and head_match:
                fact_heads.append(
                    (
                        int(layer_match.group(1)),
                        int(head_match.group(1)),
                    )
                )

        # Log head information
        self.logger.info(f"Selected {len(cfact_heads)} counterfactual heads:")
        self.logger.info(f"Cfact heads: {cfact_heads}")
        self.logger.info(f"Selected {len(fact_heads)} factual heads:")
        self.logger.info(f"Fact heads: {fact_heads}")

        # Store heads in instance variables and return them
        self.cfact_heads = cfact_heads
        self.fact_heads = fact_heads
        del cache
        if return_df:
            return df
        return cfact_heads, fact_heads

    def logit_lens_attn_and_mlp(self, return_df: bool = True):
        logit_lens = LogitLens.from_model(self.model)
        cache = self.model.extract_cache(
            [d["text_image_inputs"] for d in self.dataloader],
            target_token_positions=["last"],
            extraction_config=ExtractionConfig(
                extract_resid_out=True,
                extract_mlp_out=True,
                extract_attn_out=True,
                extract_head_out=True,
                extract_last_layernorm=True,
            ),
        )
        token_pair = self.token_pair
        token_directions = []
        for t0, t1 in token_pair:
            if hasattr(self.text_tokenizer, "tokenizer"):
                # Handle processor with tokenizer attribute
                result0 = self.text_tokenizer.tokenizer(t0, return_tensors="pt")
                result1 = self.text_tokenizer.tokenizer(t1, return_tensors="pt")
            elif hasattr(self.text_tokenizer, "__call__"):
                # Handle direct tokenizer
                result0 = self.text_tokenizer(t0, return_tensors="pt")
                result1 = self.text_tokenizer(t1, return_tensors="pt")
            else:
                raise ValueError("Invalid text tokenizer configuration")

            token_directions.append(
                (result0["input_ids"][0][1], result1["input_ids"][0][1])
            )

        # Analyze head contributions using the correct token pairs
        self.logger.info("Computing head contributions...")
        out_attn = logit_lens.compute(
            activations=cache,
            target_key="attn_out_{i}",
            token_directions=token_directions,
            metric="accuracy",
        )
        out_mlp = logit_lens.compute(
            activations=cache,
            target_key="mlp_out_{i}",
            token_directions=token_directions,
            metric="accuracy",
        )

        out_attn = {
            k: float((v.mean().detach().cpu() - 0.5)) for k, v in out_attn.items()
        }
        out_mlp = {
            k: float((v.mean().detach().cpu() - 0.5)) for k, v in out_mlp.items()
        }

        # Convert to dataframe for analysis
        df_attn = pd.DataFrame(out_attn.items(), columns=["Type", "Value"])
        df_mlp = pd.DataFrame(out_mlp.items(), columns=["Type", "Value"])

        return df_attn, df_mlp

    def get_baseline(self):
        """Get the baseline statistics"""
        assert self.baseline is not None, (
            "Baseline must be computed before accessing it"
        )
        return self.baseline

    def get_token_pair(self):
        return self.token_pair, self.baseline

    def return_generation_logits(self):
        """
        The return_generation_logits method computes the logits for the generation task. It uses the model to generate text based on the provided image and text inputs, and returns the logits for further analysis.
        """

        # first, create a dataloader for generation
        assert self.dataloader is not None, (
            "Dataloader must be initialized before generating logits"
        )

        logits = []
        generated_tokens = []
        generation_instruction = "Describe the image:"
        generation_dataloader = []
        for i, item in tqdm(enumerate(self.dataloader), desc="Generating logits", total=len(self.dataloader)):
            generation_text = self.config.prompt_templates[
                self.config.model_name + "-instruction"
            ].format(text=generation_instruction)
            generation_inputs = self.tokenizer(  # type: ignore
                text=generation_text,
                images=[item["image"].convert("RGB")],
                return_tensors="pt",
            )
            generation_output = self.model.generate(
                # inputs=self.dataloader[i]["text_image_inputs"],
                inputs=generation_inputs,
                target_token_positions=["last"],
                generation_config=GenerationConfig(
                    max_new_tokens=50,  # Adjust as needed
                    do_sample=False,  # Set to True for sampling
                    output_logits=True,  # Ensure logits are returned
                    return_dict_in_generate=True,  # Return logits in the output
                ),
                # extraction_config=ExtractionConfig(
                #     save_logits=True,
                # )
            )
            logits.append(torch.stack(generation_output["logits"]).squeeze())
            generated_tokens.append(generation_output["sequences"][:, -50:])

        max_len = max(l.shape[0] for l in logits)

        # Pad logits and tokens
        new_logits = []
        new_tokens = []
        removed_elements = []
        for i in range(len(logits)):
            l = logits[i]
            t = generated_tokens[i]

            len_diff = max_len - l.shape[0]

            if len_diff > 0:
                # skip that element if the length difference is greater than 0
                # logits.pop(i)
                # generated_tokens.pop(i)
                removed_elements.append(i)
            else:
                new_logits.append(l)
                new_tokens.append(t)
                


        # return torch.stack(padded_logits), torch.stack(padded_tokens)
        return torch.stack(new_logits), torch.stack(new_tokens), removed_elements

    def get_logprobs(self, logits, index):
        B, T, V = logits.shape

        index = index.squeeze(1)

        logits_flat = logits.view(-1, V)
        index_flat = index.view(-1)

        # Get the log probabilities for the specified indices
        log_probs_flat = selective_log_softmax(logits_flat, index_flat)

        logps = log_probs_flat.view(B, T)
        return logps

    def evaluate_generation_quality(self, base_generation_output):
        """
        The evaluate_generation_quality is a method to measure the KL divergence between the clean and intervened model outputs during generation. The model is tasked with a .generatio() task with a text like "Describe the image" and the model is expected to generate a description of the image.
        """
        self.model.use_full_model()
        base_logit, base_generated_tokens,base_removed_elements = base_generation_output
        intervened_logit, intervened_generated_tokens, intervened_removed_elements = self.return_generation_logits()

        if len(base_removed_elements) > 0:
            raise ValueError(
                f"Base generation output has removed elements: {base_removed_elements}. This should not happen."
            )
        if len(intervened_removed_elements) > 0:
            # remove from base_logit and base_generated_tokens the same elements
            self.logger.warning(
                f"Intervened generation output has removed elements: {intervened_removed_elements}. This should not happen."
            )
            base_logit = torch.stack(
                [
                    base_logit[i]
                    for i in range(len(base_logit))
                    if i not in intervened_removed_elements
                ]
            )
            base_generated_tokens = torch.stack(
                [
                    base_generated_tokens[i]
                    for i in range(len(base_generated_tokens))      
                    if i not in intervened_removed_elements
                ]
            )
            
        
        # sample some intervened tokens
        sample_intervened_tokens = intervened_generated_tokens[:10, :]
        # de-tokenize
        sample_intervened_text = self.text_tokenizer.batch_decode(
            sample_intervened_tokens.squeeze(), skip_special_tokens=True
        )

        for i, text in enumerate(sample_intervened_text):
            self.logger.info(f"--------- Sample {i} intervened text: {text}")
        # self.logger.info(
        #     f"Sample intervened tokens: {sample_intervened_tokens}, decoded text: {sample_intervened_text}"
        # )

        intervened_logprobs = self.get_logprobs(
            intervened_logit, intervened_generated_tokens
        )
        base_logprobs = self.get_logprobs(base_logit, intervened_generated_tokens)

        # per_token_kl = (
        #     torch.exp(intervened_logprobs - base_logprobs)
        #     - (intervened_logprobs - base_logprobs)
        #     - 1
        # )
        per_token_kl = (
            torch.exp(base_logprobs - intervened_logprobs)
            - (base_logprobs - intervened_logprobs)
            - 1
        )
        # make a mask for the completion tokens
        completion_mask = torch.ones_like(per_token_kl)

        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        print("Mean KL Divergence:", mean_kl.item())
        return {"mean_kl": mean_kl.item()}

    # def deprecated_run_baseline_stats(self, filename: Optional[str] = None) -> Dict[str, Any]:
    #     """Run baseline statistics on the dataset"""

    #     self.logger.info("Running baseline statistics...")

    #     token_pair, baseline_data = statistics_computer(
    #         model=self.model,
    #         dataloader=self.dataloader,
    #         write_to_file=False,
    #         filename=None,
    #         dataset_path=Path(""),
    #         given_token_pair=self.token_pair,
    #         # return_essential_data=True,
    #         compute=["image"],
    #     )
    #     assert token_pair == self.token_pair, "Token pair mismatch after statistics computation"
    #     return baseline_data
    def evaluate_coco(self):
        self.logger.info("Loading COCO dataset...")
        coco_dataset = load_dataset("yerevann/coco-karpathy", split="test")

        # Get 100 unique image ids
        unique_image_ids = list(set(coco_dataset["imgid"]))[:100]

        # Filter the dataset to only include those images
        subset_dataset = coco_dataset.filter(
            lambda example: example["imgid"] in unique_image_ids
        )

        self.logger.info("Generating captions...")
        results = []

        # Create a dictionary to hold unique images
        images_for_captioning = {}
        for item in subset_dataset:
            if item["imgid"] not in images_for_captioning:
                try:
                    with urllib.request.urlopen(item["url"]) as url_response:
                        image_data = url_response.read()
                    image = Image.open(io.BytesIO(image_data))
                    images_for_captioning[item["imgid"]] = image
                except Exception as e:
                    self.logger.warning(
                        f"Failed to download or open image for imgid {item['imgid']} from {item['url']}. Error: {e}. Skipping."
                    )

        # Filter the dataset to ensure it only contains successfully loaded images
        subset_dataset = subset_dataset.filter(
            lambda example: example["imgid"] in images_for_captioning
        )

        for image_id, image in tqdm(
            images_for_captioning.items(), desc="Generating captions"
        ):
            image = image.convert("RGB")

            generation_instruction = "A picture of"
            generation_text = self.config.prompt_templates[
                self.config.model_name + "-instruction"
            ].format(text=generation_instruction)

            inputs = self.text_tokenizer(
                text=generation_text,
                images=[image],
                return_tensors="pt",
            ).to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            generated_caption = self.text_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            results.append({"image_id": image_id, "caption": generated_caption})

        self.logger.info("Preparing ground truth annotations...")
        annotations = {
            "images": [],
            "annotations": [],
            "info": "dummy",
            "licenses": "dummy",
            "type": "captions",
        }

        img_ids = set()
        for i, item in enumerate(subset_dataset):
            if item["imgid"] not in img_ids:
                annotations["images"].append({"id": item["imgid"]})
                img_ids.add(item["imgid"])

            annotations["annotations"].append(
                {
                    "image_id": item["imgid"],
                    "id": item["sentences"][0]["sentid"],
                    "caption": item["sentences"][0]["raw"],
                }
            )

        self.logger.info("Evaluating with COCOeval...")

        annotation_file = "tmp_annotations.json"
        with open(annotation_file, "w") as f:
            json.dump(annotations, f)

        result_file = "tmp_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f)

        coco = COCO(annotation_file)
        coco_result = coco.loadRes(result_file)

        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

        return coco_eval.eval
