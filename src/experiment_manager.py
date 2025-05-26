from dataclasses import dataclass, field
import os
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


@dataclass
class BaseConfig:
    """Base configuration class for experiments"""

    model_name: str
    dataset_name: str
    # tag: str
    experiment_tag: str
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    debug: bool = False
    debug_samples: int = 10
    prompt_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "llava-hf/llava-v1.6-mistral-7b-hf": "[INST] <image>\n [/INST] {text}",
            "google/gemma-3-12b-it": "<bos><start_of_turn>user\n<start_of_image><end_of_turn>\n<start_of_turn>model\n{text}",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary"""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            # "tag": self.tag,
            "experiment_tag": self.experiment_tag,
            "extra_metadata": self.extra_metadata,
            "debug": self.debug,
            "debug_samples": self.debug_samples,
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
        self.cfact_heads: Optional[List[Tuple[int, int]]] = None
        self.fact_heads: Optional[List[Tuple[int, int]]] = None

        # Setup logging
        self.setup_logging()

        # baseline
        self.baseline = None
        self.token_pair = None

        # Output paths
        self.dataloader_dir = Path("data/manual")
        self.dataloader_dir.mkdir(parents=True, exist_ok=True)
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

    def get_model_and_dataloader(self):
        """Initialize model and dataloader"""
        self.setup_model()
        self.create_dataloader()
        return self.model, self.dataloader

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

    def setup_model(self, device_map="auto"):
        """Initialize model and tokenizers"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        self.model = HookedModel.from_pretrained(
            self.config.model_name, device_map=device_map
        )
        self.tokenizer = self.model.get_processor()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.logger.info("Model loaded successfully")

        return self
    
    def load_dataset(self):
        """Load the dataset from the dataset folder"""
        self.logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Define paths
        dataset_path = Path("dataset")
        csv_path = dataset_path / "whoops-aha!.csv"
        images_path = dataset_path / "images"

        # Load CSV data
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            self.logger.error(f"CSV file not found at {csv_path}")
            raise
        
        # Create a list to hold dataset items
        dataset_items = []
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading dataset items"):
            image_filename = row["image_filename"]
            image_file_path = images_path / image_filename
            
            if not image_file_path.exists():
                self.logger.warning(f"Image file not found: {image_file_path}, skipping this item.")
                continue

            try:
                # Load image using PIL
                from PIL import Image
                img = Image.open(image_file_path)
                
                # Ensure image is in RGB format if it's not (e.g. RGBA, P, etc.)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                dataset_item = {
                    "image": img,
                    "text": row["theme"],  # Assuming 'theme' is the text column
                    "image_id": image_filename,
                    # Explicitly get factual/counterfactual tokens, as they might be used later.
                    # Default to None if not present in the CSV.
                    "factual_tokens": row.get("factual_tokens"),
                    "counterfactual_tokens": row.get("counterfactual_tokens")
                }
                # Add all other columns from the CSV to the dataset_item,
                # skipping those already handled or special.
                for col in df.columns:
                    if col not in ["image_filename", "theme", "factual_tokens", "counterfactual_tokens"]:
                        dataset_item[col] = row[col]

                dataset_items.append(dataset_item)
            except Exception as e:
                self.logger.error(f"Error processing image {image_file_path}: {e}")
                continue
        
        if not dataset_items:
            self.logger.error("No items were successfully loaded into the dataset.")
            raise ValueError("Dataset could not be created as no items were loaded.")

        # Convert list of dicts to Hugging Face Dataset
        # We need to determine the features dynamically or define them explicitly
        # For simplicity, let's convert to a pandas DataFrame first, then to Dataset
        processed_df = pd.DataFrame(dataset_items)
        
        # Create Hugging Face Dataset from pandas DataFrame
        # This requires specifying features if there are complex types like PIL.Image.Image
        # For images, datasets library handles PIL Image objects directly.
        hf_dataset = Dataset.from_pandas(processed_df)
        
        self.logger.info(f"Dataset loaded successfully with {len(hf_dataset)} items.")
        return hf_dataset

    def create_dataloader(self, filename: Optional[Path] = None, filter: bool = True):
        """Create and save the dataloader with both image and text inputs"""
        if filename is not None and os.path.exists(self.dataloader_dir / filename):
            self.logger.info(f"Loading dataloader from {filename}")
            self.dataloader = torch.load(self.dataloader_dir / filename)
            self.logger.info(f"Dataloader loaded from {filename}")
            return self

        if not self.model or not self.tokenizer or not self.text_tokenizer:
            raise ValueError("Model must be initialized before creating dataloader")

        self.logger.info("Creating new dataloader...")
        dataset = self.load_dataset()
        # Cast to more specific type
        if isinstance(dataset, (Dataset, DatasetDict)):
            self.dataset = dataset
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        # Handle both Dataset and DatasetDict types
        train_data = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
        if not hasattr(train_data, "__len__"):
            train_data = list(train_data)
        self.logger.info(f"Dataset loaded. Train split length: {len(train_data)}")

        # In debug mode, take only the first few examples
        dataset_iter = train_data
        if self.config.debug:
            self.logger.info(
                f"Debug mode: using only first {self.config.debug_samples} examples"
            )
            dataset_iter = list(dataset_iter)[: self.config.debug_samples]

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

        if filter:
            self.logger.info("Filtering dataloader...")
            self.filter_dataloader()

        if filename is not None:
            self.logger.info(f"Saving dataloader to {filename}")
            torch.save(self.dataloader, self.dataloader_dir / filename)
            self.logger.info(f"Dataloader saved to {filename}")

        return self

    def filter_dataloader(self):
        if self.token_pair is None:
            self.get_token_pair()
        data = statistics_computer(
            model=self.model,
            dataloader=self.dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=self.token_pair,
            return_essential_data=True,
        )

        indexes_cfact_gt_fact_text = data["indexes_cfact_gt_fact_text"]
        self.logger.info(
            f"Filtering dataloader: {len(self.dataloader)} -> {len(self.dataloader) - len(indexes_cfact_gt_fact_text)}. Removing {len(indexes_cfact_gt_fact_text)} samples"
        )

        self.dataloader = [
            item
            for i, item in enumerate(self.dataloader)
            if i not in indexes_cfact_gt_fact_text
        ]
        self.token_pair = [
            item
            for i, item in enumerate(self.token_pair)
            if i not in indexes_cfact_gt_fact_text
        ]
        self.logger.info(f"Filtered dataloader length: {len(self.dataloader)}")
        self.logger.info(f"Index removed: {indexes_cfact_gt_fact_text}")
        for i, pair in enumerate(self.token_pair):
            self.logger.info(f"Token pair {i}: {pair}")
        self.logger.info("Dataloader filtered successfully")

    def select_heads(
        self, k_heads=20, return_df: bool = False, perc=0.02, sampling_mode="percentile", metric="accuracy"
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Use LogitLens to identify counterfactual and factual heads
        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: Lists of (layer, head) tuples for cfact and fact heads
        """
        if not self.model or not self.dataloader:
            raise ValueError(
                "Model and dataloader must be initialized before selecting heads"
            )

        self.logger.info("Starting head selection with LogitLens analysis...")
        logit_lens = LogitLens.from_model(self.model)

        # First get token pairs from statistics_computer
        self.logger.info("Computing statistics to get token pairs...")
        # _, _, _, token_pair = statistics_computer(
        #     model=self.model,
        #     dataloader=self.dataloader,
        #     write_to_file=False,
        #     filename=None,
        #     dataset_path=Path(""),
        #     return_essential_data=False,
        # )
        if self.token_pair is None:
            token_pair = self.get_token_pair()
            self.logger.info(f"Got {len(token_pair)} token pairs")
        else:
            token_pair = self.token_pair

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
        out = logit_lens.compute(
            activations=cache,
            target_key="head_out_L{i}H{j}",
            token_directions=token_directions,
            # metric="accuracy",
            metric=metric,
        )

        # Calculate mean impact of each head
        # mean_out = {
        #     k: float(2 * (v.mean().detach().cpu() - 0.5)) for k, v in out.items()
        # }
        # mean_out = {
        #     k: float((v.mean().detach().cpu() - 0.5)) for k, v in out.items()
        # }

        mean_out = {
            k: float((v.mean().detach().cpu() )) for k, v in out.items()
        }
        # Convert to dataframe for analysis
        df = pd.DataFrame(mean_out.items(), columns=["Head", "Value"])

        if sampling_mode=="fixed_number":

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
        self.token_pair = token_pair
        del cache
        if return_df:
            return df
        return cfact_heads, fact_heads


    def logit_lens_attn_and_mlp(
        self, return_df: bool = True
    ):
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
        
        
        

    def get_token_pair(self, filename: Optional[Path] = None) -> List[Tuple[str, str]]:
        if filename is not None and os.path.exists(self.dataloader_dir / filename):
            self.logger.info(f"Loading token pairs from {filename}")
            token_pair = torch.load(self.dataloader_dir / filename)
            self.logger.info(f"Token pairs loaded from {filename}")
            return token_pair

        if not self.model or not self.dataloader:
            raise ValueError(
                "Model and dataloader must be initialized before running baseline stats"
            )

        self.logger.info("Running baseline statistics...")
        _, _, _, token_pair = statistics_computer(
            model=self.model,
            dataloader=self.dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            return_essential_data=False,
        )
        self.token_pair = token_pair
        self.logger.info(f"Got {len(token_pair)} token pairs")

        if filename is not None:
            self.logger.info(f"Saving token pairs to {filename}")
            torch.save(token_pair, self.dataloader_dir / filename)
            self.logger.info(f"Token pairs saved to {filename}")

        return token_pair

    def run_baseline_stats(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Run baseline statistics on the dataset"""
        filename_token_pair = (
            filename.split(".")[0] + "_token_pair.pt" if filename else None
        )
        filename_baseline = (
            filename.split(".")[0] + "_baseline.pt" if filename else None
        )
        filename_token_pair = Path(filename_token_pair) if filename_token_pair else None
        filename_baseline = Path(filename_baseline) if filename_baseline else None

        if (
            filename_token_pair
            and filename_baseline
            and os.path.exists(self.dataloader_dir / filename_baseline)
            and os.path.exists(self.dataloader_dir / filename_token_pair)
        ):
            self.logger.info(f"Loading baseline stats from {filename_baseline}")
            baseline_data = torch.load(self.dataloader_dir / filename_baseline)
            self.token_pair = torch.load(self.dataloader_dir / filename_token_pair)
            self.logger.info(f"Baseline stats loaded from {filename_baseline}")
            return baseline_data
        if not self.model or not self.dataloader:
            raise ValueError(
                "Model and dataloader must be initialized before running baseline stats"
            )

        self.logger.info("Running baseline statistics...")
        if self.token_pair is None:
            token_pair = self.get_token_pair(filename=filename_token_pair)
            self.logger.info(f"Got {len(token_pair)} token pairs")
        else:
            token_pair = self.token_pair

        baseline_data = statistics_computer(
            model=self.model,
            dataloader=self.dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=token_pair,
            return_essential_data=True,
            compute=["image"],
        )
        if filename_baseline:
            self.logger.info(f"Saving baseline stats to {filename_baseline}")
            torch.save(baseline_data, self.dataloader_dir / filename_baseline)
            self.logger.info(f"Baseline stats saved to {filename_baseline}")

        return baseline_data

    def save(self, state_name: str) -> Path:
        """Save the current state of the experiment

        Args:
            state_name: A name for this saved state

        Returns:
            Path to the saved state file
        """
        self.logger.info(f"Saving experiment state: {state_name}")

        # Create timestamp for directory naming
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create the directory structure in tmp/experiment_stats/date_time
        save_dir = Path("tmp/experiment_stats") / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create filename based on state name and experiment tag
        filename = f"{self.config.experiment_tag}_{state_name}.pt"
        save_path = save_dir / filename

        # Create a state dictionary with all necessary information
        # Avoid saving the model - just save the name
        state = {
            "config": self.config.to_dict(),
            "dataloader": self.dataloader,
            "token_pair": self.token_pair,
            "cfact_heads": self.cfact_heads,
            "fact_heads": self.fact_heads,
            "baseline": self.baseline,
            "state_name": state_name,
            "timestamp": timestamp,
        }

        # Save the state using torch.save
        torch.save(state, save_path)
        self.logger.info(f"Experiment state saved to {save_path}")

        # Also save the configuration as a separate JSON file for easy inspection
        config_path = (
            save_dir / f"{self.config.experiment_tag}_{state_name}_config.json"
        )
        with open(config_path, "w") as f:
            json.dump(state["config"], f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}")

        return save_path

    @classmethod
    def load_state(
        cls, state_path: Union[str, Path], load_model: bool = False
    ) -> "ExperimentManager":
        """Load an experiment state from a saved file

        Args:
            state_path: Path to the saved state file
            load_model: Whether to also load the model (default: False)

        Returns:
            An ExperimentManager instance with the loaded state
        """
        # Ensure path is a Path object
        state_path = Path(state_path)

        # Load the state dictionary
        state = torch.load(state_path)

        # Try to load configuration from JSON if available
        json_config_path = state_path.parent / f"{state_path.stem}_config.json"
        config_data = state["config"]  # Default to state config
        if json_config_path.exists():
            try:
                with open(json_config_path, "r") as f:
                    config_data = json.load(f)
                    print(f"Loaded configuration from {json_config_path}")
            except Exception as e:
                print(f"Could not load config from JSON: {e}")

        # Create a new configuration from the loaded config data
        config = BaseConfig(
            model_name=config_data["model_name"],
            dataset_name=config_data["dataset_name"],
            experiment_tag=config_data["experiment_tag"],
        )

        # Set additional config parameters if they exist
        if "extra_metadata" in config_data:
            config.extra_metadata = config_data["extra_metadata"]
        if "debug" in config_data:
            config.debug = config_data["debug"]
        if "debug_samples" in config_data:
            config.debug_samples = config_data["debug_samples"]
        if "prompt_templates" in config_data:
            config.prompt_templates = config_data["prompt_templates"]

        # Create a new instance with the loaded config
        instance = cls(config)

        # Set up logging
        instance.setup_logging()
        instance.logger.info(f"Loading experiment state from {state_path}")

        # Restore saved state elements
        instance.dataloader = state["dataloader"]
        instance.token_pair = state["token_pair"]
        instance.cfact_heads = state["cfact_heads"]
        instance.fact_heads = state["fact_heads"]
        instance.baseline = state["baseline"]

        # Load model if requested
        if load_model:
            instance.logger.info(f"Loading model: {config.model_name}")
            instance.setup_model()

        instance.logger.info(
            f"Loaded experiment state: {state.get('state_name', 'unnamed')} "
            + f"from {state.get('timestamp', 'unknown time')}"
        )
        return instance
