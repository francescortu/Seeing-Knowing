import sys
import os
import pandas as pd
import numpy as np
import json
import datetime
import argparse
import copy  # Added for deep copying configs
from pathlib import Path
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from easyroutine.interpretability import Intervention
from src.datastatistics import statistics_computer
from src.experiment_manager import ExperimentManager, BaseConfig
from easyroutine.logger import logger, setup_logging

setup_logging(level="INFO")  # Set up logging for the experiment


@dataclass
class FullExperimentConfig(BaseConfig):
    """Configuration for full experiment with gamma-lambda interventions"""

    gamma_values: List[float] = field(
        default_factory=lambda: list(np.linspace(-3, 3, 13))
    )
    lambda_values: List[float] = field(
        default_factory=lambda: [0, 0.5, 1, 1.5, 2, 2.5, 3]
    )
    paired_gamma_lambda: List[Tuple[float, float]] = field(default_factory=list)
    use_paired_values: bool = False
    ablation_types: List[str] = field(
        default_factory=lambda: ["last-row", "last-row-img", "last-row-text", "full"]
    )
    result_filename: Optional[str] = None
    experiment_description: Optional[str] = None
    k_heads_values: List[int] = field(default_factory=list)
    rebalanced_weight: bool = True  # Add the new field with default True
    control: bool = False  # Add the control flag

    def __post_init__(self):
        if self.model_name not in self.prompt_templates:
            raise ValueError(f"No prompt template found for model {self.model_name}")

        if self.debug:
            # Reduce experiment scope in debug mode
            self.gamma_values = [-1, 0, 1]
            self.lambda_values = [0, 1]
            self.paired_gamma_lambda = [(-1, 1), (0, 0)]
            self.ablation_types = ["last-row", "last-row-img"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            # "tag": self.tag,
            "experiment_tag": self.experiment_tag,
            "extra_metadata": self.extra_metadata,
            "debug": self.debug,
            "debug_samples": self.debug_samples,
            "prompt_templates": self.prompt_templates,
            "gamma_values": self.gamma_values,
            "lambda_values": self.lambda_values,
            "paired_gamma_lambda": self.paired_gamma_lambda,
            "use_paired_values": self.use_paired_values,
            "k_heads_values": self.k_heads_values,
            "ablation_types": self.ablation_types,
            "result_filename": self.result_filename,
            "experiment_description": self.experiment_description,
            "rebalanced_weight": self.rebalanced_weight,  # Added
            "control": self.control,  # Added
        }

    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, json_path: str) -> "FullExperimentConfig":
        """Load configuration from JSON file"""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class FullExperiment(ExperimentManager):
    """Class for running full ablation experiments with gamma-lambda interventions"""

    def __init__(self, config: FullExperimentConfig):
        super().__init__(config)
        self.config: FullExperimentConfig = config  # Type hint for better IDE support
        self.results_dir = Path("results/1_heads_ablation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Initialize heads as None, they will be set by select_heads
        self.cfact_heads: Optional[List[Tuple[int, int]]] = None
        self.fact_heads: Optional[List[Tuple[int, int]]] = None

    def _select_random_heads(
        self, num_heads: int = 50
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Select a specified number of random heads for cfact and fact interventions."""
        if not self.model:
            raise ValueError("Model must be initialized before selecting random heads")

        num_layers = self.model.model_config.num_hidden_layers
        heads_per_layer = self.model.model_config.num_attention_heads
        all_heads = [
            (layer, head)
            for layer in range(num_layers)
            for head in range(heads_per_layer)
        ]

        if len(all_heads) < num_heads * 2:
            raise ValueError(
                f"Cannot select {num_heads * 2} unique heads from a total of {len(all_heads)} heads."
            )

        # Shuffle all heads
        np.random.shuffle(all_heads)

        # Select num_heads for cfact and num_heads for fact
        cfact_heads = all_heads[:num_heads]
        fact_heads = all_heads[num_heads : num_heads * 2]

        self.logger.info(f"Selected {len(cfact_heads)} random counterfactual heads.")
        self.logger.info(f"Selected {len(fact_heads)} random factual heads.")

        return cfact_heads, fact_heads

    def get_multiplication_weights(
        self,
        cfact_heads: List[Tuple[int, int]],
        fact_heads: List[Tuple[int, int]],
        gamma: Optional[float] = None,
        lambda_param: Optional[float] = None,
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
            if self.config.rebalanced_weight:
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

    def get_paired_weights(
        self,
        cfact_heads: List[Tuple[int, int]],
        fact_heads: List[Tuple[int, int]],
        gamma: Optional[float] = None,
        lambda_param: Optional[float] = None,
    ) -> Tuple[List[Tuple[int, int, float, str]], List[Tuple[int, int, float, str]]]:
        """Calculate separate weights for image and text interventions in paired mode

        For image interventions: apply gamma to counterfactual heads (lambda=0)
        For text interventions: apply lambda to factual heads (gamma=0)

        Returns:
            Tuple containing two lists:
            - Image intervention weights (with gamma for cfact heads, lambda=0)
            - Text intervention weights (with lambda for fact heads, gamma=0)
        """
        if not cfact_heads or not fact_heads:
            raise ValueError("Both cfact_heads and fact_heads must be provided")

        # For image interventions: Apply gamma to counterfactual heads (lambda=0)
        image_weights = self.get_multiplication_weights(
            cfact_heads=cfact_heads,
            fact_heads=fact_heads,
            gamma=gamma,
            lambda_param=0.0,  # Don't apply lambda to factual heads for image tokens
        )

        # For text interventions: Apply lambda to factual heads (gamma=0)
        text_weights = self.get_multiplication_weights(
            cfact_heads=cfact_heads,
            fact_heads=fact_heads,
            gamma=0.0,  # Don't apply gamma to counterfactual heads for text tokens
            lambda_param=lambda_param,
        )

        return image_weights, text_weights

    def set_interventions(
        self, heads: List[Tuple[int, int, float, str]], ablation_type: str
    ):
        """Set up interventions based on ablation type"""
        if not self.model:
            raise ValueError("Model must be initialized before setting interventions")

        self.logger.debug(f"Setting up {ablation_type} interventions...")

        intervention_list = []
        if ablation_type == "last-row":
            intervention_list = [
                Intervention(
                    type="grid",
                    activation=f"pattern_L{layer}H{head}",
                    token_positions=(["last"], ["all"]),
                    patching_values="ablation",
                    multiplication_value=weight,
                )
                for layer, head, weight, _ in heads
            ]
        elif ablation_type == "full":
            intervention_list = [
                Intervention(
                    type="grid",
                    activation=f"pattern_L{layer}H{head}",
                    token_positions=(["all"], ["all"]),
                    patching_values="ablation",
                    multiplication_value=weight,
                )
                for layer, head, weight, _ in heads
            ]
        elif ablation_type == "last-row-img-presoftmax":
            intervention_list = [
                Intervention(
                    type="grid_pre_softmax",
                    activation=f"pattern_L{layer}H{head}",
                    token_positions=(["last"], ["all-image"]),
                    patching_values="ablation",
                    multiplication_value=weight,
                )
                for layer, head, weight, _ in heads
            ]
        elif ablation_type == "last-row-img":
            intervention_list = [
                Intervention(
                    type="grid",
                    activation=f"pattern_L{layer}H{head}",
                    token_positions=(["last"], ["all-image"]),
                    patching_values="ablation",
                    multiplication_value=weight,
                )
                for layer, head, weight, _ in heads
            ]
        elif ablation_type == "last-row-text-presoftmax":
            intervention_list = [
                Intervention(
                    type="grid_pre_softmax",
                    activation=f"pattern_L{layer}H{head}",
                    token_positions=(["last"], ["all-text"]),
                    patching_values="ablation",
                    multiplication_value=weight,
                )
                for layer, head, weight, _ in heads
            ]
        elif ablation_type == "last-row-text":
            intervention_list = [
                Intervention(
                    type="grid_pre_softmax",
                    activation=f"pattern_L{layer}H{head}",
                    token_positions=(["last"], ["all-text"]),
                    patching_values="ablation",
                    multiplication_value=weight,
                )
                for layer, head, weight, _ in heads
            ]
        elif ablation_type == "last-row-paired":
            # For last-row-paired, we expect a sorted list where:
            # - First half contains image weights (for all heads)
            # - Second half contains text weights (for all heads)
            # This ensures we get two complete sets of weights

            # Split the weights into image and text groups
            total_heads = len(heads)
            mid_point = total_heads // 2

            # First half for image interventions
            image_heads = heads[:mid_point]
            intervention_list.extend(
                [
                    Intervention(
                        type="grid",
                        activation=f"pattern_L{layer}H{head}",
                        token_positions=(["last"], ["all-image"]),
                        patching_values="ablation",
                        multiplication_value=weight,
                    )
                    for layer, head, weight, _ in image_heads
                ]
            )

            # Second half for text interventions
            text_heads = heads[mid_point:]
            intervention_list.extend(
                [
                    Intervention(
                        type="grid",
                        activation=f"pattern_L{layer}H{head}",
                        token_positions=(["last"], ["all-text"]),
                        patching_values="ablation",
                        multiplication_value=weight,
                    )
                    for layer, head, weight, _ in text_heads
                ]
            )
            
        elif ablation_type == "last-row-cfact-only":
            intervention_list = [
                Intervention(
                    type="grid",
                    activation=f"pattern_L{layer}H{head}",
                    token_positions=(["last"], ["all"]),
                    patching_values="ablation",
                    multiplication_value=weight,
                )
                for layer, head, weight, type in heads if type == "cfact"
            ]
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

        self.model.register_interventions(interventions=intervention_list)

    def evaluate_heads(self, gamma, lambda_val, cfact_heads, fact_heads):
        """
        Evaluate the effect of intervening on specified heads with given gamma and lambda parameters

        Args:
            gamma: Weight multiplier for counterfactual heads
            lambda_val: Weight multiplier for factual heads
            cfact_heads: List of counterfactual heads as (layer, head) tuples
            fact_heads: List of factual heads as (layer, head) tuples

        Returns:
            List of dictionaries containing results for each ablation type
        """
        if not self.model or not self.dataloader:
            raise ValueError(
                "Model and dataloader must be initialized before evaluating heads"
            )

        # Add the token pairs if not already there
        if not hasattr(self, "token_pair") or self.token_pair is None:
            self.token_pair = self.get_token_pair()
            self.logger.info(f"Got {len(self.token_pair)} token pairs for evaluation")

        results = []

        # For each ablation type, run evaluation
        for ablation_type in self.config.ablation_types:
            self.logger.info(
                f"Evaluating with {ablation_type} ablation, gamma={gamma}, lambda={lambda_val}"
            )

            # Clean previous interventions
            self.model.clean_interventions()

            # Special handling for paired ablation
            if ablation_type == "last-row-paired":
                # Get separate weights for image and text interventions
                image_weights, text_weights = self.get_paired_weights(
                    cfact_heads,
                    fact_heads,
                    gamma=gamma,
                    lambda_param=lambda_val,
                )

                # Combine weights for intervention
                combined_weights = image_weights + text_weights
                self.set_interventions(combined_weights, ablation_type)
            else:
                # Standard single intervention set
                weights = self.get_multiplication_weights(
                    cfact_heads,
                    fact_heads,
                    gamma=gamma,
                    lambda_param=lambda_val,
                )
                self.set_interventions(weights, ablation_type)

            # Run evaluation with statistics_computer
            data = statistics_computer(
                model=self.model,
                dataloader=self.dataloader,
                write_to_file=False,
                filename=None,
                dataset_path=Path(""),
                given_token_pair=self.token_pair,
                return_essential_data=True,
            )

            # Add metadata to results
            result = {"AblationType": ablation_type, **data}
            results.append(result)

        return results

    def run_ablation_experiment(self) -> pd.DataFrame:
        """Run full ablation experiment with selected counterfactual and factual heads

        Returns:
            pd.DataFrame: Results dataframe with all metrics
        """
        # Determine base name and create output directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.config.result_filename:
            # Use the filename part if a path was provided, otherwise use the whole string
            base_name = Path(self.config.result_filename).stem
        else:
            base_name = f"full_experiment_{self.config.model_name.replace('/', '-')}_{timestamp}{self.config.experiment_tag}"

        # Ensure results_dir is a Path object if it's not already
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)

        output_dir = self.results_dir / base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths within the output directory
        csv_file_path = output_dir / f"{base_name}.csv"
        config_file_path = output_dir / "config.json"

        # Log where results and config will be saved
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Results CSV will be saved to {csv_file_path}")
        self.logger.info(f"Config JSON will be saved to {config_file_path}")
        self.logger.info(
            f"Experiment description: {self.config.experiment_description}"
        )

        # Save expanded config
        expanded_config = self.config.to_dict()
        expanded_config["run_timestamp"] = timestamp
        expanded_config["output_directory"] = str(output_dir)
        expanded_config["results_csv_path"] = str(csv_file_path)
        # Add selected heads if available
        if hasattr(self, "cfact_heads") and self.cfact_heads is not None:
            expanded_config["selected_cfact_heads"] = self.cfact_heads
        if hasattr(self, "fact_heads") and self.fact_heads is not None:
            expanded_config["selected_fact_heads"] = self.fact_heads

        with open(config_file_path, "w") as f:
            json.dump(expanded_config, f, indent=2)
        self.logger.info(f"Expanded config saved to {config_file_path}")

        # Run the experiment
        self.logger.info("Running ablation experiment...")

        # Add the token pairs if not already there
        if not hasattr(self, "token_pair") or self.token_pair is None:
            self.token_pair = self.get_token_pair()
            self.logger.info(f"Got {len(self.token_pair)} token pairs for evaluation")

        # Create empty dataframe for results
        result_df = pd.DataFrame()

        # First, run baseline measurements for each ablation type (gamma=0, lambda=0)
        console = Console(stderr=True)
        console.print(
            "[bold blue]Running baseline measurements (gamma=0, lambda=0)[/bold blue]"
        )

        for ablation_type in self.config.ablation_types:
            console.print(f"[cyan]Running baseline for {ablation_type}[/cyan]")

            # Clean previous interventions
            self.model.clean_interventions()

            # Get weights for baseline (no intervention)
            weights = self.get_multiplication_weights(
                self.cfact_heads,
                self.fact_heads,
                gamma=0,
                lambda_param=0,
            )
            self.set_interventions(weights, ablation_type)

            # Run evaluation
            data = statistics_computer(
                model=self.model,
                dataloader=self.dataloader,
                write_to_file=False,
                filename=None,
                dataset_path=Path(""),
                given_token_pair=self.token_pair,
                return_essential_data=True,
            )

            # Create baseline row
            baseline_row = pd.DataFrame(
                [
                    {
                        "ExperimentDesc": "Baseline (no intervention)",
                        "gamma": 0,
                        "lambda": 0,
                        "AblationType": ablation_type,
                        **data,
                    }
                ]
            )

            # Append baseline results
            result_df = pd.concat([result_df, baseline_row], ignore_index=True)

            # Save immediately
            result_df.to_csv(csv_file_path, index=False)  # Use new path

        # Calculate total number of experiments to run (excluding baselines)
        total_experiments = len(self.config.ablation_types)
        if self.config.use_paired_values:
            total_experiments *= len(self.config.paired_gamma_lambda)
        else:
            total_experiments *= (
                len(self.config.gamma_values) * len(self.config.lambda_values) - 1
            )  # Subtract 1 because (0,0) is already done

        # Rest of the experiments
        # Display experiment information in a panel
        console.print(
            Panel(
                f"[bold blue]Running {total_experiments} experiments[/bold blue]\n"
                f"[green]Model:[/green] {self.config.model_name}\n"
                f"[green]Dataset:[/green] {self.config.dataset_name}\n"
                f"[green]Description:[/green] {self.config.experiment_description}",
                title="Experiment Configuration",
                expand=False,
            )
        )

        # Create counters for tracking progress
        experiment_count = 0
        start_time = datetime.datetime.now()

        # Run for each ablation type
        for ablation_type in self.config.ablation_types:
            # Display current ablation type
            console.print(
                f"\n[bold cyan]Starting ablation type: {ablation_type}[/bold cyan]"
            )

            # Get number of parameter combinations for this ablation type
            if self.config.use_paired_values:
                params_count = len(self.config.paired_gamma_lambda)
                param_list = self.config.paired_gamma_lambda
            else:
                params_count = len(self.config.gamma_values) * len(
                    self.config.lambda_values
                )
                param_list = [
                    (g, l)
                    for g in self.config.gamma_values
                    for l in self.config.lambda_values
                    if not (
                        g == 0 and l == 0
                    )  # Skip baseline case as it's already done
                ]

            # Handle paired or unpaired parameter combinations
            for i, (gamma, lambda_val) in enumerate(param_list):
                # Calculate time estimation if we have at least one experiment completed
                if experiment_count > 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    avg_time_per_exp = elapsed_time / experiment_count
                    remaining_exps = total_experiments - experiment_count
                    est_time_remaining = avg_time_per_exp * remaining_exps

                    # Format time remaining for display
                    hours, remainder = divmod(est_time_remaining.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"[magenta]ETA: {int(hours)}h {int(minutes)}m {int(seconds)}s[/magenta]"
                else:
                    time_str = "[magenta]ETA: calculating...[/magenta]"

                # Display current parameter combination with time estimate
                console.print(
                    f"[green]Progress: {experiment_count}/{total_experiments} - "
                    f"{time_str} - "
                    f"Running {ablation_type} with γ={gamma:.2f}, λ={lambda_val:.2f} "
                    f"({i + 1}/{params_count})[/green]"
                )

                # Clean previous interventions
                self.model.clean_interventions()

                # Special handling for paired ablation
                if ablation_type == "last-row-paired":
                    # Get separate weights for image and text interventions
                    image_weights, text_weights = self.get_paired_weights(
                        self.cfact_heads,
                        self.fact_heads,
                        gamma=gamma,
                        lambda_param=lambda_val,
                    )

                    # Combine weights for intervention
                    combined_weights = image_weights + text_weights
                    self.set_interventions(combined_weights, ablation_type)
                else:
                    # Standard single intervention set
                    weights = self.get_multiplication_weights(
                        self.cfact_heads,
                        self.fact_heads,
                        gamma=gamma,
                        lambda_param=lambda_val,
                    )
                    self.set_interventions(weights, ablation_type)

                # Run evaluation with statistics_computer
                console.print("[cyan]Computing statistics...[/cyan]")
                data = statistics_computer(
                    model=self.model,
                    dataloader=self.dataloader,
                    write_to_file=False,
                    filename=None,
                    dataset_path=Path(""),
                    given_token_pair=self.token_pair,
                    return_essential_data=True,
                )

                # Create a single row DataFrame
                new_row = pd.DataFrame(
                    [
                        {
                            "ExperimentDesc": f"Gamma-Lambda Intervention",
                            "gamma": gamma,
                            "lambda": lambda_val,
                            "AblationType": ablation_type,
                            **data,
                        }
                    ]
                )

                result_df = pd.concat([result_df, new_row], ignore_index=True)

                # Save immediately after each experiment
                result_df.to_csv(csv_file_path, index=False)  # Use new path

                # Update progress tracking
                experiment_count += 1

                # Show key results and time estimation
                elapsed = datetime.datetime.now() - start_time
                hours, remainder = divmod(elapsed.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)

                console.print(
                    f"[yellow]Results: Image Cfact>Fact: {data['Image Cfact>Fact']:.2f}%, "
                    f"Text Cfact>Fact: {data['Text Cfact>Fact']:.2f}%[/yellow]"
                )
                console.print(
                    f"[blue]Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s - "
                    f"Avg. time per exp: {(elapsed.total_seconds() / experiment_count):.1f}s[/blue]"
                )

            # Print completion for ablation type
            console.print(
                f"[bold green]✓ Completed ablation type: {ablation_type}[/bold green]"
            )

        # Add k_heads value to the results if not already there
        if (
            "k_heads" not in result_df.columns
            and hasattr(self.config, "k_heads_values")
            and self.config.k_heads_values
        ):
            result_df["k_heads"] = self.config.k_heads_values[0]

        # Final save (although it's already been saved incrementally)
        result_df.to_csv(csv_file_path, index=False)  # Use new path

        # Display completion message
        console.print(
            Panel(
                f"[bold green]Experiment completed successfully![/bold green]\n"
                f"[yellow]Total experiments:[/yellow] {experiment_count}\n"
                f"[yellow]Results saved to:[/yellow] {csv_file_path}",  # Use new path
                title="Completion Summary",
                expand=False,
            )
        )

        self.logger.info(
            f"Completed all experiments. Final results saved to {csv_file_path}"  # Use new path
        )

        return result_df

    def _run_single_experiment(
        self, ablation, gamma, lambda_param, token_pair, result_df, file_name, pbar
    ):
        """Run a single experiment with given gamma and lambda values"""
        self.logger.info(
            f"Testing {ablation} with gamma={gamma}, lambda={lambda_param}"
        )

        # Clean previous interventions
        self.model.clean_interventions()

        # Special handling for last-row-paired ablation type
        if ablation == "last-row-paired":
            # Get separate weights for image and text interventions
            image_weights, text_weights = self.get_paired_weights(
                self.cfact_heads,
                self.fact_heads,
                gamma=gamma,
                lambda_param=lambda_param,
            )

            # Simply concatenate the two sets of weights
            # First all image weights, then all text weights
            # This matches the expectation in set_interventions for last-row-paired
            combined_weights = image_weights + text_weights

            # Set the interventions
            self.set_interventions(combined_weights, ablation)
        else:
            # Standard single intervention set for other ablation types
            heads = self.get_multiplication_weights(
                self.cfact_heads,
                self.fact_heads,
                gamma=gamma,
                lambda_param=lambda_param,
            )
            self.set_interventions(heads, ablation)

        # Run the evaluation
        data = statistics_computer(
            model=self.model,
            dataloader=self.dataloader,
            write_to_file=False,
            filename=None,
            dataset_path=Path(""),
            given_token_pair=token_pair,
            return_essential_data=True,
        )

        result_df_new = pd.concat(
            [
                result_df,
                pd.DataFrame(
                    [
                        {
                            "ExperimentDesc": "Gamma-Lambda Intervention",
                            "AblationType": ablation,
                            "Gamma": gamma,
                            "Lambda": lambda_param,
                            **data,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        # Save results after each iteration
        result_df_new.to_csv(file_name, index=False)
        pbar.update(1)

        return result_df_new


def run_multi_k_heads_experiment(
    config: FullExperimentConfig, k_heads_values: List[int], base_tag: str = "_multi_k"
):
    """
    Run parallel experiments for multiple k_heads values and merge the results

    This function runs the full ablation experiment for each k_heads value in parallel
    and then combines the results into a single master file. Each k_heads experiment
    will save its results and config into a dedicated subfolder within the batch directory.

    Args:
        config: Base configuration to use for all experiments
        k_heads_values: List of k_heads values to test
        base_tag: Base tag to append to experiment tags for each k_heads

    Returns:
        Path to the combined results file
    """
    logger.info(f"Running multi-k experiment with k_heads values: {k_heads_values}")

    # Create timestamp for this batch of experiments
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a base directory for this batch under results/1_heads_ablation
    results_base_dir = Path("results/1_heads_ablation")
    batch_dir = results_base_dir / f"multi_k_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Batch output directory: {batch_dir}")

    # Save a master config file for the entire multi-k run in the batch directory
    master_config_path = batch_dir / f"master_config_multi_k_{timestamp}.json"
    master_config_dict = config.to_dict()
    master_config_dict["k_heads_values_tested"] = (
        k_heads_values  # Add the list of k tested
    )
    master_config_dict["batch_run_timestamp"] = timestamp
    master_config_dict["batch_output_directory"] = str(batch_dir)
    with open(master_config_path, "w") as f:
        json.dump(master_config_dict, f, indent=2)
    logger.info(f"Master config for multi-k run saved to {master_config_path}")

    # Prepare base model and dataloader that will be reused
    base_experiment = FullExperiment(config)
    base_experiment.setup_model().create_dataloader().filter_dataloader()

    # Get token pairs from statistics_computer for reuse
    logger.info("Computing token pairs to be reused across experiments...")
    token_pair = base_experiment.get_token_pair()
    logger.info(f"Got {len(token_pair)} token pairs for evaluation")

    # Store paths to the individual experiment output directories
    experiment_output_dirs = []
    configs = []

    # Prepare configurations for each k_heads value
    for k_heads in k_heads_values:
        k_config = copy.deepcopy(config)
        k_config.k_heads_values = [k_heads]
        k_config.experiment_tag = f"{base_tag}_k{k_heads}"

        # Define the base name for this specific k_heads run's output folder/files
        k_base_name = f"experiment_k{k_heads}_{timestamp}"
        k_config.result_filename = k_base_name  # Pass base name, run_ablation_experiment will handle folder/file creation

        # Set description
        model_short = k_config.model_name.split("/")[-1]
        if k_config.use_paired_values:
            pairs_str = ", ".join(
                [f"({g},{l})" for g, l in k_config.paired_gamma_lambda[:3]]
            )
            if len(k_config.paired_gamma_lambda) > 3:
                pairs_str += f", ... ({len(k_config.paired_gamma_lambda)} pairs total)"
            k_config.experiment_description = f"Ablation (k_heads={k_heads}) with paired (gamma,lambda) values: {pairs_str}"
        else:
            k_config.experiment_description = f"Ablation (k_heads={k_heads}) on {model_short} with gamma={k_config.gamma_values[0]}-{k_config.gamma_values[-1]}, lambda={k_config.lambda_values[0]}-{k_config.lambda_values[-1]}"

        configs.append(k_config)
        # Store the expected output directory path for later aggregation
        experiment_output_dirs.append(batch_dir / k_base_name)

    # Run experiments for each k_heads
    results_dfs = []

    # For each k_heads, create and run a new experiment
    for i, k_heads in enumerate(k_heads_values):
        logger.info(
            f"Starting experiment {i + 1}/{len(k_heads_values)} with k_heads={k_heads}"
        )

        # Create experiment from config
        experiment = FullExperiment(configs[i])

        # Properly transfer all necessary properties from base experiment
        experiment.model = base_experiment.model
        experiment.dataloader = base_experiment.dataloader
        experiment.tokenizer = base_experiment.tokenizer
        experiment.text_tokenizer = base_experiment.text_tokenizer
        experiment.token_pair = token_pair  # Reuse token pair

        # Set the results_dir to the main batch_dir for this k_heads run
        # run_ablation_experiment will create a subfolder inside this batch_dir
        experiment.results_dir = batch_dir

        # Select heads for this k_heads value
        logger.info(f"Selecting heads for k_heads={k_heads}...")
        if experiment.config.control:
            logger.info(
                "Control mode enabled: Selecting 50 random cfact and 50 random fact heads."
            )
            experiment.cfact_heads, experiment.fact_heads = (
                experiment._select_random_heads(num_heads=50)
            )
        else:
            experiment.cfact_heads, experiment.fact_heads = (
                base_experiment.select_heads(k_heads=k_heads)
            )
        logger.info(
            f"Selected {len(experiment.cfact_heads)} counterfactual heads and {len(experiment.fact_heads)} factual heads"
        )

        # Run the experiment for this single k_heads value
        # This will create a folder like batch_dir/experiment_k{k_heads}_{timestamp}/
        # and save the CSV and config.json inside it.
        logger.info(f"Running experiment with k_heads={k_heads}...")
        result_df = experiment.run_ablation_experiment()

        # Ensure k_heads is in the results (should be added by run_ablation_experiment if needed)
        if "k_heads" not in result_df.columns:
            result_df["k_heads"] = k_heads

        results_dfs.append(result_df)
        logger.info(f"Completed experiment for k_heads={k_heads}")

    # Combine all results into a master file
    if not results_dfs:
        logger.warning("No results generated from any k_heads experiment.")
        return None

    master_df = pd.concat(results_dfs, ignore_index=True)

    # Create master CSV file path directly in batch_dir
    master_csv_file = batch_dir / f"multi_k_experiment_{timestamp}_all_k_heads.csv"
    master_df.to_csv(master_csv_file, index=False)
    logger.info(f"Combined results saved to {master_csv_file}")

    # Generate summary
    try:
        summary = master_df.groupby(["k_heads", "ExperimentDesc"])[
            "Image Cfact>Fact"
        ].agg(["mean", "std", "count"])
        logger.info("\nExperiment Summary by k_heads:")
        logger.info(f"\n{summary}")

        # Create summary file
        summary_file = batch_dir / f"multi_k_experiment_{timestamp}_summary.csv"
        summary.to_csv(summary_file)
        logger.info(f"Summary statistics saved to {summary_file}")
    except KeyError as e:
        logger.error(f"Could not generate summary statistics. Missing column: {e}")
        logger.error(f"Available columns: {master_df.columns.tolist()}")

    return str(master_csv_file)  # Return path to the master CSV


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full experiment with configurable settings"
    )
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset and parameters",
    )
    parser.add_argument("--tag", type=str, help="Custom tag for the experiment")
    parser.add_argument("--exp_desc", type=str, help="Custom description for the experiment", default="")
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument("--dataset", type=str, help="Dataset name to use")
    parser.add_argument("--control", action="store_true", default=False, help="Enable control mode")
    parser.add_argument("--not_rebalance_weight", action="store_false", default=True, help="Disable rebalancing of weights")
    parser.add_argument("--ablation_type", nargs="+", required=True, choices=["last-row", "last-row-img", "last-row-text", "full", "last-row-paired", "last-row-cfact-only"], help="Ablation type to run")
    parser.add_argument("--k_heads", type=int, default=20, help="Number of heads to select (default: 20)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    # Load config from file or create default
    if args.config:
        config = FullExperimentConfig.from_json(args.config)
    else:
        # Example configuration with both modes (cartesian product is default)
        config = FullExperimentConfig(
            model_name=args.model,  # Default model
            # model_name="google/gemma-3-12b-it",
            # tag="v5",
            experiment_tag=args.tag,
            experiment_description=args.exp_desc,
            debug=args.debug,
            rebalanced_weight=False,
            # Standard cartesian product mode (use_paired_values=False)
            gamma_values=[
                -4,
                -3.5,
                -3,
                -2,
                -2.5,
                -1,
                -1.5,
                0,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
            ],
            # lambda_values=[-3,-2,-1,0, 1, 2, 3],
            lambda_values=[
                -4,
                -3.5,
                -3,
                -2.5,
                -2,
                -1.5,
                -1,
                -0.5,
                0,
                0.5,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
            ],
            # Paired mode (set use_paired_values=True to use these)
            paired_gamma_lambda=[
                (-3, 3),
                (-2.5, 2.5),
                (-2, 2),
                (-1.5, 1.5),
                (-1, 1),
                (-0.5, 0.5),
                (0, 0),
                (0.5, -0.5),
                (1, -1),
                (1.5, -1.5),
                (2, -2),
                (2.5, -2.5),
                (3, -3)
            ],
            use_paired_values=True,  # Set to True to use paired_gamma_lambda instead of cartesian product
            ablation_types=args.ablation_type,
            control=args.control,
            k_heads_values=args.k_heads
        )


    if args.debug:
        config.experiment_tag += "_debug"  

    # Set experiment description if not already set
    if not config.experiment_description:
        model_short = config.model_name.split("/")[-1]
        if config.use_paired_values:
            pairs_str = ", ".join(
                [f"({g},{l})" for g, l in config.paired_gamma_lambda[:3]]
            )
            if len(config.paired_gamma_lambda) > 3:
                pairs_str += f", ... ({len(config.paired_gamma_lambda)} pairs total)"
            config.experiment_description = (
                f"Ablation with paired (gamma,lambda) values: {pairs_str}"
            )
        else:
            config.experiment_description = f"Ablation study on {model_short} with gamma={config.gamma_values[0]}-{config.gamma_values[-1]}, lambda={config.lambda_values[0]}-{config.lambda_values[-1]}"

    
    print(f"REBALANCE WEIGHT: {config.rebalanced_weight}")
    # Config saving is now handled within run_ablation_experiment
    # logger.info(f"Config saved to {config_filename}")
    logger.info(f"Base result filename pattern: {config.result_filename}")
    logger.info(f"Experiment description: {config.experiment_description}")

    # Run experiment
    experiment = FullExperiment(config)
    (experiment.setup_model().create_dataloader().filter_dataloader())

    # Select heads based on the control flag
    if config.control:
        logger.info(
            "Control mode enabled: Selecting 50 random cfact and 50 random fact heads."
        )
        experiment.cfact_heads, experiment.fact_heads = experiment._select_random_heads(
            num_heads=50
        )
    else:
        # Default k_heads value if not specified in config (e.g., for single run)
        k_heads_to_select = args.k_heads
        logger.info(
            f"Selecting top {k_heads_to_select} cfact and fact heads using LogitLens."
        )
        experiment.cfact_heads, experiment.fact_heads = experiment.select_heads(
            k_heads=k_heads_to_select
        )

    # Call select_heads and store the results
    # experiment.cfact_heads, experiment.fact_heads = experiment.select_heads(k_heads=15)
    results = experiment.run_ablation_experiment()

    # Log final summary
    experiment.logger.info("\nExperiment Summary:")
    summary = results.groupby("ExperimentDesc")["Image Cfact>Fact"].agg(
        ["mean", "std", "count"]
    )
    experiment.logger.info(f"\n{summary}")


def run_multi_k_main():
    """
    Entry point for running experiments across multiple k_heads values.

    This function provides a separate entry point from the standard main function,
    allowing you to run experiments for multiple k_heads values in one go.
    """
    parser = argparse.ArgumentParser(
        description="Run multiple k_heads experiments with configurable settings"
    )
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset and parameters",
    )
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument("--dataset", type=str, help="Dataset name to use")
    parser.add_argument(
        "--k_heads",
        type=str,
        default="10,5",
        help="Comma-separated list of k_heads values to test",
    )
    parser.add_argument(
        "--ablation_type",
        nargs="+",
        required=True,
        choices=[
            "last-row",
            "last-row-img",
            "last-row-text",
            "full",
            "last-row-paired",
            "last-row-cfact-only",
        ],
        help="Ablation type to run",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="_multi_k",
        help="Custom tag for the experiment",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="",
        help="Custom description for the experiment",
    )
    parser.add_argument(
        "--not_rebalance_weight",
        action="store_false",
        default=True,
        help="Disable rebalancing of weights",
    )

    args = parser.parse_args()
    # Parse k_heads values from comma-separated string
    k_heads_values = [int(k) for k in args.k_heads.split(",")]
    logger.info(f"Will run experiments for k_heads values: {k_heads_values}")

    # Load config from file or create default
    if args.config:
        config = FullExperimentConfig.from_json(args.config)
    else:
        # Default configuration using paired gamma-lambda values
        config = FullExperimentConfig(
            model_name=args.model,  # Default model
            # model_name="google/gemma-3-12b-it",
            # tag="v5",
            experiment_tag=args.tag,
            experiment_description=args.desc,
            rebalanced_weight=args.not_rebalance_weight,
            debug=False,
            # Standard cartesian product mode (use_paired_values=False)
            gamma_values=[
                -4,
                -3.5,
                -3,
                -2,
                -2.5,
                -1,
                -1.5,
                0,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
            ],
            # lambda_values=[-3,-2,-1,0, 1, 2, 3],
            lambda_values=[
                -4,
                -3.5,
                -3,
                -2.5,
                -2,
                -1.5,
                -1,
                -0.5,
                0,
                0.5,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
            ],
            # Paired mode (set use_paired_values=True to use these)
            paired_gamma_lambda=[
                (-3, 3),
                (-2.5, 2.5),
                (-2, 2),
                (-1.5, 1.5),
                (-1, 1),
                (-0.5, 0.5),
                (0, 0),
                (0.5, -0.5),
                (1, -1),
                (1.5, -1.5),
                (2, -2),
                (2.5, -2.5),
                (3, -3),
            ],
            use_paired_values=True,  # Set to True to use paired_gamma_lambda instead of cartesian product
            ablation_types=args.ablation_type,
        )

    # Override config with command line arguments
    if args.debug:
        config.debug = True
        # Use smaller set in debug mode
        k_heads_values = [5, 10]

    print(f"REBALANCE WEIGHT: {config.rebalanced_weight}")
    # Generate timestamp for this experiment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Save the master config file
    config_filename = f"configs/1_ablation_heads/multi_k_experiment_{timestamp}.json"
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    # Add k_heads_values to config for reference
    config_dict = config.to_dict()
    config_dict["k_heads_values"] = k_heads_values
    # Master config saving is now handled within run_multi_k_heads_experiment
    # with open(config_filename, "w") as f:
    #     json.dump(config_dict, f, indent=2)
    # logger.info(f"Master config saved to {config_filename}")

    # Run the multi-k experiment
    master_file = run_multi_k_heads_experiment(
        config=config, k_heads_values=k_heads_values, base_tag=f"_multi_k_{timestamp}"
    )

    logger.info(
        f"Multi-k experiment completed. Combined results saved to {master_file}"
    )

    return master_file


if __name__ == "__main__":
    import sys

    # Check for command line arguments to determine which main function to run
    if len(sys.argv) > 1 and sys.argv[1] == "--multi-k":
        # Remove the --multi-k flag from the arguments
        sys.argv.pop(1)
        run_multi_k_main()
    else:
        main()
