"""
Preprocessing script: Extract attention patterns and save to disk.

Run this once to extract attention patterns from counterfactual heads.
The notebook can then load the cached results quickly.
"""

import sys
import os
from pathlib import Path
import json
import pickle
import argparse
from datetime import datetime
import random

sys.path.append(os.path.abspath("../../../src"))
sys.path.append(os.path.abspath("../../.."))

from src.experiment_manager import ExperimentManager
from easyroutine.logger import logger, setup_logging
from easyroutine.interpretability import ExtractionConfig
from easyroutine.interpretability.activation_cache import ActivationCache, sublist

from dotenv import load_dotenv

dotenv_path = Path("../../../.env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

setup_logging(level="INFO")


def extract_and_save_attention(
    model_name: str = "google/gemma-3-12b-it",
    experiment_tag: str = "seg_attention",
    k_heads: int = 20,
    num_samples: int = 20,
    debug: bool = False,
    output_dir: str = "cache",
    extract_random: bool = True,
    random_seed: int = 42,
    extract_gradients: bool = False,
    extract_integrated_gradients: bool = False,
    ig_steps: int = 50,
):
    """
    Extract attention patterns and save everything needed for analysis.

    Args:
        model_name: Model to use
        experiment_tag: Experiment tag
        k_heads: Number of counterfactual heads
        num_samples: Number of samples to process
        debug: Debug mode
        output_dir: Directory to save cache
        extract_random: Whether to also extract random heads
        random_seed: Random seed for head selection
        extract_gradients: Whether to extract standard gradients
        extract_integrated_gradients: Whether to extract integrated gradients
        ig_steps: Number of steps for integrated gradients computation
    """
    logger.info("=" * 60)
    logger.info("ATTENTION EXTRACTION - PREPROCESSING")
    logger.info("=" * 60)

    # Initialize experiment manager
    logger.info("Initializing experiment manager...")
    manager = ExperimentManager.init(
        model_name=model_name,
        tag=experiment_tag,
    )

    if debug:
        manager.config.debug.debug = True
        manager.config.debug.debug_samples = min(100, num_samples)

    # Setup model and dataset
    logger.info("Loading dataset...")
    manager.load_dataset_from_hf()

    logger.info("Setting up model...")
    manager.setup_model()

    logger.info("Setting up dataloader...")
    manager.setup_dataloader()
    manager.setup_model_specific_variables(filter_dataloader=True)

    logger.info(f"Dataset size: {len(manager.dataloader)}")
    num_samples = min(num_samples, len(manager.dataloader))
    logger.info(f"Processing {num_samples} samples")

    # Get counterfactual heads
    logger.info("Selecting counterfactual heads...")
    cfact_heads, fact_heads = manager.select_heads(k_heads=k_heads)
    logger.info(f"Selected {len(cfact_heads)} counterfactual heads")
    logger.info(f"Top 5 heads: {cfact_heads[:5]}")

    # Set vocabulary_index for gradient computation if needed
    # This selects gradients with respect to the COUNTERFACTUAL token logit
    # paired_token[0] is the counterfactual token (e.g., "left" when image shows "right")
    # This tells the model: "compute gradients that would INCREASE the counterfactual token probability"
    if extract_gradients or extract_integrated_gradients:
        logger.info("Setting vocabulary indices for gradient computation...")
        logger.info("Gradients will be computed w.r.t. COUNTERFACTUAL token logits")
        for i in range(num_samples):
            paired_token = manager.token_pair[i]
            # paired_token[0] = counterfactual, paired_token[1] = factual
            cfact_idx = manager.tokenizer(text=paired_token[0])["input_ids"][0][1]
            manager.dataloader[i]["text_image_inputs"]["vocabulary_index"] = cfact_idx

    # Extract attention patterns
    logger.info("Extracting attention patterns (this may take a few minutes)...")

    def pattern_agr(old, new):
        if old is None:
            return [new]
        return old + [new]

    attention_cache = manager.model.extract_cache(
        [d["text_image_inputs"] for d in manager.dataloader[:num_samples]],
        target_token_positions=["all-image"],
        extraction_config=ExtractionConfig(
            extract_resid_out=False,
            extract_attn_pattern=True,
            attn_pattern_avg="none",
            attn_pattern_row_positions=["last"],
            attn_heads=[{"layer": head[0], "head": head[1]} for head in cfact_heads],
            save_logits=False,
        ),
        register_aggregation=("pattern_", pattern_agr),
    )

    pattern_keys = [k for k in attention_cache.keys() if k.startswith("pattern_")]
    logger.info(f"Extracted attention patterns for {len(pattern_keys)} heads")

    # Extract gradients if requested
    gradient_cache = None
    integrated_gradient_cache = None

    if extract_gradients:
        logger.info("Extracting standard gradients w.r.t. counterfactual token...")
        gradient_cache = manager.model.extract_cache(
            [d["text_image_inputs"] for d in manager.dataloader[:num_samples]],
            target_token_positions=["all-image"],
            extraction_config=ExtractionConfig(
                extract_embed=True,
                keep_gradient=True,
                save_logits=False,
            ),
            register_aggregation=("input_embeddings_gradients", sublist),
        )
        logger.info("✓ Standard gradients extracted")

    if extract_integrated_gradients:
        logger.info(f"Extracting integrated gradients with {ig_steps} steps...")
        logger.warning(
            f"This will perform {ig_steps} × {num_samples} gradient computations!"
        )

        integrated_gradients_list = []
        token_dict = None

        # Process each sample
        for sample_idx in range(num_samples):
            accumulated_gradients = None
            original_image = manager.dataloader[sample_idx]["text_image_inputs"][
                "pixel_values"
            ].clone()

            # Compute gradients at ig_steps interpolated inputs
            for step in range(1, ig_steps + 1):
                alpha = step / ig_steps
                manager.dataloader[sample_idx]["text_image_inputs"]["pixel_values"] = (
                    original_image * alpha
                )

                step_cache = manager.model.extract_cache(
                    [manager.dataloader[sample_idx]["text_image_inputs"]],
                    target_token_positions=["all-image"],
                    extraction_config=ExtractionConfig(
                        extract_embed=True,
                        keep_gradient=True,
                        save_logits=False,
                    ),
                    register_aggregation=("input_embeddings_gradients", sublist),
                    use_tqdm=False,
                )

                step_gradient = step_cache["input_embeddings_gradients"][0]

                if accumulated_gradients is None:
                    accumulated_gradients = step_gradient
                else:
                    accumulated_gradients += step_gradient

                if token_dict is None:
                    token_dict = step_cache["token_dict"]

            # Restore original image
            manager.dataloader[sample_idx]["text_image_inputs"]["pixel_values"] = (
                original_image
            )

            # Average gradients
            avg_gradient = accumulated_gradients / ig_steps

            # Get actual embeddings
            final_cache = manager.model.extract_cache(
                [manager.dataloader[sample_idx]["text_image_inputs"]],
                target_token_positions=["all-image"],
                extraction_config=ExtractionConfig(
                    extract_embed=True,
                    keep_gradient=False,
                    save_logits=False,
                ),
                use_tqdm=False,
            )
            actual_embedding = final_cache["input_embeddings"][0]

            # Integrated gradient = (input - 0) × average_gradient
            integrated_grad = actual_embedding * avg_gradient
            integrated_gradients_list.append(integrated_grad)

        # Create result cache
        integrated_gradient_cache = ActivationCache()
        integrated_gradient_cache["input_embeddings_gradients"] = (
            integrated_gradients_list
        )
        integrated_gradient_cache["token_dict"] = token_dict
        logger.info("✓ Integrated gradients extracted")

    # Prepare data to save
    logger.info("Preparing data for export...")

    # Save attention cache
    cache_data = {
        "attention_cache": attention_cache,
        "pattern_keys": pattern_keys,
    }

    # Save manager state
    manager_data = {
        "map_filtered_to_original_index": manager.map_filtered_to_original_index,
        "num_samples": num_samples,
    }

    # Save dataloader samples (just the images, not the full inputs)
    dataloader_samples = []
    for i in range(num_samples):
        sample = manager.dataloader[i]
        # Save only what we need for visualization
        pixel_values = sample["text_image_inputs"]["pixel_values"].cpu()

        # Squeeze all batch/singleton dimensions to get [C, H, W]
        while pixel_values.ndim > 3:
            if pixel_values.shape[0] == 1:
                pixel_values = pixel_values.squeeze(0)
            else:
                logger.warning(f"Unexpected tensor shape: {pixel_values.shape}")
                break

        if pixel_values.ndim != 3:
            logger.error(
                f"Could not reduce to 3D tensor. Final shape: {pixel_values.shape}"
            )

        dataloader_samples.append(
            {
                "pixel_values": pixel_values,
            }
        )

    # Save configuration
    config_data = {
        "model_name": model_name,
        "experiment_tag": experiment_tag,
        "k_heads": k_heads,
        "num_samples": num_samples,
        "cfact_heads": cfact_heads,
        "fact_heads": fact_heads,
        "timestamp": datetime.now().isoformat(),
        "vision_input_size": 336,
        "patch_size": 14,
        "num_patches": 576,
        "extract_gradients": extract_gradients,
        "extract_integrated_gradients": extract_integrated_gradients,
        "ig_steps": ig_steps if extract_integrated_gradients else None,
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save files
    logger.info(f"Saving to {output_path}...")

    # Save attention cache (pickle for complex objects)
    cache_file = output_path / "attention_cache.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    logger.info(f"✓ Saved attention cache to {cache_file}")

    # Save gradient caches if extracted
    if gradient_cache is not None:
        gradient_file = output_path / "gradient_cache.pkl"
        with open(gradient_file, "wb") as f:
            pickle.dump(gradient_cache, f)
        logger.info(f"✓ Saved gradient cache to {gradient_file}")

    if integrated_gradient_cache is not None:
        integrated_gradient_file = output_path / "integrated_gradient_cache.pkl"
        with open(integrated_gradient_file, "wb") as f:
            pickle.dump(integrated_gradient_cache, f)
        logger.info(f"✓ Saved integrated gradient cache to {integrated_gradient_file}")

    # Save manager data
    manager_file = output_path / "manager_data.pkl"
    with open(manager_file, "wb") as f:
        pickle.dump(manager_data, f)
    logger.info(f"✓ Saved manager data to {manager_file}")

    # Save dataloader samples
    dataloader_file = output_path / "dataloader_samples.pkl"
    with open(dataloader_file, "wb") as f:
        pickle.dump(dataloader_samples, f)
    logger.info(f"✓ Saved dataloader samples to {dataloader_file}")

    # Save config (JSON for easy reading)
    config_file = output_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"✓ Saved configuration to {config_file}")

    # Save a README
    readme_file = output_path / "README.txt"
    with open(readme_file, "w") as f:
        f.write("Cached Attention Data\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {config_data['timestamp']}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Samples: {num_samples}\n")
        f.write(f"Counterfactual heads: {k_heads}\n")
        f.write(f"Gradients extracted: {extract_gradients}\n")
        f.write(f"Integrated gradients extracted: {extract_integrated_gradients}\n")
        if extract_integrated_gradients:
            f.write(f"IG steps: {ig_steps}\n")
        f.write("\n")
        f.write("Files:\n")
        f.write("- attention_cache.pkl: Attention patterns from all heads\n")
        f.write("- manager_data.pkl: Manager state (index mappings)\n")
        f.write("- dataloader_samples.pkl: Image pixel values\n")
        f.write("- config.json: Configuration and metadata\n")
        if extract_gradients:
            f.write(
                "- gradient_cache.pkl: Standard gradients w.r.t. counterfactual token\n"
            )
        if extract_integrated_gradients:
            f.write("- integrated_gradient_cache.pkl: Integrated gradients\n")
        f.write("\n")
        f.write("Use the notebook 'segmentation_analysis.ipynb' to load and analyze.\n")

    logger.info(f"✓ Saved README to {readme_file}")

    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Cache saved to: {output_path.absolute()}")
    logger.info(
        f"Total size: {sum(f.stat().st_size for f in output_path.glob('*') if f.is_file()) / 1024 / 1024:.1f} MB"
    )

    # Extract random heads if requested
    if extract_random:
        logger.info("")
        logger.info("=" * 60)
        logger.info("EXTRACTING RANDOM HEADS")
        logger.info("=" * 60)

        random.seed(random_seed)

        # Get all possible heads from the model
        num_layers = manager.model.model_config.num_hidden_layers
        num_heads_per_layer = manager.model.model_config.num_attention_heads
        all_heads = [
            (layer, head)
            for layer in range(num_layers)
            for head in range(num_heads_per_layer)
        ]

        # Select random heads (excluding counterfactual heads)
        cfact_heads_set = set(cfact_heads)
        available_heads = [h for h in all_heads if h not in cfact_heads_set]
        random_heads = random.sample(available_heads, k_heads)

        logger.info(f"Selected {len(random_heads)} random heads")
        logger.info(f"Top 5 random heads: {random_heads[:5]}")

        # Extract attention patterns for random heads
        logger.info("Extracting attention patterns for random heads...")

        random_attention_cache = manager.model.extract_cache(
            [d["text_image_inputs"] for d in manager.dataloader[:num_samples]],
            target_token_positions=["all-image"],
            extraction_config=ExtractionConfig(
                extract_resid_out=False,
                extract_attn_pattern=True,
                attn_pattern_avg="none",
                attn_pattern_row_positions=["last"],
                attn_heads=[
                    {"layer": head[0], "head": head[1]} for head in random_heads
                ],
                save_logits=False,
            ),
            register_aggregation=("pattern_", pattern_agr),
        )

        random_pattern_keys = [
            k for k in random_attention_cache.keys() if k.startswith("pattern_")
        ]
        logger.info(
            f"Extracted attention patterns for {len(random_pattern_keys)} random heads"
        )

        # Save random heads cache
        random_output_path = Path(output_dir + "_random")
        random_output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving random heads data to {random_output_path}...")

        # Save attention cache
        random_cache_data = {
            "attention_cache": random_attention_cache,
            "pattern_keys": random_pattern_keys,
        }

        random_cache_file = random_output_path / "attention_cache.pkl"
        with open(random_cache_file, "wb") as f:
            pickle.dump(random_cache_data, f)
        logger.info(f"✓ Saved random attention cache to {random_cache_file}")

        # Save manager data (reuse same one)
        random_manager_file = random_output_path / "manager_data.pkl"
        with open(random_manager_file, "wb") as f:
            pickle.dump(manager_data, f)
        logger.info(f"✓ Saved manager data to {random_manager_file}")

        # Save dataloader samples (reuse same one)
        random_dataloader_file = random_output_path / "dataloader_samples.pkl"
        with open(random_dataloader_file, "wb") as f:
            pickle.dump(dataloader_samples, f)
        logger.info(f"✓ Saved dataloader samples to {random_dataloader_file}")

        # Save config for random heads
        random_config_data = {
            "model_name": model_name,
            "experiment_tag": experiment_tag,
            "k_heads": k_heads,
            "num_samples": num_samples,
            "random_heads": random_heads,
            "random_seed": random_seed,
            "timestamp": datetime.now().isoformat(),
            "vision_input_size": 336,
            "patch_size": 14,
            "num_patches": 576,
        }

        random_config_file = random_output_path / "config.json"
        with open(random_config_file, "w") as f:
            json.dump(random_config_data, f, indent=2)
        logger.info(f"✓ Saved configuration to {random_config_file}")

        # Save README
        random_readme_file = random_output_path / "README.txt"
        with open(random_readme_file, "w") as f:
            f.write("Cached Attention Data (Random Heads)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {random_config_data['timestamp']}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Samples: {num_samples}\n")
            f.write(f"Random heads: {k_heads}\n")
            f.write(f"Random seed: {random_seed}\n\n")
            f.write("Files:\n")
            f.write("- attention_cache.pkl: Attention patterns from random heads\n")
            f.write("- manager_data.pkl: Manager state (index mappings)\n")
            f.write("- dataloader_samples.pkl: Image pixel values\n")
            f.write("- config.json: Configuration and metadata\n\n")
            f.write("Use for comparison with counterfactual heads.\n")

        logger.info(f"✓ Saved README to {random_readme_file}")
        logger.info(f"✓ Random heads cache saved to: {random_output_path.absolute()}")
        logger.info(
            f"✓ Total size: {sum(f.stat().st_size for f in random_output_path.glob('*') if f.is_file()) / 1024 / 1024:.1f} MB"
        )

    logger.info("")
    logger.info("Next step: Run the notebook to analyze the cached data")
    logger.info("  jupyter notebook segmentation_analysis.ipynb")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract attention patterns and cache for analysis"
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-3-12b-it", help="Model name"
    )
    parser.add_argument(
        "--tag", type=str, default="seg_attention", help="Experiment tag"
    )
    parser.add_argument(
        "--k_heads", type=int, default=20, help="Number of counterfactual heads"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of samples to process"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--output_dir", type=str, default="cache", help="Directory to save cache"
    )
    parser.add_argument(
        "--extract_random",
        action="store_true",
        default=True,
        help="Also extract random heads",
    )
    parser.add_argument(
        "--no_extract_random",
        action="store_false",
        dest="extract_random",
        help="Skip random heads extraction",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for head selection"
    )
    parser.add_argument(
        "--extract_gradients",
        action="store_true",
        help="Extract standard gradients w.r.t. counterfactual token",
    )
    parser.add_argument(
        "--extract_integrated_gradients",
        action="store_true",
        help="Extract integrated gradients",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=50,
        help="Number of steps for integrated gradients (default: 50)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extract_and_save_attention(
        model_name=args.model,
        experiment_tag=args.tag,
        k_heads=args.k_heads,
        num_samples=args.num_samples,
        debug=args.debug,
        output_dir=args.output_dir,
        extract_random=args.extract_random,
        random_seed=args.random_seed,
        extract_gradients=args.extract_gradients,
        extract_integrated_gradients=args.extract_integrated_gradients,
        ig_steps=args.ig_steps,
    )
