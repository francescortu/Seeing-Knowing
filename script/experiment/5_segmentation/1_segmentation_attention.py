import sys
import os
from pathlib import Path
import json
import pickle
import argparse
from datetime import datetime
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from scipy.stats import wilcoxon

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
sys.path.append(str(project_root))

try:
    from src.visual_attn import LlavaNextImageTokenVisualizer
    from transformers import LlavaNextProcessor
except ImportError:
    pass

from src.experiment_manager import ExperimentManager
from easyroutine.logger import logger, setup_logging
from easyroutine.interpretability import ExtractionConfig
from easyroutine.interpretability.activation_cache import ActivationCache, sublist

from dotenv import load_dotenv

dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

setup_logging(level="INFO")

sample_image = {  # name, original_index
    "pirate": 457,
    "zuck": 463,
    "liberty": 12,
    "lennon": 13,
    "liberty2": 63,
    "child_newspaper": 95,
    "zebra": 100,
    "iena": 119,
    "boat": 130,
    "lighthouse": 215,
    "childfire": 216,
    "train": 350,
    "hat": 409,
    "fruit": 2,
    "mask": 26,
    "macbook": 39,
    "fire": 49,
    "sand": 70,
    "penguin": 73,
    "pizza": 99,
    # "bill_gates": 157,
}

SEG_DIR = Path("data/data_whoops_segmented")


def load_segmented_image(image_name):
    """Load the segmented image to identify object patches."""
    # Try jpg and png
    img_path = SEG_DIR / f"{image_name}_sem.jpg"
    if not img_path.exists():
        img_path = SEG_DIR / f"{image_name}_sem.png"

    # if not img_path.exists():
    #     # Try looking in current directory just in case
    #     img_path = Path("data_whoops_segmented") / f"{image_name}_sem.jpg"
    #     if not img_path.exists():
    #         img_path = Path("data_whoops_segmented") / f"{image_name}_sem.png"

    #     if not img_path.exists():
    #         # Try looking in data directory
    #         img_path = (
    #             Path("../../../data/francescortu___whoops-aha")
    #             / f"{image_name}_sem.jpg"
    #         )
    #         if not img_path.exists():
    #             # Last resort, just warn and return None
    #             logger.warning(f"Segmented image for {image_name} not found")
    #             return None

    image = Image.open(img_path).convert("RGB")
    return image


def detect_non_white_patches_gemma(
    image, patch_size=16, image_size=256, white_threshold=250
):
    """
    Detect patches that contain non-white pixels for Gemma (fixed grid).
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] == 3:  # [C, H, W]
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()
    else:
        image_np = image

    # Ensure uint8
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:  # Normalized to [0, 1]
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

    # Resize to target size if needed
    if image_np.shape[0] != image_size or image_np.shape[1] != image_size:
        pil_img = Image.fromarray(image_np)
        pil_img = pil_img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        image_np = np.array(pil_img)

    # Calculate number of patches per side
    num_patches_per_side = image_size // patch_size
    num_patches = num_patches_per_side**2

    # Initialize result tensor
    non_white_mask = torch.zeros(num_patches, dtype=torch.float32)

    # Process each patch
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            # Extract patch with bounds checking
            row_start = i * patch_size
            col_start = j * patch_size
            row_end = min((i + 1) * patch_size, image_size)
            col_end = min((j + 1) * patch_size, image_size)

            patch = image_np[row_start:row_end, col_start:col_end]

            # Check if pixel is white (all RGB channels > threshold)
            is_white = (
                (patch[:, :, 0] > white_threshold)
                & (patch[:, :, 1] > white_threshold)
                & (patch[:, :, 2] > white_threshold)
            )

            # Count non-white pixels in this patch
            num_non_white = (~is_white).sum()
            total_pixels = patch.shape[0] * patch.shape[1]

            # If majority of pixels are non-white, mark patch as 1
            patch_idx = i * num_patches_per_side + j
            if num_non_white > (total_pixels / 2):
                non_white_mask[patch_idx] = 1.0

    return non_white_mask


def detect_non_white_patches_llava(
    segmented_image, original_image, processor, white_threshold=250
):
    """
    Detect which image patches (tokens) contain non-white pixels for LLaVA.
    Uses the new LlavaNextSegmentationMapper class for accurate token mapping.

    Parameters:
        segmented_image (PIL.Image): The segmented image with white background
        original_image (PIL.Image): The original image for proper preprocessing
        processor: The LlavaNextProcessor
        white_threshold (int): Threshold for white pixel detection

    Returns:
        torch.Tensor: Binary mask [num_tokens] with 1 for non-white patches
    """
    from src.visual_attn import LlavaNextSegmentationMapper

    mapper = LlavaNextSegmentationMapper(processor, white_threshold=white_threshold)
    object_patch_mask = mapper.get_token_mask(segmented_image, original_image)

    return object_patch_mask


def compute_attention_metrics(attention_vector, object_patch_mask):
    """
    Compute attention metrics on patch-level attention.
    """
    # Ensure same device
    object_patch_mask = object_patch_mask.to(attention_vector.device)

    # Total attention
    total_attention = attention_vector.sum()

    # Attention on object patches
    attention_on_object = (attention_vector * object_patch_mask).sum()

    # Percentage of attention on object
    pct_on_object = (attention_on_object / (total_attention + 1e-9)) * 100.0

    # Number of object patches
    num_object_patches = object_patch_mask.sum()
    num_total_patches = len(object_patch_mask)
    num_background_patches = num_total_patches - num_object_patches

    # Percentage of patches that are object
    pct_area = (num_object_patches / num_total_patches) * 100.0

    # Concentration Ratio: (% attention on object) / (% area of object)
    concentration_ratio = pct_on_object / (pct_area + 1e-9)

    # Average attention per patch
    avg_attn_on_object = attention_on_object / (num_object_patches + 1e-9)
    avg_attn_on_background = (total_attention - attention_on_object) / (
        num_background_patches + 1e-9
    )

    # Ratio of average attention (object vs background)
    avg_ratio = avg_attn_on_object / (avg_attn_on_background + 1e-9)

    return {
        "pct_on_object": pct_on_object.item(),
        "concentration_ratio": concentration_ratio.item(),
        "avg_ratio": avg_ratio.item(),
        "pct_area": pct_area.item(),
        "num_object_patches": num_object_patches.item(),
        "num_total_patches": num_total_patches,
    }


def compute_gradient_metrics(gradient_vector, object_patch_mask):
    """
    Compute gradient saliency metrics on patch-level gradients.
    Gradients are w.r.t. counterfactual token logit.

    Args:
        gradient_vector: Gradient magnitudes per patch [num_patches]
        object_patch_mask: Binary mask [num_patches] with 1 for object patches

    Returns:
        dict: Metrics including percentage, concentration ratio, and average ratio
    """
    # Ensure same device
    object_patch_mask = object_patch_mask.to(gradient_vector.device)

    # Total gradient magnitude
    total_gradient = gradient_vector.sum()

    # Gradient on object patches
    gradient_on_object = (gradient_vector * object_patch_mask).sum()

    # Percentage of gradient on object
    pct_on_object = (gradient_on_object / (total_gradient + 1e-9)) * 100.0

    # Number of object patches
    num_object_patches = object_patch_mask.sum()
    num_total_patches = len(object_patch_mask)
    num_background_patches = num_total_patches - num_object_patches

    # Percentage of patches that are object
    pct_area = (num_object_patches / num_total_patches) * 100.0

    # Concentration Ratio: (% gradient on object) / (% area of object)
    concentration_ratio = pct_on_object / (pct_area + 1e-9)

    # Average gradient per patch
    avg_grad_on_object = gradient_on_object / (num_object_patches + 1e-9)
    avg_grad_on_background = (total_gradient - gradient_on_object) / (
        num_background_patches + 1e-9
    )

    # Ratio of average gradient (object vs background)
    avg_ratio = avg_grad_on_object / (avg_grad_on_background + 1e-9)

    return {
        "grad_pct_on_object": pct_on_object.item(),
        "grad_concentration_ratio": concentration_ratio.item(),
        "grad_avg_ratio": avg_ratio.item(),
        "grad_pct_area": pct_area.item(),
        "grad_num_object_patches": num_object_patches.item(),
        "grad_num_total_patches": num_total_patches,
    }


def analyze_segmentation(
    manager,
    output_path,
    cfact_heads,
    random_heads,
    model_name,
    attention_cache,
    random_attention_cache,
    gradient_cache=None,
):
    logger.info("=" * 60)
    logger.info("SEGMENTATION ANALYSIS")
    logger.info("=" * 60)

    all_metrics_cfact = []
    all_metrics_random = []

    # Determine model type
    is_llava = "llava" in model_name.lower()
    is_gemma = "gemma" in model_name.lower()

    # Setup for LLaVA if needed
    processor = None
    if is_llava:
        try:
            # We need the processor. manager.tokenizer might be it, or we load it.
            # Assuming manager.tokenizer is the processor for LLaVA
            processor = manager.tokenizer
            # Or load it if needed:
            # processor = LlavaNextProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to setup LLaVA processor: {e}")
            return

    # Map filtered index to original index to find image name
    map_filtered_to_original = manager.map_filtered_to_original_index

    # Invert sample_image to find name by original index
    original_index_to_name = {v: k for k, v in sample_image.items()}

    # Determine number of samples
    num_samples = len(map_filtered_to_original)

    logger.info(f"Total samples to analyze: {num_samples}")
    logger.info(
        f"Sample image indices (first 5): {list(original_index_to_name.keys())[:5]}..."
    )

    hits = 0
    for i in range(num_samples):
        if i not in map_filtered_to_original:
            logger.warning(f"Index {i} not found in map_filtered_to_original")
            continue

        original_idx = map_filtered_to_original[i]

        if original_idx not in original_index_to_name:
            continue

        hits += 1
        image_name = original_index_to_name[original_idx]
        logger.info(f"Analyzing sample {i} (Original {original_idx}): {image_name}")

        try:
            # Load segmented image
            seg_image = load_segmented_image(image_name)
            if seg_image is None:
                continue

            # Get object mask
            if is_llava:
                # We need the original image for processor
                # manager.dataset is the HF dataset
                original_image = manager.dataset[original_idx]["image"].convert("RGB")
                object_patch_mask = detect_non_white_patches_llava(
                    seg_image, original_image, processor
                )
            else:
                # Gemma or others with fixed grid
                # Use config values if available, else defaults
                # Gemma 3 12B uses 336x336 input, patch size 14 -> 24x24 = 576 patches
                vision_input_size = 256  # Default for Gemma 3
                patch_size = 16
                object_patch_mask = detect_non_white_patches_gemma(
                    seg_image, patch_size=patch_size, image_size=vision_input_size
                )

            # Get attention vectors
            # attention_cache keys are "pattern_{layer}_{head}"
            # We need to aggregate them

            def get_avg_attention(heads, cache):
                avg_attn = None
                count = 0
                for layer, head in heads:
                    key = f"pattern_L{layer}H{head}"
                    if key in cache:
                        attn = cache[key][i]  # [num_patches]
                        if avg_attn is None:
                            avg_attn = attn
                        else:
                            avg_attn += attn
                        count += 1
                if count > 0:
                    return avg_attn / count
                return None

            attn_cfact = get_avg_attention(cfact_heads, attention_cache).squeeze()
            attn_random = get_avg_attention(
                random_heads, random_attention_cache
            ).squeeze()

            if attn_cfact is not None:
                # Ensure mask size matches attention size
                if len(object_patch_mask) != len(attn_cfact):
                    logger.warning(
                        f"Size mismatch: mask {len(object_patch_mask)}, attn {len(attn_cfact)}"
                    )
                    # Resize mask? Or just skip?
                    # For Gemma, if sizes differ, it might be due to special tokens or different resolution
                    # If simple mismatch, maybe truncate or pad?
                    if len(attn_cfact) == 576 and len(object_patch_mask) != 576:
                        # Recompute mask with correct size?
                        pass

                # If sizes match or we can handle it
                if len(object_patch_mask) == len(attn_cfact):
                    m_cfact = compute_attention_metrics(attn_cfact, object_patch_mask)
                    m_cfact["image"] = image_name

                    # Add gradient metrics if available
                    if gradient_cache is not None:
                        try:
                            # Get gradient for this sample
                            # gradient_cache["input_embeddings_gradients"][i] has shape [1, num_patches, embed_dim] or similar
                            grad_tensor = gradient_cache["input_embeddings_gradients"][
                                i
                            ]

                            # Compute L2 norm across embedding dimension to get magnitude per patch
                            if grad_tensor.ndim == 3:  # [1, num_patches, embed_dim]
                                grad_magnitude = torch.norm(
                                    grad_tensor, dim=-1
                                ).squeeze(0)  # [num_patches]
                            elif grad_tensor.ndim == 2:  # [num_patches, embed_dim]
                                grad_magnitude = torch.norm(
                                    grad_tensor, dim=-1
                                )  # [num_patches]
                            elif grad_tensor.ndim == 1:  # Already processed
                                grad_magnitude = grad_tensor
                            else:
                                raise ValueError(
                                    f"Unexpected gradient shape: {grad_tensor.shape}"
                                )

                            # Ensure same length as attention
                            if len(grad_magnitude) == len(attn_cfact):
                                grad_metrics = compute_gradient_metrics(
                                    grad_magnitude, object_patch_mask
                                )
                                m_cfact.update(grad_metrics)
                            else:
                                logger.warning(
                                    f"Gradient size {len(grad_magnitude)} != attention size {len(attn_cfact)}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error processing gradients for {image_name}: {e}"
                            )

                    all_metrics_cfact.append(m_cfact)

            if attn_random is not None and len(object_patch_mask) == len(attn_random):
                m_random = compute_attention_metrics(attn_random, object_patch_mask)
                m_random["image"] = image_name

                # Add same gradient metrics (gradients are independent of heads)
                if gradient_cache is not None:
                    try:
                        grad_tensor = gradient_cache["input_embeddings_gradients"][i]

                        if grad_tensor.ndim == 3:
                            grad_magnitude = torch.norm(grad_tensor, dim=-1).squeeze(0)
                        elif grad_tensor.ndim == 2:
                            grad_magnitude = torch.norm(grad_tensor, dim=-1)
                        elif grad_tensor.ndim == 1:
                            grad_magnitude = grad_tensor
                        else:
                            raise ValueError(
                                f"Unexpected gradient shape: {grad_tensor.shape}"
                            )

                        if len(grad_magnitude) == len(attn_random):
                            grad_metrics = compute_gradient_metrics(
                                grad_magnitude, object_patch_mask
                            )
                            m_random.update(grad_metrics)
                    except Exception as e:
                        logger.error(
                            f"Error processing gradients for random heads {image_name}: {e}"
                        )

                all_metrics_random.append(m_random)

        except Exception as e:
            logger.error(f"Error analyzing {image_name}: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    if all_metrics_cfact:
        df_cfact = pd.DataFrame(all_metrics_cfact)

        # Determine which columns to summarize (attention + gradients if available)
        summary_cols = ["pct_on_object", "concentration_ratio", "avg_ratio"]
        if "grad_pct_on_object" in df_cfact.columns:
            summary_cols.extend(
                ["grad_pct_on_object", "grad_concentration_ratio", "grad_avg_ratio"]
            )

        # Calculate summary statistics
        summary_cfact = df_cfact[summary_cols].describe()

        # Save summary to result.csv
        summary_cfact.to_csv(output_path / "result.csv")
        logger.info(f"Saved counterfactual summary to {output_path / 'result.csv'}")

        # Save detailed metrics to a separate file
        df_cfact.to_csv(output_path / "metrics_cfact_detailed.csv", index=False)
        logger.info(
            f"Saved detailed counterfactual metrics to {output_path / 'metrics_cfact_detailed.csv'}"
        )

        # Print summary
        logger.info("Counterfactual Heads Summary:")
        logger.info("\nATTENTION METRICS:")
        logger.info(
            summary_cfact[
                ["pct_on_object", "concentration_ratio", "avg_ratio"]
            ].to_string()
        )
        if "grad_pct_on_object" in df_cfact.columns:
            logger.info("\nGRADIENT METRICS (w.r.t. Counterfactual Token):")
            logger.info(
                summary_cfact[
                    ["grad_pct_on_object", "grad_concentration_ratio", "grad_avg_ratio"]
                ].to_string()
            )

    if all_metrics_random:
        df_random = pd.DataFrame(all_metrics_random)

        # Determine which columns to summarize
        summary_cols = ["pct_on_object", "concentration_ratio", "avg_ratio"]
        if "grad_pct_on_object" in df_random.columns:
            summary_cols.extend(
                ["grad_pct_on_object", "grad_concentration_ratio", "grad_avg_ratio"]
            )

        # Calculate summary statistics
        summary_random = df_random[summary_cols].describe()

        # Save summary to result_random.csv
        summary_random.to_csv(output_path / "result_random.csv")
        logger.info(f"Saved random summary to {output_path / 'result_random.csv'}")

        # Save detailed metrics to a separate file
        df_random.to_csv(output_path / "metrics_random_detailed.csv", index=False)
        logger.info(
            f"Saved detailed random metrics to {output_path / 'metrics_random_detailed.csv'}"
        )

        logger.info("Random Heads Summary:")
        logger.info("\nATTENTION METRICS:")
        logger.info(
            summary_random[
                ["pct_on_object", "concentration_ratio", "avg_ratio"]
            ].to_string()
        )
        if "grad_pct_on_object" in df_random.columns:
            logger.info("\nGRADIENT METRICS (same gradients, independent of heads):")
            logger.info(
                summary_random[
                    ["grad_pct_on_object", "grad_concentration_ratio", "grad_avg_ratio"]
                ].to_string()
            )

    # Perform statistical tests
    statistical_tests = {}

    # Test 1: Counterfactual heads vs Random heads (attention metrics)
    if all_metrics_cfact and all_metrics_random:
        logger.info("\n" + "=" * 60)
        logger.info("STATISTICAL TESTS - Wilcoxon Signed-Rank Test")
        logger.info("=" * 60)

        df_cfact = pd.DataFrame(all_metrics_cfact)
        df_random = pd.DataFrame(all_metrics_random)

        # Ensure both dataframes have the same images in the same order
        common_images = set(df_cfact["image"]).intersection(set(df_random["image"]))
        df_cfact_aligned = (
            df_cfact[df_cfact["image"].isin(common_images)]
            .sort_values("image")
            .reset_index(drop=True)
        )
        df_random_aligned = (
            df_random[df_random["image"].isin(common_images)]
            .sort_values("image")
            .reset_index(drop=True)
        )

        logger.info(f"\nTest 1: Counterfactual Heads vs Random Heads (Attention)")
        logger.info(f"Number of paired samples: {len(df_cfact_aligned)}")

        statistical_tests["cfact_vs_random_attention"] = {}

        for metric in ["pct_on_object", "concentration_ratio", "avg_ratio"]:
            if (
                metric in df_cfact_aligned.columns
                and metric in df_random_aligned.columns
            ):
                cfact_values = df_cfact_aligned[metric].values
                random_values = df_random_aligned[metric].values

                # Wilcoxon signed-rank test (paired samples)
                statistic, p_value = wilcoxon(
                    cfact_values, random_values, alternative="greater"
                )

                statistical_tests["cfact_vs_random_attention"][metric] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "cfact_mean": float(cfact_values.mean()),
                    "random_mean": float(random_values.mean()),
                    "significant": bool(p_value < 0.05),
                }

                logger.info(f"\n  {metric}:")
                logger.info(f"    Cfact mean:  {cfact_values.mean():.4f}")
                logger.info(f"    Random mean: {random_values.mean():.4f}")
                logger.info(f"    Statistic:   {statistic:.4f}")
                logger.info(
                    f"    p-value:     {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}"
                )

    # Test 2: Attention vs Gradients (for counterfactual heads)
    if all_metrics_cfact and "grad_pct_on_object" in df_cfact.columns:
        logger.info(f"\nTest 2: Attention vs Gradients (Counterfactual Heads)")
        logger.info(f"Number of samples: {len(df_cfact)}")

        statistical_tests["attention_vs_gradient"] = {}

        metric_pairs = [
            ("pct_on_object", "grad_pct_on_object"),
            ("concentration_ratio", "grad_concentration_ratio"),
            ("avg_ratio", "grad_avg_ratio"),
        ]

        for attn_metric, grad_metric in metric_pairs:
            if attn_metric in df_cfact.columns and grad_metric in df_cfact.columns:
                attn_values = df_cfact[attn_metric].values
                grad_values = df_cfact[grad_metric].values

                # Wilcoxon signed-rank test (paired samples)
                statistic, p_value = wilcoxon(attn_values, grad_values)

                statistical_tests["attention_vs_gradient"][attn_metric] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "attention_mean": float(attn_values.mean()),
                    "gradient_mean": float(grad_values.mean()),
                    "significant": bool(p_value < 0.05),
                }

                logger.info(f"\n  {attn_metric} vs {grad_metric}:")
                logger.info(f"    Attention mean: {attn_values.mean():.4f}")
                logger.info(f"    Gradient mean:  {grad_values.mean():.4f}")
                logger.info(f"    Statistic:      {statistic:.4f}")
                logger.info(
                    f"    p-value:        {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}"
                )

    # Save statistical tests to JSON
    if statistical_tests:
        stats_file = output_path / "statistical_tests.json"
        with open(stats_file, "w") as f:
            json.dump(statistical_tests, f, indent=2)
        logger.info(f"\n✓ Saved statistical tests to {stats_file}")

        logger.info("\n" + "=" * 60)


def extract_and_save_attention(
    model_name: str,
    experiment_tag: str,
    k_heads: int = 20,
    num_samples: int = 20,
    debug: bool = False,
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
    sanitized_model_name = model_name.replace("/", "_")
    base_path = (
        Path("results") / "5_segmentation" / experiment_tag / sanitized_model_name
    )
    output_path = base_path / "cache"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save files
    logger.info(f"Saving cache to {output_path}...")

    # Save attention cache (pickle for complex objects)
    cache_file = output_path / "cache.pkl"
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
        f.write("- cache.pkl: Attention patterns from all heads\n")
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
    random_attention_cache = {}
    random_heads = []
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
        logger.info(f"Saving random heads data to {output_path}...")

        # Save attention cache
        random_cache_data = {
            "attention_cache": random_attention_cache,
            "pattern_keys": random_pattern_keys,
        }

        random_cache_file = output_path / "random_cache.pkl"
        with open(random_cache_file, "wb") as f:
            pickle.dump(random_cache_data, f)
        logger.info(f"✓ Saved random attention cache to {random_cache_file}")

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

        random_config_file = output_path / "random_config.json"
        with open(random_config_file, "w") as f:
            json.dump(random_config_data, f, indent=2)
        logger.info(f"✓ Saved random configuration to {random_config_file}")

        # Update README
        with open(readme_file, "a") as f:
            f.write("\n")
            f.write("Random Heads Data:\n")
            f.write("- random_cache.pkl: Attention patterns from random heads\n")
            f.write("- random_config.json: Configuration for random heads\n")

        logger.info(f"✓ Updated README at {readme_file}")
        logger.info(f"✓ Random heads cache saved to: {output_path.absolute()}")
        logger.info(
            f"✓ Total size: {sum(f.stat().st_size for f in output_path.glob('*') if f.is_file()) / 1024 / 1024:.1f} MB"
        )

    logger.info("")
    logger.info("Next step: Run the notebook to analyze the cached data")
    logger.info("  jupyter notebook segmentation_analysis.ipynb")

    return output_path


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-12b-it",
        help="Model name or path",
    )
    argparser.add_argument(
        "--tag",
        type=str,
        default="seg_attention",
        help="Experiment tag",
    )
    argparser.add_argument(
        "--k_heads",
        type=int,
        default=20,
        help="Number of counterfactual heads to extract",
    )
    argparser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to process",
    )

    argparser.add_argument(
        "--extract_random",
        action="store_true",
        help="Whether to extract random heads as well",
    )
    argparser.add_argument(
        "--no_extract_random",
        action="store_false",
        dest="extract_random",
        help="Do not extract random heads",
    )
    argparser.set_defaults(extract_random=True)

    argparser.add_argument(
        "--extract_gradients",
        action="store_true",
        help="Whether to extract standard gradients",
    )
    argparser.add_argument(
        "--no_extract_gradients",
        action="store_false",
        dest="extract_gradients",
        help="Do not extract standard gradients",
    )
    argparser.set_defaults(extract_gradients=False)

    argparser.add_argument(
        "--extract_integrated_gradients",
        action="store_true",
        help="Whether to extract integrated gradients",
    )
    argparser.add_argument(
        "--no_extract_integrated_gradients",
        action="store_false",
        dest="extract_integrated_gradients",
        help="Do not extract integrated gradients",
    )
    argparser.set_defaults(extract_integrated_gradients=False)
    argparser.add_argument(
        "--ig_steps",
        type=int,
        default=50,
        help="Number of steps for integrated gradients",
    )
    args = argparser.parse_args()

    # Construct cache path
    sanitized_model_name = args.model.replace("/", "_")
    output_path = Path("results") / "5_segmentation" / args.tag / sanitized_model_name
    cache_path = output_path / "cache"

    # Check if cache exists
    if cache_path.exists() and (cache_path / "cache.pkl").exists():
        logger.info(
            f"Cache for model {args.model} and tag {args.tag} already exists at {cache_path}. Skipping extraction."
        )
    else:
        extract_and_save_attention(
            model_name=args.model,
            experiment_tag=args.tag,
            k_heads=args.k_heads,
            num_samples=args.num_samples,
            extract_random=args.extract_random,
            extract_gradients=args.extract_gradients,
            extract_integrated_gradients=args.extract_integrated_gradients,
            ig_steps=args.ig_steps,
        )

    # Load data for analysis
    logger.info("Loading data for analysis...")

    # Load config
    with open(cache_path / "config.json", "r") as f:
        config = json.load(f)

    cfact_heads = config["cfact_heads"]

    # Load random config if exists
    random_heads = []
    if (cache_path / "random_config.json").exists():
        with open(cache_path / "random_config.json", "r") as f:
            random_config = json.load(f)
            random_heads = random_config["random_heads"]

    # Load attention cache
    with open(cache_path / "cache.pkl", "rb") as f:
        cache_data = pickle.load(f)
        attention_cache = cache_data["attention_cache"]

    # Load random attention cache
    random_attention_cache = {}
    if (cache_path / "random_cache.pkl").exists():
        with open(cache_path / "random_cache.pkl", "rb") as f:
            random_cache_data = pickle.load(f)
            random_attention_cache = random_cache_data["attention_cache"]

    # Load gradient cache if it exists (gradients are independent of heads)
    gradient_cache = None
    if (cache_path / "gradient_cache.pkl").exists():
        logger.info("Loading gradient cache...")
        with open(cache_path / "gradient_cache.pkl", "rb") as f:
            gradient_cache = pickle.load(f)
        logger.info(
            "✓ Gradient cache loaded (will be used for both cfact and random heads)"
        )

    # Load manager data
    with open(cache_path / "manager_data.pkl", "rb") as f:
        manager_data = pickle.load(f)
        map_filtered_to_original_index = manager_data["map_filtered_to_original_index"]

    # Initialize manager to load dataset
    # We don't need to setup the model, just the dataset
    logger.info("Initializing manager for dataset loading...")
    manager = ExperimentManager.init(
        model_name=args.model,
        tag=args.tag,
    )
    manager.load_dataset_from_hf()

    # Load dataloader samples to populate manager.dataloader
    dataloader_samples_file = cache_path / "dataloader_samples.pkl"
    if dataloader_samples_file.exists():
        with open(dataloader_samples_file, "rb") as f:
            dataloader_samples = pickle.load(f)
            # Reconstruct a minimal dataloader structure
            manager.dataloader = []
            for s in dataloader_samples:
                # s is {'pixel_values': ...}
                manager.dataloader.append({"text_image_inputs": s})
            logger.info(
                f"Restored manager.dataloader with {len(manager.dataloader)} samples"
            )
    else:
        logger.warning(
            "dataloader_samples.pkl not found, manager.dataloader will be None"
        )

    # Restore index mapping
    manager.map_filtered_to_original_index = map_filtered_to_original_index

    # For LLaVA, we might need the tokenizer/processor
    if "llava" in args.model.lower():
        try:
            from transformers import LlavaNextProcessor

            logger.info(f"Loading processor for {args.model}...")
            manager.tokenizer = LlavaNextProcessor.from_pretrained(args.model)
        except Exception as e:
            logger.warning(f"Could not load LLaVA processor: {e}")

    # Run analysis
    analyze_segmentation(
        manager=manager,
        output_path=output_path,
        cfact_heads=cfact_heads,
        random_heads=random_heads,
        model_name=args.model,
        attention_cache=attention_cache,
        random_attention_cache=random_attention_cache,
        gradient_cache=gradient_cache,
    )


if __name__ == "__main__":
    main()
