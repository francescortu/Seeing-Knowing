"""
Helper utilities for segmentation-based attention analysis.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def patch_idx_to_coordinates(
    patch_idx: int,
    num_patches_per_side: int = 24,
    patch_size: int = 14
) -> Tuple[int, int, int, int]:
    """
    Convert patch index to image coordinates.
    
    Args:
        patch_idx: Index in flattened patch grid
        num_patches_per_side: Number of patches per side (default 24 for 336/14)
        patch_size: Size of each patch in pixels (default 14)
        
    Returns:
        Tuple of (x1, y1, x2, y2) coordinates
    """
    row = patch_idx // num_patches_per_side
    col = patch_idx % num_patches_per_side
    
    x1 = col * patch_size
    y1 = row * patch_size
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    
    return x1, y1, x2, y2


def coordinates_to_patch_idx(
    x: int, 
    y: int,
    num_patches_per_side: int = 24,
    patch_size: int = 14
) -> int:
    """
    Convert image coordinates to patch index.
    
    Args:
        x, y: Pixel coordinates
        num_patches_per_side: Number of patches per side
        patch_size: Size of each patch in pixels
        
    Returns:
        Patch index
    """
    col = x // patch_size
    row = y // patch_size
    return row * num_patches_per_side + col


def bounding_box_to_patches(
    x1: int, y1: int, x2: int, y2: int,
    num_patches_per_side: int = 24,
    patch_size: int = 14,
    overlap_threshold: float = 0.5
) -> List[int]:
    """
    Convert bounding box to list of overlapping patch indices.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        num_patches_per_side: Number of patches per side
        patch_size: Size of each patch in pixels
        overlap_threshold: Minimum overlap ratio to include patch
        
    Returns:
        List of patch indices
    """
    patches = []
    
    # Get range of patches that could overlap
    col_start = max(0, x1 // patch_size)
    col_end = min(num_patches_per_side - 1, x2 // patch_size)
    row_start = max(0, y1 // patch_size)
    row_end = min(num_patches_per_side - 1, y2 // patch_size)
    
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            # Calculate patch boundaries
            patch_x1 = col * patch_size
            patch_y1 = row * patch_size
            patch_x2 = patch_x1 + patch_size
            patch_y2 = patch_y1 + patch_size
            
            # Calculate intersection
            intersect_x1 = max(x1, patch_x1)
            intersect_y1 = max(y1, patch_y1)
            intersect_x2 = min(x2, patch_x2)
            intersect_y2 = min(y2, patch_y2)
            
            if intersect_x2 > intersect_x1 and intersect_y2 > intersect_y1:
                # Calculate overlap ratio
                intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
                patch_area = patch_size * patch_size
                overlap_ratio = intersect_area / patch_area
                
                if overlap_ratio >= overlap_threshold:
                    patch_idx = row * num_patches_per_side + col
                    patches.append(patch_idx)
    
    return patches


def mask_to_patches(
    mask: np.ndarray,
    target_size: int = 336,
    num_patches_per_side: int = 24,
    patch_size: int = 14,
    overlap_threshold: float = 0.5
) -> List[int]:
    """
    Convert binary mask to list of patch indices.
    
    Args:
        mask: Binary mask [H, W]
        target_size: Target image size (default 336)
        num_patches_per_side: Number of patches per side
        patch_size: Size of each patch
        overlap_threshold: Minimum overlap to include patch
        
    Returns:
        List of patch indices
    """
    from torchvision.transforms.functional import resize
    
    # Resize mask to target size
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    mask_resized = resize(mask_tensor, [target_size, target_size])
    mask_resized = mask_resized.squeeze().numpy()
    
    patches = []
    for row in range(num_patches_per_side):
        for col in range(num_patches_per_side):
            patch_mask = mask_resized[
                row * patch_size:(row + 1) * patch_size,
                col * patch_size:(col + 1) * patch_size
            ]
            
            if patch_mask.mean() >= overlap_threshold:
                patch_idx = row * num_patches_per_side + col
                patches.append(patch_idx)
    
    return patches


def visualize_patches(
    image: np.ndarray,
    patch_indices: List[int],
    num_patches_per_side: int = 24,
    patch_size: int = 14,
    figsize: Tuple[int, int] = (10, 10),
    title: str = "Highlighted Patches"
):
    """
    Visualize image with highlighted patches.
    
    Args:
        image: Image array [H, W, 3]
        patch_indices: List of patch indices to highlight
        num_patches_per_side: Number of patches per side
        patch_size: Size of each patch
        figsize: Figure size
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    
    for patch_idx in patch_indices:
        x1, y1, x2, y2 = patch_idx_to_coordinates(
            patch_idx, num_patches_per_side, patch_size
        )
        
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='cyan', facecolor='cyan', alpha=0.3
        )
        ax.add_patch(rect)
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def compute_baseline_random(
    num_cfact_patches: int,
    num_total_patches: int = 576,
    num_trials: int = 1000
) -> Dict[str, float]:
    """
    Compute baseline statistics if patches were selected randomly.
    
    Args:
        num_cfact_patches: Number of counterfactual patches
        num_total_patches: Total number of patches
        num_trials: Number of random trials
        
    Returns:
        Dictionary with mean, std, min, max of random proportions
    """
    proportions = []
    
    for _ in range(num_trials):
        # Randomly select patches
        random_patches = np.random.choice(
            num_total_patches, 
            size=num_cfact_patches, 
            replace=False
        )
        
        # Simulate uniform attention distribution
        attention = np.random.rand(num_total_patches)
        attention = attention / attention.sum()  # Normalize
        
        # Calculate proportion to random patches
        proportion = attention[random_patches].sum()
        proportions.append(proportion)
    
    return {
        "mean": float(np.mean(proportions)),
        "std": float(np.std(proportions)),
        "min": float(np.min(proportions)),
        "max": float(np.max(proportions)),
        "expected": num_cfact_patches / num_total_patches
    }


def get_top_attention_patches(
    attention_pattern: np.ndarray,
    k: int = 10
) -> List[Tuple[int, float]]:
    """
    Get top-k patches by attention weight.
    
    Args:
        attention_pattern: Attention weights for image patches [num_patches]
        k: Number of top patches to return
        
    Returns:
        List of (patch_idx, attention_weight) tuples
    """
    top_indices = np.argsort(attention_pattern)[-k:][::-1]
    return [(int(idx), float(attention_pattern[idx])) for idx in top_indices]


def compute_iou(patches_a: List[int], patches_b: List[int]) -> float:
    """
    Compute Intersection over Union between two sets of patches.
    
    Args:
        patches_a, patches_b: Lists of patch indices
        
    Returns:
        IoU score
    """
    set_a = set(patches_a)
    set_b = set(patches_b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def aggregate_attention_across_heads(
    attention_cache,
    sample_idx: int,
    pattern_keys: List[str],
    num_patches: int = 576
) -> np.ndarray:
    """
    Aggregate attention patterns across multiple heads.
    
    Args:
        attention_cache: Cache containing attention patterns
        sample_idx: Sample index
        pattern_keys: List of pattern keys (one per head)
        num_patches: Number of image patches
        
    Returns:
        Aggregated attention array [num_patches]
    """
    start_img = attention_cache["token_dict"][sample_idx]["all-image"][0]
    
    total_attention = np.zeros(num_patches)
    
    for key in pattern_keys:
        pattern = attention_cache[key][sample_idx][0, 0].cpu().numpy()
        img_attention = pattern[start_img:start_img + num_patches]
        total_attention += img_attention
    
    # Average across heads
    return total_attention / len(pattern_keys)


def create_annotation_dict_template(
    num_samples: int,
    output_path: str = "annotations.json"
):
    """
    Create a template JSON file for manual annotations.
    
    Args:
        num_samples: Number of samples to annotate
        output_path: Path to save template
    """
    import json
    
    template = {
        "metadata": {
            "description": "Manual annotations of counterfactual patches",
            "format": "Each key is a sample index, value is list of patch indices",
            "num_samples": num_samples,
            "num_patches_per_image": 576,
            "grid_size": "24x24",
            "patch_size": 14
        },
        "annotations": {
            str(i): {
                "patch_indices": [],
                "notes": "",
                "counterfactual_description": ""
            } for i in range(num_samples)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Annotation template saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Segmentation utilities module")
    print("\nExample: Convert bounding box to patches")
    
    # Example: Object at coordinates (100, 100) to (200, 200)
    patches = bounding_box_to_patches(100, 100, 200, 200)
    print(f"Bounding box (100, 100, 200, 200) covers patches: {patches}")
    
    # Example: Baseline random attention
    baseline = compute_baseline_random(num_cfact_patches=50)
    print(f"\nBaseline for 50 random patches: {baseline}")
