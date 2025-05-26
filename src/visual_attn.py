import math
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from transformers.image_processing_utils import select_best_resolution
import matplotlib.pyplot as plt


class LlavaNextImageTokenVisualizer:
    def __init__(self, original_image, processor):
        """
        Parameters:
            original_image (PIL.Image): e.g. a 1024x1024 image.
            inputs (dict): should contain:
                - "image_sizes": list of tuples like [(orig_height, orig_width)]
                - "pixel_values": a tensor of shape [B, 3, final_height, final_width]
            processor: an object (or processor) with attributes:
                - patch_size (e.g. 14)
                - image_processor.image_grid_pinpoints (list of int)
                - num_additional_image_tokens (optional)
        """
        # self.original_image = original_image
        self.inputs = processor.image_processor.preprocess(
            images=[original_image], do_normalize=False, return_tensors="pt"
        )
        self.processor = processor
        # Compute the mapping between tokens and patches.
        self.map_dict = self._compute_token_mapping()
        self.img = self.merge_subimages_clockwise(self.inputs["pixel_values"][0])

    def _compute_token_mapping(self):
        """
        Computes the mapping between token indices and patch coordinates.
        Returns a dictionary with:
          - "base_patches_map": mapping from token index to (row, col) in the base grid.
          - "scaled_patches_map": mapping from token index to (row, col) in the zoomed (scaled) grid.
          - "base_grid_size": (num_rows, num_cols) of the base grid.
          - "scaled_grid_size": (num_rows, num_cols) of the scaled grid.
          - "base_to_scaled_map": mapping from a base patch coordinate (i,j) to a list of scaled patch coordinates.
          - "n_lines": number of new-line tokens (here, equal to the number of rows in the scaled grid).
        """
        # --- 1) Basic info ---
        (orig_height, orig_width) = self.inputs["image_sizes"][0]
        final_height, final_width = self.inputs["pixel_values"][0].shape[
            -2:
        ]  # e.g., (336, 336)
        patch_size = self.processor.patch_size
        grid_pinpoints = self.processor.image_processor.image_grid_pinpoints
        num_additional_tokens = getattr(
            self.processor, "num_additional_image_tokens", 1
        )

        # --- 2) Best resolution & scale factors ---
        h_best, w_best = select_best_resolution(
            [orig_height, orig_width], grid_pinpoints
        )
        scale_h = h_best // final_height
        scale_w = w_best // final_width

        # --- 3) Compute base grid dimensions ---
        base_patches_h = final_height // patch_size
        base_patches_w = final_width // patch_size
        base_patches_count = base_patches_h * base_patches_w

        # --- 4) Compute scaled grid dimensions ---
        scaled_patches_h = base_patches_h * scale_h
        scaled_patches_w = base_patches_w * scale_w

        # Adjust the scaled grid dimensions to respect the original image aspect ratio.
        original_aspect_ratio = orig_width / orig_height
        current_aspect_ratio = scaled_patches_w / scaled_patches_h
        if original_aspect_ratio > current_aspect_ratio:
            new_height = (orig_height * scaled_patches_w) // orig_width
            padding = (scaled_patches_h - new_height) // 2
            scaled_patches_h -= padding * 2
        else:
            new_width = (orig_width * scaled_patches_h) // orig_height
            padding = (scaled_patches_w - new_width) // 2
            scaled_patches_w -= padding * 2

        scaled_patches_count = scaled_patches_h * scaled_patches_w
        new_line_count = scaled_patches_h  # one new-line per row

        print(
            "Number of tokens:",
            base_patches_count + scaled_patches_count + new_line_count,
        )
        self.number_of_tokens = (
            base_patches_count + scaled_patches_count + new_line_count
        )
        # --- 5) Create the base patches map ---
        base_patches_map = {}
        for i in range(base_patches_count):
            # Map token index (for base grid tokens) to a (row, col)
            base_patches_map[i] = (i // base_patches_w, i % base_patches_w)

        # --- 6) Create the scaled patches map ---
        scaled_patches_map = {}
        # Ensure scaled_patches_w is an integer.
        spw = int(scaled_patches_w)
        for i in range(scaled_patches_count):
            row = i // spw
            col = i % spw
            # In this example we treat the last patch in each row as a newline token.
            # (You can modify this behavior as needed.)
            if (i + 1) % spw == 0:
                scaled_patches_map[i + base_patches_count] = (row, col)
            else:
                scaled_patches_map[i + base_patches_count] = (row, col)

        # --- 7) Map from base grid patch coordinates to a list of scaled grid patch coordinates ---
        base_to_scaled_map = {}
        for _, base_patch in base_patches_map.items():
            i, j = base_patch
            base_to_scaled_map[(i, j)] = []
            for i_ in range(i * scale_h, (i + 1) * scale_h):
                for j_ in range(j * scale_w, (j + 1) * scale_w):
                    base_to_scaled_map[(i, j)].append((i_, j_))

        return {
            "base_patches_map": base_patches_map,
            "scaled_patches_map": scaled_patches_map,
            "base_grid_size": (base_patches_h, base_patches_w),
            "scaled_grid_size": (int(scaled_patches_h), spw),
            "base_to_scaled_map": base_to_scaled_map,
            "n_lines": new_line_count,
        }

    def merge_subimages_clockwise(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Merges a tensor of sub-images into one.

        Parameters:
            img_tensor (torch.Tensor): shape [5, 3, 336, 336]
                - index 0: base image
                - indices [1..4]: sub-images (each 3x336x336)

        Returns:
            torch.Tensor: a merged image of shape [3, 672, 672] arranged in a 2x2 grid:
                top-left, top-right, bottom-right, bottom-left (clockwise order).
        """
        subimgs = img_tensor[1:]  # take the four sub-images
        merged = torch.zeros((3, 672, 672), dtype=subimgs.dtype, device=subimgs.device)
        # Arrange sub-images in a 2x2 grid.
        merged[:, 0:336, 0:336] = subimgs[0]  # top-left
        merged[:, 0:336, 336:672] = subimgs[1]  # top-right
        merged[:, 336:672, 0:336] = subimgs[2]  # bottom-left
        merged[:, 336:672, 336:672] = subimgs[3]  # bottom-right
        return merged

    def visualize_token(self, token_index: int, show: bool = True) -> torch.Tensor:
        """
        Highlights the patch corresponding to a single token by drawing a red outline.

        Parameters:
            img (torch.Tensor): image tensor of shape [3, pxl, pxl] (CHW).
            token_index (int): index of the token to highlight.

        Returns:
            torch.Tensor: a copy of the image with a red outline around the patch.
        """
        # Determine which patch references this token has.
        if token_index in self.map_dict["base_patches_map"]:
            # For a base token, map it to its corresponding scaled patches.
            base_coord = self.map_dict["base_patches_map"][token_index]
            patches = self.map_dict["base_to_scaled_map"][base_coord]
        elif token_index in self.map_dict["scaled_patches_map"]:
            patches = [self.map_dict["scaled_patches_map"][token_index]]
        else:
            raise ValueError(f"Token index {token_index} not found in mapping.")

        # Use the scaled grid to compute patch dimensions.
        grid_rows, grid_cols = self.map_dict["scaled_grid_size"]
        pxl = self.img.shape[-1]
        patch_height = pxl // grid_rows
        patch_width = pxl // grid_cols

        img_copy = self.img.clone()

        def draw_red_outline(tensor, row_start, row_end, col_start, col_end):
            # Draw a red border (RGB: 255,0,0)
            tensor[0, row_start, col_start:col_end] = 255  # top edge
            tensor[1, row_start, col_start:col_end] = 0
            tensor[2, row_start, col_start:col_end] = 0
            tensor[0, row_end - 1, col_start:col_end] = 255  # bottom edge
            tensor[1, row_end - 1, col_start:col_end] = 0
            tensor[2, row_end - 1, col_start:col_end] = 0
            tensor[0, row_start:row_end, col_start] = 255  # left edge
            tensor[1, row_start:row_end, col_start] = 0
            tensor[2, row_start:row_end, col_start] = 0
            tensor[0, row_start:row_end, col_end - 1] = 255  # right edge
            tensor[1, row_start:row_end, col_end - 1] = 0
            tensor[2, row_start:row_end, col_end - 1] = 0

        # For each patch reference, draw the red outline.
        for ref in patches:
            if isinstance(ref, tuple) and len(ref) == 2:
                I, J = ref
                row_start = I * patch_height
                row_end = min((I + 1) * patch_height, pxl)
                col_start = J * patch_width
                col_end = min((J + 1) * patch_width, pxl)
                draw_red_outline(img_copy, row_start, row_end, col_start, col_end)
            else:
                print(f"Warning: unrecognized patch reference: {ref}")
        if show:
            plt.imshow(img_copy.permute(1, 2, 0))
        else:
            return img_copy

    def visualize_tokens_heatmap(
        self,
        token_indexes,
        alpha_max=0.5,
        show: bool = True,
        title="",
        color: str = "yellow",
        normalize: bool = True,
        min_val: float = None,
        max_val: float = None,
        threshold: float = None,
    ) -> torch.Tensor:
        """
        Overlays a transparent heatmap on the image based on a set of token values.

        Parameters:
            token_indexes (list or torch.Tensor): sequence of token values (floats) whose length equals the number
                of tokens in the mapping.
            alpha_max (float): maximum overlay opacity.
            show (bool): whether to display the image using plt.imshow or return it.
            title (str): title for the plot if shown.
            color (str): color for the heatmap overlay ("red", "green", "blue", "yellow", "cyan", "magenta").
            normalize (bool): whether to normalize token values for better visualization.
            min_val (float, optional): override the minimum value for normalization.
            max_val (float, optional): override the maximum value for normalization.
            threshold (float, optional): only show values above this threshold.

        Returns:
            torch.Tensor: the image with a heatmap overlay.
        """
        if not torch.is_tensor(token_indexes):
            token_indexes = torch.tensor(token_indexes, dtype=torch.float32)
        else:
            token_indexes = token_indexes.to(torch.float32)

        img = self.img.clone().float()
        C, H, W = img.shape
        grid_rows, grid_cols = self.map_dict["scaled_grid_size"]
        patch_height = H // grid_rows
        patch_width = W // grid_cols

        # Create a heatmap for the original area.
        heatmap_orig = torch.zeros((H, W), dtype=torch.float32, device=img.device)

        # Apply threshold if specified
        if threshold is not None:
            token_indexes = torch.where(
                token_indexes >= threshold,
                token_indexes,
                torch.zeros_like(token_indexes),
            )

        num_tokens = len(token_indexes)
        for token_idx in range(num_tokens):
            token_val = token_indexes[token_idx].item()
            patches = None
            if token_idx in self.map_dict.get("base_patches_map", {}):
                base_ref = self.map_dict["base_patches_map"][token_idx]
                patches = self.map_dict["base_to_scaled_map"].get(base_ref, [])
                if not isinstance(patches, list):
                    patches = [patches]
            elif token_idx in self.map_dict.get("scaled_patches_map", {}):
                patches = [self.map_dict["scaled_patches_map"][token_idx]]
            else:
                continue

            for ref in patches:
                if isinstance(ref, tuple) and len(ref) == 2:
                    I, J = ref
                    row_start = I * patch_height
                    row_end = min((I + 1) * patch_height, H)
                    col_start = J * patch_width
                    col_end = min((J + 1) * patch_width, W)
                    heatmap_orig[row_start:row_end, col_start:col_end] += token_val

        # Apply normalization to enhance contrast
        if normalize:
            # Handle empty or uniform heatmaps
            # Handle empty or uniform heatmaps
            if heatmap_orig.max() == heatmap_orig.min():
                heatmap_orig_norm = torch.zeros_like(heatmap_orig)
            else:
                # If min/max not provided, compute from non-zero values if possible
                if min_val is None or max_val is None:
                    non_zero = heatmap_orig[heatmap_orig > 0]
                    if len(non_zero) > 0:
                        computed_min = non_zero.min().item()
                        computed_max = heatmap_orig.max().item()
                    else:
                        computed_min = 0
                        computed_max = 1

                    min_val = min_val if min_val is not None else computed_min
                    max_val = max_val if max_val is not None else computed_max

                # First normalize to [0, 1] range
                value_range = max(max_val - min_val, 1e-6)  # Avoid division by zero
                
                # Create a mask for non-zero values only
                mask = heatmap_orig > 0
                
                # Initialize normalized heatmap with zeros
                heatmap_orig_norm = torch.zeros_like(heatmap_orig)
                
                # Only normalize non-zero values
                if mask.sum() > 0:  # Check if there are any non-zero values
                    # Apply normalization only to non-zero values
                    heatmap_orig_norm[mask] = (heatmap_orig[mask] - min_val) / value_range
                    
                    # Apply square root normalization to enhance differences
                    heatmap_orig_norm[mask] = torch.sqrt(heatmap_orig_norm[mask])
                
                # Clamp to ensure values are in [0, 1] range
                heatmap_orig_norm = torch.clamp(heatmap_orig_norm, 0, 1)
        else:
            # For no normalization, just use raw values but ensure they're within [0,1]
            heatmap_orig_norm = torch.clamp(heatmap_orig, 0, 1)

        # Prepare the overlay color.
        overlay_color_map = {
            "red": torch.tensor([255.0, 0.0, 0.0], device=img.device).view(3, 1, 1),
            "green": torch.tensor([0.0, 255.0, 0.0], device=img.device).view(3, 1, 1),
            "blue": torch.tensor([0.0, 0.0, 255.0], device=img.device).view(3, 1, 1),
            "yellow": torch.tensor([255.0, 255.0, 0.0], device=img.device).view(
                3, 1, 1
            ),
            "cyan": torch.tensor([0.0, 255.0, 255.0], device=img.device).view(3, 1, 1),
            "magenta": torch.tensor([255.0, 0.0, 255.0], device=img.device).view(
                3, 1, 1
            ),
        }
        if color not in overlay_color_map:
            raise ValueError(
                f"Color '{color}' not recognized. Choose from {list(overlay_color_map.keys())}."
            )
        overlay_color = overlay_color_map[color]

        hm_expanded = heatmap_orig_norm.unsqueeze(0)  # shape [1, H, W]

        # Use a mask to only blend where there are non-zero values
        mask = (hm_expanded > 0).float()
        blended = (1 - alpha_max * hm_expanded) * img + (
            alpha_max * hm_expanded
        ) * overlay_color
        img = mask * blended + (1 - mask) * img

        img = img.clamp(0, 255).to(img.dtype)
        if show:
            plt.figure(figsize=(10, 10))
            plt.title(title)
            plt.imshow(img.permute(1, 2, 0))
            plt.axis("off")

            # Add info about normalization range if enabled
            if normalize and min_val is not None and max_val is not None:
                value_text = f"Range: [{min_val:.2e}, {max_val:.2e}]"
                plt.annotate(
                    value_text,
                    (0.5, 0.05),
                    xycoords="figure fraction",
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8
                    ),
                )

            plt.tight_layout()
        else:
            return img


class Gemma3ImageTokenVisualizer:
    def __init__(self, original_image, processor):
        """
        Parameters:
            original_image (PIL.Image): e.g. a 1024x1024 image.
            inputs (dict): should contain:
                - "image_sizes": list of tuples like [(orig_height, orig_width)]
                - "pixel_values": a tensor of shape [B, 3, final_height, final_width]
            processor: an object (or processor) with attributes:
                - patch_size (e.g. 14)
                - image_processor.image_grid_pinpoints (list of int)
                - num_additional_image_tokens (optional)
        """
        # self.original_image = original_image
        self.inputs = processor.image_processor.preprocess(
            images=[original_image], do_normalize=False, return_tensors="pt"
        )
        self.processor = processor

        self.patch_size = 16
        # Compute the mapping between tokens and patches.
        self.map_dict = self._compute_token_mapping()
        self.img = self.inputs["pixel_values"][
            0
        ]  # Store the image tensor for visualization
        self.number_of_tokens = 256

    def _compute_token_mapping(self):
        """
        Computes the mapping between token indices and patch coordinates.
        Returns a dictionary with:
            - "base_patches_map": mapping from token index to [(start_row_pixel, start_col_pixel), (end_row_pixel, end_col_pixel)]

        """

        (orig_height, orig_width) = self.inputs["pixel_values"][0, 0].shape[
            -2:
        ]  # 896 x 896

        num_tokens = self.patch_size**2  # 16 x 16 = 256

        num_pixels_per_patch = orig_height // self.patch_size  # 896/16 = 56

        # Create mapping from token index to pixel coordinates
        base_patches_map = {}
        for i in range(num_tokens):
            # Convert token index to grid position (row, col)
            row = i // self.patch_size
            col = i % self.patch_size

            # Calculate pixel coordinates for this patch
            row_start = row * num_pixels_per_patch
            row_end = (row + 1) * num_pixels_per_patch
            col_start = col * num_pixels_per_patch
            col_end = (col + 1) * num_pixels_per_patch

            # Store the patch boundaries in the map
            base_patches_map[i] = [(row_start, col_start), (row_end, col_end)]

        return {
            "base_patches_map": base_patches_map,
            "grid_size": (self.patch_size, self.patch_size),
        }

    def visualize_tokens_heatmap(
        self,
        token_indexes,
        alpha_max=0.5,
        show=True,
        title="",
        color="yellow",
        normalize=True,
        min_val=None,
        max_val=None,
        threshold=None,
    ) -> torch.Tensor:
        """
        Overlays a transparent heatmap on the image based on a set of token values.

        Parameters:
            token_indexes (list or torch.Tensor): sequence of token values (floats) whose length equals the number
                of tokens in the mapping (256 for Gemma3).
            alpha_max (float): maximum overlay opacity.
            show (bool): whether to display the image using plt.imshow or return it.
            title (str): title for the plot if shown.
            color (str): color for the heatmap overlay ("red", "green", "blue", "yellow", "cyan", "magenta").
            normalize (bool): whether to normalize token values to [0,1] range for better visualization.
            min_val (float, optional): override the minimum value for normalization.
            max_val (float, optional): override the maximum value for normalization.
            threshold (float, optional): only show values above this threshold.

        Returns:
            torch.Tensor: the image with a heatmap overlay.
        """
        if not torch.is_tensor(token_indexes):
            token_indexes = torch.tensor(token_indexes, dtype=torch.float32)
        else:
            token_indexes = token_indexes.to(torch.float32)

        img = self.img.clone().float()
        C, H, W = img.shape

        # Create a heatmap for the original area
        heatmap = torch.zeros((H, W), dtype=torch.float32, device=img.device)

        # Apply token values to the corresponding patches in the heatmap
        num_tokens = min(len(token_indexes), self.patch_size**2)

        # Apply threshold if specified
        if threshold is not None:
            token_indexes = torch.where(
                token_indexes >= threshold,
                token_indexes,
                torch.zeros_like(token_indexes),
            )

        # Handle normalization if enabled
        if normalize:
            # If min/max not provided, compute them from the data
            if min_val is None or max_val is None:
                # Filter out non-zero values to handle cases where most values are 0
                non_zero_values = token_indexes[token_indexes != 0]
                if len(non_zero_values) > 0:
                    computed_min = non_zero_values.min().item()
                    computed_max = token_indexes.max().item()
                else:
                    # If all values are 0, avoid division by zero
                    computed_min = 0
                    computed_max = 1

                min_val = min_val if min_val is not None else computed_min
                max_val = max_val if max_val is not None else computed_max

            # Ensure we don't divide by zero
            value_range = max(max_val - min_val, 1e-6)
        else:
            # If not normalizing, just use the raw values
            min_val = 0
            value_range = 1

        for token_idx in range(num_tokens):
            token_val = token_indexes[token_idx].item()

            if token_idx in self.map_dict["base_patches_map"]:
                coordinates = self.map_dict["base_patches_map"][token_idx]
                start_coords, end_coords = coordinates
                row_start, col_start = start_coords
                row_end, col_end = end_coords

                # Normalize the token value to [0,1] range
                norm_val = (
                    (token_val - min_val) / value_range
                    if normalize and value_range > 0
                    else token_val
                )

                # Only apply non-zero values to avoid affecting the entire image
                if norm_val > 0:
                    # Apply the normalized value to the patch area in the heatmap
                    heatmap[row_start:row_end, col_start:col_end] = norm_val

        # Prepare the overlay color
        overlay_color_map = {
            "red": torch.tensor([255.0, 0.0, 0.0], device=img.device).view(3, 1, 1),
            "green": torch.tensor([0.0, 255.0, 0.0], device=img.device).view(3, 1, 1),
            "blue": torch.tensor([0.0, 0.0, 255.0], device=img.device).view(3, 1, 1),
            "yellow": torch.tensor([255.0, 255.0, 0.0], device=img.device).view(
                3, 1, 1
            ),
            "cyan": torch.tensor([0.0, 255.0, 255.0], device=img.device).view(3, 1, 1),
            "magenta": torch.tensor([255.0, 0.0, 255.0], device=img.device).view(
                3, 1, 1
            ),
        }

        if color not in overlay_color_map:
            raise ValueError(
                f"Color '{color}' not recognized. Choose from {list(overlay_color_map.keys())}."
            )

        overlay_color = overlay_color_map[color]

        # Expand heatmap to match the image dimensions
        hm_expanded = heatmap.unsqueeze(0)  # shape [1, H, W]

        # FIX: Only blend where heatmap has values > 0
        # This ensures we don't affect the entire image, just the highlighted patches
        mask = (hm_expanded > 0).float()
        blended = (1 - alpha_max * hm_expanded) * img + (
            alpha_max * hm_expanded
        ) * overlay_color
        img = mask * blended + (1 - mask) * img

        img = img.clamp(0, 255).to(img.dtype)

        if show:
            plt.figure(figsize=(10, 10))
            plt.title(title)
            plt.imshow(img.permute(1, 2, 0))
            plt.axis("off")

            # Create a separate colorbar figure if normalization is enabled
            if normalize:
                # Add a text annotation showing min/max values
                value_text = f"Range: [{min_val:.2e}, {max_val:.2e}]"
                plt.annotate(
                    value_text,
                    (0.5, 0.05),
                    xycoords="figure fraction",
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8
                    ),
                )

            plt.tight_layout()
            plt.show()

        return img
