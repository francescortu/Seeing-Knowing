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

        # Ensure image is in proper range [0, 255]
        if self.img.max() <= 1.0:
            self.img = self.img * 255.0

        print(
            f"LLaVA Image range: [{self.img.min().item():.1f}, {self.img.max().item():.1f}]"
        )

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
            (orig_height, orig_width), grid_pinpoints
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

        # print(
        #     "Number of tokens:",
        #     base_patches_count + scaled_patches_count + new_line_count,
        # )
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

    def visualize_token(self, token_index: int, show: bool = True):
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
            display_img = img_copy.permute(1, 2, 0)
            if display_img.is_floating_point():
                display_img = (display_img / 255.0).clamp(0, 1)
            plt.imshow(display_img)
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
        min_val=None,
        max_val=None,
        threshold=None,
    ):
        """
        Overlays a transparent heatmap on the image based on token values.

        Parameters:
            token_indexes (list or torch.Tensor): Sequence of token values.
            alpha_max (float): Maximum overlay opacity.
            show (bool): Whether to display the image using plt.imshow or return it.
            title (str): Title for the plot if shown.
            color (str): Color for the heatmap overlay.
            normalize (bool): Whether to normalize token values.
            min_val (float, optional): Override min value for normalization.
            max_val (float, optional): Override max value for normalization.
            threshold (float, optional): Threshold for token values.

        Returns:
            torch.Tensor: The image with heatmap overlay.
        """
        if not torch.is_tensor(token_indexes):
            token_indexes = torch.tensor(
                token_indexes, dtype=torch.float32, device=self.img.device
            )
        else:
            token_indexes = token_indexes.to(torch.float32).to(self.img.device)

        img = self.img.clone().float()
        C, H, W = img.shape
        grid_rows, grid_cols = self.map_dict["scaled_grid_size"]
        patch_height = H // grid_rows
        patch_width = W // grid_cols

        # Create heatmap
        heatmap = torch.zeros((H, W), dtype=torch.float32, device=img.device)

        # Apply threshold if specified
        if threshold is not None:
            token_indexes_eff = torch.where(
                token_indexes >= threshold,
                token_indexes,
                torch.zeros_like(token_indexes),
            )
        else:
            token_indexes_eff = token_indexes

        # Populate heatmap with token values
        num_tokens = len(token_indexes_eff)
        for token_idx in range(num_tokens):
            token_val = token_indexes_eff[token_idx].item()
            if token_val == 0:
                continue

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
                    i, j = ref
                    row_start = i * patch_height
                    row_end = min((i + 1) * patch_height, H)
                    col_start = j * patch_width
                    col_end = min((j + 1) * patch_width, W)
                    heatmap[row_start:row_end, col_start:col_end] += token_val

        # Normalize heatmap
        if normalize:
            if heatmap.max() == heatmap.min():
                heatmap_norm = torch.zeros_like(heatmap)
                min_val_eff = 0
                max_val_eff = 1
            else:
                if min_val is None or max_val is None:
                    non_zero = heatmap[heatmap > 0]
                    computed_min = non_zero.min().item() if len(non_zero) > 0 else 0
                    computed_max = heatmap.max().item() if len(non_zero) > 0 else 1
                    min_val_eff = min_val if min_val is not None else computed_min
                    max_val_eff = max_val if max_val is not None else computed_max
                else:
                    min_val_eff = min_val
                    max_val_eff = max_val

                value_range = max(max_val_eff - min_val_eff, 1e-6)
                mask_gt_0 = heatmap > 0
                heatmap_norm = torch.zeros_like(heatmap)
                if mask_gt_0.sum() > 0:
                    heatmap_norm[mask_gt_0] = (
                        heatmap[mask_gt_0] - min_val_eff
                    ) / value_range
                heatmap_norm = torch.clamp(heatmap_norm, 0, 1)
        else:
            heatmap_norm = torch.clamp(heatmap, 0, 1)
            min_val_eff = 0
            max_val_eff = 1

        # Color overlay
        overlay_color_map = {
            "red": torch.tensor([255.0, 0.0, 0.0], device=img.device).view(3, 1, 1),
            "paper_red": torch.tensor([241.0, 156.0, 153.0], device=img.device).view(
                3, 1, 1
            ),
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
            raise ValueError(f"Color '{color}' not recognized.")

        overlay_color = overlay_color_map[color]
        hm_expanded = heatmap_norm.unsqueeze(0)

        # Clamp alpha_max to reasonable range
        alpha_max = min(alpha_max, 1.0)

        # Only blend where heatmap has values > 0
        mask = (hm_expanded > 0).float()
        blended = (1 - alpha_max * hm_expanded) * img + (
            alpha_max * hm_expanded
        ) * overlay_color
        img = mask * blended + (1 - mask) * img

        img = img.clamp(0, 255).to(self.img.dtype)

        if show:
            plt.figure(figsize=(10, 10))
            plt.title(title)
            display_img = img.permute(1, 2, 0)
            if display_img.is_floating_point():
                display_img = (display_img / 255.0).clamp(0, 1)
            plt.imshow(display_img)
            plt.axis("off")

            # Create a separate colorbar figure if normalization is enabled
            if normalize:
                # Add a text annotation showing min/max values
                value_text = f"Range ({color}): [{min_val_eff:.2e}, {max_val_eff:.2e}]"
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

    def calculate_coverage_percentage(self, token_indexes, threshold=None):
        """
        Calculates the percentage of image pixels covered by patches corresponding to non-zero token values.

        Parameters:
            token_indexes (list or torch.Tensor): Sequence of token values.
            threshold (float, optional): Threshold for token values. Values below this are treated as zero.

        Returns:
            float: Percentage of image pixels covered (0-100).
        """
        if not torch.is_tensor(token_indexes):
            token_indexes = torch.tensor(
                token_indexes, dtype=torch.float32, device=self.img.device
            )
        else:
            token_indexes = token_indexes.to(torch.float32).to(self.img.device)

        # Apply threshold if specified
        if threshold is not None:
            token_indexes_eff = torch.where(
                token_indexes >= threshold,
                token_indexes,
                torch.zeros_like(token_indexes),
            )
        else:
            token_indexes_eff = token_indexes

        C, H, W = self.img.shape
        total_pixels = H * W
        grid_rows, grid_cols = self.map_dict["scaled_grid_size"]
        patch_height = H // grid_rows
        patch_width = W // grid_cols

        # Create a binary mask to track covered pixels
        covered_mask = torch.zeros((H, W), dtype=torch.bool, device=self.img.device)

        # Process each token
        num_tokens = len(token_indexes_eff)
        for token_idx in range(num_tokens):
            token_val = token_indexes_eff[token_idx].item()
            if token_val == 0:
                continue

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
                    i, j = ref
                    row_start = i * patch_height
                    row_end = min((i + 1) * patch_height, H)
                    col_start = j * patch_width
                    col_end = min((j + 1) * patch_width, W)
                    covered_mask[row_start:row_end, col_start:col_end] = True

        # Calculate coverage percentage
        covered_pixels = covered_mask.sum().item()
        coverage_percentage = (covered_pixels / total_pixels) * 100.0

        return coverage_percentage

    def get_raw_heatmap_values(self, token_indexes, token_attention_threshold=None):
        """
        Generates a raw 2D heatmap based on token attention values.

        Parameters:
            token_indexes (list or torch.Tensor): Sequence of token attention values.
            token_attention_threshold (float, optional): Tokens with attention below this value will be treated as zero.

        Returns:
            torch.Tensor: A 2D tensor representing the spatial heatmap.
        """
        if not torch.is_tensor(token_indexes):
            token_indexes = torch.tensor(
                token_indexes, dtype=torch.float32, device=self.img.device
            )
        else:
            token_indexes = token_indexes.to(torch.float32).to(self.img.device)

        if token_attention_threshold is not None:
            actual_token_indexes = torch.where(
                token_indexes >= token_attention_threshold,
                token_indexes,
                torch.zeros_like(token_indexes),
            )
        else:
            actual_token_indexes = token_indexes

        H, W = self.img.shape[-2:]
        grid_rows, grid_cols = self.map_dict["scaled_grid_size"]
        patch_height = H // grid_rows
        patch_width = W // grid_cols
        heatmap_values = torch.zeros(
            (H, W), dtype=torch.float32, device=self.img.device
        )

        num_tokens_to_process = len(actual_token_indexes)
        for token_idx in range(num_tokens_to_process):
            token_val = actual_token_indexes[token_idx].item()
            if token_val == 0:  # Skip if token's attention is zero
                continue

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
                    i, j = ref  # Changed I, J to i, j
                    row_start = i * patch_height
                    row_end = min((i + 1) * patch_height, H)
                    col_start = j * patch_width
                    col_end = min((j + 1) * patch_width, W)
                    heatmap_values[row_start:row_end, col_start:col_end] += token_val
        return heatmap_values


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

    def calculate_coverage_percentage(self, token_indexes, threshold=None):
        """
        Calculates the percentage of image pixels covered by patches corresponding to non-zero token values.

        Parameters:
            token_indexes (list or torch.Tensor): Sequence of token values.
            threshold (float, optional): Threshold for token values. Values below this are treated as zero.

        Returns:
            float: Percentage of image pixels covered (0-100).
        """
        if not torch.is_tensor(token_indexes):
            token_indexes = torch.tensor(
                token_indexes, dtype=torch.float32, device=self.img.device
            )
        else:
            token_indexes = token_indexes.to(torch.float32).to(self.img.device)

        # Apply threshold if specified
        if threshold is not None:
            token_indexes_eff = torch.where(
                token_indexes >= threshold,
                token_indexes,
                torch.zeros_like(token_indexes),
            )
        else:
            token_indexes_eff = token_indexes

        C, H, W = self.img.shape
        total_pixels = H * W

        # Create a binary mask to track covered pixels
        covered_mask = torch.zeros((H, W), dtype=torch.bool, device=self.img.device)

        # Process each token
        num_tokens = min(len(token_indexes_eff), self.patch_size**2)
        for token_idx in range(num_tokens):
            token_val = token_indexes_eff[token_idx].item()
            if token_val == 0:
                continue

            if token_idx in self.map_dict["base_patches_map"]:
                coordinates = self.map_dict["base_patches_map"][token_idx]
                start_coords, end_coords = coordinates
                row_start, col_start = start_coords
                row_end, col_end = end_coords

                # Mark all pixels covered by this token's patch
                covered_mask[row_start:row_end, col_start:col_end] = True

        # Calculate coverage percentage
        covered_pixels = covered_mask.sum().item()
        coverage_percentage = (covered_pixels / total_pixels) * 100.0

        return coverage_percentage

    def get_raw_heatmap_values(self, token_indexes, token_attention_threshold=None):
        """
        Generates a raw 2D heatmap based on token attention values.

        Parameters:
            token_indexes (list or torch.Tensor): Sequence of token attention values.
            token_attention_threshold (float, optional): Tokens with attention below this value will be treated as zero.

        Returns:
            torch.Tensor: A 2D tensor representing the spatial heatmap.
        """
        if not torch.is_tensor(token_indexes):
            token_indexes = torch.tensor(
                token_indexes, dtype=torch.float32, device=self.img.device
            )
        else:
            token_indexes = token_indexes.to(torch.float32).to(self.img.device)

        if token_attention_threshold is not None:
            actual_token_indexes = torch.where(
                token_indexes >= token_attention_threshold,
                token_indexes,
                torch.zeros_like(token_indexes),
            )
        else:
            actual_token_indexes = token_indexes

        C, H, W = self.img.shape
        heatmap_values = torch.zeros(
            (H, W), dtype=torch.float32, device=self.img.device
        )  # Fixed: used self.img.device

        num_tokens_to_process = min(len(actual_token_indexes), self.patch_size**2)

        for token_idx in range(num_tokens_to_process):
            token_val = actual_token_indexes[token_idx].item()
            if token_val == 0:  # Skip if token's attention is zero or below threshold
                continue

            if token_idx in self.map_dict["base_patches_map"]:
                coordinates = self.map_dict["base_patches_map"][token_idx]
                start_coords, end_coords = coordinates
                row_start, col_start = start_coords
                row_end, col_end = end_coords

                # Add the token value to the corresponding patch area in the heatmap
                heatmap_values[row_start:row_end, col_start:col_end] += token_val

        return heatmap_values


# Add the new HeatmapComparer class here
class HeatmapComparer:
    def __init__(
        self,
        llava_visualizer: LlavaNextImageTokenVisualizer,
        gemma_visualizer: Gemma3ImageTokenVisualizer,
    ):
        self.llava_visualizer = llava_visualizer
        self.gemma_visualizer = gemma_visualizer

    def _normalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Normalizes a heatmap to the [0, 1] range."""
        if heatmap.numel() == 0:  # Handle empty tensor
            return heatmap

        min_val = heatmap.min()
        max_val = heatmap.max()

        if max_val == min_val:
            if max_val > 0:
                return torch.ones_like(heatmap)
            else:
                return torch.zeros_like(heatmap)

        # Consider only positive values for min if they exist, otherwise use 0
        # This helps if heatmaps have negative noise but we're interested in positive activations
        positive_values = heatmap[heatmap > 0]
        if positive_values.numel() > 0:
            current_min = positive_values.min().item()
        else:
            current_min = 0  # Or min_val.item() if negative values should set the floor

        current_max = max_val.item()

        # If all values were <=0 and current_min became 0, and max is also 0 or less
        if current_max <= current_min:  # Handles all zero or all negative/zero cases
            if (
                current_max > 0
            ):  # Should not happen if current_max <= current_min unless current_min is negative
                current_max = max(
                    current_max, current_min + 1e-6
                )  # ensure range is not zero
            else:  # all zero or negative
                return torch.zeros_like(heatmap)

        value_range = max(current_max - current_min, 1e-6)  # Avoid division by zero

        normalized_heatmap = torch.zeros_like(heatmap)

        # Apply normalization only to positive parts if that's the logic desired
        # or to the whole map based on the determined min/max
        # For simplicity here, let's normalize based on the determined min/max for all values
        # that could be positive after shifting by current_min.

        # Normalize, ensuring we only scale values that would be positive in the range.
        # Values below current_min will become <=0, then clamped.
        normalized_heatmap = (heatmap - current_min) / value_range
        normalized_heatmap = torch.clamp(normalized_heatmap, 0, 1)

        return normalized_heatmap

    # def calculate_jaccard_index(
    #     self,
    #     token_indexes_llava,
    #     token_indexes_gemma3,
    #     attention_threshold_llava=None,
    #     attention_threshold_gemma3=None,
    #     binarization_threshold=0.1,
    # ):
    #     """
    #     Calculates the Jaccard index between the heatmaps of LLaVA-NeXT and Gemma3.
    #     Heatmaps are normalized to [0,1] before binarization.

    #     Parameters:
    #         token_indexes_llava (list or torch.Tensor): Token attention values for LLaVA-NeXT.
    #         token_indexes_gemma3 (list or torch.Tensor): Token attention values for Gemma3.
    #         attention_threshold_llava (float, optional): Threshold for LLaVA-NeXT token attention.
    #         attention_threshold_gemma3 (float, optional): Threshold for Gemma3 token attention.
    #         binarization_threshold (float): Value above which *normalized* heatmap pixels are considered 'active' (1), else 'inactive' (0).

    #     Returns:
    #         float: The Jaccard index.
    #     """
    #     heatmap_llava_raw = self.llava_visualizer.get_raw_heatmap_values(
    #         token_indexes_llava, token_attention_threshold=attention_threshold_llava
    #     )
    #     heatmap_gemma_raw = self.gemma_visualizer.get_raw_heatmap_values(
    #         token_indexes_gemma3, token_attention_threshold=attention_threshold_gemma3
    #     )

    #     # Ensure heatmaps are on the same device for comparison and resizing
    #     target_device = (
    #         heatmap_gemma_raw.device
    #         if heatmap_gemma_raw.numel() > 0
    #         else heatmap_llava_raw.device
    #     )
    #     if heatmap_llava_raw.numel() > 0:
    #         heatmap_llava_raw = heatmap_llava_raw.to(target_device)
    #     if heatmap_gemma_raw.numel() > 0:
    #         heatmap_gemma_raw = heatmap_gemma_raw.to(target_device)

    #     # Resize heatmap_llava to match heatmap_gemma dimensions
    #     if heatmap_llava_raw.numel() > 0 and heatmap_gemma_raw.numel() > 0:
    #         gemma_H, gemma_W = heatmap_gemma_raw.shape
    #         resizer = T.Resize(
    #             (gemma_H, gemma_W),
    #             interpolation=T.InterpolationMode.BILINEAR,
    #             antialias=True,
    #         )
    #         heatmap_llava_resized = resizer(heatmap_llava_raw.unsqueeze(0)).squeeze(0)
    #     elif (
    #         heatmap_llava_raw.numel() > 0
    #     ):  # Gemma is empty, use Llava's shape or a default
    #         heatmap_llava_resized = heatmap_llava_raw
    #     else:  # Llava is empty or both are empty
    #         heatmap_llava_resized = heatmap_llava_raw  # which is empty

    #     # Normalize both heatmaps to [0, 1] range before binarization
    #     heatmap_llava_norm = self._normalize_heatmap(heatmap_llava_resized)
    #     heatmap_gemma_norm = self._normalize_heatmap(heatmap_gemma_raw)

    #     #

    #     # Binarize both normalized heatmaps
    #     heatmap_llava_bin = (heatmap_llava_norm > binarization_threshold).float()
    #     heatmap_gemma_bin = (heatmap_gemma_norm > binarization_threshold).float()

    #     # Calculate Jaccard Index
    #     intersection = (
    #         heatmap_llava_bin * heatmap_gemma_bin
    #     ).sum()  # Logical AND for binary tensors
    #     union = (
    #         ((heatmap_llava_bin + heatmap_gemma_bin) > 0).float().sum()
    #     )  # Logical OR for binary tensors

    #     if union == 0:
    #         return 0.0

    #     jaccard_index = intersection / union
    #     return jaccard_index.item()
    def calculate_jaccard_index(
        self,
        token_indexes_llava,
        token_indexes_gemma3,
        attention_threshold_llava=None,
        attention_threshold_gemma3=None,
    ):
        """
        Jaccard where Llava’s active pixels are chosen to match the count of Gemma’s non-zero pixels.
        """
        # 1) get raw heatmaps
        hl_raw = self.llava_visualizer.get_raw_heatmap_values(
            token_indexes_llava, token_attention_threshold=attention_threshold_llava
        )
        hg_raw = self.gemma_visualizer.get_raw_heatmap_values(
            token_indexes_gemma3, token_attention_threshold=attention_threshold_gemma3
        )

        # 2) align devices
        device = hg_raw.device if hg_raw.numel() > 0 else hl_raw.device
        hl_raw = hl_raw.to(device)
        hg_raw = hg_raw.to(device)

        # 3) resize Llava → Gemma resolution
        if hl_raw.numel() > 0 and hg_raw.numel() > 0:
            H, W = hg_raw.shape
            resizer = T.Resize(
                (H, W), interpolation=T.InterpolationMode.BILINEAR, antialias=True
            )
            hl_resized = resizer(hl_raw.unsqueeze(0)).squeeze(0)
        else:
            hl_resized = hl_raw

        # 4) normalize both to [0,1]
        hl_norm = self._normalize_heatmap(hl_resized)
        hg_norm = self._normalize_heatmap(hg_raw)

        # 5) count Gemma’s non-zero pixels
        n_g = int((hg_norm > 0).sum().item())

        # 6) pick Llava’s top-n_g pixels
        flat = hl_norm.flatten()
        if n_g > 0:
            k = min(n_g, flat.numel())
            threshold = torch.topk(flat, k, largest=True).values[-1]
            hl_bin = (hl_norm >= threshold).float()
        else:
            hl_bin = torch.zeros_like(hl_norm)

        # 7) binarize Gemma by >0
        hg_bin = (hg_norm > 0).float()

        # 8) compute Jaccard
        inter = (hl_bin * hg_bin).sum()
        union = ((hl_bin + hg_bin) > 0).float().sum()
        return 0.0 if union == 0 else (inter / union).item()


# import torch
# import torchvision.transforms as T

# class HeatmapComparer:

#     def __init__(self,
#                  llava_visualizer: LlavaNextImageTokenVisualizer,
#                  gemma_visualizer: Gemma3ImageTokenVisualizer):
# #         self.llava = llava_visualizer
# #         self.gemma = gemma_visualizer


#     def calculate_jaccard_index(
#         self,
#         token_indexes_llava,
#         token_indexes_gemma,
#         *,
#         method="patch",        # "patch" | "pixel_abs" | "pixel_pct"
#         pixel_threshold=0.1,   # for "pixel_abs"
#         keep_frac=0.05,        # for "pixel_pct"
#         resize="bilinear"      # "bilinear" or "nearest" for pixel‐level
#     ):
#         # --- PATCH-LEVEL JACCARD ON GEMMA’S 16×16 GRID ---
#         if method == "patch":
#             P =  16
#             # Gemma’s own vector of length P*P
#             g_vals = torch.tensor(token_indexes_gemma[:P*P], dtype=torch.float32)

#             # Build LLaVA raw heatmap (H_ll × W_ll) and aggregate into P×P
#             h_ll = self.llava_visualizer.get_raw_heatmap_values(token_indexes_llava)
#             H_ll, W_ll = h_ll.shape
#             pix_per_patch_h = H_ll // P
#             pix_per_patch_w = W_ll // P

#             l_agg = []
#             for idx in range(P*P):
#                 r, c = divmod(idx, P)
#                 patch = h_ll[
#                     r*pix_per_patch_h:(r+1)*pix_per_patch_h,
#                     c*pix_per_patch_w:(c+1)*pix_per_patch_w
#                 ]
#                 l_agg.append(patch.max().item())
#             l_vals = torch.tensor(l_agg, dtype=torch.float32)

#             # threshold by percentile so both have same #active patches
#             cut_l = torch.quantile(l_vals, 1-keep_frac)
#             cut_g = torch.quantile(g_vals, 1-keep_frac)
#             b_l = (l_vals >= cut_l)
#             b_g = (g_vals >= cut_g)

#             inter = (b_l & b_g).sum().item()
#             union = (b_l | b_g).sum().item()
#             return inter/union if union>0 else 0.0

#         # --- PIXEL-LEVEL JACCARD ---
#         # 1) get raw heatmaps
#         h_ll = self.llava_visualizer.get_raw_heatmap_values(token_indexes_llava)
#         h_gg = self.gemma_visualizer.get_raw_heatmap_values(token_indexes_gemma)

#         # 2) resize LLaVA → Gemma’s resolution
#         H_gg, W_gg = h_gg.shape
#         mode = T.InterpolationMode.BILINEAR if resize=="bilinear" else T.InterpolationMode.NEAREST
#         h_ll = T.Resize((H_gg, W_gg), interpolation=mode)(h_ll.unsqueeze(0)).squeeze(0)

#         # 3) normalize each independently to [0,1]
#         def norm01(x):
#             mn, mx = float(x.min()), float(x.max())
#             return (x - mn)/max(mx-mn,1e-6)
#         h_ll = norm01(h_ll)
#         h_gg = norm01(h_gg)

#         # 4) binarize
#         if method == "pixel_abs":
#             b_l = (h_ll >= pixel_threshold)
#             b_g = (h_gg >= pixel_threshold)
#         else:  # "pixel_pct"
#             cut_l = torch.quantile(h_ll.flatten(), 1-keep_frac)
#             cut_g = torch.quantile(h_gg.flatten(), 1-keep_frac)
#             b_l = (h_ll >= cut_l)
#             b_g = (h_gg >= cut_g)

#         # 5) Jaccard
#         inter = (b_l & b_g).sum().item()
#         union = (b_l | b_g).sum().item()
#         return inter/union if union>0 else 0.0
