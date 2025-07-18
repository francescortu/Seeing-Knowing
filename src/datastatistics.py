from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
import os
import sys


from easyroutine.interpretability import HookedModel
from rich import print
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
from rich.live import Live
from rich.layout import Layout
import io
import base64
from PIL import Image
import json
import re
import datetime
from tqdm import tqdm
from typing import Literal
from src.utils import get_whoops_element_by_id, start_ollama, ollama_model_map
from pathlib import Path
from easyroutine.interpretability import (
    HookedModel,
    ExtractionConfig,
    ActivationSaver,
    ActivationLoader,
)
from easyroutine.interpretability.tools import LogitLens
from easyroutine.logger import logger, enable_debug_logging, enable_info_logging
from tqdm import tqdm
import torch
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import logging
from dataclasses import dataclass

from easyroutine.interpretability import HookedModel


@dataclass
class StatisticsResult:
    """Container for statistics computation results"""

    total_cfact_logit_images: float = 0
    total_fact_logit_images: float = 0
    total_cfact_logit_text_only: float = 0
    total_fact_logit_text_only: float = 0
    position_cfact_logit_images: List[int] = None
    position_fact_logit_images: List[int] = None
    position_cfact_logit_text_only: List[int] = None
    position_fact_logit_text_only: List[int] = None
    cfact_higher_images: int = 0
    valid_images: int = 0
    cfact_higher_text: int = 0
    valid_text: int = 0
    higher_pos_image: List[float] = None
    higher_pos_text: List[float] = None

    def __post_init__(self):
        if self.position_cfact_logit_images is None:
            self.position_cfact_logit_images = []
        if self.position_fact_logit_images is None:
            self.position_fact_logit_images = []
        if self.position_cfact_logit_text_only is None:
            self.position_cfact_logit_text_only = []
        if self.position_fact_logit_text_only is None:
            self.position_fact_logit_text_only = []
        if self.higher_pos_image is None:
            self.higher_pos_image = []
        if self.higher_pos_text is None:
            self.higher_pos_text = []


class TokenPairExtractor:
    """Handles token pair extraction and validation"""

    def __init__(self, model: HookedModel):
        self.text_tokenizer = model.get_text_tokenizer()
        self.processor = model.get_processor()

    def clean_token(self, token: str) -> str:
        """Remove stopwords and clean token string"""
        stop_words = [
            ".",
            " .",
            ",",
            " ,",
            "-",
            "_",
            ";",
            " ;",
            ":",
            " :",
            "!",
            " !",
            "?",
            " ?",
            "(",
            " (",
            ")",
            " )",
            "[",
            "]",
            "{",
            "}",
            "'",
            '"',
            '"',
            "`",
            "~",
            '"',
            "",
            "``",
            "''",
            "''",
            "``",
        ]
        return token if token not in stop_words else ""

    def extract_token_pairs(
        self, element: Dict, given_token_pair: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[str, str]:
        """Extract and validate token pairs from data element"""
        if given_token_pair is not None:
            return [p[0] for p in given_token_pair], [p[1] for p in given_token_pair]

        ctoken = element["counterfactual_tokens"]
        ftoken = element["factual_tokens"]

        ctoken_strings = [
            self.clean_token(
                self.text_tokenizer.decode(self.text_tokenizer(c)["input_ids"][1])
            )
            for c in ctoken
        ]
        ftoken_strings = [
            self.clean_token(
                self.text_tokenizer.decode(self.text_tokenizer(f)["input_ids"][1])
            )
            for f in ftoken
        ]

        ctoken_strings = [c for c in ctoken_strings if c]
        ftoken_strings = [f for f in ftoken_strings if f]

        return ctoken_strings, ftoken_strings


class StatisticsComputer:
    """Main class for computing model statistics"""

    def __init__(self, model: HookedModel):
        self.model = model
        self.token_extractor = TokenPairExtractor(model)
        self.logger = logging.getLogger(__name__)

    def compute_token_predictions(
        self, element: Dict, compute_image: bool, compute_text: bool, disable_text_interventions: bool = False
    ) -> Tuple[Dict, Optional[Dict]]:
        """Compute model predictions for both image and text inputs"""
        prediction_with_img = None
        prediction_text_only = None

        if compute_image:
            self.model.use_full_model()
            prediction_with_img = self.model.predict(
                inputs=element["text_image_inputs"], k=100
            )

        if compute_text:
            self.model.use_language_model_only()
            # self.model.clean_interventions()
            prediction_text_only = self.model.predict(
                inputs=element["text_inputs"], k=100
            )

        return prediction_with_img, prediction_text_only

    def process_predictions(
        self, predictions: Dict[str, Any], token_strings: List[str]
    ) -> Tuple[str, float, int]:
        """Process predictions and find best matching token"""
        best_token = None
        max_logit = float("-inf")
        # position = -1

        if predictions is not None:
            candidates = []
            for token in token_strings:
                if token in predictions:
                    candidates.append((token, predictions[token]))

            if candidates:
                best_token, max_logit = max(candidates, key=lambda x: x[1])
                # position = list(predictions).index(best_token)
            else:
                best_token = token_strings[0]
        return best_token, max_logit

    def find_logit_and_position(
        self, predictions: Dict[str, Any], best_token: str
    ) -> Tuple[float, int]:
        """Find logit and position for the best token"""
        max_logit = float("-inf")
        position = -1

        if predictions is not None:
            if best_token in predictions:
                return predictions[best_token], list(predictions).index(best_token)
        return max_logit, position

    def compute_statistics(
        self,
        dataloader: List[Dict],
        compute: List[str] = ["image", "text"],
        given_token_pair: Optional[List[Tuple[str, str]]] = None,
        return_essential_data: bool = False,
        return_topk: int = 20,
        interventions: Optional[Dict[int,List]] = None,
        disable_text_interventions: bool = False,
    ) -> Dict:
        """Main method to compute statistics"""
        stats = StatisticsResult()
        compute_image = "image" in compute
        compute_text = "text" in compute

        # Lists to store logits with indices for top-k analysis
        image_cfact_logits = []
        image_fact_logits = []
        text_cfact_logits = []
        text_fact_logits = []
        token_pairs = []

        # Set up Rich console with no stdout capture to avoid interference with other progress bars
        console = Console(
            stderr=True
        )  # Use stderr instead of stdout to avoid collision

        # Create a simple panel for the stats processing
        with console.status(
            f"[bold green]Processing statistics for {len(dataloader)} items...",
            spinner="dots",
        ) as status:
            start_time = datetime.datetime.now()

            for i, element in enumerate(dataloader):
                if interventions:
                    self.model.clean_interventions()
                    self.model.register_interventions(
                        interventions=interventions[i]
                    )
                # Print status update every 10 items without using progress bar
                # if i % 10 == 0:
                #     console.print(
                #         f"[cyan]Processing item {i}/{len(dataloader)}[/cyan]", end="\r"
                #     )

                # Get predictions
                pred_img, pred_text = self.compute_token_predictions(
                    element, compute_image, compute_text, disable_text_interventions
                )

                # Extract tokens
                if given_token_pair is None:
                    ctoken_strings, ftoken_strings = (
                        self.token_extractor.extract_token_pairs(element, given_token_pair)
                    )
                else:
                    ctoken_strings = [given_token_pair[i][0]]
                    ftoken_strings = [given_token_pair[i][1]]
                # First get c_token and f_token
                if compute_image and compute_text:
                    c_token, _ = self.process_predictions(pred_img, ctoken_strings)
                    f_token, _ = self.process_predictions(pred_text, ftoken_strings)
                elif compute_image:
                    c_token, _ = self.process_predictions(pred_img, ctoken_strings)
                    f_token, _ = self.process_predictions(pred_img, ftoken_strings)
                elif compute_text:
                    c_token, _ = self.process_predictions(pred_text, ctoken_strings)
                    f_token, _ = self.process_predictions(pred_text, ftoken_strings)
                else:
                    raise ValueError(
                        "At least one of compute_image or compute_text must be True"
                    )

                if compute_image:
                    c_logit, c_pos = self.find_logit_and_position(pred_img, c_token)
                    f_logit, f_pos = self.find_logit_and_position(pred_img, f_token)

                    stats.total_cfact_logit_images += (
                        c_logit if c_logit != float("-inf") else 0
                    )
                    stats.total_fact_logit_images += (
                        f_logit if f_logit != float("-inf") else 0
                    )
                    stats.position_cfact_logit_images.append(c_pos)
                    stats.position_fact_logit_images.append(f_pos)
                    image_cfact_logits.append((i, c_logit))
                    image_fact_logits.append((i, f_logit))

                if compute_text:

                    c_logit, c_pos = self.find_logit_and_position(pred_text, c_token)
                    f_logit, f_pos = self.find_logit_and_position(pred_text, f_token)

                    stats.total_cfact_logit_text_only += (
                        c_logit if c_logit != float("-inf") else 0
                    )
                    stats.total_fact_logit_text_only += (
                        f_logit if f_logit != float("-inf") else 0
                    )
                    stats.position_cfact_logit_text_only.append(c_pos)
                    stats.position_fact_logit_text_only.append(f_pos)
                    text_cfact_logits.append((i, c_logit, c_token))
                    text_fact_logits.append((i, f_logit, f_token))

                # Store token pairs for return
                token_pairs.append((c_token, f_token))
                

                status.update(
                    f"[bold green]Processing statistics for {i}/{len(dataloader)} items...",
                )

            # Print completion message
            console.print("[bold green]Statistics processing complete!        ")

        # Calculate statistics
        stats = self._calculate_final_statistics(stats)

        # if return_essential_data:
        #     return self._prepare_essential_data(
        #         stats,
        #         image_cfact_logits,
        #         image_fact_logits,
        #         text_cfact_logits,
        #         text_fact_logits,
        #         return_topk,
        #     )
        essential_data = self._prepare_essential_data(
            stats,
            image_cfact_logits,
            image_fact_logits,
            text_cfact_logits,
            text_fact_logits,
            return_topk,
        )

        # report = self._generate_report(stats)
        return essential_data, token_pairs

    def _calculate_final_statistics(self, stats: StatisticsResult) -> StatisticsResult:
        """Calculate final statistics from raw data"""
        if stats.position_cfact_logit_images:
            stats.cfact_higher_images, stats.valid_images, _, stats.higher_pos_image = (
                self._count_cfact_higher_filtered(
                    stats.position_cfact_logit_images, stats.position_fact_logit_images
                )
            )

        if stats.position_cfact_logit_text_only:
            stats.cfact_higher_text, stats.valid_text, _, stats.higher_pos_text = (
                self._count_cfact_higher_filtered(
                    stats.position_cfact_logit_text_only,
                    stats.position_fact_logit_text_only,
                )
            )

        return stats

    def _count_cfact_higher_filtered(
        self, cfact_positions: List[int], fact_positions: List[int]
    ) -> Tuple[int, int, List[int], List[float]]:
        """Count cases where counterfactual token appears before factual token"""
        cfact_wins = 0
        valid_examples = 0
        cfact_wins_index = []
        better_pos = []

        for i, (c_pos, f_pos) in enumerate(zip(cfact_positions, fact_positions)):
            if c_pos == -1 and f_pos == -1:
                better_pos.append(101)
                continue

            valid_examples += 1
            c_adj = c_pos if c_pos != -1 else 101
            f_adj = f_pos if f_pos != -1 else 101

            if c_adj < f_adj:
                cfact_wins += 1
                cfact_wins_index.append(i)

            better_pos.append(
                min(c_pos if c_pos != -1 else 101, f_pos if f_pos != -1 else 101)
            )

        return cfact_wins, valid_examples, cfact_wins_index, better_pos

    def _prepare_essential_data(
        self,
        stats: StatisticsResult,
        image_cfact_logits: List[Tuple[int, float]],
        image_fact_logits: List[Tuple[int, float]],
        text_cfact_logits: List[Tuple[int, float]],
        text_fact_logits: List[Tuple[int, float]],
        return_topk: int,
    ) -> Dict:
        """Prepare essential data including top-k analysis"""
        topk_data = {}

        if image_cfact_logits:
            image_cfact_sorted = sorted(
                image_cfact_logits, key=lambda x: x[1], reverse=True
            )
            image_fact_sorted = sorted(
                image_fact_logits, key=lambda x: x[1], reverse=True
            )

            topk_data.update(
                {
                    "image_top_cfact": [
                        (idx, logit) for idx, logit in image_cfact_sorted[:return_topk]
                    ],
                    "image_bottom_cfact": [
                        (idx, logit) for idx, logit in image_cfact_sorted[-return_topk:]
                    ],
                    "image_top_fact": [
                        (idx, logit) for idx, logit in image_fact_sorted[:return_topk]
                    ],
                    "image_bottom_fact": [
                        (idx, logit) for idx, logit in image_fact_sorted[-return_topk:]
                    ],
                }
            )

        if text_cfact_logits:
            text_cfact_sorted = sorted(
                text_cfact_logits, key=lambda x: x[1], reverse=True
            )
            text_fact_sorted = sorted(
                text_fact_logits, key=lambda x: x[1], reverse=True
            )

            topk_data.update(
                {
                    "text_top_cfact": [
                        (idx, logit, tok)
                        for idx, logit, tok in text_cfact_sorted[:return_topk]
                    ],
                    "text_bottom_cfact": [
                        (idx, logit, tok)
                        for idx, logit, tok in text_cfact_sorted[-return_topk:]
                    ],
                    "text_top_fact": [
                        (idx, logit, tok)
                        for idx, logit, tok in text_fact_sorted[:return_topk]
                    ],
                    "text_bottom_fact": [
                        (idx, logit, tok)
                        for idx, logit, tok in text_fact_sorted[-return_topk:]
                    ],
                }
            )
            
            # find the index where cfact > fact
            indexes_cfact_gt_fact_text = []
            for i in range(len(text_cfact_logits)):
                if text_cfact_logits[i][1] > text_fact_logits[i][1]:
                    indexes_cfact_gt_fact_text.append(i)

        return {
            "Fact Acc": 100 - (stats.cfact_higher_images / stats.valid_images * 100)
            if stats.valid_images
            else 0,
            "Image Cfact logit": stats.total_cfact_logit_images,
            "Image Fact Logit": stats.total_fact_logit_images,
            "Image Pos Higher": np.array(stats.higher_pos_image, dtype=float).mean(),
            "indexes_cfact_gt_fact_text": indexes_cfact_gt_fact_text,
        }

    def _generate_report(self, stats: StatisticsResult) -> str:
        """Generate a readable report from statistics"""
        return (
            "\n ======== Total Statistics ========\n"
            f"Total cfact logit images: {stats.total_cfact_logit_images}\n"
            f"Total fact logit images: {stats.total_fact_logit_images}\n"
            f"Number of valid examples (image+text): {stats.valid_images}\n"
        )


# Maintain backward compatibility
def statistics_computer(
    model: HookedModel,
    dataloader: List[Dict],
    filename: Optional[str] = None,
    write_to_file: bool = False,
    dataset_path: Optional[Path] = None,
    compute: List[str] = ["image", "text"],
    given_token_pair: Optional[List[Tuple[str, str]]] = None,
    # return_essential_data: bool = False,
    return_topk: int = 20,
    interventions: Optional[Dict[int,List]] = None,
    disable_text_interventions: bool = False,
):
    """Backward compatible wrapper for StatisticsComputer"""
    computer = StatisticsComputer(model)
    data, token_pairs = computer.compute_statistics(
        dataloader=dataloader,
        compute=compute,
        given_token_pair=given_token_pair,
        # return_essential_data=return_essential_data,
        return_topk=return_topk,
        interventions=interventions,
        disable_text_interventions=disable_text_interventions,
    )

    if write_to_file and filename:
        with open(filename, "w") as f:
            if isinstance(result, tuple):
                f.write(result[0])
            else:
                f.write(str(result))

    return token_pairs, data
