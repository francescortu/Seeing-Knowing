import os
import sys
import json
import base64
import time
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Literal, Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

# Add paths to sys.path for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

# Local imports after path modification
from easyroutine.inference.litellm_model_interface import (  # noqa: E402
    LiteLLMInferenceModel,
    LiteLLMInferenceModelConfig,
)

# Model configuration aliases for convenience
MODEL_ALIASES = {
    "gemini-flash": "openrouter/google/gemini-2.5-flash-image-preview",
    "gpt5-mini": "openrouter/openai/gpt-5-mini",
    "gpt5": "openrouter/openai/gpt-5",
    "claude-3.7-sonnet": "openrouter/anthropic/claude-3.7-sonnet",
    "claude-3.5-haiku": "openrouter/anthropic/claude-3.5-haiku",
    "claude-3.5-sonnet": "openrouter/anthropic/claude-3.5-sonnet",
    "gpt-4o": "openrouter/openai/gpt-4o",
    "gpt-4o-mini": "openrouter/openai/gpt-4o-mini",
}


@dataclass
class ValidationResult:
    """Results from LLM judge validation"""

    appropriate_meaningful: Literal[
        "Yes", "Some do not make sense", "None are appropriate"
    ]
    grammatically_correct: Literal["Yes", "No", "Some do not make sense"]
    knowledge_reflection_score: int  # 1-5 scale
    raw_response: str


class LLMJudgeValidator:
    """LLM-as-a-judge validator for whoops-aha dataset with parallel processing support"""

    def __init__(
        self,
        model_config: LiteLLMInferenceModelConfig,
        max_workers: int = 3,
        rate_limit_delay: float = 0.5,
    ):
        self.model = LiteLLMInferenceModel(model_config)
        self.max_workers = max_workers  # Number of concurrent requests
        self.rate_limit_delay = rate_limit_delay  # Delay between requests (seconds)

    def encode_image_to_base64(
        self,
        image: Image.Image,
        format: str = "JPEG",
        max_dim: int = 1024,
        quality: int = 85,
    ) -> Tuple[str, str]:
        """Convert image to base64 string for API calls, enforcing RGB and size.

        Returns: (base64_str, mime_type)
        """
        try:
            # Ensure PIL Image
            if not isinstance(image, Image.Image):
                # Best-effort conversion if dataset provides other types
                image = Image.fromarray(image)

            # Ensure RGB (drop alpha if present)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Downscale very large images to reduce payload/provider errors
            if max_dim is not None:
                # Choose a high-quality downsampling filter depending on PIL version
                try:
                    resample_filter = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
                    image.thumbnail((max_dim, max_dim), resample=resample_filter)
                except Exception:
                    # Fallback: let PIL choose default resample
                    image.thumbnail((max_dim, max_dim))

            buffered = BytesIO()
            mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
            save_kwargs = {}
            if format.upper() == "JPEG":
                save_kwargs = {"quality": quality, "optimize": True}
            image.save(buffered, format=format.upper(), **save_kwargs)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str, mime
        except Exception:
            # Final fallback: try PNG if JPEG path failed
            try:
                buffered = BytesIO()
                image = image.convert("RGB") if image.mode != "RGB" else image
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return img_str, "image/png"
            except Exception as e2:
                raise e2

    def create_validation_prompt_improved(
        self, text: str, tokens: List[str], with_image: bool = False
    ) -> str:
        """Create improved validation prompt with precise criteria"""

        if with_image:
            task_description = """You will evaluate sentence completions that should describe UNUSUAL or ANOMALOUS elements visible in the provided image. Focus on identifying completions that capture strange, unexpected, or contradictory visual elements."""
            knowledge_focus = "unusual/anomalous visual elements in the image"
        else:
            task_description = """You will evaluate sentence completions that should reflect NORMAL, EXPECTED real-world scenarios based on common knowledge and typical experiences."""
            knowledge_focus = "normal, expected real-world knowledge"

        prompt = f"""{task_description}

    EVALUATION CRITERIA:

    1. SEMANTIC APPROPRIATENESS: Do the completions make logical sense?
    
    Evaluate EACH completion by inserting it into: "{text} [COMPLETION]"
    
    → "Yes" = ALL completions create logical, coherent sentences
    → "Some do not make sense" = 1+ completions create illogical/incoherent sentences  
    → "None are appropriate" = ALL completions create illogical sentences
    
    Examples of INAPPROPRIATE: contradictory logic, nonsensical combinations, impossible scenarios

    2. GRAMMATICAL CORRECTNESS: Are the completed sentences grammatically valid?
    
    Check syntax, word order, agreement, and grammatical structure of each completed sentence.
    
    → "Yes" = ALL completed sentences follow proper grammar rules
    → "Some do not make sense" = SOME completed sentences have grammatical errors
    → "No" = ALL completed sentences contain grammatical errors
    
    Focus on: subject-verb agreement, article usage, word order, tense consistency

    3. KNOWLEDGE ALIGNMENT: How well do completions reflect {knowledge_focus}?
    
    {"IMAGE CONTEXT: Rate how accurately completions describe the strange/unusual elements you can SEE in the image." if with_image else "REAL-WORLD CONTEXT: Rate how well completions reflect typical, widely-accepted real-world scenarios."}
    
    SCORING RUBRIC:
    • 5 = Perfectly captures {knowledge_focus} - highly accurate and relevant
    • 4 = Mostly accurate - minor gaps or imprecisions  
    • 3 = Partially accurate - some correct elements, some missing/wrong
    • 2 = Minimally accurate - mostly incorrect with few relevant elements
    • 1 = Completely inaccurate - fails to reflect {knowledge_focus}

    SENTENCE: "{text}"
    COMPLETIONS TO EVALUATE: {tokens}

    {"INSTRUCTION: Look carefully at the image and identify what appears unusual, unexpected, or anomalous. Then evaluate how well the completions capture these strange elements." if with_image else "INSTRUCTION: Consider what would be normal, expected, and typical in real-world scenarios. Then evaluate how well the completions reflect this common knowledge."}

    IMPORTANT: You must respond with valid JSON only. Do not include any text before or after the JSON.
    
    Required JSON format:
    {{
        "appropriate_meaningful": "Yes, all are appropriate and meaningful" | "Some do not make sense" | "No, they do not make sense",
        "grammatically_correct": "Yes, all are grammatically correct" | "Some are not grammatically correct" | "No, they are not grammatically correct", 
        "knowledge_reflection_score": 1-5,
        "reasoning": "Explain your evaluation for each criterion, citing specific completions"
    }}"""

        return prompt

    def validate_factual_tokens(
        self, text: str, factual_tokens: List[str]
    ) -> ValidationResult:
        """Validate factual tokens WITHOUT image (should reflect common knowledge)"""
        prompt = self.create_validation_prompt_improved(
            text, factual_tokens, with_image=False
        )

        messages = [{"role": "user", "content": prompt}]

        response = self.model.chat(messages)
        raw_response = self.model.get_last_text_from_response(response)

        # Enhanced response validation
        if not isinstance(raw_response, str):
            print(
                f"Warning: Response is not a string, got {type(raw_response)}: {response}"
            )
            raw_response = ""

        # Check for empty response
        if not raw_response or not raw_response.strip():
            print("Warning: Empty response received from model")
            print(f"Full response object: {response}")
            # Return default values for empty response
            return ValidationResult(
                appropriate_meaningful="Error: Empty response",
                grammatically_correct="Error: Empty response",
                knowledge_reflection_score=1,
                raw_response="<EMPTY_RESPONSE>",
            )

        # Parse JSON response
        try:
            # Handle responses wrapped in code blocks
            json_str = raw_response.strip()
            if "```json" in json_str:
                # Extract JSON from code block
                json_start = json_str.find("```json") + 7
                json_end = json_str.find("```", json_start)
                if json_end == -1:  # No closing ```
                    json_end = len(json_str)
                json_str = json_str[json_start:json_end].strip()
            elif "```" in json_str and "{" in json_str:
                # Handle case where JSON is in code block without "json" label
                start_brace = json_str.find("{")
                end_brace = json_str.rfind("}")
                if start_brace != -1 and end_brace != -1:
                    json_str = json_str[start_brace : end_brace + 1]

            # Try to find JSON object if response has extra text
            if not json_str.startswith("{"):
                start_brace = json_str.find("{")
                if start_brace != -1:
                    end_brace = json_str.rfind("}")
                    if end_brace != -1:
                        json_str = json_str[start_brace : end_brace + 1]

            # First attempt: parse as-is
            try:
                result_data = json.loads(json_str)
            except json.JSONDecodeError:
                # Handle truncated JSON by attempting to fix it
                json_str = self._fix_truncated_json(json_str)
                result_data = json.loads(json_str)

            return ValidationResult(
                appropriate_meaningful=result_data["appropriate_meaningful"],
                grammatically_correct=result_data["grammatically_correct"],
                knowledge_reflection_score=result_data["knowledge_reflection_score"],
                raw_response=raw_response,
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: '{raw_response}'")
            print(f"Attempted JSON string: '{json_str}'")
            print(f"Response length: {len(raw_response)}")

            # Try to extract partial information from non-JSON response
            fallback_result = self._extract_fallback_validation(raw_response)
            return ValidationResult(
                appropriate_meaningful=fallback_result.get(
                    "appropriate_meaningful", "Error: Parse failed"
                ),
                grammatically_correct=fallback_result.get(
                    "grammatically_correct", "Error: Parse failed"
                ),
                knowledge_reflection_score=fallback_result.get(
                    "knowledge_reflection_score", 1
                ),
                raw_response=raw_response,
            )

    def _extract_fallback_validation(self, raw_response: str) -> dict:
        """
        Extract validation information from non-JSON responses using pattern matching.
        This is a fallback when JSON parsing fails.
        """
        result = {}

        # Convert to lowercase for easier matching
        response_lower = raw_response.lower()

        # Look for appropriate/meaningful indicators
        if "appropriate" in response_lower or "meaningful" in response_lower:
            if any(
                word in response_lower
                for word in ["yes", "all", "appropriate", "meaningful", "good"]
            ):
                result["appropriate_meaningful"] = (
                    "Yes, all are appropriate and meaningful"
                )
            elif any(word in response_lower for word in ["some", "partially", "mixed"]):
                result["appropriate_meaningful"] = "Some do not make sense"
            else:
                result["appropriate_meaningful"] = "No, they do not make sense"

        # Look for grammatical indicators
        if "gramm" in response_lower or "correct" in response_lower:
            if any(
                word in response_lower for word in ["yes", "all", "correct", "good"]
            ):
                result["grammatically_correct"] = "Yes, all are grammatically correct"
            elif any(word in response_lower for word in ["some", "partially", "mixed"]):
                result["grammatically_correct"] = "Some are not grammatically correct"
            else:
                result["grammatically_correct"] = (
                    "No, they are not grammatically correct"
                )

        # Look for knowledge reflection score
        import re

        score_match = re.search(
            r"(\d+)(?:/5|/10|\s*out\s*of\s*(?:5|10))", response_lower
        )
        if score_match:
            score = int(score_match.group(1))
            # Normalize to 1-5 scale
            if "10" in score_match.group(0):
                score = max(
                    1, min(5, (score + 1) // 2)
                )  # Convert 10-point to 5-point scale
            result["knowledge_reflection_score"] = score
        elif any(
            word in response_lower for word in ["high", "good", "well", "excellent"]
        ):
            result["knowledge_reflection_score"] = 4
        elif any(word in response_lower for word in ["low", "poor", "bad", "terrible"]):
            result["knowledge_reflection_score"] = 2
        else:
            result["knowledge_reflection_score"] = 3  # Default middle value

        return result

    def validate_counterfactual_tokens(
        self, text: str, counterfactual_tokens: List[str], image: Image.Image
    ) -> ValidationResult:
        """Validate counterfactual tokens WITH image (should reflect strange/anomalous things)"""
        prompt = self.create_validation_prompt_improved(
            text, counterfactual_tokens, with_image=True
        )

        last_err = None
        raw_response = None
        # Try JPEG first (smaller), then PNG as fallback
        for enc_format in ("JPEG", "PNG"):
            try:
                image_b64, mime = self.encode_image_to_base64(image, format=enc_format)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                            },
                        ],
                    }
                ]
                response = self.model.chat(messages)
                raw_response = self.model.get_last_text_from_response(response)
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue
        if raw_response == "" and last_err is not None:
            # Re-raise the last error to be handled by caller (dataset loop will save progress)
            raise last_err
        if not isinstance(raw_response, str):
            raw_response = ""

        # Parse JSON response
        try:
            # Handle responses wrapped in code blocks
            if "```json" in raw_response:
                # Extract JSON from code block
                json_start = raw_response.find("```json") + 7
                json_end = raw_response.find("```", json_start)
                json_str = raw_response[json_start:json_end].strip()
            else:
                json_str = raw_response.strip()

            result_data = json.loads(json_str)
            return ValidationResult(
                appropriate_meaningful=result_data["appropriate_meaningful"],
                grammatically_correct=result_data["grammatically_correct"],
                knowledge_reflection_score=result_data["knowledge_reflection_score"],
                raw_response=raw_response,
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {raw_response}")
            # Return default values if parsing fails
            return ValidationResult(
                appropriate_meaningful="Some do not make sense",
                grammatically_correct="Some do not make sense",
                knowledge_reflection_score=1,
                raw_response=raw_response,
            )

    def validate_dataset_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single dataset sample"""
        text = sample["text"]
        factual_tokens = sample["factual_tokens"]
        counterfactual_tokens = sample["counterfactual_tokens"]
        image = sample["image"]

        # Validate factual tokens (without image)
        factual_result = self.validate_factual_tokens(text, factual_tokens)

        # Validate counterfactual tokens (with image)
        counterfactual_result = self.validate_counterfactual_tokens(
            text, counterfactual_tokens, image
        )

        return {
            "index": sample["index"],
            "image_id": sample["image_id"],
            "text": text,
            "factual_tokens": factual_tokens,
            "counterfactual_tokens": counterfactual_tokens,
            "factual_validation": {
                "appropriate_meaningful": factual_result.appropriate_meaningful,
                "grammatically_correct": factual_result.grammatically_correct,
                "knowledge_reflection_score": factual_result.knowledge_reflection_score,
                "raw_response": factual_result.raw_response,
            },
            "counterfactual_validation": {
                "appropriate_meaningful": counterfactual_result.appropriate_meaningful,
                "grammatically_correct": counterfactual_result.grammatically_correct,
                "knowledge_reflection_score": counterfactual_result.knowledge_reflection_score,
                "raw_response": counterfactual_result.raw_response,
            },
        }

    def _validate_sample_with_retry(
        self, sample: Dict[str, Any], max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Validate a single sample with retry logic for parallel processing"""

        sample_index = sample.get("index", "unknown")

        for attempt in range(max_retries):
            try:
                # Add rate limiting delay for retries
                if attempt > 0:
                    delay = self.rate_limit_delay * (2**attempt)  # Exponential backoff
                    time.sleep(delay)

                result = self.validate_dataset_sample(sample)

                # Check if we got an empty response and should retry
                if (
                    result
                    and hasattr(result, "raw_response")
                    and (
                        result.raw_response == "<EMPTY_RESPONSE>"
                        or result.appropriate_meaningful.startswith("Error:")
                    )
                ):
                    if attempt < max_retries - 1:
                        tqdm.write(
                            f"Sample {sample_index}: Empty/error response on attempt {attempt + 1}, retrying..."
                        )
                        continue  # Retry for empty responses
                    else:
                        tqdm.write(
                            f"Sample {sample_index}: Got empty/error response after {max_retries} attempts"
                        )

                # Use tqdm.write for thread-safe logging
                if attempt > 0:
                    tqdm.write(
                        f"Sample {sample_index}: Success on attempt {attempt + 1}"
                    )

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    tqdm.write(
                        f"Sample {sample_index}: Attempt {attempt + 1} failed ({e}), retrying..."
                    )
                else:
                    tqdm.write(
                        f"Sample {sample_index}: Failed after {max_retries} attempts: {e}"
                    )
                    return None

        return None

    def validate_dataset_parallel(
        self,
        dataset_name: str = "francescortu/whoops-aha",
        split: str = "train",
        max_samples: Optional[int] = None,
        output_file: Optional[str] = None,
        save_frequency: int = 10,  # Save after every N samples
        resume: bool = True,
        indices: Optional[List[int]] = None,
        max_workers: Optional[int] = None,
        rate_limit_delay: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Parallel validation of the dataset with rate limiting and error handling"""

        # Use instance settings or override
        workers = max_workers or self.max_workers
        delay = rate_limit_delay or self.rate_limit_delay

        print(
            f"Starting parallel validation with {workers} workers and {delay}s rate limit delay"
        )

        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        # Prepare selection filters
        indices_set = set(indices) if indices else None

        results: List[Dict[str, Any]] = []
        processed_indices: set[int] = set()

        # Try to resume from existing file
        if resume and output_file and os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    results = json.load(f)
                for r in results:
                    if isinstance(r, dict) and "index" in r:
                        processed_indices.add(r["index"])
                print(f"Resuming with {len(results)} existing results")
            except Exception as e:
                print(f"Could not resume from {output_file}: {e}")
                results = []
                processed_indices = set()

        # Collect samples to process
        samples_to_process = []
        for i, sample in enumerate(dataset):
            sample_index = sample.get("index", i)

            # Apply filters
            if indices_set is not None and sample_index not in indices_set:
                continue
            if sample_index in processed_indices:
                continue
            if max_samples is not None and len(samples_to_process) >= max_samples:
                break

            samples_to_process.append(sample)

        print(f"Processing {len(samples_to_process)} samples in parallel...")

        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks with submission progress
            future_to_sample = {}

            tqdm.write("Submitting tasks to thread pool...")
            for i, sample in enumerate(
                tqdm(samples_to_process, desc="Submitting tasks", unit="task")
            ):
                # Add staggered delay to respect rate limits
                if i > 0:  # Don't delay the first task
                    time.sleep(delay)
                future = executor.submit(self._validate_sample_with_retry, sample)
                future_to_sample[future] = sample

            # Collect results as they complete
            processed_count = 0
            failed_count = 0

            # Enhanced progress bar with success/failure tracking
            with tqdm(
                total=len(samples_to_process),
                desc="Processing samples",
                unit="sample",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    sample_index = sample.get("index", "unknown")

                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            processed_indices.add(result["index"])
                            processed_count += 1
                        else:
                            failed_count += 1

                        # Update progress bar description with current stats
                        pbar.set_description(
                            f"Processing samples (✓{processed_count} ✗{failed_count})"
                        )

                        # Save intermediate results
                        if output_file and processed_count % save_frequency == 0:
                            self._save_results(results, output_file)
                            tqdm.write(
                                f"Progress saved: {processed_count} processed, {failed_count} failed"
                            )

                    except Exception as e:
                        failed_count += 1
                        tqdm.write(
                            f"Error collecting result for sample {sample_index}: {e}"
                        )

                    pbar.update(1)

        print(
            f"Parallel validation completed: {processed_count} successful, {failed_count} failed"
        )

        # Save final results
        if output_file:
            self._save_results(results, output_file)
            print(f"Final results saved to {output_file}")

        return results

    def validate_dataset(
        self,
        dataset_name: str = "francescortu/whoops-aha",
        split: str = "train",
        max_samples: Optional[int] = None,
        output_file: Optional[str] = None,
        save_frequency: int = 1,  # Save after every N samples
        resume: bool = True,  # Try to resume from existing file
        indices: Optional[
            List[int]
        ] = None,  # Only validate these dataset indices if provided
    ) -> List[Dict[str, Any]]:
        """Validate the entire dataset or a subset with incremental saving"""

        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        # Prepare selection filters
        indices_set = set(indices) if indices else None

        results: List[Dict[str, Any]] = []
        processed_indices: set[int] = set()

        # Try to resume from existing file
        if resume and output_file and os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    results = json.load(f)
                # Build set of already processed sample indices to skip them safely
                for r in results:
                    if isinstance(r, dict) and "index" in r:
                        processed_indices.add(r["index"])
                print(
                    f"Resuming with {len(results)} existing results (will skip {len(processed_indices)} indices already processed)"
                )
            except (json.JSONDecodeError, Exception) as e:
                print(f"Could not resume from {output_file}: {e}")
                results = []
                processed_indices = set()

        # Iterate dataset and process only requested/remaining indices
        processed_count = 0
        total_to_process = None
        if indices_set is not None:
            # Only those indices that are not already processed
            remaining = [idx for idx in indices_set if idx not in processed_indices]
            total_to_process = len(remaining)
        elif max_samples is not None:
            total_to_process = max_samples

        processed_since_save = 0
        for i, sample in enumerate(tqdm(dataset, desc="Validating dataset")):
            sample_index = sample.get("index", i)

            # If a subset of indices is provided, skip others
            if indices_set is not None and sample_index not in indices_set:
                continue

            # Skip already processed indices when resuming
            if sample_index in processed_indices:
                continue

            try:
                result = self.validate_dataset_sample(sample)
                results.append(result)
                processed_indices.add(sample_index)
                processed_count += 1
                processed_since_save += 1

                # Save intermediate results more frequently
                if output_file and processed_since_save >= save_frequency:
                    self._save_results(results, output_file)
                    print(
                        f"Progress saved: {processed_count} samples processed in this run (total results: {len(results)})"
                    )
                    processed_since_save = 0

            except Exception as e:
                print(f"Error processing sample {sample.get('index', 'unknown')}: {e}")
                # Save progress even on error
                if output_file:
                    self._save_results(results, output_file)
                continue

            # Respect max_samples if provided (applies after filtering by indices)
            if max_samples is not None and processed_count >= max_samples:
                break

        if total_to_process is not None:
            print(
                f"Requested to process up to: {total_to_process} samples; actually processed: {processed_count}"
            )

        # Save final results
        if output_file:
            self._save_results(results, output_file)
            print(f"Final results saved to {output_file}")

        return results

    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Helper method to safely save results to file"""
        try:
            # Write to temporary file first, then rename (atomic operation)
            temp_file = output_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(results, f, indent=2)

            # Atomic rename
            os.rename(temp_file, output_file)
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Example usage of the LLM judge validator"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate whoops-aha dataset using LLM as judge"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'openrouter/google/gemini-2.5-flash-image-preview')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to validate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--openrouter-key",
        type=str,
        default=None,
        help="OpenRouter API key (if not set as environment variable)",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=1,
        help="Save results every N samples (default: 1)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing output file, start fresh",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing for faster validation",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers (default: 3)",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay between API requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of specific dataset indices to validate",
    )

    args = parser.parse_args()

    # Resolve model alias to full model name
    model_name = args.model
    if model_name in MODEL_ALIASES:
        resolved_model = MODEL_ALIASES[model_name]
        print(f"Resolved model alias '{model_name}' -> '{resolved_model}'")
        model_name = resolved_model
    else:
        print(f"Using model name directly: '{model_name}'")

    # Setup model configuration
    api_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or use --openrouter-key"
        )

    config = LiteLLMInferenceModelConfig(
        model_name=model_name,
        openrouter_api_key=api_key,
        temperature=0.1,  # Low temperature for consistent evaluations
        max_new_tokens=1000,  # Increased for detailed JSON responses with reasoning
    )

    # Initialize validator with parallel processing settings
    validator = LLMJudgeValidator(
        config, max_workers=args.max_workers, rate_limit_delay=args.rate_limit_delay
    )

    # Parse indices if provided
    indices = None
    if args.indices:
        try:
            indices = [int(x.strip()) for x in args.indices.split(",")]
            print(f"Validating specific indices: {indices}")
        except ValueError:
            print("Error: Invalid indices format. Use comma-separated integers.")
            return

    # Run validation (parallel or sequential)
    if args.parallel:
        print("Using parallel processing...")
        results = validator.validate_dataset_parallel(
            max_samples=args.max_samples,
            output_file=args.output,
            save_frequency=args.save_frequency,
            resume=not args.no_resume,
            indices=indices,
        )
    else:
        print("Using sequential processing...")
        results = validator.validate_dataset(
            max_samples=args.max_samples,
            output_file=args.output,
            save_frequency=args.save_frequency,
            resume=not args.no_resume,
            indices=indices,
        )

    print(f"Validation completed. {len(results)} samples processed.")

    # Print summary statistics
    factual_appropriate = sum(
        1 for r in results if r["factual_validation"]["appropriate_meaningful"] == "Yes"
    )
    counterfactual_appropriate = sum(
        1
        for r in results
        if r["counterfactual_validation"]["appropriate_meaningful"] == "Yes"
    )

    factual_grammar = sum(
        1 for r in results if r["factual_validation"]["grammatically_correct"] == "Yes"
    )
    counterfactual_grammar = sum(
        1
        for r in results
        if r["counterfactual_validation"]["grammatically_correct"] == "Yes"
    )

    factual_avg_score = sum(
        r["factual_validation"]["knowledge_reflection_score"] for r in results
    ) / len(results)
    counterfactual_avg_score = sum(
        r["counterfactual_validation"]["knowledge_reflection_score"] for r in results
    ) / len(results)

    print("\nSummary Statistics:")
    print("Factual tokens:")
    print(
        f"  - Appropriate: {factual_appropriate}/{len(results)} ({factual_appropriate / len(results) * 100:.1f}%)"
    )
    print(
        f"  - Grammatical: {factual_grammar}/{len(results)} ({factual_grammar / len(results) * 100:.1f}%)"
    )
    print(f"  - Avg knowledge score: {factual_avg_score:.2f}/5")
    print("Counterfactual tokens:")
    print(
        f"  - Appropriate: {counterfactual_appropriate}/{len(results)} ({counterfactual_appropriate / len(results) * 100:.1f}%)"
    )
    print(
        f"  - Grammatical: {counterfactual_grammar}/{len(results)} ({counterfactual_grammar / len(results) * 100:.1f}%)"
    )
    print(f"  - Avg anomaly score: {counterfactual_avg_score:.2f}/5")


if __name__ == "__main__":
    main()
