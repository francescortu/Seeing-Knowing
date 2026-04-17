import os
import sys
from dotenv import load_dotenv
from typing import Optional

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
from llm_judge_validator import LLMJudgeValidator  # noqa: E402

MODEL_CONFIGS = {
    "gemini-flash": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="openrouter/google/gemini-2.5-flash-image-preview",
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ),
    },
    "gpt5-mini": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="openrouter/openai/gpt-5-mini",
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ),
    },
    "gpt5": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="openrouter/openai/gpt-5",
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ),
    },
    "claude-3.7-sonnet": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="openrouter/anthropic/claude-3.7-sonnet",
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ),
    },
    "claude-3.5-haiku": {
        "interface": LiteLLMInferenceModel,
        "config": LiteLLMInferenceModelConfig(
            model_name="openrouter/anthropic/claude-3.5-haiku",
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ),
    },
}


def run_validation(
    model_name: str = "gemini-flash",
    max_samples: int = 500,
    output_file: Optional[str] = None,
    save_frequency: int = 1,
    resume: bool = True,
    indices: Optional[list[int]] = None,
):
    """Run validation using specified model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Available models: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]["config"]
    validator = LLMJudgeValidator(config)

    if output_file is None:
        output_file = f"validation_results_{model_name}_{max_samples}samples.json"

    print(f"Running validation with {model_name} on {max_samples} samples...")
    print(f"Save frequency: every {save_frequency} samples")
    print(f"Resume from existing file: {resume}")

    results = validator.validate_dataset(
        max_samples=max_samples,
        output_file=output_file,
        save_frequency=save_frequency,
        resume=resume,
        indices=indices,
    )

    print(f"Validation completed. Results saved to {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run data validation on whoops-aha dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-flash",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use for validation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of samples to validate",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results"
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=1,
        help="Save results every N samples (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Don't resume from existing output file, start fresh",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of dataset indices to process (e.g., 2,6,37)",
    )
    parser.add_argument(
        "--indices-file",
        type=str,
        default=None,
        help="Path to a JSON or TXT file containing indices (JSON: [..]; TXT: one index per line)",
    )

    args = parser.parse_args()

    # Parse indices from CLI
    selected_indices: list[int] | None = None
    if args.indices_file:
        import json as _json

        try:
            if args.indices_file.lower().endswith((".json", ".jsonl")):
                with open(args.indices_file, "r") as f:
                    data = _json.load(f)
                if isinstance(data, dict) and "indices" in data:
                    selected_indices = list(map(int, data["indices"]))
                elif isinstance(data, list):
                    selected_indices = list(map(int, data))
                else:
                    raise ValueError("Unsupported JSON structure for indices")
            else:
                # TXT: one index per line, allow comma separation too
                with open(args.indices_file, "r") as f:
                    lines = f.read().replace("\n", ",")
                selected_indices = [
                    int(x.strip()) for x in lines.split(",") if x.strip()
                ]
        except Exception as e:
            print(f"Failed to load indices from file {args.indices_file}: {e}")
            selected_indices = None
    elif args.indices:
        try:
            selected_indices = [
                int(x.strip()) for x in args.indices.split(",") if x.strip()
            ]
        except Exception as e:
            print(f"Failed to parse --indices: {e}")
            selected_indices = None

    run_validation(
        model_name=args.model,
        max_samples=args.max_samples,
        output_file=args.output,
        save_frequency=args.save_frequency,
        resume=args.resume,
        indices=selected_indices,
    )
