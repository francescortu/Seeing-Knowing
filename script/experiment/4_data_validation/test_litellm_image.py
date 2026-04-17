#!/usr/bin/env python3
"""
Test script to verify LiteLLM image functionality works with the whoops-aha dataset
"""

import os
import sys
import base64
from io import BytesIO
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

# Add paths to sys.path for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

# Local imports after path modification
from easyroutine.inference.litellm_model_interface import (
    LiteLLMInferenceModel,
    LiteLLMInferenceModelConfig,
)  # noqa: E402


def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def test_litellm_with_image():
    """Test basic LiteLLM functionality with image"""

    # Check if API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        return False

    print("✓ API key found")

    # Initialize model
    try:
        config = LiteLLMInferenceModelConfig(
            model_name="openrouter/google/gemini-2.5-flash-image-preview",
            openrouter_api_key=api_key,
            temperature=0.1,
            max_new_tokens=100,
        )
        model = LiteLLMInferenceModel(config)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        return False

    # Load a sample from the dataset
    try:
        dataset = load_dataset("francescortu/whoops-aha", split="train")
        sample = dataset[0]  # Get first sample
        print(f"✓ Dataset loaded, sample text: '{sample['text']}'")
        print(f"✓ Factual tokens: {sample['factual_tokens']}")
        print(f"✓ Counterfactual tokens: {sample['counterfactual_tokens']}")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return False

    # Test text-only request (factual validation)
    try:
        messages = [
            {
                "role": "user",
                "content": f"Look at this sentence: '{sample['text']}' and these possible completions: {sample['factual_tokens']}. Are these completions appropriate and meaningful? Answer briefly.",
            }
        ]

        response = model.chat(messages)
        text_response = model.get_last_text_from_response(response)
        print(f"✓ Text-only request successful: {text_response[:100]}...")

    except Exception as e:
        print(f"ERROR: Text-only request failed: {e}")
        return False

    # Test image + text request (counterfactual validation)
    try:
        image = sample["image"]
        image_b64 = encode_image_to_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Look at this image and sentence: '{sample['text']}' with these completions: {sample['counterfactual_tokens']}. Do these completions reflect strange or anomalous things in the image? Answer briefly.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ]

        response = model.chat(messages)
        image_response = model.get_last_text_from_response(response)
        print(f"✓ Image + text request successful: {image_response[:100]}...")

    except Exception as e:
        print(f"ERROR: Image + text request failed: {e}")
        return False

    print("\n✓ All tests passed! LiteLLM image functionality is working correctly.")
    return True


if __name__ == "__main__":
    success = test_litellm_with_image()
    if not success:
        sys.exit(1)
    print("\nYou can now run the full validation with:")
    print("poetry run python data_validation.py --model gemini-flash --max-samples 5")
