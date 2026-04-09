# LLM-as-a-Judge Dataset Validation

This directory contains tools for validating the `francescortu/whoops-aha` dataset using Large Language Models as judges, following the same evaluation criteria as human annotators.

## Overview

The validation system evaluates both **factual** and **counterfactual** token completions with three criteria:

1. **Appropriateness and Meaningfulness**: Are the completions appropriate and meaningful in the sentence?
2. **Grammar**: Is the sentence grammatically correct?
3. **Knowledge Reflection**: How well do completions reflect common knowledge (factual) or strange/anomalous things in the image (counterfactual)?

### Key Differences:
- **Factual tokens** are evaluated **WITHOUT** the image (should reflect common knowledge)
- **Counterfactual tokens** are evaluated **WITH** the image (should reflect strange/anomalous elements)

## Files

- `llm_judge_validator.py`: Main validation implementation
- `data_validation.py`: Convenience script with pre-configured models
- `test_litellm_image.py`: Test script to verify image functionality
- `README.md`: This documentation

## Setup

1. **Install dependencies** (if not already done):
   ```bash
   cd /path/to/Seeing-Knowing
   poetry install
   ```

2. **Set up API keys**:
   ```bash
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   ```

## Summary

**✅ Implementation Complete and Working!**

The LLM-as-a-Judge validation system is now fully functional and correctly:

- ✅ **Loads environment variables** from `.env` file using `python-dotenv`
- ✅ **Parses JSON responses** correctly (handles both plain JSON and ```json code blocks)
- ✅ **Processes images** properly using LiteLLM with base64 encoding
- ✅ **Evaluates factual tokens** WITHOUT images (reflecting common knowledge)
- ✅ **Evaluates counterfactual tokens** WITH images (reflecting anomalous elements)
- ✅ **Follows human evaluation criteria** from the original form

## Usage

### Quick Test

First, verify that image functionality works:

```bash
cd /path/to/Seeing-Knowing/script/experiment/4_data_validation
poetry run python test_litellm_image.py
```

### Run Validation

#### Using the convenience script (recommended):

```bash
# Validate 10 samples with Gemini Flash
poetry run python data_validation.py --model gemini-flash --max-samples 10

# Validate 50 samples with GPT-5
poetry run python data_validation.py --model gpt5 --max-samples 50 --output results_gpt5.json

# Available models: gemini-flash, gpt5, claude-3.7-sonnet
```

#### Using the full validator directly:

```bash
poetry run python llm_judge_validator.py \
    --model "openrouter/google/gemini-2.5-flash-image-preview" \
    --max-samples 100 \
    --output validation_results.json \
    --openrouter-key "your_api_key"
```

## Output Format

The validation results are saved as JSON with the following structure:

```json
{
  "index": 0,
  "image_id": "some_id",
  "text": "The boy is holding a",
  "factual_tokens": ["ball", "book", "balloon"],
  "counterfactual_tokens": ["cigar", "smoke", "cigarette"],
  "factual_validation": {
    "appropriate_meaningful": "Yes",
    "grammatically_correct": "Yes",
    "knowledge_reflection_score": 5,
    "raw_response": "..."
  },
  "counterfactual_validation": {
    "appropriate_meaningful": "Some do not make sense",
    "grammatically_correct": "Yes",
    "knowledge_reflection_score": 4,
    "raw_response": "..."
  }
}
```

## Available Models

The system supports any LiteLLM-compatible model via OpenRouter. Pre-configured models include:

- `gemini-flash`: Google Gemini 2.5 Flash (with image support)
- `gpt5`: OpenAI GPT-5 (with image support)
- `claude-3.7-sonnet`: Anthropic Claude 3.7 Sonnet (with image support)

## Evaluation Criteria (from Human Form)

### 1. Appropriateness and Meaningfulness
- **Yes**: All completions are appropriate and meaningful
- **Some do not make sense**: At least one completion fails (ungrammatical, unnatural, or semantically odd)
- **None are appropriate**: All completions fail

### 2. Grammar
- **Yes**: Base sentence is grammatically well-formed
- **No**: Contains grammar errors or is incomplete

### 3. Knowledge Reflection (1-5 scale)
- **For factual tokens** (no image): How well do completions reflect common knowledge?
- **For counterfactual tokens** (with image): How well do completions reflect strange/anomalous things in the image?

Scale:
- 1 = Not at all accurate/appropriate
- 3 = Neutral or partially accurate  
- 5 = Very accurate and appropriate

## Implementation Details

The system uses LiteLLM with the following image format (compatible with OpenAI Vision API):

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Your prompt here"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
    }
]
```

## Troubleshooting

1. **"No API key" error**: Make sure `OPENROUTER_API_KEY` is set in your environment
2. **Model not found**: Check that the model name is correct and available on OpenRouter
3. **Image encoding issues**: The system automatically converts PIL Images to base64 PNG format
4. **JSON parsing errors**: The system has fallback handling for malformed responses

## Example Analysis

After running validation, you can analyze results:

```python
import json

# Load results
with open('validation_results.json') as f:
    results = json.load(f)

# Calculate statistics
factual_appropriate = sum(1 for r in results 
                         if r["factual_validation"]["appropriate_meaningful"] == "Yes")

print(f"Factual appropriateness: {factual_appropriate}/{len(results)} samples")
```
