#!/usr/bin/env python3
"""
Test script for parallel LLM judge validation
"""

import subprocess
import sys
from pathlib import Path


def test_parallel_validation():
    """Test the parallel validation with a small sample"""

    # Test with a small subset first
    cmd = [
        "poetry",
        "run",
        "python",
        "llm_judge_validator.py",
        "--model",
        "openrouter/google/gemini-2.5-flash-image-preview",
        "--max-samples",
        "5",
        "--parallel",
        "--max-workers",
        "2",
        "--rate-limit-delay",
        "0.3",
        "--output",
        "test_parallel_results.json",
        "--save-frequency",
        "2",
    ]

    print("Testing parallel validation with 5 samples...")
    print("Command:", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}")

    except Exception as e:
        print(f"Error running test: {e}")


if __name__ == "__main__":
    test_parallel_validation()
