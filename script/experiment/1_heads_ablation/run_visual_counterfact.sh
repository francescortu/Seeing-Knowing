#!/bin/bash
# Example script to run heads ablation experiment with Visual-Counterfact dataset

# Run with debug mode to test quickly
poetry run python script/experiment/1_heads_ablation/2_full.py \
    --model "llava-hf/llava-v1.6-mistral-7b-hf" \
    --dataset "mgolov/Visual-Counterfact" \
    --tag "visual_counterfact_test" \
    --debug \
    --debug_samples 10

# Full experiment example (remove --debug flags for production)
# poetry run python script/experiment/1_heads_ablation/2_full.py \
#     --model "llava-hf/llava-v1.6-mistral-7b-hf" \
#     --dataset "mgolov/Visual-Counterfact" \
#     --tag "visual_counterfact_full" \
#     --k_heads 20 \
#     --gamma -50 -25 -10 -5 -3 -2.5 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 2.5 3 5 10 25 50
