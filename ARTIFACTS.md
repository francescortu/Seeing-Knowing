# Artifact Map

This file maps the checked-in paper artifacts to the main experiment families.

## Head Selection

- LLaVA-NeXT: `results/0_heads_selection/v16_arXiv/llava_2025-07-07_16-26-26/`
- Gemma3: `results/0_heads_selection/v16_arXiv/gemma_2025-07-03_18-20-56/`

Contents:

- `selected_heads.csv`
- `logit_lens_attn.csv`
- `logit_lens_mlp.csv`
- `full_attn_to_img.csv`
- `stats.json`

## Main Paired Intervention

- LLaVA-NeXT: `results/1_heads_ablation/v16_arXiv/llava-hf-llava-v1.6-mistral-7b-hf_2025-07-12_19-05-07/v16_arXiv.csv`
- Gemma3: `results/1_heads_ablation/v16_arXiv/google-gemma-3-12b-it_2025-07-14_17-28-15/v16_arXiv.csv`

## Multi-k Sweep

- LLaVA-NeXT: `results/1_heads_ablation/v16_arXiv/multik_llava-hf-llava-v1.6-mistral-7b-hf_2025-07-10_14-01-00/multi_k_results.csv`
- Gemma3: `results/1_heads_ablation/v16_arXiv/multik_google-gemma-3-12b-it_2025-07-03_20-21-45/multi_k_results.csv`

## MLP Ablation

- LLaVA-NeXT: `results/1_heads_ablation/v16_MLP/llava-hf-llava-v1.6-mistral-7b-hf_2025-09-03_18-34-40/v16_MLP.csv`
- Gemma3: `results/1_heads_ablation/v16_MLP/google-gemma-3-12b-it_2025-09-04_14-41-05/v16_MLP.csv`

## Localization Summaries

- Attention-head localization:
  - `results/2_ImgCfactLocalization/v16_arXiv/llava-hf-llava-v1.6-mistral-7b-hf_2025-07-07_17-25-14/results.csv`
  - `results/2_ImgCfactLocalization/v16_arXiv/google-gemma-3-12b-it_2025-07-03_18-56-49/results.csv`
- Integrated gradients:
  - `results/2_ImgCfactLocalization/v17_integrated_gradients/llava-hf-llava-v1.6-mistral-7b-hf_2025-10-16_11-31-19/results.csv`
  - `results/2_ImgCfactLocalization/v17_integrated_gradients/google-gemma-3-12b-it_2025-10-22_21-05-46/results.csv`

Raw `top_pixels.json` files are intentionally not committed here because they are large intermediate artifacts.

## Validation

- `results/4_data_validation/`

Includes human validation summaries and model-based validation outputs used in the appendix discussion.

## Segmentation

- `results/5_segmentation/seg_attention/`

Includes per-model segmentation summaries and statistical test outputs.

## POPE Control

- `results/6_RebuttalFEB2026/POPE_No_IMG/`

Includes no-image POPE control results for both paper models.
