# Seeing-Knowing

Official repository for the paper _When Seeing Overrides Knowing: Disentangling Knowledge Conflict in Vision-Language Models_.

Canonical dataset: [`francescortu/whoops-aha`](https://huggingface.co/datasets/francescortu/whoops-aha)

## What Is Here

- `script/1_logitlens.py`: top-level wrapper for head identification.
- `script/2_intervention.py`: top-level wrapper for head intervention.
- `script/3_pixel_localization.py`: top-level wrapper for visual localization.
- `script/experiment/`: paper experiment code, including appendix experiments used in the final draft.
- `src/`: shared experiment and evaluation utilities.
- `plots/`: Python plotting helpers plus a checked-in artifact plotting entrypoint.
- `results/`: compact paper artifacts and summaries.

Heavy intermediate localization artifacts are intentionally not tracked in git.

## Installation

Requirements:

- Python `>=3.10,<3.13`
- Poetry
- access to the models used in the paper

Install:

```bash
poetry install
```

## Main Commands

Use one of:

- `llava-hf/llava-v1.6-mistral-7b-hf`
- `google/gemma-3-12b-it`

Head identification:

```bash
poetry run python script/1_logitlens.py --model llava-hf/llava-v1.6-mistral-7b-hf
```

Paired head intervention:

```bash
poetry run python script/2_intervention.py --model llava-hf/llava-v1.6-mistral-7b-hf --use_paired --lambda_values -3 -2.5 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 2.5 3
```

Pixel localization:

```bash
poetry run python script/3_pixel_localization.py --model llava-hf/llava-v1.6-mistral-7b-hf --saliency
```

Appendix experiments:

```bash
poetry run python script/experiment/1_heads_ablation/3_multik.py --model google/gemma-3-12b-it --use_paired --ks 1,5,10,20,30,40,50,60
poetry run python script/experiment/3_mlp_ablation/mlp_ablation.py --model google/gemma-3-12b-it
poetry run python script/experiment/5_segmentation/1_segmentation_attention.py --model llava-hf/llava-v1.6-mistral-7b-hf
```

Plot checked-in artifacts:

```bash
poetry run python plots/example_plots.py --output-dir results/figures
```

## Dataset Notes

The default and canonical dataset source in the experiment code is the Hugging Face dataset `francescortu/whoops-aha`.

Local exported dataset folders from paper submission preparation are treated as derived artifacts, not as the source of truth. In particular, older local ARR exports may contain recompressed image files and should not be used as the canonical reference for reproducing the main results.

The checked-in `results/` tree contains the compact result artifacts:

- selected heads
- paired intervention summaries
- multi-`k` sweeps
- MLP ablation summaries
- localization summary CSVs
- validation summaries
- segmentation summaries
- POPE control results

The experiment scripts write the canonical result tables directly into `results/`.
