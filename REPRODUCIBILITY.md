# Reproducibility

## Scope

This repository is the public paper repository. It includes the experiment code and the compact result artifacts needed to inspect the paper outputs.

Tracked in git:

- code
- configs
- compact CSV and JSON artifacts
- segmentation reference assets

Not tracked in git:

- heavy raw localization maps such as `top_pixels.json`
- local temporary logs
- private exploratory notebooks and scratch experiments

## Canonical Dataset

Use:

```bash
francescortu/whoops-aha
```

This is the default dataset source in the main experiment code and should be treated as the source of truth for reruns.

## Recommended Rerun Order

1. Head identification

```bash
poetry run python script/1_logitlens.py --model llava-hf/llava-v1.6-mistral-7b-hf --tag v16_arXiv
poetry run python script/1_logitlens.py --model google/gemma-3-12b-it --tag v16_arXiv
```

2. Main paired intervention

```bash
poetry run python script/2_intervention.py --model llava-hf/llava-v1.6-mistral-7b-hf --tag v16_arXiv --use_paired
poetry run python script/2_intervention.py --model google/gemma-3-12b-it --tag v16_arXiv --use_paired
```

3. Multi-`k` appendix sweep

```bash
poetry run python script/experiment/1_heads_ablation/3_multik.py --model llava-hf/llava-v1.6-mistral-7b-hf --tag v16_arXiv
poetry run python script/experiment/1_heads_ablation/3_multik.py --model google/gemma-3-12b-it --tag v16_arXiv
```

4. MLP ablation

```bash
poetry run python script/experiment/3_mlp_ablation/mlp_ablation.py --model llava-hf/llava-v1.6-mistral-7b-hf --tag v16_MLP
poetry run python script/experiment/3_mlp_ablation/mlp_ablation.py --model google/gemma-3-12b-it --tag v16_MLP
```

5. Localization

```bash
poetry run python script/3_pixel_localization.py --model llava-hf/llava-v1.6-mistral-7b-hf --tag v16_arXiv --experiments baseline multiple_resid_ablation_with_control
poetry run python script/3_pixel_localization.py --model google/gemma-3-12b-it --tag v16_arXiv --experiments baseline multiple_resid_ablation_with_control
```

6. Plotting

```bash
poetry run python plots/example_plots.py --output-dir results/paper_figures
```

## Validation And Controls

The repository also includes:

- LLM-based and human validation summaries in `results/4_data_validation/`
- segmentation summaries in `results/5_segmentation/seg_attention/`
- POPE control results in `results/6_RebuttalFEB2026/POPE_No_IMG/`

These are archived artifacts for inspection. They are not required for a minimal rerun of the main paper pipeline.

## Lightweight Checks

Useful sanity checks before launching long runs:

```bash
python3 -m py_compile script/1_logitlens.py script/2_intervention.py script/3_pixel_localization.py
python3 -m py_compile script/experiment/0_logit_lens/1_logitlens.py script/experiment/1_heads_ablation/2_full.py script/experiment/2_ImgCfactLocalization/1_full.py script/experiment/3_mlp_ablation/mlp_ablation.py
python3 -m py_compile src/*.py plots/example_plots.py plots/plot_functions.py
```

## Known Constraint

Raw localization maps are large enough to make the repository unwieldy. The checked-in localization CSVs are the summary outputs used for paper inspection; regenerate the raw `top_pixels.json` artifacts locally if you need the full intermediate maps.
