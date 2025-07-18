


# When Seeing Overrides Knowing: Disentangling Knowledge Conflict in Vision-Language Models
**Dataset: [WHOOPS-AHA!](https://huggingface.co/datasets/francescortu/whoops-aha)**


This repository contains the code and scripts for running experiments on the paper *"When Seeing Overrides Knowing: Disentangling Knowledge Conflict in Vision-Language Models"*. 
## Installation

### Prerequisites

Ensure you have the following installed:

- **Python**: Version `3.8+`
- **Poetry**: For dependency and virtual environment management (version >`1.8`)
- **Git**: For cloning repositories and handling submodules

### Install
```bash
poetry install
```


## Run Experiments
```bash
    - <model_name>: either `llava-hf/llava-v1.6-mistral-7b-hf` or `google/gemma-3-12b-it`
```

### Identification of components and heads
```bash
poetry run python script/experiment/0_logit_lens/1_logitlens.py /
                            --model <model_name>

```

### Intervention
```bash 
poetry run python script/2_intervention.py --model <model_name> --ablation_type last-row-paired  
```

### Pixel Localization
```bash
poetry run python script/3_pixel_localization.py --experiments baseline multiple_resid_ablation_with_control  --model <model_name>
```

### Run All Main Experiments and Produce Plots

To run the main experiments and automatically produce the corresponding plots, use the following commands:

#### Identification of components and heads
```bash
poetry run python script/1_logitlens.py --model <model_name>
```

#### Intervention
```bash
poetry run python script/2_intervention.py --model <model_name> --not_rebalance_weight --ablation_type last-row-paired
```

#### Pixel Localization
```bash
poetry run python script/3_pixel_localization.py --experiments baseline multiple_resid_ablation_with_control --model <model_name>
```

#### Plotting Results
After running the experiments, generate all main plots with:
```bash
poetry run python plots/example_plots.py
```

Replace `<model_name>` with either `llava-hf/llava-v1.6-mistral-7b-hf` or `google/gemma-3-12b-it` as appropriate.