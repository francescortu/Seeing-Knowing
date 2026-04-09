from __future__ import annotations

from pathlib import Path
import re
from typing import Dict

import pandas as pd


MODEL_LABELS = {
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-NeXT",
    "google/gemma-3-12b-it": "Gemma3",
}

MODEL_SLUGS = {
    "llava-hf/llava-v1.6-mistral-7b-hf": "llava-next",
    "google/gemma-3-12b-it": "gemma3",
}

LOCALIZATION_METHODS = {
    "resid_ablation": "Through Attn Heads",
    "resid_ablation_control": "Random",
    "resid_ablation_grad": "Through Gradients",
}


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def model_slug(model_name: str) -> str:
    return MODEL_SLUGS.get(model_name, model_name.replace("/", "-"))


def _parse_head_id(head_id: str) -> tuple[int, int]:
    match = re.search(r"L(\d+)H(\d+)", head_id)
    if match is None:
        raise ValueError(f"Could not parse head identifier: {head_id}")
    return int(match.group(1)), int(match.group(2))


def _factual_accuracy_from_internal(counterfactual_win_pct: pd.Series) -> pd.Series:
    return 100.0 - counterfactual_win_pct.astype(float)


def _ensure_lambda(df: pd.DataFrame) -> pd.Series:
    if "Lambda" in df.columns:
        return df["Lambda"].astype(float)
    if "Gamma" in df.columns:
        return -df["Gamma"].astype(float)
    raise ValueError("Expected either a 'Lambda' or 'Gamma' column")


def normalize_head_selection(
    raw_df: pd.DataFrame, model_name: str, top_k: int = 20
) -> pd.DataFrame:
    normalized = raw_df.copy()
    layers_heads = normalized["Head"].map(_parse_head_id)
    normalized["layer"] = layers_heads.map(lambda item: item[0])
    normalized["head"] = layers_heads.map(lambda item: item[1])
    # The saved raw score is oriented toward the counterfactual token.
    # We invert it here so the public table matches the paper:
    # higher values mean stronger factual preference.
    normalized["factual_accuracy_pct"] = (0.5 - normalized["Value"].astype(float)) * 100
    normalized["factual_preference_pct_points"] = -normalized["Value"].astype(float) * 100
    normalized["model"] = model_label(model_name)
    normalized["model_slug"] = model_slug(model_name)
    normalized = normalized.sort_values(["layer", "head"]).reset_index(drop=True)

    ranked = normalized.sort_values("factual_accuracy_pct", ascending=False).reset_index(
        drop=True
    )
    factual_ids = {
        (row.layer, row.head)
        for row in ranked.head(top_k).itertuples(index=False)
    }
    counterfactual_ids = {
        (row.layer, row.head)
        for row in ranked.tail(top_k).itertuples(index=False)
    }

    def assign_group(row: pd.Series) -> str:
        key = (row["layer"], row["head"])
        if key in factual_ids:
            return "Factual"
        if key in counterfactual_ids:
            return "Counterfactual"
        return "Other"

    normalized["head_group"] = normalized.apply(assign_group, axis=1)
    return normalized[
        [
            "model",
            "model_slug",
            "layer",
            "head",
            "factual_accuracy_pct",
            "factual_preference_pct_points",
            "head_group",
        ]
    ]


def summarize_attention_by_group(
    normalized_heads: pd.DataFrame, full_attention_df: pd.DataFrame
) -> pd.DataFrame:
    merged = full_attention_df.copy()
    merged["layer"] = merged["layer"].astype(int)
    merged["head"] = merged["head"].astype(int)
    merged["attention_to_image_pct"] = merged["value"].astype(float) * 100
    merged = merged.merge(
        normalized_heads[["layer", "head", "head_group", "model", "model_slug"]],
        on=["layer", "head"],
        how="left",
    )

    summary_rows = []
    for group_name in ["Counterfactual", "Factual"]:
        group_df = merged[merged["head_group"] == group_name]
        summary_rows.append(
            {
                "model": normalized_heads["model"].iloc[0],
                "model_slug": normalized_heads["model_slug"].iloc[0],
                "group": group_name,
                "attention_to_image_pct": group_df["attention_to_image_pct"].mean(),
            }
        )

    summary_rows.append(
        {
            "model": normalized_heads["model"].iloc[0],
            "model_slug": normalized_heads["model_slug"].iloc[0],
            "group": "All",
            "attention_to_image_pct": merged["attention_to_image_pct"].mean(),
        }
    )
    return pd.DataFrame(summary_rows)


def normalize_intervention(raw_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    normalized = raw_df.copy()
    normalized["lambda"] = _ensure_lambda(normalized)
    normalized["factual_accuracy_pct"] = _factual_accuracy_from_internal(
        normalized["Image Cfact>Fact"]
    )
    normalized["counterfactual_accuracy_pct"] = normalized["Image Cfact>Fact"].astype(
        float
    )
    if "Text Cfact>Fact" in normalized.columns:
        normalized["text_factual_accuracy_pct"] = _factual_accuracy_from_internal(
            normalized["Text Cfact>Fact"]
        )
    normalized["model"] = model_label(model_name)
    normalized["model_slug"] = model_slug(model_name)
    normalized["intervention_target"] = normalized["AblationType"].replace(
        {
            "last-row-paired": "Paired head intervention",
            "mlp": "MLP intervention",
        }
    )
    normalized["direction"] = normalized["lambda"].map(
        lambda value: (
            "Favor visual evidence"
            if value < 0
            else "Favor parametric knowledge"
            if value > 0
            else "Baseline"
        )
    )
    keep_columns = [
        "model",
        "model_slug",
        "intervention_target",
        "lambda",
        "direction",
        "factual_accuracy_pct",
        "counterfactual_accuracy_pct",
        "Image Valid Examples",
        "Text Valid Examples",
        "mean_kl",
    ]
    keep_columns = [column for column in keep_columns if column in normalized.columns]
    return normalized[keep_columns].sort_values("lambda").reset_index(drop=True)


def normalize_multik(raw_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    normalized = normalize_intervention(raw_df, model_name)
    normalized["num_heads"] = raw_df["k_heads"].astype(int).values
    columns = [
        "model",
        "model_slug",
        "num_heads",
        "lambda",
        "direction",
        "factual_accuracy_pct",
        "counterfactual_accuracy_pct",
    ]
    return normalized[columns].sort_values(["num_heads", "lambda"]).reset_index(
        drop=True
    )


def normalize_localization(raw_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    baseline = raw_df[raw_df["ExperimentDesc"] == "baseline"].iloc[0]
    baseline_factual_accuracy = 100.0 - float(baseline["Image Cfact>Fact"])

    rows = []
    for method_key, method_label in LOCALIZATION_METHODS.items():
        method_df = raw_df[raw_df["ExperimentDesc"] == method_key].copy()
        if method_df.empty:
            continue

        threshold = method_df["threshold"].astype(float)
        pixels_removed = threshold * 100 if threshold.max() <= 1 else threshold
        method_df["pixels_removed_pct"] = pixels_removed
        method_df["factual_accuracy_pct"] = _factual_accuracy_from_internal(
            method_df["Image Cfact>Fact"]
        )

        rows.append(
            pd.DataFrame(
                [
                    {
                        "model": model_label(model_name),
                        "model_slug": model_slug(model_name),
                        "method": method_label,
                        "pixels_removed_pct": 0.0,
                        "factual_accuracy_pct": baseline_factual_accuracy,
                    }
                ]
            )
        )

        rows.append(
            method_df.assign(
                model=model_label(model_name),
                model_slug=model_slug(model_name),
                method=method_label,
            )[
                [
                    "model",
                    "model_slug",
                    "method",
                    "pixels_removed_pct",
                    "factual_accuracy_pct",
                ]
            ]
        )

    normalized = pd.concat(rows, ignore_index=True)
    method_order = ["Through Attn Heads", "Random", "Through Gradients"]
    normalized["method"] = pd.Categorical(
        normalized["method"], categories=method_order, ordered=True
    )
    return normalized.sort_values(["model", "method", "pixels_removed_pct"]).reset_index(
        drop=True
    )


def write_table(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def default_raw_artifact_paths() -> Dict[str, Path]:
    return {
        "heads_llava": Path(
            "results/0_heads_selection/v16_arXiv/llava_2025-07-07_16-26-26/selected_heads.csv"
        ),
        "heads_gemma": Path(
            "results/0_heads_selection/v16_arXiv/gemma_2025-07-03_18-20-56/selected_heads.csv"
        ),
        "heads_attn_llava": Path(
            "results/0_heads_selection/v16_arXiv/llava_2025-07-07_16-26-26/full_attn_to_img.csv"
        ),
        "heads_attn_gemma": Path(
            "results/0_heads_selection/v16_arXiv/gemma_2025-07-03_18-20-56/full_attn_to_img.csv"
        ),
        "intervention_llava": Path(
            "results/1_heads_ablation/v16_arXiv/llava-hf-llava-v1.6-mistral-7b-hf_2025-07-12_19-05-07/v16_arXiv.csv"
        ),
        "intervention_gemma": Path(
            "results/1_heads_ablation/v16_arXiv/google-gemma-3-12b-it_2025-07-14_17-28-15/v16_arXiv.csv"
        ),
        "multik_llava": Path(
            "results/1_heads_ablation/v16_arXiv/multik_llava-hf-llava-v1.6-mistral-7b-hf_2025-07-10_14-01-00/multi_k_results.csv"
        ),
        "multik_gemma": Path(
            "results/1_heads_ablation/v16_arXiv/multik_google-gemma-3-12b-it_2025-07-03_20-21-45/multi_k_results.csv"
        ),
        "mlp_llava": Path(
            "results/1_heads_ablation/v16_MLP/llava-hf-llava-v1.6-mistral-7b-hf_2025-09-03_18-34-40/v16_MLP.csv"
        ),
        "mlp_gemma": Path(
            "results/1_heads_ablation/v16_MLP/google-gemma-3-12b-it_2025-09-04_14-41-05/v16_MLP.csv"
        ),
        "localization_llava": Path(
            "results/2_ImgCfactLocalization/v16_arXiv/llava-hf-llava-v1.6-mistral-7b-hf_2025-07-07_17-25-14/results.csv"
        ),
        "localization_gemma": Path(
            "results/2_ImgCfactLocalization/v16_arXiv/google-gemma-3-12b-it_2025-07-03_18-56-49/results.csv"
        ),
    }


def build_default_paper_tables(output_dir: Path = Path("results/paper_tables")) -> Dict[str, Path]:
    raw_paths = default_raw_artifact_paths()
    output_dir.mkdir(parents=True, exist_ok=True)

    llava_heads = normalize_head_selection(
        pd.read_csv(raw_paths["heads_llava"]),
        "llava-hf/llava-v1.6-mistral-7b-hf",
    )
    gemma_heads = normalize_head_selection(
        pd.read_csv(raw_paths["heads_gemma"]),
        "google/gemma-3-12b-it",
    )
    llava_attention = summarize_attention_by_group(
        llava_heads, pd.read_csv(raw_paths["heads_attn_llava"])
    )
    gemma_attention = summarize_attention_by_group(
        gemma_heads, pd.read_csv(raw_paths["heads_attn_gemma"])
    )

    intervention = pd.concat(
        [
            normalize_intervention(
                pd.read_csv(raw_paths["intervention_llava"]),
                "llava-hf/llava-v1.6-mistral-7b-hf",
            ),
            normalize_intervention(
                pd.read_csv(raw_paths["intervention_gemma"]),
                "google/gemma-3-12b-it",
            ),
        ],
        ignore_index=True,
    )

    localization = pd.concat(
        [
            normalize_localization(
                pd.read_csv(raw_paths["localization_llava"]),
                "llava-hf/llava-v1.6-mistral-7b-hf",
            ),
            normalize_localization(
                pd.read_csv(raw_paths["localization_gemma"]),
                "google/gemma-3-12b-it",
            ),
        ],
        ignore_index=True,
    )

    multik = pd.concat(
        [
            normalize_multik(
                pd.read_csv(raw_paths["multik_llava"]),
                "llava-hf/llava-v1.6-mistral-7b-hf",
            ),
            normalize_multik(
                pd.read_csv(raw_paths["multik_gemma"]),
                "google/gemma-3-12b-it",
            ),
        ],
        ignore_index=True,
    )

    mlp = pd.concat(
        [
            normalize_intervention(
                pd.read_csv(raw_paths["mlp_llava"]),
                "llava-hf/llava-v1.6-mistral-7b-hf",
            ),
            normalize_intervention(
                pd.read_csv(raw_paths["mlp_gemma"]),
                "google/gemma-3-12b-it",
            ),
        ],
        ignore_index=True,
    )

    figure4_intervention = intervention[intervention["lambda"].between(-3, 3)].copy()

    outputs = {
        "figure3_heads_llava": write_table(
            llava_heads, output_dir / "figure3_heads_llava-next.csv"
        ),
        "figure3_heads_gemma": write_table(
            gemma_heads, output_dir / "figure3_heads_gemma3.csv"
        ),
        "figure3_attention_summary_llava": write_table(
            llava_attention, output_dir / "figure3_attention_summary_llava-next.csv"
        ),
        "figure3_attention_summary_gemma": write_table(
            gemma_attention, output_dir / "figure3_attention_summary_gemma3.csv"
        ),
        "figure4_intervention": write_table(
            figure4_intervention, output_dir / "figure4_intervention.csv"
        ),
        "figure4_intervention_full": write_table(
            intervention, output_dir / "figure4_intervention_full.csv"
        ),
        "figure5_localization": write_table(
            localization, output_dir / "figure5_localization.csv"
        ),
        "appendix_multik": write_table(multik, output_dir / "appendix_multik.csv"),
        "appendix_mlp": write_table(mlp, output_dir / "appendix_mlp.csv"),
    }
    return outputs
