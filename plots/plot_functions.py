import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter


def set_custom_theme():
    """Set a custom theme for matplotlib plots similar to the R theme_custom"""
    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams["font.size"] = 13
    mpl.rcParams["axes.titlesize"] = 15
    mpl.rcParams["axes.labelsize"] = 15
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["ytick.labelsize"] = 14
    mpl.rcParams["legend.fontsize"] = 14
    mpl.rcParams["figure.titlesize"] = 16

    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.color"] = "grey"
    mpl.rcParams["grid.alpha"] = 0.3

    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    mpl.rcParams["figure.figsize"] = (8, 6)
    mpl.rcParams["figure.dpi"] = 100

    # Custom colors similar to the R color scheme
    custom_colors = {
        "EOL": "#FFC107",
        "Internal_image": "#D55E00",
        "Last_image": "#5F9ED1",
        "EOI": "#0C8A68",
        "llava": "#009E73",
        "gemma": "#D55E00",
    }

    return custom_colors


def plot_mlp_logit_diff(df, save_path=None):
    """
    Plot MLP logit difference barplot.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing logit attribution data, with columns:
        - label: string with "mlp_out" for filtering
        - position: position value (12 in the example)
        - diff_mean: difference value to plot
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    """
    # Set custom theme
    set_custom_theme()

    # Filter data
    mlp_out_df = df[
        (df["position"] == 12) & (df["label"].str.contains("mlp_out", case=False))
    ]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate x values from 0 to 11
    x_values = np.arange(12)

    # Plot the bars
    ax.bar(x_values, mlp_out_df["diff_mean"], width=0.7, color="#CC6677")

    # Set labels and title
    ax.set_xlabel("Layer", fontsize=15)
    ax.set_ylabel("Logit Diff", fontsize=15)
    ax.set_title("MLP", fontsize=15)

    # Set the y-axis limits
    ax.set_ylim(-1.5, 1.5)

    # Set x-ticks
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(i) for i in range(12)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    return fig, ax


def plot_experiment1_combined(df_llava, df_gemma, save_path=None):
    """
    Plot combined experiment 1 results for LLaVA and Gemma models.

    Parameters:
    -----------
    df_llava : pandas DataFrame
        DataFrame containing LLaVA results with columns:
        - Lambda: intervention parameter
        - Fact Acc: accuracy metric (will be converted to image_diff)
    df_gemma : pandas DataFrame
        DataFrame containing Gemma results with columns:
        - Lambda: intervention parameter
        - Fact Acc: accuracy metric (will be converted to image_diff)
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    """
    # Set custom theme
    custom_colors = set_custom_theme()

    # Prepare data for both models
    for df in [df_llava, df_gemma]:
        if "image_diff" not in df.columns:
            # Rename the column if it doesn't exist
            if "Fact Acc" in df.columns:
                df["image_diff"] = df["Fact Acc"]
            else:
                raise ValueError("Column 'Fact Acc' not found in dataframe")

    # Extract baseline values
    baseline_llava = df_llava[df_llava["Lambda"] == 0]["image_diff"].unique()[0]
    baseline_gemma = df_gemma[df_gemma["Lambda"] == 0]["image_diff"].unique()[0]

    # Filter for Lambda values between -3 and 3
    df_llava = df_llava[(df_llava["Lambda"] >= -3) & (df_llava["Lambda"] <= 3)]
    df_gemma = df_gemma[(df_gemma["Lambda"] >= -3) & (df_gemma["Lambda"] <= 3)]

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot LLaVA data
    ax.plot(
        df_llava["Lambda"],
        df_llava["image_diff"],
        marker="o",
        markersize=8,
        linewidth=2,
        color=custom_colors["llava"],
        label="LLaVA-NeXT",
    )

    # Plot Gemma data
    ax.plot(
        df_gemma["Lambda"],
        df_gemma["image_diff"],
        marker="o",
        markersize=8,
        linewidth=2,
        color=custom_colors["gemma"],
        label="Gemma3",
    )

    # Add baseline reference lines
    ax.axhline(
        y=baseline_llava, linestyle="--", linewidth=1, color=custom_colors["llava"]
    )
    ax.axhline(
        y=baseline_gemma, linestyle="--", linewidth=1, color=custom_colors["gemma"]
    )

    # Set labels
    ax.set_xlabel("λ", fontsize=15)
    ax.set_ylabel("Factual Accuracy (%)", fontsize=15)

    # Set y-axis limits
    ax.set_ylim(15, 85)

    # Add a legend
    ax.legend(loc="best")

    # Add x-axis note
    plt.text(
        0.5,
        -0.15,
        "Enhance Counterfactual Heads                Enhance Factual Heads",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")

    return fig, ax


def plot_experiment2(df, model_name="LLaVA-NeXT", save_path=None):
    """
    Plot experiment 2 results showing factual accuracy vs pixels removed.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing experiment 2 results with columns:
        - ExperimentDesc: type of experiment
        - threshold: percentage of pixels removed (0-100)
        - Fact Acc: accuracy metric (will be converted)
    model_name : str, optional
        Name of the model to display in the title (default: 'LLaVA-NeXT')
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    """
    # Set custom theme
    set_custom_theme()

    # Process data similar to the R code
    # 1. Filter out baseline and get full-pixel counts per condition
    full_pixels = df[(df["ExperimentDesc"] != "baseline") & (df["threshold"] == 0)][
        ["ExperimentDesc", "avg_num_pixel"]
    ].rename(columns={"avg_num_pixel": "full_pixels"})

    # 2. Join and compute percentage removed
    df_plot = df[df["ExperimentDesc"] != "baseline"].merge(
        full_pixels, on="ExperimentDesc", how="left"
    )

    # 3. Convert Fact Acc to factual accuracy

    # 4. Multiply threshold by 100 if needed
    if df_plot["threshold"].max() <= 1:
        df_plot["threshold"] = df_plot["threshold"] * 100

    # Extract the baseline value
    baseline_value = (
         df[df["ExperimentDesc"] == "baseline"]["Fact Acc"].values[0]
    )

    # Add baseline rows for zero threshold
    ablation_types = ["resid_ablation", "resid_ablation_control", "resid_ablation_grad"]
    new_rows = []

    for exp_type in ablation_types:
        new_row = {
            "ExperimentDesc": exp_type,
            "threshold": 0,
            "Fact Acc": baseline_value,
        }
        # Add other needed columns
        for col in df_plot.columns:
            if col not in new_row:
                new_row[col] = None
        new_rows.append(new_row)

    # Append new rows to df_plot
    df_plot = pd.concat([df_plot, pd.DataFrame(new_rows)], ignore_index=True)

    # Map experiment descriptions to labels and styles
    exp_map = {
        "resid_ablation": "Through Attn Heads",
        "resid_ablation_control": "Random",
        "resid_ablation_grad": "Through Gradients",
    }

    color_map = {
        "resid_ablation": "#009E73",
        "resid_ablation_control": "darkgray",
        "resid_ablation_grad": "#D55E00",
    }

    linestyle_map = {
        "resid_ablation": "solid",
        "resid_ablation_control": "dotted",
        "resid_ablation_grad": "solid",
    }

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each experiment type
    for exp_type in ablation_types:
        data = df_plot[df_plot["ExperimentDesc"] == exp_type]
        label = exp_map.get(exp_type, exp_type)
        ax.plot(
            data["threshold"],
            data["Fact Acc"],
            color=color_map.get(exp_type, "blue"),
            linestyle=linestyle_map.get(exp_type, "solid"),
            marker="o",
            markersize=8,
            linewidth=2,
            label=label,
        )

    # Set labels and title
    ax.set_xlabel("% Pixels Removed", fontsize=15)
    ax.set_ylabel("Factual Accuracy (%)", fontsize=15)
    ax.set_title(model_name, fontsize=16)

    # Set axis limits and ticks
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))

    # Add legend
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")

    return fig, ax


def plot_heads_heatmap(df, stats=None, save_path=None):
    """
    Plot attention heads heatmap with optional stats panel.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing head information with columns:
        - Head: head identifier in the format 'LxHy' (where x is layer, y is head)
        - Value: score for each head
    stats : dict or None
        Dictionary containing statistics for bar plot panel
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    """
    # Set custom theme
    set_custom_theme()

    # Process the data
    df2 = df.copy()

    # Extract layer and head numbers from Head column
    df2["layer"] = df2["Head"].str.extract(r"L(\d+)H").astype(int)
    df2["head"] = df2["Head"].str.extract(r"H(\d+)$").astype(int)

    # Process values similar to R code
    df2["Value"] = (df2["Value"] + 0.5) * 100 - 50

    # Identify factual and counterfactual heads
    fact_heads = df2[df2["Value"] > 24.5]
    cfact_heads = df2[df2["Value"] < -24.5]

    # Create a figure with two subplots if stats is provided, otherwise just the heatmap
    if stats is not None:
        fig = plt.figure(figsize=(17, 9))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
        ax_heatmap = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
    else:
        fig, ax_heatmap = plt.subplots(figsize=(12, 9))
        ax_bar = None

    # Create pivot table for heatmap
    heatmap_data = df2.pivot(index="head", columns="layer", values="Value")

    # Set max value for color scale
    lim = max(abs(df2["Value"].max()), abs(df2["Value"].min()))

    # Create custom colormap like in R
    colors = ["#da1e28", "white", "#0072c3"]  # red, white, blue
    cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)

    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        vmin=-lim,
        vmax=lim,
        ax=ax_heatmap,
        cbar_kws={"label": "Factual Accuracy (%)"},
    )

    # Customize heatmap
    ax_heatmap.set_xlabel("Layer", fontsize=28)
    ax_heatmap.set_ylabel("Head", fontsize=28)

    # Customize ticks (showing only every other number)
    n_layers = len(df2["layer"].unique())
    n_heads = len(df2["head"].unique())

    layer_ticks = list(range(0, n_layers, 2))
    head_ticks = list(range(0, n_heads, 2))

    ax_heatmap.set_xticks([i + 0.5 for i in layer_ticks])
    ax_heatmap.set_xticklabels([str(i) for i in layer_ticks], fontsize=22)

    ax_heatmap.set_yticks([i + 0.5 for i in head_ticks])
    ax_heatmap.set_yticklabels([str(i) for i in head_ticks], fontsize=22)

    # Get the colorbar and customize it
    if len(ax_heatmap.collections) > 0 and hasattr(
        ax_heatmap.collections[0], "colorbar"
    ):
        cbar = ax_heatmap.collections[0].colorbar
        if cbar is not None:
            cbar.set_ticks([-lim, -25, 0, 25, lim])
            cbar.set_ticklabels(["Counter-\nfactual", "25", "50", "75", "Factual"])
            if hasattr(cbar, "ax"):
                cbar.ax.tick_params(labelsize=20)
                cbar.ax.set_title("Factual Accuracy (%)", fontsize=22)

    # Add bar plot if stats is provided
    if stats is not None and ax_bar is not None:
        # Create summary dataframe for bar plot
        summary_df = pd.DataFrame(
            {
                "metric": ["Counterfactual", "Factual", "All"],
                "mean": [
                    cfact_heads["value"].mean()
                    if "value" in cfact_heads.columns
                    else cfact_heads["Value"].mean(),
                    fact_heads["value"].mean()
                    if "value" in fact_heads.columns
                    else fact_heads["Value"].mean(),
                    df2["value"].mean()
                    if "value" in df2.columns
                    else df2["Value"].mean(),
                ],
                "se": [
                    cfact_heads["value"].std() / np.sqrt(len(cfact_heads))
                    if "value" in cfact_heads.columns
                    else cfact_heads["Value"].std() / np.sqrt(len(cfact_heads)),
                    fact_heads["value"].std() / np.sqrt(len(fact_heads))
                    if "value" in fact_heads.columns
                    else fact_heads["Value"].std() / np.sqrt(len(fact_heads)),
                    df2["value"].std() / np.sqrt(len(df2))
                    if "value" in df2.columns
                    else df2["Value"].std() / np.sqrt(len(df2)),
                ],
            }
        )

        # Set colors for the bar plot
        colors = ["#DA1E28", "#0072C3", "darkgrey"]  # red, blue, grey

        # Plot bars
        ax_bar.bar(
            summary_df["metric"],
            summary_df["mean"],
            yerr=summary_df["se"],
            color=colors,
            width=0.7,
        )

        # Customize bar plot
        ax_bar.set_ylabel("% Attention to Image", fontsize=28)
        ax_bar.set_xticks(range(len(summary_df["metric"])))
        ax_bar.set_xticklabels(
            summary_df["metric"], fontsize=22, rotation=45, ha="right"
        )
        ax_bar.tick_params(axis="y", labelsize=22)

        # Format y-axis as percentage
        ax_bar.yaxis.set_major_formatter(PercentFormatter(1.0))

        # Remove x-axis label
        ax_bar.set_xlabel("")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# Main function to demonstrate usage of all plot functions
def main():
    # Example usage for each function
    # Note: This is placeholder code that would need to be adapted to your data

    # Example for plot_mlp_logit_diff
    mlp_df = pd.DataFrame(
        {
            "position": [12] * 12,
            "label": ["mlp_out"] * 12,
            "diff_mean": np.random.uniform(-1, 1, 12),
        }
    )
    plot_mlp_logit_diff(mlp_df, save_path="mlp_plot.png")

    # Example for plot_experiment1_combined
    lambda_values = np.arange(-3, 4, 1)
    llava_df = pd.DataFrame(
        {
            "Lambda": lambda_values,
            "Fact Acc": np.random.uniform(20, 80, len(lambda_values)),
        }
    )
    gemma_df = pd.DataFrame(
        {
            "Lambda": lambda_values,
            "Fact Acc": np.random.uniform(20, 80, len(lambda_values)),
        }
    )
    plot_experiment1_combined(llava_df, gemma_df, save_path="experiment1.png")

    # Example for plot_experiment2
    exp_types = ["resid_ablation", "resid_ablation_control", "resid_ablation_grad"]
    thresholds = np.linspace(0, 1, 6)

    exp2_data = []
    for exp in exp_types:
        for threshold in thresholds:
            exp2_data.append(
                {
                    "ExperimentDesc": exp,
                    "threshold": threshold,
                    "Fact Acc": np.random.uniform(20, 80),
                    "avg_num_pixel": 1000 * (1 - threshold),
                }
            )

    # Add baseline row
    exp2_data.append(
        {
            "ExperimentDesc": "baseline",
            "threshold": 0,
            "Fact Acc": 50,
            "avg_num_pixel": 1000,
        }
    )

    exp2_df = pd.DataFrame(exp2_data)
    plot_experiment2(exp2_df, model_name="LLaVA-NeXT", save_path="experiment2.png")

    # Example for plot_heads_heatmap
    n_layers = 32
    n_heads = 32
    heads_data = []

    for layer in range(n_layers):
        for head in range(n_heads):
            heads_data.append(
                {"Head": f"L{layer}H{head}", "Value": np.random.uniform(-0.5, 0.5)}
            )

    heads_df = pd.DataFrame(heads_data)
    plot_heads_heatmap(heads_df, stats=None, save_path="heads_heatmap.png")


if __name__ == "__main__":
    main()
