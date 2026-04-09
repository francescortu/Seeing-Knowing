#!/usr/bin/env python3
"""
Script for running experiments across multiple k_heads values.
"""

import sys
import os
import argparse
import json
import datetime
from pathlib import Path

import pandas as pd

# ensure src imports work
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
from src.paper_results import model_slug, normalize_multik, write_table

import importlib.util


full_path = Path(__file__).parent / "2_full.py"
spec = importlib.util.spec_from_file_location("full_experiment", str(full_path))
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)

FullExperimentConfig = full.FullExperimentConfig
FullExperimentRunner = full.FullExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments for multiple k_heads values"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset name or path"
    )
    parser.add_argument(
        "--tag", type=str, default="multi_k", help="Base experiment tag"
    )
    parser.add_argument(
        "--ks",
        type=str,
        required=True,
        help="Comma-separated list of k_heads values, e.g. 10,20,30",
    )
    parser.add_argument("--gamma", nargs="+", type=float, help="List of gamma values")
    parser.add_argument(
        "--lambda", dest="lambda_", nargs="+", type=float, help="List of lambda values"
    )
    parser.add_argument(
        "--ablation_types", nargs="+", type=str, help="List of ablation types"
    )
    parser.add_argument(
        "--use_paired", action="store_true", help="Use paired gamma-lambda"
    )
    parser.add_argument(
        "--control", action="store_true", help="Select random heads for control"
    )
    parser.add_argument(
        "--no_rebalance_weight",
        action="store_false",
        dest="rebalanced_weight",
        help="Disable rebalancing of other heads weights",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode with limited samples"
    )
    parser.add_argument(
        "--debug_samples", type=int, default=None, help="Number of debug samples"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # parse k_heads list
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    # timestamp and batch directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create folder structure: results/1_heads_ablation/tag_name/model_timestamp
    tag_folder = Path("results/1_heads_ablation") / args.tag
    batch_dir = tag_folder / f"multik_{args.model.replace('/', '-')}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # base config
    base_cfg = FullExperimentConfig(model_name=args.model, experiment_tag=args.tag)
    if args.dataset:
        base_cfg.dataset_name = args.dataset
    if args.gamma:
        base_cfg.gamma_values = args.gamma
    if args.lambda_:
        base_cfg.lambda_values = args.lambda_
    if args.ablation_types:
        base_cfg.ablation_types = args.ablation_types
    base_cfg.use_paired = args.use_paired
    base_cfg.control = args.control
    if args.rebalanced_weight is not None:
        base_cfg.rebalanced_weight = args.rebalanced_weight
    if args.debug:
        base_cfg.debug = True
    if args.debug_samples:
        base_cfg.debug_samples = args.debug_samples

    # Save master config
    master_config = base_cfg.__dict__.copy()
    master_config["k_heads_values"] = ks
    master_config["batch_run_timestamp"] = timestamp
    master_config_path = batch_dir / "master_config.json"
    with open(master_config_path, "w") as f:
        json.dump(master_config, f, default=str, indent=2)

    results_list = []
    # run experiments for each k
    for k in ks:
        cfg = FullExperimentConfig(
            model_name=base_cfg.model_name,
            experiment_tag=f"{args.model.replace('/', '-')}_{args.tag}_k{k}",
        )
        # copy other attributes
        cfg.dataset_name = base_cfg.dataset_name
        cfg.gamma_values = base_cfg.gamma_values
        # cfg.lambda_values = base_cfg.lambda_values
        cfg.ablation_types = base_cfg.ablation_types
        cfg.use_paired = base_cfg.use_paired
        cfg.rebalanced_weight = base_cfg.rebalanced_weight
        cfg.control = base_cfg.control
        cfg.debug = base_cfg.debug
        cfg.debug_samples = base_cfg.debug_samples
        cfg.k_heads = k

        print(f"Running k_heads={k}")
        runner = FullExperimentRunner(cfg)
        df_k = runner.evaluate()
        df_k["k_heads"] = k
        results_list.append(df_k)

        # save individual results and config
        subdir = batch_dir / f"k_{k}"
        subdir.mkdir(exist_ok=True)
        df_k.to_csv(subdir / f"results_k{k}.csv", index=False)
        with open(subdir / "config.json", "w") as f:
            json.dump(cfg.__dict__, f, default=str, indent=2)

    # combine
    if results_list:
        master_df = pd.concat(results_list, ignore_index=True)
        master_csv = batch_dir / "multi_k_results.csv"
        master_df.to_csv(master_csv, index=False)
        write_table(
            normalize_multik(master_df, args.model),
            Path("results/paper_tables")
            / f"appendix_multik_{model_slug(args.model)}.csv",
        )
        print(f"Combined results saved to {master_csv}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
