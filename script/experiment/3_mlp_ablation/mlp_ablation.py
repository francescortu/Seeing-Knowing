import argparse
import importlib.util
from pathlib import Path

from src.paper_results import save_mlp_results


full_path = Path(__file__).parent.parent / "1_heads_ablation" / "2_full.py"
spec = importlib.util.spec_from_file_location("full_experiment", str(full_path))
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)

FullExperimentConfig = full.FullExperimentConfig
FullExperimentRunner = full.FullExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser("Run MLP ablation experiments")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--tag", type=str, default="v16_MLP")
    parser.add_argument(
        "--lambda_values",
        "--lambda",
        nargs="+",
        type=float,
        default=[-50, -25, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 25, 50],
        help="Signed intervention strengths reported as lambda in the saved results.",
    )
    parser.add_argument(
        "--mlp-layers",
        nargs="+",
        type=int,
        default=[29, 30, 31],
        help="MLP layers to ablate.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = FullExperimentConfig(
        model_name=args.model,
        experiment_tag=args.tag,
        gamma_values=[-value for value in args.lambda_values],
        ablation_types=["mlp"],
        mlp_ablation_layers=args.mlp_layers,
    )
    if args.dataset:
        cfg.dataset_name = args.dataset
    if args.debug:
        cfg.debug = True
    if args.debug_samples:
        cfg.debug_samples = args.debug_samples

    runner = FullExperimentRunner(cfg)
    df = runner.run()
    save_mlp_results(df, args.model)


if __name__ == "__main__":
    main()
