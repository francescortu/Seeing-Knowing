import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from src.experiment_manager import ExperimentManager
from src.datastatistics import statistics_computer
from easyroutine.interpretability import Intervention
from easyroutine.logger import logger, setup_logging

# Setup logging for the experiment
setup_logging(level="INFO")


@dataclass
class FullExperimentConfig:
    model_name: str
    experiment_tag: str
    dataset_name: str = "francescortu/whoops-aha"
    k_heads: int = 20
    gamma_values: List[float] = field(  # -lambda
        default_factory=lambda: [
            -40,
            -20,
            -10,
            -5,
            -3,
            -2.5,
            -2,
            -1.5,
            -1,
            -0.5,
            0,
            0.5,
            1,
            1.5,
            2,
            2.5,
            3,
            5,
            10,
            20,
            40,
        ]
    )

    ablation_types: List[str] = field(
        default_factory=lambda: [
            "last-row",
            "last-row-img",
            "last-row-text",
            "full",
            "mlp",
        ]
    )
    use_paired: bool = False
    rebalanced_weight: bool = True
    control: bool = False
    debug: bool = False
    debug_samples: int = 10
    output_dir: Path = Path("results/1_heads_ablation")


class FullExperimentRunner:
    def __init__(self, cfg: FullExperimentConfig):
        self.cfg = cfg
        self.manager = ExperimentManager.init(
            model_name=cfg.model_name, tag=cfg.experiment_tag
        )
        if cfg.debug:
            self.manager.config.debug.debug = True
            self.manager.config.debug.debug_samples = cfg.debug_samples
        self.manager.load_dataset_from_hf(cfg.dataset_name)
        self.manager.setup_model()
        self.manager.setup_dataloader()
        self.manager.setup_model_specific_variables(filter_dataloader=True)

    def select_heads(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        if self.cfg.control:
            num_layers = self.manager.model.model_config.num_hidden_layers
            heads_per_layer = self.manager.model.model_config.num_attention_heads
            all_heads = [
                (l, h) for l in range(num_layers) for h in range(heads_per_layer)
            ]
            np.random.shuffle(all_heads)
            k = self.cfg.k_heads
            return all_heads[:k], all_heads[k : 2 * k]
        return self.manager.select_heads(k_heads=self.cfg.k_heads)

    def compute_weights(
        self,
        cfact_heads: List[Tuple[int, int]],
        fact_heads: List[Tuple[int, int]],
        gamma: Optional[float],
        lambda_param: Optional[float],
    ) -> List[Tuple[int, int, float, str]]:
        nh = self.manager.model.model_config.num_attention_heads
        # group by layer
        layer_map = {}
        for l, h in cfact_heads:
            layer_map.setdefault(l, {"cfact": set(), "fact": set(), "other": set()})[
                "cfact"
            ].add(h)
        for l, h in fact_heads:
            layer_map.setdefault(l, {"cfact": set(), "fact": set(), "other": set()})[
                "fact"
            ].add(h)
        for l, groups in layer_map.items():
            other = set(range(nh)) - groups["cfact"] - groups["fact"]
            groups["other"] = other
        weights = []
        for l, groups in layer_map.items():
            N, F, M = len(groups["cfact"]), len(groups["fact"]), len(groups["other"])
            w_c = 1 + (gamma or 0)
            w_f = 1 + (lambda_param or 0)
            if self.cfg.rebalanced_weight and M > 0:
                w_o = (nh - N * w_c - F * w_f) / M
            else:
                w_o = 1.0
            for h in groups["cfact"]:
                weights.append((l, h, w_c, "cfact"))
            for h in groups["fact"]:
                weights.append((l, h, w_f, "fact"))
            for h in groups["other"]:
                weights.append((l, h, w_o, "other"))
        return weights

    def compute_paired_weights(
        self,
        cfact_heads: List[Tuple[int, int]],
        fact_heads: List[Tuple[int, int]],
        gamma: float,
        lambda_param: float,
    ) -> Tuple[List[Tuple[int, int, float, str]], List[Tuple[int, int, float, str]]]:
        img_w = self.compute_weights(cfact_heads, fact_heads, gamma, 0.0)
        txt_w = self.compute_weights(cfact_heads, fact_heads, 0.0, lambda_param)
        return img_w, txt_w

    def set_interventions(
        self, heads: List[Tuple[int, int, float, str]], ablation: str
    ) -> None:
        inters = []
        for l, h, w, ht in heads:
            if ablation == "last-row":
                ipos, tpos, pre = ["last"], ["all"], False
            elif ablation == "full":
                ipos, tpos, pre = ["all"], ["all"], False
            elif ablation == "last-row-img":
                ipos, tpos, pre = ["last"], ["all-image"], False
            elif ablation == "last-row-text":
                ipos, tpos, pre = ["last"], ["all-text"], False
            elif ablation == "last-row-img-presoftmax":
                ipos, tpos, pre = ["last"], ["all-image"], True
            elif ablation == "last-row-text-presoftmax":
                ipos, tpos, pre = ["last"], ["all-text"], True
            elif ablation == "last-row-paired":
                ipos = ["last"]
                tpos = ["all-image"] if ht == "cfact" else ["all-text"]
                pre = False
            elif ablation == "last-row-cfact-only":
                if ht != "cfact":
                    continue
                ipos, tpos, pre = ["last"], ["all"], False
            else:
                raise ValueError(f"Unknown ablation type {ablation}")
            itype = "grid_pre_softmax" if pre else "grid"
            act = f"pattern_L{l}H{h}"
            inters.append(
                Intervention(
                    type=itype,
                    activation=act,
                    token_positions=(ipos, tpos),
                    patching_values="ablation",
                    multiplication_value=w,
                )
            )
        self.manager.model.register_interventions(interventions=inters)

    def set_mlp_intervention(self, gamma: float, lambda_param: float) -> None:
        interventions = [
            Intervention(
                type="full",
                activation="mlp_out_29",
                token_positions=["last"],
                patching_values="ablation",
                multiplication_value=gamma,
            ),
            Intervention(
                type="full",
                activation="mlp_out_30",
                token_positions=["last"],
                patching_values="ablation",
                multiplication_value=gamma,
            ),
            Intervention(
                type="full",
                activation="mlp_out_31",
                token_positions=["last"],
                patching_values="ablation",
                multiplication_value=gamma,
            ),
        ]
        self.manager.model.register_interventions(interventions=interventions)

    def evaluate(
        self, evaluate_generation_quality: bool = False, evaluate_coco: bool = False
    ) -> pd.DataFrame:
        cfact, fact = self.select_heads()
        results = []

        lambda_values = [-g for g in self.cfg.gamma_values]
        print("#################################################")
        print(
            f"Starting evaluation: \n \t - Counterfactual Heads: {cfact} \n \t - Factual Heads: {fact} \n \t - Gamma Values: {self.cfg.gamma_values} \n \t - Lambda Values: {lambda_values} \n \t - Ablation Types: {self.cfg.ablation_types if not self.cfg.use_paired else 'last-row'}  \n \t - Rebalanced Weight: {self.cfg.rebalanced_weight} \n \t - Control: {self.cfg.control} \n \t - Debug: {self.cfg.debug} \n \t - Debug Samples: {self.cfg.debug_samples} \n \t - Evaluate Generation Quality: {evaluate_generation_quality}"
        )
        print("-----------------------------------------------------")
        base_generation_output = None
        if evaluate_generation_quality:
            base_generation_output = self.manager.return_generation_logits()

        for g, lam in zip(self.cfg.gamma_values, lambda_values):
            if self.cfg.use_paired:
                img_w, txt_w = self.compute_paired_weights(cfact, fact, g, lam)
                weights = img_w + txt_w
                ablation_list = ["last-row-paired"]
            else:
                weights = self.compute_weights(cfact, fact, g, lam)
                ablation_list = self.cfg.ablation_types
            for ab in ablation_list:
                self.manager.model.clean_interventions()
                if ab == "mlp":
                    self.set_mlp_intervention(gamma=g, lambda_param=lam)
                    _, data = statistics_computer(
                        model=self.manager.model,
                        dataloader=self.manager.dataloader,
                        write_to_file=False,
                        filename=None,
                        dataset_path=Path(""),
                        given_token_pair=self.manager.token_pair,
                        # return_essential_data=True,
                    )
                else:
                    self.set_interventions(weights, ab)
                    _, data = statistics_computer(
                        model=self.manager.model,
                        dataloader=self.manager.dataloader,
                        write_to_file=False,
                        filename=None,
                        dataset_path=Path(""),
                        given_token_pair=self.manager.token_pair,
                        # return_essential_data=True,
                    )
                if evaluate_generation_quality:
                    data_gen = self.manager.evaluate_generation_quality(
                        base_generation_output
                    )
                    row = {"AblationType": ab, "Lambda": lam, **data, **data_gen}
                elif evaluate_coco:
                    data_coco = self.manager.evaluate_coco()
                    row = {"AblationType": ab, "Lambda": lam, **data, **data_coco}
                else:
                    row = {"AblationType": ab, "Lambda": lam, **data}
                results.append(row)
        return pd.DataFrame(results)

    def run(
        self, evaluate_generation_quality: bool = False, evaluate_coco: bool = False
    ) -> pd.DataFrame:
        df = self.evaluate(
            evaluate_generation_quality=evaluate_generation_quality,
            evaluate_coco=evaluate_coco,
        )
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Create folder structure: output_dir/tag_name/model_experiment_timestamp
        tag_folder = self.cfg.output_dir / self.cfg.experiment_tag
        out = tag_folder / f"{self.cfg.model_name.replace('/', '-')}_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        csv = out / f"{self.cfg.experiment_tag}.csv"
        cfgf = out / "config.json"
        df.to_csv(csv, index=False)
        with open(cfgf, "w") as f:
            json.dump(self.cfg.__dict__, f, default=str, indent=2)
        logger.info(f"Results saved to {csv}")
        return df

    # def run_generation_quality_evaluation(self) -> pd.DataFrame:
    #     """
    #     Run generation quality evaluation for the model under the interventions set in the model.
    #     """
    #     df = self.evaluate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run full head ablation experiments")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--k_heads", type=int, default=None)
    parser.add_argument("--lambd", nargs="+", type=float)
    # parser.add_argument("--lambda", dest="lambda_", nargs="+", type=float)
    parser.add_argument("--ablation_types", nargs="+", type=str)
    parser.add_argument("--use_paired", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_samples", type=int)
    parser.add_argument(
        "--evaluate_generation_quality",
        action="store_true",
        dest="evaluate_generation_quality",
        default=False,
    )
    parser.add_argument(
        "--evaluate_coco", action="store_true", dest="evaluate_coco", default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = FullExperimentConfig(model_name=args.model, experiment_tag=args.tag)
    if args.dataset:
        base.dataset_name = args.dataset
    if args.k_heads:
        base.k_heads = args.k_heads
    if args.lambd:
        base.gamma_values = args.lambd
    if args.ablation_types:
        base.ablation_types = args.ablation_types
    base.use_paired = args.use_paired
    base.control = args.control
    base.rebalanced_weight = False
    if args.debug:
        base.debug = True
    if args.debug_samples:
        base.debug_samples = args.debug_samples
    runner = FullExperimentRunner(base)
    runner.run(
        evaluate_generation_quality=args.evaluate_generation_quality,
        evaluate_coco=args.evaluate_coco,
    )


if __name__ == "__main__":
    main()
