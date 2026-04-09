from typing import List, Tuple
from easyroutine.interpretability import Intervention


def compute_weights(
    num_attention_heads: int,
    cfact_heads: List[Tuple[int, int]],
    fact_heads: List[Tuple[int, int]],
    gamma: float = 0.0,
    lambda_param: float = 0.0,
    rebalanced: bool = True,
) -> List[Tuple[int, int, float, str]]:
    """
    Compute multiplicative weights for heads.
    Returns list of (layer, head, weight, type) tuples.
    Types: 'cfact', 'fact', 'other'.
    """
    # Group heads by layer
    layer_map = {}
    for layer, head in cfact_heads:
        layer_map.setdefault(layer, {"cfact": set(), "fact": set(), "other": set()})
        layer_map[layer]["cfact"].add(head)
    for layer, head in fact_heads:
        layer_map.setdefault(layer, {"cfact": set(), "fact": set(), "other": set()})
        layer_map[layer]["fact"].add(head)

    # Determine other heads per layer
    for layer, groups in layer_map.items():
        all_heads = set(range(num_attention_heads))
        groups["other"] = all_heads - groups["cfact"] - groups["fact"]

    weights = []
    for layer, groups in layer_map.items():
        N = len(groups["cfact"])
        F = len(groups["fact"])
        M = len(groups["other"])
        w_c = 1 + gamma
        w_f = 1 + lambda_param
        if rebalanced and M > 0:
            w_o = (num_attention_heads - N * w_c - F * w_f) / M
        else:
            w_o = 1.0
        for h in groups["cfact"]:
            weights.append((layer, h, w_c, "cfact"))
        for h in groups["fact"]:
            weights.append((layer, h, w_f, "fact"))
        for h in groups["other"]:
            weights.append((layer, h, w_o, "other"))
    return weights


def compute_paired_weights(
    num_attention_heads: int,
    cfact_heads: List[Tuple[int, int]],
    fact_heads: List[Tuple[int, int]],
    gamma: float,
    lambda_param: float,
    rebalanced: bool = True,
) -> Tuple[List[Tuple[int, int, float, str]], List[Tuple[int, int, float, str]]]:
    """
    Compute separate weights for image (cfact) and text (fact) interventions.
    Returns two lists: (image_weights, text_weights)
    """
    image_weights = compute_weights(
        num_attention_heads, cfact_heads, fact_heads, gamma, 0.0, rebalanced
    )
    text_weights = compute_weights(
        num_attention_heads, cfact_heads, fact_heads, 0.0, lambda_param, rebalanced
    )
    return image_weights, text_weights


def set_interventions(
    model,
    heads: List[Tuple[int, int, float, str]],
    ablation_type: str,
) -> None:
    """
    Register interventions on model based on ablation type.
    """
    interventions = []
    for layer, head, weight, htype in heads:
        if ablation_type == "last-row":
            ipos = ["last"]
            tpos = ["all"]
            presoft = False
        elif ablation_type == "full":
            ipos = ["all"]
            tpos = ["all"]
            presoft = False
        elif ablation_type == "last-row-img":
            ipos = ["last"]
            tpos = ["all-image"]
            presoft = False
        elif ablation_type == "last-row-text":
            ipos = ["last"]
            tpos = ["all-text"]
            presoft = False
        elif ablation_type == "last-row-img-presoftmax":
            ipos = ["last"]
            tpos = ["all-image"]
            presoft = True
        elif ablation_type == "last-row-text-presoftmax":
            ipos = ["last"]
            tpos = ["all-text"]
            presoft = True
        elif ablation_type == "last-row-paired":
            ipos = ["last"]
            tpos = ["all-image"] if htype == "cfact" else ["all-text"]
            presoft = False
        elif ablation_type == "last-row-cfact-only":
            if htype != "cfact":
                continue
            ipos = ["last"]
            tpos = ["all"]
            presoft = False
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

        itype = "grid_pre_softmax" if presoft else "grid"
        act = f"pattern_L{layer}H{head}"
        interventions.append(
            Intervention(
                type=itype,
                activation=act,
                token_positions=(ipos, tpos),
                patching_values="ablation",
                multiplication_value=weight,
            )
        )
    model.register_interventions(interventions=interventions)
 
    
    
    