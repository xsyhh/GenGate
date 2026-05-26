from __future__ import annotations

import math


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def preference_bce_loss(margin: float, target_prob: float) -> float:
    pos = max(_sigmoid(margin), 1e-12)
    neg = max(_sigmoid(-margin), 1e-12)
    return -target_prob * math.log(pos) - (1.0 - target_prob) * math.log(neg)


def dpo_margin(policy_a: float, policy_b: float, ref_a: float, ref_b: float, beta: float) -> float:
    return beta * ((policy_a - policy_b) - (ref_a - ref_b))
