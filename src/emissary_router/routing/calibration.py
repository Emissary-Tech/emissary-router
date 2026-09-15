"""Classifier output -> the probabilities the gateway reasons with.

One global logit shift (AppConfig.confidence_bias) is applied to every pass head; length
heads ("<label>:len", routing/labels.py) carry a normalized output length, not a pass
probability, and are never shifted. With bias 0 the values are returned untouched.
"""
from __future__ import annotations

import math

from emissary_router.routing.labels import LEN_SUFFIX

_EPS = 1e-6


def sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p: float) -> float:
    q = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(q / (1.0 - q))


def shift_probability(p: float, bias: float) -> float:
    """sigmoid(logit(p) + bias); bias 0 returns p unchanged (exact)."""
    if not bias:
        return p
    return sigmoid(logit(p) + bias)


def to_probabilities(values: dict[str, float], data_format: str, bias: float = 0.0) -> dict[str, float]:
    """Turn the classifier response into probabilities.

    data_format "logits": pass heads -> sigmoid(z + bias), length heads -> sigmoid(z).
    data_format "probs":  pass heads -> sigmoid(logit(p) + bias) (unchanged when bias is 0),
                          length heads -> unchanged.
    """
    out: dict[str, float] = {}
    for label, v in values.items():
        v = float(v)
        if data_format == "logits":
            out[label] = sigmoid(v) if label.endswith(LEN_SUFFIX) else sigmoid(v + bias)
        else:
            out[label] = v if label.endswith(LEN_SUFFIX) else shift_probability(v, bias)
    return out
