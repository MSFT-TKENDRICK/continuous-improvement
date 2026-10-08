"""Confidence formulas published by TypeSafe for System One answers.

Choice (K options):  clip((K * p_max - 1) / (K - 1), 0, 1)
Score  (K levels):   clip(1 - sum_i p_i * |i - peak| / mean_i |i - (K-1)/2|, 0, 1)

Noul answers carry no confidence on the TypeSafe wire. For gating we derive
|2p - 1| (0 at p=0.5, 1 at p in {0, 1}); this is an s1eval convention, not a
TypeSafe field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _clip(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def choice_confidence(probabilities: Mapping[str, float] | Sequence[float]) -> float:
    ps = list(probabilities.values()) if isinstance(probabilities, Mapping) else list(probabilities)
    k = len(ps)
    if k < 2:
        raise ValueError("choice confidence needs at least 2 options")
    return _clip((k * max(ps) - 1.0) / (k - 1.0))


def score_confidence(probabilities: Sequence[float]) -> float:
    ps = list(probabilities)
    k = len(ps)
    if k < 2:
        raise ValueError("score confidence needs at least 2 levels")
    peak = max(range(k), key=lambda i: (ps[i], -i))
    centre = (k - 1) / 2.0
    spread = sum(abs(i - centre) for i in range(k)) / k
    dispersion = sum(p * abs(i - peak) for i, p in enumerate(ps))
    return _clip(1.0 - dispersion / spread)


def noul_confidence(p_true: float) -> float:
    return _clip(abs(2.0 * p_true - 1.0))
