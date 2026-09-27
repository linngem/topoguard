"""Formal bounds for the k-of-n gate: the quantitative justification to document in a
technical file (e.g. ISO 14971 risk management / validation).

Model: n modules each emit (or not) a finding. Each module asserts it falsely with
probability p_fp and detects it when true with sensitivity `sens`. Errors of LLMs sharing a
base model or data are correlated: modelled as a beta-binomial with intraclass correlation
ρ (ρ=0 → binomial, true independence)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _tail(n: int, k: int, p: float, rho: float) -> float:
    """P(X ≥ k), X ~ BetaBinom(n, p, ρ)."""
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    if rho == 0 or p in (0.0, 1.0):
        return float(stats.binom.sf(k - 1, n, p))
    s = 1 / rho - 1
    return float(stats.betabinom.sf(k - 1, n, p * s, (1 - p) * s))


def false_acceptance(n: int, k: int, p_fp: float, rho: float = 0.0) -> float:
    """Probability that a hallucination enters the workspace."""
    return _tail(n, k, p_fp, rho)


def true_acceptance(n: int, k: int, sens: float, rho: float = 0.0) -> float:
    """Probability that a real finding enters the workspace."""
    return _tail(n, k, sens, rho)


def effective_n(n: int, rho: float) -> float:
    """Effective number of independent modules (Kish design effect)."""
    return n / (1 + (n - 1) * rho)


def estimate_rho(errors: np.ndarray) -> float:
    """ρ from a binary (cases × modules) matrix of validation errors:
    mean pairwise Pearson correlation between modules."""
    e = np.asarray(errors, float)
    e = e[:, e.std(0) > 0]
    if e.shape[1] < 2:
        return 0.0
    c = np.corrcoef(e.T)
    return float(np.clip(c[np.triu_indices_from(c, 1)].mean(), 0, 0.999))


@dataclass
class GateDesign:
    n: int
    k: int
    false_acceptance: float
    true_acceptance: float
    effective_n: float


def design_gate(n: int, p_fp: float, sens: float, rho: float = 0.0,
                max_false_acceptance: float = 1e-3) -> GateDesign | None:
    """Smallest k meeting the false-acceptance bound (maximises sensitivity)."""
    for k in range(1, n + 1):
        far = false_acceptance(n, k, p_fp, rho)
        if far <= max_false_acceptance:
            return GateDesign(n, k, far, true_acceptance(n, k, sens, rho), effective_n(n, rho))
    return None
