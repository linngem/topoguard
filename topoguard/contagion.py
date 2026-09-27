"""Linear threshold model (LTM) contagion over the agent topology.

Design use: a correct finding usually enters through several modules at once (redundant
signal); a hallucination usually starts in a single agent. With the right threshold θ the
topology spreads the former and smothers the latter. `firewall_curve` measures that margin."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import networkx as nx
import numpy as np


def ltm_cascade(G: nx.Graph, seeds: Iterable, theta: float | Mapping = 0.5,
                max_steps: int = 1000) -> tuple[float, int]:
    """Synchronous update; an active node stays active. Returns (active fraction, steps)."""
    active = set(seeds)
    th = theta if isinstance(theta, Mapping) else {v: theta for v in G}
    for step in range(1, max_steps + 1):
        new = {v for v in G if v not in active
               and G.degree[v] > 0
               and sum(u in active for u in G[v]) / G.degree[v] >= th[v]}
        if not new:
            return len(active) / len(G), step - 1
        active |= new
    return len(active) / len(G), max_steps


@dataclass
class FirewallPoint:
    theta: float
    truth_reach: float     # corroborated finding seeded at the hub + neighbours
    error_reach: float     # single-agent hallucination (mean over nodes)

    @property
    def margin(self) -> float:
        return self.truth_reach - self.error_reach


def firewall_curve(G: nx.Graph, thetas: Iterable[float] = np.linspace(0.05, 0.95, 19),
                   truth_seed=None, n_truth: int = 3, error_samples: int | None = None,
                   rng: np.random.Generator | None = None) -> list[FirewallPoint]:
    """Real finding: `n_truth` modules assert it at once (the hub and its highest-degree
    neighbours). Error: a single agent. A fixed n_truth keeps the topology comparison fair."""
    rng = rng or np.random.default_rng(0)
    deg = dict(G.degree)
    hub = truth_seed if truth_seed is not None else max(deg, key=deg.get)
    neigh = sorted(G[hub], key=deg.get, reverse=True)[: n_truth - 1]
    truth_seeds = {hub, *neigh}
    nodes = list(G)
    if error_samples and error_samples < len(nodes):
        nodes = list(rng.choice(nodes, error_samples, replace=False))
    out = []
    for th in thetas:
        t, _ = ltm_cascade(G, truth_seeds, th)
        e = float(np.mean([ltm_cascade(G, [v], th)[0] for v in nodes]))
        out.append(FirewallPoint(float(th), t, e))
    return out


def best_threshold(curve: list[FirewallPoint], min_truth: float = 0.9) -> FirewallPoint | None:
    """θ that minimises error spread while keeping reach ≥ min_truth for corroborated findings."""
    ok = [p for p in curve if p.truth_reach >= min_truth]
    return min(ok, key=lambda p: (p.error_reach, -p.theta)) if ok else None
