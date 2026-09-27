"""Spectral metrics and leader–follower collective response (Savari et al., Sci. Rep. 2026;
Horsevad et al., Nat. Commun. 2022). Pure linear algebra: zero LLM calls."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


def kirchhoff_index(G: nx.Graph) -> float:
    """R_g = N · Σ_{i≥2} 1/λ_i on the combinatorial Laplacian (mean effective resistance)."""
    lam = np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray().astype(float))
    return float(len(G) * np.sum(1.0 / lam[1:]))


def normalized_spectrum(G: nx.Graph) -> np.ndarray:
    """Normalized Laplacian spectrum ∈ [0, 2]: region A (communities), B (motifs), C (bipartiteness)."""
    return np.linalg.eigvalsh(nx.normalized_laplacian_matrix(G).toarray())


def collective_response(G: nx.Graph, leader, omegas, w0: float = 1.0) -> np.ndarray:
    """H²(ω)/N_f: fraction of followers tracking the leader at frequency ω (1 = perfect synchrony).

    dx_i/dt = ω0 Σ_j (a_ij/k_i)(x_j − x_i),  x_leader = u(t)
    H(ω) = (jωI − W_F)^{-1} W_L
    """
    nodes = list(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodes)
    k = A.sum(1)
    W = w0 * (A / k[:, None] - np.eye(len(A)))
    li = nodes.index(leader)
    f = [i for i in range(len(A)) if i != li]
    WF, WL = W[np.ix_(f, f)], W[f, li]
    I = np.eye(len(f))
    omegas = np.atleast_1d(omegas)
    return np.array([np.linalg.norm(np.linalg.solve(1j * w * I - WF, WL)) ** 2 for w in omegas]) / len(f)


def bandwidth(G: nx.Graph, leader, level: float = 0.5, w0: float = 1.0,
              omegas: np.ndarray | None = None) -> float:
    """Frequency at which the collective response drops below `level`.
    Interpretation: how fast an instruction can change and still reach the whole system."""
    omegas = np.logspace(-3, 2, 200) if omegas is None else omegas
    h = collective_response(G, leader, omegas, w0)
    below = np.flatnonzero(h < level)
    return float(omegas[below[0]]) if below.size else float(omegas[-1])


@dataclass
class TopologyReport:
    n: int
    mean_degree: float
    clustering: float
    transitivity: float
    avg_shortest_path: float
    kirchhoff: float
    hub: object
    bandwidth_hub: float
    bandwidth_periphery: float

    @property
    def hub_advantage(self) -> float:
        """How much the system gains when the orchestrator sits at the hub rather than the periphery."""
        return self.bandwidth_hub / max(self.bandwidth_periphery, 1e-12)


def topology_report(G: nx.Graph, w0: float = 1.0) -> TopologyReport:
    if not nx.is_connected(G):
        raise ValueError("The topology must be connected")
    deg = dict(G.degree)
    hub = max(deg, key=deg.get)
    periph = min(deg, key=deg.get)
    return TopologyReport(
        n=len(G),
        mean_degree=2 * G.number_of_edges() / len(G),
        clustering=nx.average_clustering(G),
        transitivity=nx.transitivity(G),
        avg_shortest_path=nx.average_shortest_path_length(G),
        kirchhoff=kirchhoff_index(G),
        hub=hub,
        bandwidth_hub=bandwidth(G, hub, w0=w0),
        bandwidth_periphery=bandwidth(G, periph, w0=w0),
    )
