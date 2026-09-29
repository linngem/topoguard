"""Analysis: does the topology predict real propagation?

An *adoption rule* gives, for an agent with n neighbours of which m assert a finding:
    p_adopt(m, n)  probability of adopting it if it did not assert it before
    p_keep(m, n)   probability of keeping it if it already asserted it
Given a rule, a stochastic contagion over the network predicts each finding's reach.

Two sources of rules:
1. `fit_all`: fitted on the network data itself (fraction rule, leave-one-topology-out).
2. `micro.py`: measured in an isolated micro-experiment (one agent, controlled inbox) and then
   used to predict the network WITHOUT fitting anything on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import networkx as nx
import numpy as np
from scipy import optimize, stats

from .harness import cond_key, error_seeds, iter_trials, seeds  # noqa: F401 (seeds re-exported)


class AdoptionRule(Protocol):
    def p_adopt(self, m: np.ndarray, n: np.ndarray) -> np.ndarray: ...
    def p_keep(self, m: np.ndarray, n: np.ndarray) -> np.ndarray: ...


@dataclass
class AdoptionFit:
    """Fraction rule: P(adopt) = σ(s·(m/n − θ)); constant retention."""
    key: str
    theta: float
    slope: float
    keep: float
    base_rate: float
    n_events: int

    def p_adopt_frac(self, f):
        return 1 / (1 + np.exp(-self.slope * (np.asarray(f, float) - self.theta)))

    def p_adopt(self, m, n):
        n = np.asarray(n, float)
        return self.p_adopt_frac(np.divide(m, n, out=np.zeros_like(n), where=n > 0))

    def p_keep(self, m, n):
        return np.full(np.shape(m), self.keep)


def _events(records, keyfn: Callable[[dict], str], seed_field: str):
    by = {(*cond_key(r), r["round"], r["agent"]): r for r in records}
    X, Y, keep = [], [], []
    for (*c, rnd, a), r in by.items():
        if rnd == 0 or r[seed_field]:
            continue
        prev = by.get((*c, rnd - 1, a))
        if prev is None:
            continue
        key = keyfn(r)
        if key in prev["claims"]:
            keep.append(key in r["claims"])
        else:
            X.append(r["inbox_frac"].get(key, 0.0))
            Y.append(key in r["claims"])
    return np.array(X, float), np.array(Y, float), np.array(keep, float)


def fit_adoption(records: list[dict], key, seed_field: str) -> AdoptionFit:
    keyfn = key if callable(key) else (lambda r, k=key: k)
    X, Y, keep = _events(records, keyfn, seed_field)
    if len(X) < 5:
        raise ValueError(f"Too few events to fit: {len(X)}")

    def nll(p):
        th, log_s = p
        q = np.clip(1 / (1 + np.exp(-np.exp(log_s) * (X - th))), 1e-6, 1 - 1e-6)
        return -np.sum(Y * np.log(q) + (1 - Y) * np.log(1 - q))

    # With no adoptions at all the MLE runs to θ→∞, so θ is bounded at 1.5 (≈ "never adopted").
    res = optimize.minimize(nll, x0=[0.5, np.log(5)], method="L-BFGS-B",
                            bounds=[(-0.5, 1.5), (np.log(0.5), np.log(60))])
    th, log_s = res.x
    zero = X == 0
    name = key if isinstance(key, str) else seed_field
    return AdoptionFit(name, float(th), float(np.exp(log_s)),
                       float(keep.mean()) if keep.size else 1.0,
                       float(Y[zero].mean()) if zero.any() else float("nan"), int(len(X)))


def fit_all(records: list[dict]) -> dict[str, AdoptionFit]:
    kinds = sorted({r["error_kind"] for r in records})
    fits = {k: fit_adoption([r for r in records if r["error_kind"] == k],
                            lambda r: r["error_key"], "error_seed") for k in kinds}
    fits["truth"] = fit_adoption(records, lambda r: r["truth_key"], "truth_seed")
    return fits


def simulate(G: nx.Graph, seed_nodes: list, rule: AdoptionRule, rounds: int = 4,
             local_k: int | None = None, n_mc: int = 400, rng=None) -> float:
    """Mean reach (fraction of agents asserting the finding in the last round)."""
    rng = rng or np.random.default_rng(0)
    nodes = list(G)
    idx = {v: i for i, v in enumerate(nodes)}
    A = nx.to_numpy_array(G, nodelist=nodes)
    deg = A.sum(1)
    is_seed = np.zeros(len(nodes), bool)
    is_seed[[idx[s] for s in seed_nodes]] = True
    reach = []
    for _ in range(n_mc):
        s = is_seed.copy()
        for _ in range(rounds):
            cnt = A @ s
            if local_k:
                cnt = np.where(cnt >= local_k, cnt, 0)
            u = rng.random(len(nodes))
            s = is_seed | np.where(s, u < rule.p_keep(cnt, deg), u < rule.p_adopt(cnt, deg))
        reach.append(s.mean())
    return float(np.mean(reach))


def simulate_fast(G: nx.Graph, seed_nodes: list, rule: AdoptionRule, rounds: int = 4,
                  local_k: int | None = None, n_mc: int = 2000, seed: int = 0) -> float:
    """Vectorised version of `simulate` (all Monte-Carlo runs at once). Same model, different
    random stream, so values differ from `simulate` by Monte-Carlo noise only. Used for grid
    searches; the pre-registered analysis keeps `simulate`."""
    rng = np.random.default_rng(seed)
    nodes = list(G)
    idx = {v: i for i, v in enumerate(nodes)}
    A = nx.to_numpy_array(G, nodelist=nodes)
    deg = np.broadcast_to(A.sum(1), (n_mc, len(nodes))).ravel()
    is_seed = np.zeros(len(nodes), bool)
    is_seed[[idx[s] for s in seed_nodes]] = True
    s = np.tile(is_seed, (n_mc, 1))
    for _ in range(rounds):
        cnt = s.astype(float) @ A
        if local_k:
            cnt = np.where(cnt >= local_k, cnt, 0)
        c = cnt.ravel()
        pa = np.asarray(rule.p_adopt(c, deg)).reshape(s.shape)
        pk = np.asarray(rule.p_keep(c, deg)).reshape(s.shape)
        u = rng.random(s.shape)
        s = is_seed | np.where(s, u < pk, u < pa)
    return float(s.mean())


def observed(records: list[dict]) -> dict[tuple, dict[str, float]]:
    """Final reach per trial: 'error' (that trial's error) and 'truth'."""
    out = {}
    for cond, recs in iter_trials(records).items():
        last = max(r["round"] for r in recs)
        fin = [r for r in recs if r["round"] == last]
        ek, tk = recs[0]["error_key"], recs[0]["truth_key"]
        out[cond] = {"error": float(np.mean([ek in r["claims"] for r in fin])),
                     "truth": float(np.mean([tk in r["claims"] for r in fin]))}
    return out


@dataclass
class Comparison:
    rows: list[dict]

    def summary(self) -> dict:
        res = {}
        for k in sorted({r["error_kind"] for r in self.rows}):
            rows = [r for r in self.rows if r["error_kind"] == k]
            res[f"error_{k}"] = _score([r["pred_error"] for r in rows], [r["obs_error"] for r in rows])
        res["truth"] = _score([r["pred_truth"] for r in self.rows], [r["obs_truth"] for r in self.rows])
        return res


def _score(p, o):
    p, o = np.asarray(p), np.asarray(o)
    rho = stats.spearmanr(p, o).statistic if np.ptp(p) > 0 and np.ptp(o) > 0 else float("nan")
    return {"spearman": float(rho), "mae": float(np.mean(np.abs(p - o))), "n": int(len(p))}


RuleProvider = Callable[[str, str], AdoptionRule]   # (case_id, "truth"|error_kind) -> rule


def compare(records: list[dict], topologies: dict[str, nx.Graph], rounds: int | None = None,
            leave_one_topology_out: bool = True, n_truth: int = 3,
            rules: RuleProvider | None = None) -> Comparison:
    """Predicted vs observed per cell (case, topology, error, position, gate).
    With `rules`, external rules are used (e.g. from the micro-experiment) and nothing is fitted."""
    rounds = rounds or max(r["round"] for r in records)
    obs = observed(records)
    cells: dict[tuple, list] = {}
    for (case, t, ek, pos, k, _rep), v in obs.items():
        cells.setdefault((case, t, ek, pos, k), []).append(v)
    fit_cache: dict = {}
    rows = []
    for (case, t, ek, pos, k), vals in sorted(cells.items(), key=str):
        if rules is not None:
            r_err, r_tru = rules(case, ek), rules(case, "truth")
        else:
            if t not in fit_cache:
                train = [r for r in records if r["topology"] != t] if leave_one_topology_out else records
                fit_cache[t] = fit_all(train)
            r_err, r_tru = fit_cache[t][ek], fit_cache[t]["truth"]
        G = topologies[t]
        errs, truth = error_seeds(G, pos, n_truth)      # one source, or two for pos == "pair"
        rows.append({
            "case": case, "topology": t, "error_kind": ek, "error_pos": pos, "local_k": k,
            "n_rep": len(vals),
            "pred_error": simulate(G, errs, r_err, rounds, k),
            "obs_error": float(np.mean([v["error"] for v in vals])),
            "pred_truth": simulate(G, truth, r_tru, rounds, k),
            "obs_truth": float(np.mean([v["truth"] for v in vals])),
        })
    return Comparison(rows)
