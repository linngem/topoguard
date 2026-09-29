"""Adoption micro-experiment: a single agent with a controlled inbox.

An agent is shown a case plus the reports of n neighbours, m of which assert a candidate
finding X. Varying n, m and X measures P(adopt X | m, n, plausibility) directly. With n varying
independently, one can tell whether LLMs respond to the FRACTION (m/n) or to the NUMBER (m) of
voices. `keep` condition: the agent already asserted X in its previous report → retention.

The rules measured here are then used to predict the network experiment without fitting
anything on it (see `compare(..., rules=...)`).
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize

from .cases import CASES, LEGACY_CASE_IDS, Case
from .scenario import Scenario, parse, render_user

ADOPT_GRID = {2: (0, 1, 2), 4: (0, 1, 2, 3, 4), 8: (0, 1, 2, 4, 6, 8)}
KEEP_GRID = {4: (0, 2, 4)}
_SEED_ID = {new: old for old, new in LEGACY_CASE_IDS.items()}   # keeps prompts byte-identical


def _inbox(case: Case, target: str | None, n: int, m: int, seed: str) -> dict[str, list[str]]:
    rng = random.Random(seed)
    ids = rng.sample(range(1, 30), n)
    yes = set(rng.sample(ids, m))
    base = case.base_key
    return {f"m{i}": sorted([base, target]) if (i in yes and target) else [base] for i in ids}


def inbox_for(cid: str, target: str | None, n: int, m: int, role, rep: int) -> dict[str, list[str]]:
    return _inbox(CASES[cid], target, n, m, f"{_SEED_ID.get(cid, cid)}-{n}-{m}-{role}-{rep}")


def jobs(cases=None, reps: int = 4):
    cases = cases or list(CASES)
    out = []
    for cid in cases:
        for rep in range(reps):
            for n, ms in ADOPT_GRID.items():
                for m in ms:
                    roles = Case.ROLES if m > 0 else (None,)
                    for role in roles:
                        out.append(("adopt", cid, n, m, role, rep))
            for n, ms in KEEP_GRID.items():
                for m in ms:
                    for role in Case.ROLES:
                        out.append(("keep", cid, n, m, role, rep))
    return out


def run(backend, cases=None, reps: int = 4, out_path="micro.jsonl", workers: int = 32) -> Path:
    js = jobs(cases, reps)
    stats = {"parse_failures": 0}

    def one(job):
        cond, cid, n, m, role, rep = job
        c = CASES[cid]
        scn = Scenario(cid)
        target = c.candidate(role).key if role else None
        inbox = inbox_for(cid, target, n, m, role, rep)
        prev = {c.base_key} | ({target} if cond == "keep" else set())
        txt = backend.complete(scn.system(), render_user(scn, [], inbox, prev), replica=rep)
        claims = parse(txt, scn.vocab)
        if claims is None:
            stats["parse_failures"] += 1
        return {"cond": cond, "case": cid, "n": n, "m": m, "role": role, "rep": rep,
                "claims": sorted(claims) if claims is not None else None}

    with ThreadPoolExecutor(workers) as pool:
        recs = list(pool.map(one, js))
    out_path = Path(out_path)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    print(f"micro: {len(recs)} responses, parse failures={stats['parse_failures']}")
    return out_path


def to_rows(recs: list[dict], plaus: dict[tuple[str, str], float]) -> list[dict]:
    """One row per (response, evaluated finding). At m=0 the same response counts for all 4."""
    rows = []
    for r in recs:
        if r["claims"] is None:
            continue
        c = CASES[r["case"]]
        roles = Case.ROLES if r["role"] is None else (r["role"],)
        for role in roles:
            key = c.candidate(role).key
            rows.append({"cond": r["cond"], "case": r["case"], "n": r["n"],
                         "m": r["m"] if r["role"] is not None else 0, "role": role, "key": key,
                         "plaus": plaus[(r["case"], key)], "y": float(key in r["claims"])})
    return rows


# --- logistic regression (no statsmodels dependency) ---------------------------
def plaus_t(p):
    """Panel plausibility on the log-odds scale (fixed before seeing the Haiku data)."""
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


FEATURES = {
    "fraction":         lambda d: [d["plaus"], d["frac"], d["plaus"] * d["frac"]],
    "count":            lambda d: [d["plaus"], d["m"], d["plaus"] * d["m"]],
    "both":             lambda d: [d["plaus"], d["frac"], d["m"], d["plaus"] * d["frac"], d["plaus"] * d["m"]],
    "plausibility_only": lambda d: [d["plaus"]],
    "social_only":      lambda d: [d["frac"]],
}
COEF_NAMES = {"both": ["intercept", "plausibility (log-odds)", "fraction", "m/8",
                       "plausibility×fraction", "plausibility×m"]}


def _design(rows, model):
    X = []
    for r in rows:
        d = {"plaus": plaus_t(r["plaus"]), "m": r["m"] / 8.0, "frac": r["m"] / r["n"] if r["n"] else 0.0}
        X.append([1.0, *FEATURES[model](d)])
    return np.array(X), np.array([r["y"] for r in rows])


def _fit(X, y, l2: float = 1e-3):
    def f(b):
        z = X @ b
        ll = np.sum(y * z - np.logaddexp(0, z))
        return -ll + l2 * np.sum(b[1:] ** 2)

    def g(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        gr = -(X.T @ (y - p))
        gr[1:] += 2 * l2 * b[1:]
        return gr

    return optimize.minimize(f, np.zeros(X.shape[1]), jac=g, method="L-BFGS-B").x


def _logloss(X, y, b):
    p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class LogitRule:
    model: str
    beta: np.ndarray

    def prob(self, plaus, m, n):
        m = np.asarray(m, float); n = np.asarray(n, float)
        frac = np.divide(m, n, out=np.zeros_like(m), where=n > 0)
        d = {"plaus": np.full_like(m, plaus_t(plaus)), "m": m / 8.0, "frac": frac}
        X = np.column_stack([np.ones_like(m), *FEATURES[self.model](d)])
        return 1 / (1 + np.exp(-(X @ self.beta)))


def fit_rule(rows, model: str) -> LogitRule:
    X, y = _design(rows, model)
    return LogitRule(model, _fit(X, y))


def compare_models(rows, models=("plausibility_only", "social_only", "fraction", "count", "both")):
    """Leave-one-case-out log-loss (lower = better)."""
    cases = sorted({r["case"] for r in rows})
    res = {}
    for model in models:
        losses, ns = [], []
        for c in cases:
            tr = [r for r in rows if r["case"] != c]
            te = [r for r in rows if r["case"] == c]
            b = _fit(*_design(tr, model))
            Xt, yt = _design(te, model)
            losses.append(_logloss(Xt, yt, b) * len(yt)); ns.append(len(yt))
        res[model] = sum(losses) / sum(ns)
    return res


def bootstrap_coefs(rows, model: str, B: int = 300, seed: int = 0):
    """95 % CIs of the coefficients by resampling CASES (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    cases = sorted({r["case"] for r in rows})
    by = {c: [r for r in rows if r["case"] == c] for c in cases}
    draws = []
    for _ in range(B):
        samp = [r for c in rng.choice(cases, len(cases)) for r in by[c]]
        draws.append(_fit(*_design(samp, model)))
    draws = np.array(draws)
    return np.percentile(draws, [2.5, 50, 97.5], axis=0)


@dataclass
class MicroRule:
    """(p_adopt, p_keep) rule for one specific finding, ready for `simulate`."""
    adopt: LogitRule
    keep: LogitRule
    plaus: float

    def p_adopt(self, m, n):
        return self.adopt.prob(self.plaus, m, n)

    def p_keep(self, m, n):
        return self.keep.prob(self.plaus, m, n)


def rule_provider(adopt: LogitRule, keep: LogitRule, plaus: dict[tuple[str, str], float]):
    from .cases import ERROR_ROLE

    def provider(case_id: str, kind: str) -> MicroRule:
        c = CASES[case_id]
        key = c.truth.key if kind == "truth" else c.candidate(ERROR_ROLE[kind]).key
        return MicroRule(adopt, keep, plaus[(case_id, key)])
    return provider


def load(path) -> list[dict]:
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l]
    for r in recs:
        r["case"] = LEGACY_CASE_IDS.get(r["case"], r["case"])
    return recs


@dataclass
class DampedRule:
    """Micro-experiment rule with a network-context damping of adoption:
    logit p_adopt' = logit p_adopt − λ; retention unchanged. λ = 0 is the original rule.
    Motivation: isolated agents over-predict how far errors travel in the network, plausibly
    because in the network each agent also reads competing findings and its own previous report."""
    base: MicroRule
    lam: float

    def p_adopt(self, m, n):
        p = np.clip(self.base.p_adopt(m, n), 1e-9, 1 - 1e-9)
        return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - self.lam)))

    def p_keep(self, m, n):
        return self.base.p_keep(m, n)


def damped_provider(provider, lam: float):
    """Wraps a `rule_provider` so every rule it returns is damped by λ."""
    return lambda case_id, kind: DampedRule(provider(case_id, kind), lam)
