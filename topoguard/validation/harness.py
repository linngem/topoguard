"""Trial execution: synchronous message-passing rounds over a topology.

Round 0: each agent answers using only the vignette and its private data.
Round r: it receives what its neighbours said in round r-1.
`local_k` mode: an agent only sees findings asserted by ≥ k of its neighbours
(the k-of-n gate applied locally = complex contagion with an integer threshold).
"""
from __future__ import annotations

import itertools
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx

from .cases import LEGACY_CASE_IDS, LEGACY_TOPOLOGY_NAMES
from .scenario import Scenario

COND_FIELDS = ("case", "topology", "error_kind", "error_pos", "local_k", "replica")


@dataclass(frozen=True)
class Condition:
    topology: str
    error_pos: str            # "hub" | "periphery" (one source) | "pair" (two sources)
    local_k: int | None       # None = unfiltered messages
    replica: int
    error_kind: str = "implausible"
    case: str = "pneumonia"


def seeds(G: nx.Graph, error_pos: str, n_truth: int = 3):
    """Error seed (highest- or lowest-degree node) and the n_truth truth seeds."""
    deg = dict(G.degree)
    order = sorted(G, key=lambda v: (-deg[v], v))
    err = order[0] if error_pos == "hub" else order[-1]
    truth = [v for v in order if v != err][:n_truth]
    return err, truth


def pair_seeds(G: nx.Graph) -> tuple[int, int]:
    """Two error sources placed in the worst case for a k = 2 gate: the pair of nodes with the
    most common neighbours (ties: higher summed degree, then lowest ids). Every common
    neighbour receives the error from two sources and can therefore pass it through the gate."""
    deg = dict(G.degree)
    best = max(itertools.combinations(sorted(G), 2),
               key=lambda p: (len(set(G[p[0]]) & set(G[p[1]])), deg[p[0]] + deg[p[1]],
                              -p[0], -p[1]))
    return best


def error_seeds(G: nx.Graph, error_pos: str, n_truth: int = 3):
    """Error source(s) and truth seeds. error_pos: 'hub' | 'periphery' (one source) or
    'pair' (two sources, see `pair_seeds`)."""
    if error_pos != "pair":
        err, truth = seeds(G, error_pos, n_truth)
        return [err], truth
    errs = list(pair_seeds(G))
    deg = dict(G.degree)
    order = sorted(G, key=lambda v: (-deg[v], v))
    return errs, [v for v in order if v not in errs][:n_truth]


def _inbox(G, v, prev: dict, local_k: int | None) -> dict[str, list[str]]:
    msgs = {f"m{u}": sorted(prev[u]) for u in G[v]}
    if not local_k:
        return msgs
    counts: dict[str, int] = {}
    for keys in msgs.values():
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
    allowed = {k for k, c in counts.items() if c >= local_k}
    return {src: [k for k in keys if k in allowed] for src, keys in msgs.items()}


def run_trial(G: nx.Graph, policy, cond: Condition, rounds: int = 4, n_truth: int = 3,
              workers: int = 10) -> list[dict]:
    scn = Scenario(cond.case, cond.error_kind)
    errs, truth = error_seeds(G, cond.error_pos, n_truth)
    private = {v: ([scn.error_fact] if v in errs else []) + ([scn.truth_fact] if v in truth else [])
               for v in G}
    state = {v: set() for v in G}
    records = []
    with ThreadPoolExecutor(workers) as pool:
        for r in range(rounds + 1):
            prev = dict(state)

            def act(v):
                inbox = _inbox(G, v, prev, cond.local_k) if r > 0 else {}
                rep = hash((cond.replica, r)) & 0xFFFF
                return v, inbox, policy.step(f"m{v}", private[v], inbox, prev[v], rep, scn=scn)

            for v, inbox, out in pool.map(act, list(G)):
                state[v] = out
                records.append({
                    **cond.__dict__, "backend": policy.name, "round": r, "agent": v,
                    "degree": G.degree[v], "error_key": scn.error_key, "truth_key": scn.truth_key,
                    "error_seed": v in errs, "truth_seed": v in truth,
                    "claims": sorted(out),
                    "inbox_frac": {k: (sum(k in ks for ks in inbox.values()) / len(inbox)
                                       if inbox else 0.0) for k in scn.tracked},
                })
    return records


def conditions(topologies, *, error_kinds=("implausible", "plausible"),
               error_pos=("hub", "periphery"), local_k=(None, 2), replicas: int = 3,
               cases=("pneumonia",)):
    return [Condition(t, p, k, rep, ek, c) for c, t, ek, p, k, rep in
            itertools.product(cases, topologies, error_kinds, error_pos, local_k, range(replicas))]


def run_grid(topologies: dict[str, nx.Graph], policy, *, error_kinds=("implausible", "plausible"),
             error_pos=("hub", "periphery"), local_k=(None, 2), replicas: int = 3, rounds: int = 4,
             out_path: str | Path = "trials.jsonl", workers: int = 10, parallel_trials: int = 8,
             progress: bool = True, cases=("pneumonia",)) -> Path:
    """Runs trials in parallel; each trial is written in full when it finishes (the file never
    contains a half-written trial)."""
    out_path = Path(out_path)
    conds = conditions(topologies, error_kinds=error_kinds, error_pos=error_pos,
                       local_k=local_k, replicas=replicas, cases=cases)
    lock = threading.Lock()
    done = 0
    with out_path.open("w", encoding="utf-8") as fh, ThreadPoolExecutor(parallel_trials) as pool:
        futs = {pool.submit(run_trial, topologies[c.topology], policy, c, rounds, 3, workers): c
                for c in conds}
        for f in as_completed(futs):
            recs = f.result()
            with lock:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                if progress:
                    print(f"  trial {done}/{len(conds)}  {futs[f]}", flush=True)
    return out_path


def n_calls(topologies: dict[str, nx.Graph], *, error_kinds=("implausible", "plausible"),
            error_pos=("hub", "periphery"), local_k=(None, 2), replicas: int = 3,
            rounds: int = 4, cases=("pneumonia",)) -> int:
    """Upper bound on model calls for a grid (before caching)."""
    per = sum(len(G) for G in topologies.values()) * (rounds + 1)
    return per * len(error_kinds) * len(error_pos) * len(local_k) * replicas * len(cases)


def load(path: str | Path) -> list[dict]:
    """Loads trial records, normalising older logs (missing fields, Spanish ids)."""
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l]
    for r in recs:
        r.setdefault("error_kind", "implausible")
        r.setdefault("error_key", "derrame_pericardico")
        r.setdefault("case", "pneumonia")
        r.setdefault("truth_key", "hiponatremia")
        r["case"] = LEGACY_CASE_IDS.get(r["case"], r["case"])
        r["topology"] = LEGACY_TOPOLOGY_NAMES.get(r["topology"], r["topology"])
    return recs


def cond_key(r: dict) -> tuple:
    return tuple(r[f] for f in COND_FIELDS)


def iter_trials(records: Iterable[dict]) -> dict[tuple, list[dict]]:
    groups: dict = {}
    for r in records:
        groups.setdefault(cond_key(r), []).append(r)
    return groups
