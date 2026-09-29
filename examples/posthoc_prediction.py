"""POST-HOC analyses of the network prediction (not in the Phase-2 pre-registration).

The pre-registered result (analyze_multicase.py, Q3) is unchanged. This script only helps read it:

1. Ungated cells. With a single source, a local k = 2 gate keeps the error at the source by
   construction, so most gated cells sit exactly at the floor (0.10). Tied observations make
   Spearman uninformative there; the correlation is also reported on ungated cells only.
2. Noise ceiling. Split-half reliability of the observed cell means across replicates
   (Spearman–Brown corrected): how high any predictor could correlate given replicate noise.
3. Network damping (exploratory, then tested prospectively). The micro-experiment rule
   over-predicts how far errors travel in the network. One parameter λ lowers the logit of
   adoption in the network (retention unchanged). λ is fitted here on the single-source network
   data and then FROZEN; its prospective test is the two-source experiment, whose data did not
   exist when λ was fixed (see results/two_sources/PREREGISTRATION.md).

    cd examples && PYTHONPATH=.. python posthoc_prediction.py
Writes results/multicase/posthoc_prediction.json.
"""
import json

import numpy as np
from scipy import stats

from run_validation import topologies
from topoguard.validation import compare, error_seeds, load, micro
from topoguard.validation.analysis import _score, observed, simulate_fast
from topoguard.validation.plausibility import load_scores

OUT = "results/multicase"
LAMBDAS = np.round(np.arange(0.0, 3.01, 0.1), 2)


def prediction_rows(recs, topo, provider):
    return compare(recs, topo, rules=provider).rows


def summary(rows, ungated=False):
    rows = [r for r in rows if not (ungated and r["local_k"])]
    res = {}
    for k in sorted({r["error_kind"] for r in rows}):
        rr = [r for r in rows if r["error_kind"] == k]
        res[f"error_{k}"] = _score([r["pred_error"] for r in rr], [r["obs_error"] for r in rr])
    res["truth"] = _score([r["pred_truth"] for r in rows], [r["obs_truth"] for r in rows])
    return res


def noise_ceiling(recs, n_split=1000, seed=0):
    """Split-half Spearman of cell means across replicates, Spearman–Brown corrected."""
    rng = np.random.default_rng(seed)
    cells = {}
    for (case, t, ek, pos, k, rep), v in observed(recs).items():
        cells.setdefault((case, t, ek, pos, k), {})[rep] = v
    out = {}
    for label, sel, what in [("error_plausible", lambda c: c[2] == "plausible", "error"),
                             ("error_plausible_ungated",
                              lambda c: c[2] == "plausible" and not c[4], "error"),
                             ("truth", lambda c: True, "truth"),
                             ("truth_ungated", lambda c: not c[4], "truth")]:
        keys = [c for c in cells if sel(c) and len(cells[c]) >= 2]
        rs = []
        for _ in range(n_split):
            a, b = [], []
            for c in keys:
                reps = list(cells[c])
                rng.shuffle(reps)
                h = len(reps) // 2
                a.append(np.mean([cells[c][r][what] for r in reps[:h]]))
                b.append(np.mean([cells[c][r][what] for r in reps[h:2 * h]]))
            if np.ptp(a) > 0 and np.ptp(b) > 0:
                rs.append(stats.spearmanr(a, b).statistic)
        r = float(np.nanmean(rs))
        out[label] = {"split_half": r, "ceiling": 2 * r / (1 + r), "cells": len(keys)}
    return out


def cell_table(recs, topo, base, rounds=4, n_truth=3):
    """Observed means per cell and damped predictions for every λ in LAMBDAS (fast simulator)."""
    cells = {}
    for (case, t, ek, pos, k, _rep), v in observed(recs).items():
        cells.setdefault((case, t, ek, pos, k), []).append(v)
    table = []
    for (case, t, ek, pos, k), vals in sorted(cells.items(), key=str):
        G = topo[t]
        errs, truth = error_seeds(G, pos, n_truth)
        row = {"case": case, "topology": t, "error_kind": ek, "error_pos": pos, "local_k": k,
               "obs_error": float(np.mean([v["error"] for v in vals])),
               "obs_truth": float(np.mean([v["truth"] for v in vals])), "pred": {}}
        for lam in LAMBDAS:
            prov = micro.damped_provider(base, float(lam))
            row["pred"][float(lam)] = (simulate_fast(G, errs, prov(case, ek), rounds, k),
                                       simulate_fast(G, truth, prov(case, "truth"), rounds, k))
        table.append(row)
    return table


def rows_at(table, lam):
    return [{**{k: v for k, v in r.items() if k != "pred"},
             "pred_error": r["pred"][lam][0], "pred_truth": r["pred"][lam][1]} for r in table]


def fit_lambda(table):
    """λ minimising the mean absolute error over error and truth cells."""
    def mae(lam):
        rr = rows_at(table, lam)
        return float(np.mean([abs(r["pred_error"] - r["obs_error"]) for r in rr]
                             + [abs(r["pred_truth"] - r["obs_truth"]) for r in rr]))
    losses = {float(l): mae(float(l)) for l in LAMBDAS}
    return min(losses, key=losses.get), losses


def main():
    pl = load_scores(f"{OUT}/plausibility.csv")
    rows_m = micro.to_rows(micro.load(f"{OUT}/micro.jsonl"), pl)
    adopt = [r for r in rows_m if r["cond"] == "adopt"]
    keep = [r for r in rows_m if r["cond"] == "keep"]
    base = micro.rule_provider(micro.fit_rule(adopt, "both"), micro.fit_rule(keep, "both"), pl)
    recs = (load("results/anthropic_claude-haiku-4-5-20251001/trials.jsonl")
            + load(f"{OUT}/network_trials.jsonl"))
    topo = topologies(10)

    rows0 = prediction_rows(recs, topo, base)
    res = {"note": "POST-HOC; the pre-registered Q3 result is in network_prediction_summary.json",
           "preregistered_rule": {"all_cells": summary(rows0), "ungated_cells": summary(rows0, True)},
           "noise_ceiling": noise_ceiling(recs)}

    table = cell_table(recs, topo, base)
    lam, losses = fit_lambda(table)
    rows_d = rows_at(table, lam)
    res["damping"] = {"lambda": lam, "fitted_on": "single-source networks, 3 cases",
                      "objective": "mean absolute error over error and truth cells",
                      "simulator": "simulate_fast, n_mc = 2000",
                      "grid": {str(k): v for k, v in losses.items()},
                      "undamped_same_simulator": {"all_cells": summary(rows_at(table, 0.0))},
                      "in_sample": {"all_cells": summary(rows_d),
                                    "ungated_cells": summary(rows_d, True)}}

    # leave-one-case-out: λ fitted on two cases, evaluated on the third
    loco_rows, loco_lams = [], {}
    for held in ("pneumonia", "inferior_mi", "lupus"):
        l_h, _ = fit_lambda([r for r in table if r["case"] != held])
        loco_lams[held] = l_h
        loco_rows += rows_at([r for r in table if r["case"] == held], l_h)
    res["damping"]["leave_one_case_out"] = {"lambdas": loco_lams, "all_cells": summary(loco_rows),
                                            "ungated_cells": summary(loco_rows, True)}
    json.dump(res, open(f"{OUT}/posthoc_prediction.json", "w"), indent=2)

    def show(title, s):
        print(f"  {title}")
        for k, d in s.items():
            print(f"    {k:22} spearman={d['spearman']:.2f} mae={d['mae']:.3f} n={d['n']}")
    print("POST-HOC network prediction (the pre-registered Q3 is unchanged)")
    show("pre-registered rule, all cells", res["preregistered_rule"]["all_cells"])
    show("pre-registered rule, ungated cells", res["preregistered_rule"]["ungated_cells"])
    print("  noise ceiling (Spearman–Brown):")
    for k, d in res["noise_ceiling"].items():
        print(f"    {k:24} {d['ceiling']:.2f} (cells={d['cells']})")
    print(f"  damping λ = {lam}")
    show("undamped rule, same fast simulator, all cells",
         res["damping"]["undamped_same_simulator"]["all_cells"])
    show("damped rule, in sample, all cells", res["damping"]["in_sample"]["all_cells"])
    show(f"damped rule, leave-one-case-out (λ per fold {loco_lams}), all cells",
         res["damping"]["leave_one_case_out"]["all_cells"])
    show("damped rule, leave-one-case-out, ungated cells",
         res["damping"]["leave_one_case_out"]["ungated_cells"])


if __name__ == "__main__":
    main()
