"""Analysis of the two-source error experiment (one folder per model in results/two_sources/).

    cd examples && PYTHONPATH=.. python analyze_two_sources.py

For every model × case × topology × error type × gate, from the final round of each trial:
  reach           share of the 10 agents asserting the error (2 sources = 0.20)
  escaped         share of trials in which at least one non-source agent asserts it
  second_order    share of trials in which it reached an agent that is NOT a common neighbour
                  of the two sources, i.e. it passed the gate a second time downstream
  truth_reach     share asserting the real finding
Single-source reach (Haiku 4.5, same case/topology/error/gate, mean over hub and periphery
entry) is added for comparison when available.

Pre-registered prospective test (Haiku 4.5 only): the two-source networks are predicted from the
micro-experiment rule, with nothing fitted on network data, both undamped (λ = 0) and with the
network damping λ frozen in results/multicase/posthoc_prediction.json before these data existed.

Writes results/two_sources/_summary/{cells.csv, pooled.csv, report.md, prediction.json}.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_validation import topologies
from topoguard.validation import compare, load, micro, pair_seeds
from topoguard.validation.analysis import _score, iter_trials, observed
from topoguard.validation.plausibility import load_scores

MC = Path("results/multicase")

ROOT = Path("results/two_sources")
SINGLE = [Path("results/anthropic_claude-haiku-4-5-20251001/trials.jsonl"),
          Path("results/multicase/network_trials.jsonl")]
N = 10


def per_trial(recs, topo):
    out = []
    for cond, rs in iter_trials(recs).items():
        case, t, ek, pos, k, rep = cond
        G = topo[t]
        s1, s2 = pair_seeds(G)
        common = set(G[s1]) & set(G[s2])
        last = max(r["round"] for r in rs)
        fin = {r["agent"]: r for r in rs if r["round"] == last}
        ek_key = rs[0]["error_key"]
        has = {v for v, r in fin.items() if ek_key in r["claims"]}
        others = has - {s1, s2}
        out.append({"case": case, "topology": t, "error_kind": ek, "gate_k": k or 1,
                    "reach": len(has) / N, "escaped": float(bool(others)),
                    "second_order": float(bool(others - common)),
                    "truth_reach": np.mean([rs[0]["truth_key"] in r["claims"] for r in fin.values()]),
                    "n_common": len(common)})
    return out


def single_source(topo):
    recs = [r for p in SINGLE if p.exists() for r in load(p)]
    cells = defaultdict(list)
    for (case, t, ek, pos, k, _r), v in observed(recs).items():
        cells[(case, t, ek, k or 1)].append(v["error"])
    return {k: float(np.mean(v)) for k, v in cells.items()}


def ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    d = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return float(x.mean()), *np.percentile(d, [2.5, 97.5])


def prospective_prediction(recs, topo):
    """Micro rule (λ = 0) versus the frozen damped rule on the two-source networks."""
    pl = load_scores(MC / "plausibility.csv")
    rows = micro.to_rows(micro.load(MC / "micro.jsonl"), pl)
    base = micro.rule_provider(micro.fit_rule([r for r in rows if r["cond"] == "adopt"], "both"),
                               micro.fit_rule([r for r in rows if r["cond"] == "keep"], "both"), pl)
    lam = json.load(open(MC / "posthoc_prediction.json"))["damping"]["lambda"]
    res = {"lambda_frozen": lam}
    for label, prov in (("undamped", base), ("damped", micro.damped_provider(base, lam))):
        cmp = compare(recs, topo, rules=prov).rows
        res[label] = {}
        for k in sorted({r["error_kind"] for r in cmp}):
            rr = [r for r in cmp if r["error_kind"] == k]
            res[label][f"error_{k}"] = _score([r["pred_error"] for r in rr],
                                              [r["obs_error"] for r in rr])
        res[label]["truth"] = _score([r["pred_truth"] for r in cmp], [r["obs_truth"] for r in cmp])
    return res


def main():
    topo = topologies(N)
    out = ROOT / "_summary"
    out.mkdir(parents=True, exist_ok=True)
    single = single_source(topo)
    cells, pooled, report = [], [], ["# Two-source error experiment", ""]

    for d in sorted(p for p in ROOT.glob("*/") if not p.name.startswith("_")):
        if not (d / "trials.jsonl").exists():
            continue
        rows = per_trial(load(d / "trials.jsonl"), topo)
        by = defaultdict(list)
        for r in rows:
            by[(r["case"], r["topology"], r["error_kind"], r["gate_k"])].append(r)
        for (case, t, ek, k), rs in sorted(by.items()):
            cells.append({"model": d.name, "case": case, "topology": t, "error_kind": ek,
                          "gate_k": k, "trials": len(rs), "n_common": rs[0]["n_common"],
                          **{m: float(np.mean([r[m] for r in rs]))
                             for m in ("reach", "escaped", "second_order", "truth_reach")},
                          "single_source_reach": single.get((case, t, ek, k))
                          if d.name.startswith("claude-haiku") else None})
        pool = defaultdict(list)
        for r in rows:
            pool[(r["topology"], r["error_kind"], r["gate_k"])].append(r)
        report += [f"## {d.name}", "",
                   "| topology | common neighbours | error | gate k | reach | escaped (95 % CI) | "
                   "second-order | truth reach |", "|---|---|---|---|---|---|---|---|"]
        for (t, ek, k), rs in sorted(pool.items()):
            e, lo, hi = ci([r["escaped"] for r in rs])
            row = {"model": d.name, "topology": t, "error_kind": ek, "gate_k": k,
                   "trials": len(rs), "reach": float(np.mean([r["reach"] for r in rs])),
                   "escaped": e, "escaped_lo": lo, "escaped_hi": hi,
                   "second_order": float(np.mean([r["second_order"] for r in rs])),
                   "truth_reach": float(np.mean([r["truth_reach"] for r in rs]))}
            pooled.append(row)
            report.append(f"| {t} | {rs[0]['n_common']} | {ek} | {k} | {row['reach']:.2f} | "
                          f"{e:.0%} [{lo:.0%}, {hi:.0%}] | {row['second_order']:.0%} | "
                          f"{row['truth_reach']:.2f} |")
        report.append("")

    if not cells:
        raise SystemExit("No results yet: run run_two_sources.py first.")
    haiku = [d for d in ROOT.glob("claude-haiku-4-5*/") if (d / "trials.jsonl").exists()]
    if haiku:
        pred = prospective_prediction(load(haiku[0] / "trials.jsonl"), topo)
        json.dump(pred, open(out / "prediction.json", "w"), indent=2)
        report += ["## Prospective prediction (Haiku 4.5, nothing fitted on these data)", "",
                   f"λ frozen before the run: {pred['lambda_frozen']}", "",
                   "| finding | undamped ρ · MAE | damped ρ · MAE |", "|---|---|---|"]
        for k in pred["undamped"]:
            u, d = pred["undamped"][k], pred["damped"][k]
            report.append(f"| {k} | {u['spearman']:.2f} · {u['mae']:.3f} | "
                          f"{d['spearman']:.2f} · {d['mae']:.3f} |")
        report.append("")
    for name, data in (("cells.csv", cells), ("pooled.csv", pooled)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    report += ["Reach 0.20 = the two sources only. Escaped: at least one other agent asserts the "
               "error at the end. Second-order: it reached an agent that does not neighbour both "
               "sources, so it passed the gate again downstream.", ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
