"""Original (shared samples) vs re-run (independent agents) for Haiku 4.5's network experiments.

    cd examples && PYTHONPATH=.. python compare_rerun.py [--B 2000]

Writes results/original_shared_samples/comparison/:
  cells.csv       per case × topology × error × position × gate: final reach old / new, Δ, 95 % CI
  by_topology.csv the same pooled over cases and positions
  report.md       the checks behind each claim of the article, old vs new

Control built into the design: in the ring every agent has a different neighbour set, so no two
prompts can coincide and the old run never shared samples there. Ring Δ therefore measures pure
re-sampling noise; a topology whose Δ exceeds it was distorted by the shared samples.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_validation import topologies
from topoguard.validation import compare, load, micro
from topoguard.validation.analysis import iter_trials, observed
from topoguard.validation.plausibility import load_scores

ARCHIVE = Path("results/original_shared_samples")
OLD = {"pneumonia": ARCHIVE / "pneumonia_trials.jsonl",
       "multicase": ARCHIVE / "multicase_network_trials.jsonl"}
# official data (independent agents); a fresh run of rerun_haiku_independent.py can be compared
# instead with --new-dir results/rerun_independent
NEW = {"pneumonia": Path("results/anthropic_claude-haiku-4-5-20251001/trials.jsonl"),
       "multicase": Path("results/multicase/network_trials.jsonl")}
MC = Path("results/multicase")
N = 10


def load_arm(paths):
    return [r for p in paths if p.exists() for r in load(p)]


def cell_values(recs):
    """(case, topology, error_kind, pos, gate) -> list of per-trial final reaches."""
    cells = defaultdict(list)
    for (case, t, ek, pos, k, _rep), v in observed(recs).items():
        cells[(case, t, ek, pos, k or 1)].append(v)
    return cells


def boot_diff(old, new, B, rng):
    """Δ = mean(new) − mean(old), 95 % CI resampling trials within each arm."""
    old, new = np.asarray(old, float), np.asarray(new, float)
    d = [rng.choice(new, len(new)).mean() - rng.choice(old, len(old)).mean() for _ in range(B)]
    return float(new.mean() - old.mean()), *np.percentile(d, [2.5, 97.5])


def exposure(recs):
    """Share of agent-rounds (round ≥ 1) whose prompt coincided with another agent's in the same
    trial and round in the ORIGINAL run: same neighbour set (→ same inbox), same own previous
    report and same private seeds."""
    topo = topologies(N)
    prev = {}
    for cond, rs in iter_trials(recs).items():
        for r in rs:
            prev[(cond, r["round"], r["agent"])] = tuple(sorted(r["claims"]))
    groups = defaultdict(int)
    keys = []
    for cond, rs in iter_trials(recs).items():
        G = topo[cond[1]]
        for r in rs:
            if r["round"] == 0:
                continue
            key = (cond, r["round"], frozenset(G.neighbors(r["agent"])),
                   prev.get((cond, r["round"] - 1, r["agent"])), r["error_seed"], r["truth_seed"])
            groups[key] += 1
            keys.append((cond[1], key))
    tot, dup = defaultdict(int), defaultdict(int)
    for t, key in keys:
        tot[t] += 1
        dup[t] += groups[key] > 1
    return {t: dup[t] / tot[t] for t in tot}


def gate_containment(cells, arm):
    """Share of gated trials (k = 2) in which the error never left its source agent."""
    vals = [v["error"] for (c, t, ek, pos, k), vs in cells.items() if k == 2 for v in vs]
    return float(np.mean([x <= 1 / N + 1e-9 for x in vals])), len(vals)


def q3(recs, topo):
    pl = load_scores(MC / "plausibility.csv")
    rows = micro.to_rows(micro.load(MC / "micro.jsonl"), pl)
    adopt = [r for r in rows if r["cond"] == "adopt"]
    keep = [r for r in rows if r["cond"] == "keep"]
    prov = micro.rule_provider(micro.fit_rule(adopt, "both"), micro.fit_rule(keep, "both"), pl)
    return compare(recs, topo, rules=prov).summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--new-dir", help="compare a fresh re-run (…/pneumonia_trials.jsonl, "
                                      "…/multicase_trials.jsonl) instead of the official data")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = ARCHIVE / "comparison"
    new_paths = ([Path(a.new_dir) / f"{k}_trials.jsonl" for k in OLD] if a.new_dir
                 else list(NEW.values()))
    out.mkdir(parents=True, exist_ok=True)

    old = load_arm(OLD.values())
    new = load_arm(new_paths)
    if not new:
        raise SystemExit("No re-run found: run rerun_haiku_independent.py first.")
    c_old, c_new = cell_values(old), cell_values(new)
    common = sorted(set(c_old) & set(c_new), key=str)
    print(f"cells: {len(common)} in common (old {len(c_old)}, new {len(c_new)})")

    # per cell
    rows = []
    for cell in common:
        row = dict(zip(("case", "topology", "error_kind", "error_pos", "gate_k"), cell))
        for what in ("error", "truth"):
            o = [v[what] for v in c_old[cell]]
            n = [v[what] for v in c_new[cell]]
            d, lo, hi = boot_diff(o, n, a.B, rng)
            row.update({f"{what}_old": np.mean(o), f"{what}_new": np.mean(n),
                        f"{what}_delta": d, f"{what}_lo": lo, f"{what}_hi": hi})
        row.update({"trials_old": len(c_old[cell]), "trials_new": len(c_new[cell])})
        rows.append(row)

    # pooled by topology × error kind × gate
    pooled = defaultdict(lambda: {"old": defaultdict(list), "new": defaultdict(list)})
    for cell in common:
        key = (cell[1], cell[2], cell[4])
        for arm, cells in (("old", c_old), ("new", c_new)):
            for v in cells[cell]:
                pooled[key][arm]["error"].append(v["error"])
                pooled[key][arm]["truth"].append(v["truth"])
    exp = exposure(old)
    by_topo = []
    for (t, ek, k), arms in sorted(pooled.items()):
        row = {"topology": t, "error_kind": ek, "gate_k": k, "shared_prompt_share_old": exp.get(t)}
        for what in ("error", "truth"):
            d, lo, hi = boot_diff(arms["old"][what], arms["new"][what], a.B, rng)
            row.update({f"{what}_old": np.mean(arms["old"][what]),
                        f"{what}_new": np.mean(arms["new"][what]),
                        f"{what}_delta": d, f"{what}_lo": lo, f"{what}_hi": hi})
        by_topo.append(row)

    for path, data in (("cells.csv", rows), ("by_topology.csv", by_topo)):
        with open(out / path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    # claims of the article
    def star_hub_max(cells):
        v = [np.mean([x["error"] for x in vs]) for (c, t, ek, pos, k), vs in cells.items()
             if t == "star" and pos == "hub" and ek == "plausible" and k == 1]
        return max(v) if v else float("nan")
    cont_old, n_old = gate_containment({c: c_old[c] for c in common}, "old")
    cont_new, n_new = gate_containment({c: c_new[c] for c in common}, "new")
    topo = topologies(N)
    q_old = q3([r for r in old if (r["case"], r["topology"]) in {(c[0], c[1]) for c in common}], topo)
    q_new = q3(new, topo)

    L = ["# Haiku 4.5 network: shared samples (original) vs independent agents (re-run)", ""]
    L += ["## Exposure of the original run", "",
          "Share of agent-rounds whose prompt coincided with another agent's (→ shared answer):", ""]
    L += [f"- {t}: {v:.0%}" for t, v in sorted(exp.items(), key=lambda kv: -kv[1])]
    L += ["", "## Claims of the article", "",
          "| check | original | re-run |", "|---|---|---|",
          f"| max final reach, plausible error at star hub, no gate | {star_hub_max(c_old):.0%} | "
          f"{star_hub_max(c_new):.0%} |",
          f"| gated trials (k = 2) where the error stayed in its source | {cont_old:.0%} (n={n_old}) | "
          f"{cont_new:.0%} (n={n_new}) |"]
    for k in ("error_plausible", "error_implausible", "truth"):
        if k in q_old and k in q_new:
            L.append(f"| Q3 micro→network {k}: Spearman / MAE | {q_old[k]['spearman']:.2f} / "
                     f"{q_old[k]['mae']:.3f} | {q_new[k]['spearman']:.2f} / {q_new[k]['mae']:.3f} |")
    L += ["", "## Final reach by topology (pooled over cases and positions)", "",
          "| topology | error | gate k | shared prompts (old) | error reach old → new (Δ, 95 % CI) | "
          "truth reach old → new (Δ, 95 % CI) |", "|---|---|---|---|---|---|"]
    for r in by_topo:
        L.append(f"| {r['topology']} | {r['error_kind']} | {r['gate_k']} | "
                 f"{(r['shared_prompt_share_old'] or 0):.0%} | "
                 f"{r['error_old']:.2f} → {r['error_new']:.2f} ({r['error_delta']:+.2f}, "
                 f"[{r['error_lo']:+.2f}, {r['error_hi']:+.2f}]) | "
                 f"{r['truth_old']:.2f} → {r['truth_new']:.2f} ({r['truth_delta']:+.2f}, "
                 f"[{r['truth_lo']:+.2f}, {r['truth_hi']:+.2f}]) |")
    L += ["", "Ring never shared prompts, so its Δ is the re-sampling noise floor.", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
