"""Practitioner's lookup table and trade-off figure for the k-of-n corroboration gate.

Part 1 (theory, beta-binomial model of `topoguard.reliability`): for n modules, intraclass error
correlation ρ, per-module false-positive rate p_fp and a false-acceptance target, the smallest k
that meets the target, the resulting true-acceptance rate (sensitivity s), and the lowest FAR any
k can reach (k = n). "none" means no k meets the target: more independent modules are needed,
not a higher k.

Part 2 (data, Haiku 4.5 networks): the latency cost of the gate, as the share of agents holding
the real finding after each round, with and without the local k = 2 gate.

    cd examples && PYTHONPATH=.. python make_gate_design.py
Writes results/gate_design/{lookup.csv, lookup.md, lookup_table.tex} and
docs/figures/fig9_gate_tradeoff.png.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topoguard.reliability import design_gate, effective_n, false_acceptance, true_acceptance
from topoguard.validation import load
from topoguard.validation.analysis import iter_trials

OUT = Path("results/gate_design")
FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
NS, RHOS, PFPS, TARGETS, SENS = (3, 5, 10), (0.0, 0.2, 0.5), (0.01, 0.05), (1e-3, 1e-4), 0.9
NETWORK = [Path("results/anthropic_claude-haiku-4-5-20251001/trials.jsonl"),
           Path("results/multicase/network_trials.jsonl")]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.titlelocation": "left", "legend.frameon": False,
})


def lookup():
    rows = []
    for p_fp in PFPS:
        for target in TARGETS:
            for rho in RHOS:
                for n in NS:
                    g = design_gate(n, p_fp, SENS, rho, target)
                    rows.append({"p_fp": p_fp, "target_far": target, "rho": rho, "n": n,
                                 "n_eff": round(effective_n(n, rho), 2),
                                 "k": g.k if g else None,
                                 "far": g.false_acceptance if g else None,
                                 "tar": g.true_acceptance if g else None,
                                 "floor_far": false_acceptance(n, n, p_fp, rho)})
    return rows


def write_tables(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "lookup.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def cell(r):
        return (f"k={r['k']} (TAR {r['tar']:.2f})" if r["k"]
                else f"none (min FAR {r['floor_far']:.1e})")

    md = [f"# k-of-n gate lookup (true-positive sensitivity s = {SENS})", ""]
    tex = [r"\begin{tabular}{llccc}", r"\toprule",
           r"$p_{\mathrm{fp}}$, target FAR & $\rho$ & $n=3$ & $n=5$ & $n=10$ \\", r"\midrule"]
    for p_fp in PFPS:
        for target in TARGETS:
            md += [f"## p_fp = {p_fp}, FAR < {target:g}", "",
                   "| ρ | n = 3 | n = 5 | n = 10 |", "|---|---|---|---|"]
            for i, rho in enumerate(RHOS):
                rr = [r for r in rows if r["p_fp"] == p_fp and r["target_far"] == target
                      and r["rho"] == rho]
                md.append(f"| {rho} | " + " | ".join(cell(r) for r in rr) + " |")
                label = (f"{p_fp:g}, $10^{{{int(np.log10(target))}}}$" if i == 0 else "")
                tex.append(f"{label} & {rho:g} & " + " & ".join(
                    (f"{r['k']} ({r['tar']:.2f})" if r["k"] else r"--") for r in rr) + r" \\")
            md.append("")
            tex.append(r"\midrule" if (p_fp, target) != (PFPS[-1], TARGETS[-1]) else r"\bottomrule")
    tex.append(r"\end{tabular}")
    (OUT / "lookup.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "lookup_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(md))


def latency():
    """Share of non-seed agents holding the truth after each round, gate vs no gate."""
    recs = [r for p in NETWORK if p.exists() for r in load(p)]
    by = defaultdict(list)
    for cond, rs in iter_trials(recs).items():
        k = cond[4] or 1
        for rnd in sorted({r["round"] for r in rs}):
            fin = [r for r in rs if r["round"] == rnd]
            by[(k, rnd)].append(np.mean([r["truth_key"] in r["claims"] for r in fin]))
    return {k: [float(np.mean(by[(k, r)])) for r in range(5)] for k in (1, 2)}


def figure(lat):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.1))
    n, p_fp = 5, 0.05
    for rho, col in zip(RHOS, (BLUE, ORANGE, GRAY)):
        far = [false_acceptance(n, k, p_fp, rho) for k in range(1, n + 1)]
        tar = [true_acceptance(n, k, SENS, rho) for k in range(1, n + 1)]
        a1.plot(far, tar, "-o", color=col, lw=2, ms=6, mec=SURF, mew=1.2, label=f"ρ = {rho}")
        for k, (x, y) in enumerate(zip(far, tar), start=1):
            if rho == 0.0:
                a1.annotate(f"k={k}", (x, y), xytext=(4, -12), textcoords="offset points",
                            fontsize=8.5, color=INK2)
    for t in TARGETS:
        a1.axvline(t, color=INK2, lw=1, ls=(0, (4, 3)))
    a1.set_xscale("log")
    a1.set_xlabel("False-acceptance rate (log)")
    a1.set_ylabel("True-acceptance rate")
    a1.set_title(f"Gate trade-off, n = {n}, p_fp = {p_fp}, s = {SENS}")
    a1.legend(loc="lower right")

    rounds = np.arange(5)
    a2.plot(rounds, lat[1], "-o", color=BLUE, lw=2, ms=6, mec=SURF, mew=1.2, label="No gate")
    a2.plot(rounds, lat[2], "-o", color=AQUA, lw=2, ms=6, mec=SURF, mew=1.2, label="Local k = 2")
    a2.set_ylim(0, 1.02)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_xticks(rounds)
    a2.set_xlabel("Communication round")
    a2.set_ylabel("Agents holding the real finding")
    a2.set_title("Latency cost of the gate (Haiku 4.5, 3 cases)")
    a2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_gate_tradeoff.png", dpi=180, bbox_inches="tight")
    print("✓", FIG / "fig9_gate_tradeoff.png")


if __name__ == "__main__":
    rows = lookup()
    write_tables(rows)
    lat = latency()
    print("truth reach by round, no gate:", [round(x, 2) for x in lat[1]])
    print("truth reach by round, k = 2:  ", [round(x, 2) for x in lat[2]])
    figure(lat)
