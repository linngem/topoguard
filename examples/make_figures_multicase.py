"""Phase-2 figures (12 cases: micro-experiment and multi-case network) + a LinkedIn summary card.

    cd examples && PYTHONPATH=.. python make_figures_multicase.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topoguard.validation import load, micro
from topoguard.validation.analysis import observed
from topoguard.validation.plausibility import load_scores

RES = Path("results/multicase")
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.titlelocation": "left", "legend.frameon": False,
})
ROLE_LABEL = {"truth": "Truth", "err_high": "Plausible error",
              "err_mid": "Intermediate error", "err_low": "Implausible error"}
TOPO_LABEL = {"star": "Star", "ring": "Ring", "small_world": "Small-world",
              "scale_free_cluster": "Clustered\nscale-free"}
PCT = matplotlib.ticker.PercentFormatter(1.0)


def save(fig, name, dpi=180, tight=True):
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print("✓", OUT / name)


pl = load_scores(RES / "plausibility.csv")
rows = micro.to_rows(micro.load(RES / "micro.jsonl"), pl)
ad = [r for r in rows if r["cond"] == "adopt"]
t = defaultdict(list)
for r in ad:
    t[(r["role"], r["n"], r["m"])].append(r["y"])

# ---- Fig 6: micro-experiment --------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7), sharey=True)
cols = {2: AQUA, 4: BLUE, 8: ORANGE}
for ax, role in zip(axes, ROLE_LABEL):
    for n, ms in micro.ADOPT_GRID.items():
        ys = [np.mean(t[(role, n, m)]) for m in ms]
        ax.plot(ms, ys, "-o", color=cols[n], lw=2, ms=6, mec=SURF, mew=1.2, label=f"{n} neighbours")
    ax.set_title(ROLE_LABEL[role])
    ax.set_xticks([0, 1, 2, 4, 6, 8])
    ax.set_xlabel("Neighbours asserting it (m)")
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(PCT)
axes[0].set_ylabel("Probability of adopting it")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.06), fontsize=10)
fig.suptitle("One agent facing m of n neighbours: plausibility rules, the number of voices pushes",
             x=0.06, ha="left", y=1.1, fontsize=12.5)
save(fig, "fig6_micro_experiment.png")

# ---- Fig 7: lupus network (plausible error) -----------------------------------
obs = observed(load(RES / "network_trials.jsonl"))
cell = defaultdict(list)
for (case, topo, ek, pos, k, rep), v in obs.items():
    cell[(case, topo, ek, pos, k)].append(v["error"])
ORDER = ["star", "ring", "small_world", "scale_free_cluster"]
x = np.arange(len(ORDER)); w = 0.36
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, k, title in [(axes[0], None, "No gate"), (axes[1], 2, "Local gate k = 2")]:
    for off, pos, col, lab in [(-w/2, "hub", ORANGE, "Error enters at the hub"),
                               (w/2, "periphery", BLUE, "Error enters at the periphery")]:
        vals = [np.mean(cell[("lupus", tp, "plausible", pos, k)]) for tp in ORDER]
        ax.bar(x + off, vals, w - 0.03, color=col, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.axhline(0.1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x, [TOPO_LABEL[tp] for tp in ORDER], fontsize=9.5)
    ax.set_title(title); ax.grid(axis="x", visible=False); ax.set_ylim(0, 0.95)
    ax.yaxis.set_major_formatter(PCT)
axes[0].set_ylabel("Agents asserting the error")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.04), fontsize=10)
fig.suptitle("Lupus: a made-up antiphospholipid syndrome in the network (4 replicates; line = source only)",
             x=0.06, ha="left", y=1.1, fontsize=12.5)
save(fig, "fig7_lupus_network.png")

# ---- Fig 8: prediction from the micro-experiment ------------------------------
pr = list(csv.DictReader(open(RES / "network_pred_micro_both.csv", encoding="utf-8")))
fig, ax = plt.subplots(figsize=(5.6, 5.3))
ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=0)
for lab, sel, col, field in [
        ("Truth", lambda r: True, BLUE, "truth"),
        ("Plausible error", lambda r: r["error_kind"] == "plausible", ORANGE, "error"),
        ("Implausible error", lambda r: r["error_kind"] == "implausible", GRAY, "error")]:
    rr = [r for r in pr if sel(r)]
    ax.scatter([float(r[f"pred_{field}"]) for r in rr], [float(r[f"obs_{field}"]) for r in rr],
               s=34, color=col, edgecolor=SURF, lw=1.2, alpha=0.85, label=lab)
ax.set_xlim(0, 1.03); ax.set_ylim(0, 1.03)
ax.set_xlabel("Predicted with the micro-experiment rule")
ax.set_ylabel("Observed in the network")
ax.set_title("Predicting the network without having seen it")
ax.text(0.02, 0.97, "3 cases · 5 or 4 topologies · no network fitting\nρ truth 0.68 · plausible error 0.42",
        transform=ax.transAxes, va="top", fontsize=9.5, color=INK2)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=10)
save(fig, "fig8_prediction_from_micro.png")

# ---- LinkedIn card (1080 × 1350) ----------------------------------------------
fig = plt.figure(figsize=(6, 7.5))
fig.text(0.06, 0.955, "Can the shape of an AI agent team", fontsize=17, weight="medium")
fig.text(0.06, 0.918, "stop its hallucinations?", fontsize=17, weight="medium")
fig.text(0.06, 0.878, "12 synthetic clinical cases · Claude Haiku 4.5 agents · ~12,000 model calls",
         fontsize=9.5, color=INK2)

a1 = fig.add_axes([0.15, 0.49, 0.79, 0.32])
for role, col in [("truth", BLUE), ("err_high", ORANGE), ("err_low", GRAY)]:
    ms = micro.ADOPT_GRID[8]
    ys = [np.mean(t[(role, 8, m)]) for m in ms]
    a1.plot(ms, ys, "-o", color=col, lw=2.4, ms=6, mec=SURF, mew=1.2)
    anchor = {"truth": (1, ys[1], (-8, 6), "right"), "err_high": (4, ys[3], (8, -16), "left"),
              "err_low": (8, ys[-1], (-4, 8), "right")}[role]
    a1.annotate(ROLE_LABEL[role], anchor[:2], xytext=anchor[2], textcoords="offset points",
                ha=anchor[3], fontsize=10.5, color=col, weight="medium")
a1.set_xticks([0, 1, 2, 4, 6, 8]); a1.set_ylim(-0.03, 1.12); a1.yaxis.set_major_formatter(PCT)
a1.set_xlabel("Of 8 neighbouring agents, how many assert it", fontsize=10)
a1.set_ylabel("Adoption", fontsize=10)
a1.set_title("1 · A plausible falsehood repeated by all is adopted", fontsize=11.5)

a2 = fig.add_axes([0.15, 0.12, 0.79, 0.26])
labels = ["Enters at the\nstar's hub", "Enters at the\nperiphery", "Hub, with the\nk=2 gate"]
vals = [np.mean(cell[("lupus", "star", "plausible", "hub", None)]),
        np.mean(cell[("lupus", "star", "plausible", "periphery", None)]),
        np.mean(cell[("lupus", "star", "plausible", "hub", 2)])]
bars = a2.bar(range(3), vals, 0.55, color=[ORANGE, BLUE, AQUA])
for i, v in enumerate(vals):
    a2.text(i, v + 0.03, f"{v:.0%}", ha="center", fontsize=12, weight="medium", color=INK)
a2.set_xticks(range(3), labels, fontsize=9.5); a2.set_ylim(0, 1); a2.yaxis.set_major_formatter(PCT)
a2.grid(axis="x", visible=False)
a2.set_ylabel("Team contaminated", fontsize=10)
a2.set_title("2 · Lupus case: a made-up antiphospholipid syndrome", fontsize=11.5)

fig.text(0.06, 0.03, "Open code, data and prompts: github.com/linngem/topoguard",
         fontsize=10, color=INK2)
save(fig, "linkedin_card.png", dpi=180, tight=False)
