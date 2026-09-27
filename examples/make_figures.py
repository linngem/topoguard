"""Phase-1 figures, generated from the experiment results.

    cd examples && PYTHONPATH=.. python make_figures.py results/anthropic_claude-haiku-4-5-20251001
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from run_validation import topologies
from topoguard.validation import ERRORS, TRUTH_KEY, compare, fit_all, load
from topoguard.validation.analysis import _events

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/anthropic_claude-haiku-4-5-20251001")
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
TOPO_LABEL = {"star": "Star", "small_world": "Small-world",
              "scale_free_cluster": "Clustered\nscale-free", "ring": "Ring", "complete": "Complete"}
ORDER = ["star", "ring", "small_world", "scale_free_cluster", "complete"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "medium", "axes.titlelocation": "left",
    "legend.frameon": False,
})

recs = load(RES / "trials.jsonl")
topo = topologies(10)
fits = fit_all(recs)


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("✓", OUT / name)


# ---- Fig 1: topologies --------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(13, 3.1))
for ax, name in zip(axes, ORDER):
    G = topo[name]
    pos = (nx.circular_layout(G) if name in ("ring", "complete", "small_world")
           else nx.kamada_kawai_layout(G))
    if name == "star":
        pos = {0: np.zeros(2), **{k: np.array([np.cos(2*np.pi*k/9), np.sin(2*np.pi*k/9)]) for k in range(1, 10)}}
    hub = (max(dict(G.degree).items(), key=lambda kv: kv[1])[0]
           if name not in ("ring", "complete") else None)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRAY, width=0.9, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[140 if v == hub else 70 for v in G],
                           node_color=[ORANGE if v == hub else BLUE for v in G],
                           edgecolors=SURF, linewidths=1.5)
    ax.set_title(TOPO_LABEL[name].replace("\n", " "), loc="center")
    ax.set_axis_off()
fig.text(0.5, -0.02, "10 agents per network. Orange: the most connected agent (hub). "
         "In the ring and complete networks all agents have the same degree.",
         ha="center", color=INK2, fontsize=10)
save(fig, "fig1_topologies.png")

# ---- Fig 2: empirical adoption rule ------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
series = [("Truth (hyponatremia)", TRUTH_KEY, "truth_seed", recs, fits["truth"], BLUE),
          ("Plausible error (PE)", ERRORS["plausible"][0], "error_seed",
           [r for r in recs if r["error_kind"] == "plausible"], fits["plausible"], ORANGE),
          ("Implausible error (pericardial effusion)", ERRORS["implausible"][0], "error_seed",
           [r for r in recs if r["error_kind"] == "implausible"], fits["implausible"], GRAY)]
bins = np.array([-0.01, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
xs = np.linspace(0, 1, 200)
for label, key, sf, rr, fit, col in series:
    X, Y, _ = _events(rr, lambda r, k=key: k, sf)
    idx = np.digitize(X, bins[1:-1], right=True)
    cx, cy, cn = [], [], []
    for b in np.unique(idx):
        m = idx == b
        if m.sum() >= 5:
            cx.append(X[m].mean()); cy.append(Y[m].mean()); cn.append(m.sum())
    ax.plot(xs, fit.p_adopt_frac(xs), color=col, lw=2)
    ax.scatter(cx, cy, s=[min(20 + n / 4, 160) for n in cn], color=col, edgecolor=SURF, lw=1.5, zorder=3)
    ax.annotate(label, (1.0, float(fit.p_adopt_frac(np.array([1.0]))[0])), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK, fontsize=10)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Fraction of neighbours asserting the finding")
ax.set_ylabel("Probability of adopting it")
ax.set_title("LLMs filter by content: same social pressure, different adoption")
save(fig, "fig2_adoption_rule.png")

# ---- aggregated data ----------------------------------------------------------
cmp = compare(recs, topo)
df = pd.DataFrame(cmp.rows)
df.to_csv(RES / "comparison.csv", index=False)

# ---- Fig 3: where the plausible error enters matters --------------------------
d = df[(df.error_kind == "plausible") & (df.local_k.isna())].set_index(["topology", "error_pos"]).obs_error
fig, ax = plt.subplots(figsize=(7.6, 4.0))
x = np.arange(len(ORDER)); w = 0.36
for off, pos, col, lab in [(-w/2, "hub", ORANGE, "Error enters at the hub"),
                           (w/2, "periphery", BLUE, "Error enters at the periphery")]:
    vals = [d[(t, pos)] for t in ORDER]
    ax.bar(x + off, vals, w - 0.03, color=col, label=lab)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9.5, color=INK2)
ax.axhline(0.1, color=INK2, lw=1, ls=(0, (4, 3)), label="Source agent only (10 %)")
ax.set_xticks(x, [TOPO_LABEL[t] for t in ORDER]); ax.set_ylim(0, 0.62)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylabel("Agents ending up asserting the error")
ax.set_title("A plausible error at the hub spreads; at the periphery it dies out")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9.5)
ax.grid(axis="x", visible=False)
save(fig, "fig3_error_by_position.png")

# ---- Fig 4: the k=2 gate — firewall and its cost ------------------------------
p = df[df.error_kind == "plausible"].copy()
p["gate"] = p.local_k.fillna(0).astype(int)
g = p.groupby(["topology", "gate"])[["obs_error", "obs_truth"]].mean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, gate, title in [(axes[0], 0, "No gate"), (axes[1], 2, "Local gate k = 2")]:
    for off, col, lab, colname in [(-w/2, BLUE, "Truth spread", "obs_truth"),
                                   (w/2, ORANGE, "Error spread", "obs_error")]:
        vals = [g.loc[(t, gate), colname] for t in ORDER]
        ax.bar(x + off, vals, w - 0.03, color=col, label=lab)
    ax.set_xticks(x, [TOPO_LABEL[t] for t in ORDER], fontsize=9.5)
    ax.set_title(title); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
axes[0].set_ylabel("Fraction of agents")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0), fontsize=10)
fig.suptitle("The gate blocks the error in every network, but in the star and ring it also slows the truth",
             x=0.06, ha="left", fontsize=12.5, y=1.08)
save(fig, "fig4_gate_k2.png")

# ---- Fig 5: predicted vs observed ---------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=0)
pts = [("Truth", df.pred_truth, df.obs_truth, BLUE),
       ("Plausible error", df[df.error_kind == "plausible"].pred_error,
        df[df.error_kind == "plausible"].obs_error, ORANGE),
       ("Implausible error", df[df.error_kind == "implausible"].pred_error,
        df[df.error_kind == "implausible"].obs_error, GRAY)]
for lab, pr, ob, col in pts:
    ax.scatter(pr, ob, s=46, color=col, edgecolor=SURF, lw=1.5, label=lab, alpha=0.9)
s = cmp.summary()
ax.text(0.02, 0.97, f"Spearman ρ (cross-validated)\ntruth {s['truth']['spearman']:.2f} · "
        f"plausible error {s['error_plausible']['spearman']:.2f}",
        transform=ax.transAxes, va="top", fontsize=9.5, color=INK2)
ax.set_xlim(0, 1.03); ax.set_ylim(0, 1.03)
ax.set_xlabel("Reach predicted from the topology"); ax.set_ylabel("Reach observed with LLMs")
ax.set_title("Does the network predict what LLMs do?")
ax.legend(loc="lower right")
save(fig, "fig5_predicted_vs_observed.png")
