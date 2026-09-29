"""Analysis of the out-of-domain control (software incidents), as pre-registered in
results/software/PREREGISTRATION.md.

    cd examples && PYTHONPATH=.. python analyze_software_control.py
Writes results/software/{summary.json, report.md} and docs/figures/fig10_software_control.png.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topoguard.validation import SOFTWARE_CASES, micro
from topoguard.validation.plausibility import load_scores

SW = Path("results/software")
MC = Path("results/multicase")
FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
ROLES = ("truth", "err_high", "err_mid", "err_low")
LABEL = {"truth": "Truth", "err_high": "Plausible error", "err_mid": "Intermediate error",
         "err_low": "Implausible error"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
COL = {"truth": "#2a78d6", "err_high": "#eb6834", "err_mid": "#1baf7a", "err_low": "#8a8984"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.titlelocation": "left", "legend.frameon": False,
})


def curves(rows):
    t = defaultdict(list)
    for r in rows:
        if r["cond"] == "adopt":
            t[(r["role"], r["n"], r["m"])].append(r["y"])
    return t


def logloss(rule_beta, model, rows):
    X, y = micro._design(rows, model)
    return micro._logloss(X, y, rule_beta)


def main():
    pl_c = load_scores(MC / "plausibility.csv")
    pl_s = load_scores(SW / "plausibility.csv")
    clin = micro.to_rows(micro.load(MC / "micro.jsonl"), pl_c)
    recs = micro.load(SW / "micro.jsonl")
    soft = micro.to_rows(recs, pl_s)
    clin_ad = [r for r in clin if r["cond"] == "adopt"]
    soft_ad = [r for r in soft if r["cond"] == "adopt"]
    res = {"parse_failures": sum(r["claims"] is None for r in recs), "responses": len(recs)}

    # manipulation check: panel plausibility by role
    res["panel_plausibility"] = {cid: {role: pl_s[(cid, c.candidate(role).key)] for role in ROLES}
                                 for cid, c in SOFTWARE_CASES.items()}

    # S1: plausible > implausible error under unanimity (n in {4, 8}), in each case
    s1 = {}
    for cid in SOFTWARE_CASES:
        unan = {role: [r["y"] for r in soft_ad if r["case"] == cid and r["role"] == role
                       and r["n"] in (4, 8) and r["m"] == r["n"]] for role in ("err_high", "err_low")}
        s1[cid] = {"p_high": float(np.mean(unan["err_high"])), "p_low": float(np.mean(unan["err_low"])),
                   "n": len(unan["err_high"]),
                   "supported": bool(np.mean(unan["err_high"]) > np.mean(unan["err_low"]))}
    res["S1_unanimity"] = s1

    # S2: transfer of the clinical rules to software (log-loss on software adoption)
    rules = {m: micro.fit_rule(clin_ad, m) for m in ("both", "social_only", "plausibility_only")}
    ll = {m: logloss(rules[m].beta, m, soft_ad) for m in rules}
    base = np.clip(np.mean([r["y"] for r in soft_ad]), 1e-6, 1 - 1e-6)
    ll["base_rate_software"] = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    res["S2_transfer_logloss"] = ll
    res["S2_supported"] = bool(ll["both"] < ll["social_only"])

    # S3 (descriptive): plausibility coefficient fitted on software data alone
    res["S3_software_both_coefs"] = dict(zip(micro.COEF_NAMES["both"],
                                             map(float, micro.fit_rule(soft_ad, "both").beta)))
    (SW / "summary.json").write_text(json.dumps(res, indent=2))

    # figure: clinical vs software adoption with 8 neighbours
    tc, ts = curves(clin), curves(soft)
    ms = micro.ADOPT_GRID[8]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, t, title in ((axes[0], tc, "Clinical (12 cases)"), (axes[1], ts, "Software (2 cases)")):
        for role in ROLES:
            ax.plot(ms, [np.mean(t[(role, 8, m)]) for m in ms], "-o", color=COL[role], lw=2,
                    ms=6, mec=SURF, mew=1.2, label=LABEL[role])
        ax.set_title(title)
        ax.set_xticks(ms)
        ax.set_xlabel("Of 8 neighbours, how many assert it")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Probability of adopting it")
    axes[1].legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_software_control.png", dpi=180, bbox_inches="tight")

    L = ["# Out-of-domain control: software incidents (Haiku 4.5)", "",
         f"Responses: {res['responses']}, parse failures: {res['parse_failures']}", "",
         "## Panel plausibility (manipulation check)", "",
         "| case | truth | high | mid | low |", "|---|---|---|---|---|"]
    for cid, d in res["panel_plausibility"].items():
        L.append(f"| {cid} | " + " | ".join(f"{d[r]:.2f}" for r in ROLES) + " |")
    L += ["", "## S1 · plausible vs implausible error under unanimity (n = 4, 8)", "",
          "| case | plausible | implausible | supported |", "|---|---|---|---|"]
    for cid, d in s1.items():
        L.append(f"| {cid} | {d['p_high']:.0%} | {d['p_low']:.0%} | {d['supported']} |")
    L += ["", "## S2 · clinical rules applied to software (log-loss, lower = better)", ""]
    L += [f"- {k}: {v:.3f}" for k, v in ll.items()]
    L += [f"- supported (both < social_only): {res['S2_supported']}", "",
          "## S3 · 'both' coefficients fitted on software alone (descriptive)", ""]
    L += [f"- {k}: {v:.2f}" for k, v in res["S3_software_both_coefs"].items()]
    (SW / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("✓", FIG / "fig10_software_control.png")


if __name__ == "__main__":
    main()
