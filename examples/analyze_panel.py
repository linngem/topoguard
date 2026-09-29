"""Cross-model analysis of the panel (Haiku 4.5 reference + every folder in results/panel/).

    cd examples && PYTHONPATH=.. python analyze_panel.py --family "<gemma>=gemma,<medgemma>=gemma"

Outputs in results/panel/_summary/:
  adoption.csv        P(adopt) per model × finding role × social pressure (m = 0 vs m = n)
  micro_rules.json    per model: Q1 log-loss, 'both' coefficients + 95 % CI, H1, H2, H4 flags
  network.csv         per model × topology × position × gate: mean final reach of the error / truth
  network_pred.json   per model: network predicted from ITS OWN micro rule, nothing fitted (Q3)
  error_corr.csv      pairwise correlation of error adoption between models (raw and residual)
  gate_design.json    ρ within model (replicas) vs ρ between models → k needed by design_gate
  h3_correlation.json pre-registered H3: within-model vs between-family residual ρ, 95 % CI
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_validation import topologies
from topoguard.reliability import design_gate, effective_n
from topoguard.validation import compare, load, micro
from topoguard.validation.analysis import observed
from topoguard.validation.plausibility import load_scores

ERR_ROLES = ("err_high", "err_mid", "err_low")
MC = Path("results/multicase")
PANEL = Path("results/panel")


def discover() -> dict[str, dict]:
    """Panel folders plus Haiku 4.5's Phase-2 data (micro-experiment and the independent-agent
    network re-run), unless a panel folder for Haiku exists."""
    models = {}
    for d in sorted(PANEL.glob("*/")):
        if d.name.startswith("_"):
            continue
        micro_p = d / "micro.jsonl"
        if not micro_p.exists() and d.name.startswith("claude-haiku-4-5"):
            micro_p = MC / "micro.jsonl"
        if not micro_p.exists():
            continue
        net = [d / "network_trials.jsonl"] if (d / "network_trials.jsonl").exists() else []
        models[d.name] = {"micro": micro_p, "network": net}
    if not any(n.startswith("claude-haiku-4-5") for n in models):
        models = {"claude-haiku-4-5": {
            "micro": MC / "micro.jsonl",
            "network": [MC / "network_trials.jsonl",
                        Path("results/anthropic_claude-haiku-4-5-20251001/trials.jsonl")]},
            **models}
    return models


def adoption_table(name, recs, rows):
    out = []
    fails = sum(r["claims"] is None for r in recs)
    for role in ("truth", *ERR_ROLES):
        base = [r["y"] for r in rows if r["cond"] == "adopt" and r["role"] == role and r["m"] == 0]
        full = [r["y"] for r in rows if r["cond"] == "adopt" and r["role"] == role
                and r["m"] == r["n"] and r["m"] > 0]
        out.append({"model": name, "role": role, "p_adopt_m0": np.mean(base),
                    "p_adopt_unanimous": np.mean(full), "n_m0": len(base), "n_unan": len(full),
                    "parse_fail_rate": fails / len(recs)})
    return out


def residual_items(rows, rule):
    """Error-adoption items (m ≥ 0) keyed by (case, n, m, role, rep) → (y, y − p̂ of the model's own rule)."""
    items = {}
    for r in rows:
        if r["cond"] != "adopt" or r["role"] not in ERR_ROLES:
            continue
        p = float(rule.prob(r["plaus"], np.array([r["m"]]), np.array([r["n"]]))[0])
        items[(r["case"], r["n"], r["m"], r["role"], r["rep"])] = (r["y"], r["y"] - p)
    return items


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def within_model_rho(items, idx):
    """Replica-vs-replica correlation of the same model on the same item (same case, n, m, role)."""
    by = defaultdict(dict)
    for (c, n, m, role, rep), v in items.items():
        by[(c, n, m, role)][rep] = v[idx]
    reps = sorted({k[-1] for k in items})
    vals = []
    for r1, r2 in itertools.combinations(reps, 2):
        pairs = [(d[r1], d[r2]) for d in by.values() if r1 in d and r2 in d]
        if pairs:
            vals.append(corr(*zip(*pairs)))
    return float(np.nanmean(vals)) if vals else float("nan")


def _resample(items, cases):
    """Items of the cases drawn (with repetition); the draw index replaces the case id."""
    by_case = defaultdict(list)
    for k, v in items.items():
        by_case[k[0]].append((k, v))
    return {(i, *k[1:]): v for i, c in enumerate(cases) for k, v in by_case[c]}


def h3_stats(items, names, family, cases):
    """Mean residual ρ within model (replica vs replica), between models of the same family and
    between families, on the given (possibly resampled) cases. Pre-registered test (phase 3):
    between-family < within-model."""
    it = {n: _resample(items[n], cases) for n in names}
    within = [within_model_rho(it[n], 1) for n in names]
    same, diff = [], []
    for m1, m2 in itertools.combinations(names, 2):
        common = sorted(set(it[m1]) & set(it[m2]))
        r = corr([it[m1][k][1] for k in common], [it[m2][k][1] for k in common])
        (same if family[m1] == family[m2] else diff).append(r)
    mean = lambda v: float(np.nanmean(v)) if len(v) else float("nan")  # noqa: E731
    w, b = mean(within), mean(diff)
    return {"within_model": w, "same_family": mean(same), "between_family": b,
            "within_minus_between": w - b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000, help="case-bootstrap resamples")
    ap.add_argument("--panel-size", type=int, default=4, help="n modules for the gate illustration")
    ap.add_argument("--family", default="",
                    help="model families for H3, e.g. 'gemma3_27b=gemma,medgemma_27b=gemma'; "
                         "unlisted models are their own family")
    a = ap.parse_args()

    out = PANEL / "_summary"
    out.mkdir(parents=True, exist_ok=True)
    pl = load_scores(MC / "plausibility.csv")
    topo = {k: v for k, v in topologies(10).items() if k != "complete"}
    models = discover()
    print("models:", ", ".join(models))

    adoption, rules_out, net_rows, net_pred, items = [], {}, [], {}, {}
    for name, src in models.items():
        recs = micro.load(src["micro"])
        rows = micro.to_rows(recs, pl)
        adopt = [r for r in rows if r["cond"] == "adopt"]
        keep = [r for r in rows if r["cond"] == "keep"]
        adoption += adoption_table(name, recs, rows)

        # Q1 + Q2 per model
        loco = micro.compare_models(adopt)
        ci = micro.bootstrap_coefs(adopt, "both", B=a.B)
        rules_out[name] = {"loco_logloss": loco, "winner": min(loco, key=loco.get),
                           "both_coefs": {n: {"est": float(ci[1, i]), "lo": float(ci[0, i]),
                                              "hi": float(ci[2, i])}
                                          for i, n in enumerate(micro.COEF_NAMES["both"])}}
        ad_m = {r["role"]: r for r in adoption if r["model"] == name}
        rules_out[name]["h1_plausibility_positive"] = bool(ci[0, 1] > 0)
        rules_out[name]["h2_high_over_low_unanimous"] = bool(
            ad_m["err_high"]["p_adopt_unanimous"] > ad_m["err_low"]["p_adopt_unanimous"])
        print(f"\n[{name}] Q1 winner={rules_out[name]['winner']}  "
              f"plausibility coef={ci[1, 1]:.2f} [{ci[0, 1]:.2f}, {ci[2, 1]:.2f}]")

        # per-item residuals for the cross-model correlation (rep recovered from the raw records)
        rule_both = micro.fit_rule(adopt, "both")
        rows_rep = []
        for r in recs:
            if r["claims"] is None or r["cond"] != "adopt":
                continue
            for role in ([r["role"]] if r["role"] else ERR_ROLES):   # m = 0 scores every role
                if role not in ERR_ROLES:
                    continue
                key = micro.CASES[r["case"]].candidate(role).key
                rows_rep.append({"cond": "adopt", "case": r["case"], "n": r["n"], "m": r["m"],
                                 "role": role, "rep": r["rep"], "plaus": pl[(r["case"], key)],
                                 "y": float(key in r["claims"])})
        items[name] = residual_items(rows_rep, rule_both)

        # network: observed reach + prediction from this model's own micro rule
        if src["network"]:
            nrecs = [t for p in src["network"] if p.exists() for t in load(p)
                     if t["topology"] in topo]
            obs = observed(nrecs)
            cells = defaultdict(list)
            for (case, t, ek, pos, k, _rep), v in obs.items():
                cells[(t, ek, pos, k)].append(v)
            for (t, ek, pos, k), vs in sorted(cells.items(), key=str):
                net_rows.append({"model": name, "topology": t, "error_kind": ek, "error_pos": pos,
                                 "gate_k": k or 1, "error_reach": np.mean([v["error"] for v in vs]),
                                 "truth_reach": np.mean([v["truth"] for v in vs]), "trials": len(vs)})
            gated = [v["error"] for (c, t, ek, pos, k, _r), v in obs.items() if k == 2]
            contained = float(np.mean([x <= 0.1 + 1e-9 for x in gated])) if gated else float("nan")
            rules_out[name]["h4_gate_containment"] = {"share": contained, "trials": len(gated),
                                                      "supported": contained >= 0.95}
            prov = micro.rule_provider(rule_both, micro.fit_rule(keep, "both"), pl)
            net_pred[name] = compare(nrecs, topo, rules=prov).summary()
            e = net_pred[name].get("error_plausible", {})
            print(f"[{name}] Q3 plausible-error reach predicted from micro: "
                  f"spearman={e.get('spearman', float('nan')):.2f} mae={e.get('mae', float('nan')):.3f}")

    # cross-model error correlation on the SAME items (identical prompts and inboxes)
    names = list(items)
    corr_rows = []
    for m1, m2 in itertools.combinations(names, 2):
        common = sorted(set(items[m1]) & set(items[m2]))
        raw = corr([items[m1][k][0] for k in common], [items[m2][k][0] for k in common])
        res = corr([items[m1][k][1] for k in common], [items[m2][k][1] for k in common])
        corr_rows.append({"model_a": m1, "model_b": m2, "items": len(common),
                          "rho_raw": raw, "rho_residual": res})
    within = {n: {"rho_raw": within_model_rho(items[n], 0),
                  "rho_residual": within_model_rho(items[n], 1)} for n in names}

    # H3 (pre-registered, phase 3): within-model vs between-family residual ρ, case bootstrap
    h3 = {}
    if len(names) >= 2:
        fam = dict(kv.split("=", 1) for kv in a.family.split(",") if "=" in kv)
        family = {n: fam.get(n, "claude" if n.startswith("claude") else n) for n in names}
        cases = sorted({k[0] for n in names for k in items[n]})
        rng = np.random.default_rng(0)
        point = h3_stats(items, names, family, cases)
        draws = [h3_stats(items, names, family, list(rng.choice(cases, len(cases))))
                 for _ in range(a.B)]
        h3 = {"family": family, "B": a.B, **{
            k: {"est": point[k], "lo": float(np.nanpercentile([d[k] for d in draws], 2.5)),
                "hi": float(np.nanpercentile([d[k] for d in draws], 97.5))} for k in point}}
        h3["supported"] = bool(h3["within_minus_between"]["lo"] > 0)

    # gate illustration: same model n times (ρ within) vs one module per model (ρ between)
    gate = {}
    if len(names) >= 2:
        ad = {(r["model"], r["role"]): r for r in adoption}
        # spontaneous rates (m = 0): what one module asserts with no social pressure, i.e. what
        # the gate sees coming from each module before any propagation
        p_fp = float(np.mean([ad[(n, "err_high")]["p_adopt_m0"] for n in names]))
        sens = float(np.mean([ad[(n, "truth")]["p_adopt_m0"] for n in names]))
        rho_between = float(np.nanmean([c["rho_raw"] for c in corr_rows]))
        rho_within = float(np.nanmean([w["rho_raw"] for w in within.values()]))
        n = a.panel_size
        for label, rho in (("homogeneous_panel", rho_within), ("heterogeneous_panel", rho_between)):
            r_ = float(np.clip(np.nan_to_num(rho), 0.0, 0.999))
            g = design_gate(n, p_fp, sens, r_, max_false_acceptance=0.05)
            gate[label] = {"rho": rho, "effective_n": effective_n(n, r_),
                           "k": g.k if g else None,
                           "false_acceptance": g.false_acceptance if g else None,
                           "true_acceptance": g.true_acceptance if g else None}
        gate["assumptions"] = {"n": n, "p_fp": p_fp, "sens": sens, "max_false_acceptance": 0.05,
                               "note": "illustrative: p_fp and sens are spontaneous (m = 0) rates"}

    # write everything
    def dump_csv(path, rows):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    dump_csv(out / "adoption.csv", adoption)
    dump_csv(out / "network.csv", net_rows)
    dump_csv(out / "error_corr.csv", corr_rows)
    json.dump(rules_out, open(out / "micro_rules.json", "w"), indent=2)
    json.dump(net_pred, open(out / "network_pred.json", "w"), indent=2)
    json.dump({"within_model": within, **gate}, open(out / "gate_design.json", "w"), indent=2)
    json.dump(h3, open(out / "h3_correlation.json", "w"), indent=2)

    print("\nError-adoption correlation (raw | residual after each model's own rule)")
    for n, w in within.items():
        print(f"  within {n:24} {w['rho_raw']:.2f} | {w['rho_residual']:.2f}")
    for c in corr_rows:
        print(f"  {c['model_a']:>20} × {c['model_b']:<20} {c['rho_raw']:.2f} | {c['rho_residual']:.2f}"
              f"  (n={c['items']})")
    if gate:
        for label in ("homogeneous_panel", "heterogeneous_panel"):
            g = gate[label]
            print(f"  {label:20} ρ={g['rho']:.2f} n_eff={g['effective_n']:.2f} k={g['k']}")
    if h3:
        print("\nH3 · residual error correlation (95 % CI, case bootstrap)")
        for k in ("within_model", "same_family", "between_family", "within_minus_between"):
            d = h3[k]
            print(f"  {k:22} {d['est']:6.2f}  [{d['lo']:6.2f}, {d['hi']:6.2f}]")
        print(f"  supported (lower bound of within − between > 0): {h3['supported']}")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
