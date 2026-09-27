"""Phase-2 pre-registered analysis: micro-experiment model comparison and network prediction.

    cd examples && PYTHONPATH=.. python analyze_multicase.py
"""
import csv
import json

from run_validation import topologies
from topoguard.validation import compare, load, micro
from topoguard.validation.analysis import _score
from topoguard.validation.plausibility import load_scores

OUT = "results/multicase"
pl = load_scores(f"{OUT}/plausibility.csv")
rows = micro.to_rows(micro.load(f"{OUT}/micro.jsonl"), pl)
adopt = [r for r in rows if r["cond"] == "adopt"]
keep = [r for r in rows if r["cond"] == "keep"]

print("Q1 · leave-one-case-out log-loss (adoption, lower = better)")
cm = micro.compare_models(adopt)
for k, v in sorted(cm.items(), key=lambda kv: kv[1]):
    print(f"  {k:18} {v:.4f}")
json.dump(cm, open(f"{OUT}/micro_model_comparison.json", "w"), indent=2)

print("\nQ2 · coefficients of the 'both' model (95 % CI, case bootstrap, B=2000)")
ci = micro.bootstrap_coefs(adopt, "both", B=2000)
for i, name in enumerate(micro.COEF_NAMES["both"]):
    print(f"  {name:26} {ci[1, i]:7.2f}  [{ci[0, i]:6.2f}, {ci[2, i]:6.2f}]")

print("\nQ3 · predicting the network with the micro-experiment rule (no network fitting)")
recs = load("results/anthropic_claude-haiku-4-5-20251001/trials.jsonl") + load(f"{OUT}/network_trials.jsonl")
topo = topologies(10)
res = {}
for model in ("both", "fraction", "count"):
    prov = micro.rule_provider(micro.fit_rule(adopt, model), micro.fit_rule(keep, model), pl)
    cmp = compare(recs, topo, rules=prov)
    res[model] = cmp.summary()
    print(f"  rule={model}")
    for k, d in res[model].items():
        print(f"    {k:22} spearman={d['spearman']:.3f} mae={d['mae']:.3f} n={d['n']}")
    if model == "both":
        with open(f"{OUT}/network_pred_micro_both.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cmp.rows[0]))
            w.writeheader()
            w.writerows(cmp.rows)
        for case in ("pneumonia", "inferior_mi", "lupus"):
            rr = [r for r in cmp.rows if r["case"] == case]
            e = _score([r["pred_error"] for r in rr if r["error_kind"] == "plausible"],
                       [r["obs_error"] for r in rr if r["error_kind"] == "plausible"])
            t = _score([r["pred_truth"] for r in rr], [r["obs_truth"] for r in rr])
            print(f"      {case:12} plausible error rho={e['spearman']:.2f} mae={e['mae']:.3f} | "
                  f"truth rho={t['spearman']:.2f} mae={t['mae']:.3f}")
cmp = compare(recs, topo, leave_one_topology_out=True)
res["network_fitted_fraction_LOTO"] = cmp.summary()
print("  baseline: fraction rule fitted on network data (leave-one-topology-out)")
for k, d in cmp.summary().items():
    print(f"    {k:22} spearman={d['spearman']:.3f} mae={d['mae']:.3f} n={d['n']}")
json.dump(res, open(f"{OUT}/network_prediction_summary.json", "w"), indent=2)
