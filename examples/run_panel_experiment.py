"""Heterogeneous-panel experiment: does model diversity make the k-of-n gate safer?

Panel (default): Claude Haiku 4.5 (API) + Qwen3.8 27B + gpt-oss 20B + Gemma 4 26B + MedGemma 27B
(Ollama). MedGemma shares its lineage with Gemma, so both form ONE independence group; the
experiment measures whether their errors are in fact more correlated than across families.

    # 0) cost and plan
    python run_panel_experiment.py --dry-run
    # 1) pipeline test at no cost (simulated models with known error structure)
    python run_panel_experiment.py --sim
    # 2) real run (resumable: re-running skips calls already stored)
    ollama pull qwen3.8:27b gpt-oss:20b gemma4:26b medgemma:27b
    export ANTHROPIC_API_KEY=...          # never commit it
    python run_panel_experiment.py
    # 3) re-analyse without calls
    python run_panel_experiment.py --analyze-only

Models run one after another, so a single 24 GB GPU is enough (Ollama loads one at a time).
"""
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from topoguard.validation import AnthropicBackend, OpenAICompatBackend
from topoguard.validation.cases import CASES
from topoguard.validation.panel import (ModelSpec, SimulatedPanelBackend, collect, evaluate,
                                        heterogeneous, homogeneous, load_responses, n_calls,
                                        predicted_far_injected, rho_report)

# name, family, provider, model id
MODELS = [
    ("haiku", "anthropic", "anthropic", "claude-haiku-4-5-20251001"),
    ("qwen", "qwen", "ollama", "qwen3.8:27b"),
    ("gptoss", "openai", "ollama", "gpt-oss:20b"),
    ("gemma", "gemma", "ollama", "gemma4:26b"),
    ("medgemma", "gemma", "ollama", "medgemma:27b"),
]
ERROR_KINDS = ("plausible", "implausible")
SIM_PROFILES = {   # p_fp, sens — only used with --sim
    "haiku": (0.05, 0.85), "qwen": (0.07, 0.80), "gptoss": (0.06, 0.75),
    "gemma": (0.08, 0.80), "medgemma": (0.06, 0.85),
}


def build_backends(a, specs):
    out = {}
    for name, family, provider, model_id in MODELS:
        if name not in specs:
            continue
        if a.sim:
            p_fp, sens = SIM_PROFILES[name]
            out[name] = SimulatedPanelBackend(name, family, p_fp=p_fp, sens=sens)
        elif provider == "anthropic":
            out[name] = AnthropicBackend(model_id, max_tokens=400,
                                         cache_dir=Path(a.out) / "cache" / name)
        else:
            out[name] = OpenAICompatBackend(model_id, base_url=a.ollama_url, api_key="ollama",
                                            temperature=0.7, max_tokens=a.local_max_tokens,
                                            cache_dir=Path(a.out) / "cache" / name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="simulated models, no calls")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--models", default=",".join(m[0] for m in MODELS),
                    help="subset, e.g. haiku,qwen,gemma")
    ap.add_argument("--replicas", type=int, default=3, help="≥3 for the homogeneous panels")
    ap.add_argument("--ollama-url", default="http://localhost:11434/v1")
    ap.add_argument("--local-max-tokens", type=int, default=2048,
                    help="reasoning models (gpt-oss, qwen) spend tokens thinking first")
    ap.add_argument("--out", default="results/panel")
    a = ap.parse_args()

    names = [n for n in a.models.split(",") if n]
    specs = {n: ModelSpec(n, f) for n, f, *_ in MODELS if n in names}
    models = list(specs.values())
    out = Path(a.out) / ("sim" if a.sim else "real")
    out.mkdir(parents=True, exist_ok=True)
    a.out = str(out)
    resp_path = out / "responses.jsonl"

    total = n_calls(len(CASES), len(models), ERROR_KINDS, a.replicas)
    print(f"Models: {', '.join(f'{m.name}[{m.family}]' for m in models)}")
    print(f"Calls: {total} ({total // max(len(models), 1)} per model; "
          f"Haiku ≈ {total // max(len(models), 1)} if included)")
    if a.dry_run:
        return

    if not a.analyze_only:
        t0 = time.time()
        workers = {n: (8 if n == "haiku" else 2) for n in specs}
        collect(build_backends(a, specs), error_kinds=ERROR_KINDS, replicas=a.replicas,
                out_path=resp_path, workers=workers)
        print(f"collection: {time.time() - t0:.0f}s")

    resp = load_responses(resp_path)
    fails = {m.name: sum(v is None for (mm, *_), v in resp.items() if mm == m.name)
             for m in models}
    print("unparseable responses:", fails)

    # ---- error correlation
    rho = {k: rho_report(resp, models, k) for k in ("fp", "fn")}
    (out / "rho.json").write_text(json.dumps(rho, indent=2), encoding="utf-8")
    for k, r in rho.items():
        label = "false positives" if k == "fp" else "missed truth"
        fmt = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]" if c else "n/a"   # noqa: E731
        print(f"\nρ ({label}, clean condition, {r['rows']} rows / {r['cases']} cases): "
              f"all={r['rho_all']:.3f} {fmt(r['rho_all_ci95'])}")
        print(f"  within-family={r['rho_within_family']:.3f} across-family="
              f"{r['rho_across_family']:.3f}  difference CI95 {fmt(r['within_minus_across_ci95'])}")
        print("  rates:", {m: round(v, 3) for m, v in r["rates"].items()})
        for p, v in sorted(r["pairs"].items(), key=lambda t: -abs(t[1]) if t[1] == t[1] else 0):
            print(f"  {p:22} {v:+.3f}")

    # ---- panels
    panels = [heterogeneous(models, "hetero_family"),
              heterogeneous(models, "hetero_naive", by_family=False)]
    if "medgemma" in specs:
        panels.append(heterogeneous([m for m in models if m.name != "medgemma"], "hetero_no_medgemma"))
    if a.replicas >= 3:
        panels += [homogeneous(m, 3) for m in models]
    rows = []
    for p in panels:
        rows += [asdict(r) for r in evaluate(p, resp, error_kinds=ERROR_KINDS)]

    rho_fp = rho["fp"]["rho_all"]
    for r in rows:
        p_spont = sum(rho["fp"]["rates"].values()) / len(models) / 3   # per false key
        r["far_injected_prior"] = predicted_far_injected(
            r["source_adopted"], p_spont, r["n_groups"], r["k"], rho_fp if rho_fp == rho_fp else 0.0)
    with (out / "panel_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for ek in ERROR_KINDS:
        print(f"\n=== injected error: {ek} ===")
        print(f"{'panel':24} {'groups':>6} {'k':>2} {'FAR inj':>8} {'prior':>7} {'FAR spont':>9} "
              f"{'sens truth':>10} {'adopt':>6}")
        for r in rows:
            if r["error_kind"] == ek and r["k"] in (1, 2, 3):
                print(f"{r['panel']:24} {r['n_groups']:6} {r['k']:2} {r['far_injected']:8.3f} "
                      f"{r['far_injected_prior']:7.3f} {r['far_spontaneous']:9.3f} "
                      f"{r['sens_truth']:10.3f} {r['source_adopted']:6.2f}")
    print(f"\nSaved to {out}/ (responses.jsonl, rho.json, panel_results.csv)")


if __name__ == "__main__":
    main()
