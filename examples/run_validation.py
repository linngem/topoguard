"""Phase-1 validation experiment (one case, five topologies).

    # 0) cost: how many calls it will make
    python run_validation.py --backend anthropic --dry-run
    # 1) pipeline test at no cost (simulated threshold agents, known θ=0.4)
    python run_validation.py --backend sim
    # 2) real run
    python run_validation.py --backend anthropic --model claude-haiku-4-5-20251001
    python run_validation.py --backend openai --model qwen2.5:14b --base-url http://localhost:11434/v1
"""
import argparse
import csv
import json
import time
from pathlib import Path

import networkx as nx

from topoguard.validation import (AnthropicBackend, LLMPolicy, OpenAICompatBackend,
                                  SimulatedPolicy, compare, fit_all, load, n_calls, run_grid)


def topologies(n: int, seed: int = 1) -> dict[str, nx.Graph]:
    return {
        "star": nx.star_graph(n - 1),
        "small_world": nx.connected_watts_strogatz_graph(n, 4, 0.2, seed=seed),
        "scale_free_cluster": nx.powerlaw_cluster_graph(n, 2, 0.8, seed=seed),
        "ring": nx.cycle_graph(n),
        "complete": nx.complete_graph(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "anthropic", "openai"], default="sim")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--base-url")
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10, help="agents in parallel per trial")
    ap.add_argument("--parallel-trials", type=int, default=8)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true", help="reuse an existing trials.jsonl")
    a = ap.parse_args()

    topo = topologies(a.n_agents)
    print(f"Expected LLM calls: {n_calls(topo, replicas=a.replicas, rounds=a.rounds)}")
    if a.dry_run:
        return

    tag = a.backend if a.backend == "sim" else f"{a.backend}_{a.model.replace('/', '_').replace(':', '_')}"
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    trials = out / "trials.jsonl"

    if not a.analyze_only:
        if a.backend == "sim":
            policy = SimulatedPolicy()
        elif a.backend == "anthropic":
            policy = LLMPolicy(AnthropicBackend(a.model, cache_dir=out / "cache"))
        else:
            policy = LLMPolicy(OpenAICompatBackend(a.model, a.base_url, cache_dir=out / "cache"))
        t0 = time.time()
        run_grid(topo, policy, replicas=a.replicas, rounds=a.rounds, out_path=trials,
                 workers=a.workers, parallel_trials=a.parallel_trials)
        print(f"time: {time.time() - t0:.0f}s")
        if isinstance(policy, LLMPolicy):
            b = policy.backend
            print(f"calls={b.calls} cache={b.cache_hits} parse_failures={policy.parse_failures}")
            if hasattr(b, "input_tokens"):
                print(f"tokens in={b.input_tokens} out={b.output_tokens}")

    recs = load(trials)
    fits = fit_all(recs)
    print("\nEffective adoption rule (all data):")
    for name, f in fits.items():
        print(f"  {name:12} θ={f.theta:5.2f} slope={f.slope:5.1f} p_keep={f.keep:.2f} "
              f"spontaneous adoption={f.base_rate:.3f} n={f.n_events}")

    cmp = compare(recs, topo, rounds=a.rounds)
    print(f"\n{'topology':20} {'error':12} {'err@':10} {'k':>4}  {'error pred/obs':>15}  {'truth pred/obs':>16}")
    for r in cmp.rows:
        print(f"{r['topology']:20} {r['error_kind']:12} {r['error_pos']:10} {str(r['local_k']):>4}  "
              f"{r['pred_error']:6.2f}/{r['obs_error']:<6.2f}   {r['pred_truth']:6.2f}/{r['obs_truth']:<6.2f}")
    print("\nLeave-one-topology-out:")
    for k, d in cmp.summary().items():
        print(f"  {k:22} spearman={d['spearman']:.3f} mae={d['mae']:.3f} n={d['n']}")

    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cmp.rows[0]))
        w.writeheader()
        w.writerows(cmp.rows)
    with (out / "fits.json").open("w", encoding="utf-8") as fh:
        json.dump({k: v.__dict__ for k, v in fits.items()}, fh, indent=2)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
