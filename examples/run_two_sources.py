"""Two-source error experiment: the case where the k = 2 gate is actually tested.

With one source, a local k = 2 gate cannot let the error through by construction: no agent ever
hears it from two neighbours unless another agent invents it spontaneously. Here the same false
datum is given to TWO agents, placed in the worst case for the gate (the pair with the most
common neighbours, `pair_seeds`): every common neighbour hears the error twice and may adopt it,
and from there it may travel further.

Design: 3 cases (pneumonia, inferior MI, lupus) × 5 topologies × 2 error types × 2 gates
(none, local k = 2) × 4 replicates × 4 rounds, independent agents. ≤ 12,000 calls per model.

    cd examples
    PYTHONPATH=.. python run_two_sources.py --dry-run
    PYTHONPATH=.. python run_two_sources.py                                   # Haiku 4.5
    PYTHONPATH=.. python run_two_sources.py --backend openai --model gpt-oss:20b \\
        --base-url http://localhost:11434/v1                                  # any open model
    PYTHONPATH=.. python run_two_sources.py --backend sim                     # cost-free test
    PYTHONPATH=.. python analyze_two_sources.py

Interrupted? Run the same command again: answers are cached per model.
"""
import argparse
import json
import time
from pathlib import Path

from run_validation import topologies
from topoguard.validation import (AnthropicBackend, LLMPolicy, OpenAICompatBackend,
                                  SimulatedPolicy, n_calls, run_grid)

CASES = ("pneumonia", "inferior_mi", "lupus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "openai", "sim"], default="anthropic")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: provider default for Anthropic (as in phases 1-2), 1.0 otherwise")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default: 400 for Anthropic (as in phases 1-2), 1024 otherwise")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--parallel-trials", type=int, default=16)
    ap.add_argument("--out", default="results/two_sources")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    topo = topologies(10)
    grid = dict(error_pos=("pair",), replicas=a.replicas, rounds=a.rounds, cases=tuple(a.cases))
    n = n_calls(topo, **grid)
    print(f"calls ≤ {n} ({len(a.cases)} cases × {len(topo)} topologies × 2 errors × 2 gates × "
          f"{a.replicas} replicates)")
    if a.dry_run:
        return

    tag = "sim" if a.backend == "sim" else a.model.replace("/", "_").replace(":", "_")
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    if a.backend == "sim":
        pol, b = SimulatedPolicy(), None
    elif a.backend == "anthropic":
        b = AnthropicBackend(a.model, temperature=a.temperature, max_tokens=a.max_tokens or 400,
                             cache_dir=out / "cache")
        pol = LLMPolicy(b, independent_agents=True)
    else:
        b = OpenAICompatBackend(a.model, a.base_url, temperature=a.temperature or 1.0,
                                max_tokens=a.max_tokens or 1024, cache_dir=out / "cache")
        pol = LLMPolicy(b, independent_agents=True)

    t = time.time()
    run_grid(topo, pol, out_path=out / "trials.jsonl", workers=a.workers,
             parallel_trials=a.parallel_trials, progress=False, **grid)
    meta = {"model": a.model if b else "simulated", "backend": getattr(b, "name", "sim"),
            "independent_agents": True, "cases": a.cases, "replicas": a.replicas,
            "rounds": a.rounds, "seconds": round(time.time() - t),
            "parse_failures": getattr(pol, "parse_failures", 0)}
    if b:
        meta.update({"calls": b.calls, "cache_hits": b.cache_hits,
                     "input_tokens": getattr(b, "input_tokens", None),
                     "output_tokens": getattr(b, "output_tokens", None)})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
