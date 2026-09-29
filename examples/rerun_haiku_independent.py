"""Re-runs Haiku 4.5's network experiments with independent agent samples.

The original runs keyed the response cache on (model, prompt, temperature, replica) only, so
agents that received an identical prompt within a trial (mostly star leaves) shared ONE sampled
answer. This script repeats both original designs exactly, but each agent is its own draw:

  pneumonia                 5 topologies (incl. complete) × 2 errors × 2 positions × 2 gates × 3 reps
  inferior_mi, lupus        4 topologies                  × 2 × 2 × 2 × 4 reps

    cd examples
    PYTHONPATH=.. python rerun_haiku_independent.py --dry-run     # calls and rough cost
    PYTHONPATH=.. python rerun_haiku_independent.py               # both designs
    PYTHONPATH=.. python compare_rerun.py                         # original vs re-run

Within a trial every agent is now an independent draw. Across conditions, the same agent with the
same replica and an identical prompt (e.g. round 0 with and without the gate) still reuses its
answer: common random numbers, which pair the gate/no-gate comparison and should be reported.

Interrupted? Just run it again: every answer is cached under results/rerun_independent/cache.
"""
import argparse
import json
import time
from pathlib import Path

from run_validation import topologies
from topoguard.validation import (CASES, AnthropicBackend, LLMPolicy, OpenAICompatBackend, Scenario,
                                  n_calls, render_user, run_grid)

MODEL = "claude-haiku-4-5-20251001"
DESIGNS = {
    # name: (cases, include 'complete', replicas) — identical to the original runs
    "pneumonia": (("pneumonia",), True, 3),
    "multicase": (("inferior_mi", "lupus"), False, 4),
}


def rough_input_tokens() -> int:
    """Typical prompt size (chars / 3.5), for the cost estimate only."""
    scn = Scenario("lupus")
    base = CASES["lupus"].base_key
    inbox = {f"m{i}": [base] for i in range(4)}
    return int(len(scn.system() + render_user(scn, [], inbox, {base})) / 3.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=[*DESIGNS, "all"], default="all")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base-url", help="test against an OpenAI-compatible server instead")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--parallel-trials", type=int, default=16)
    ap.add_argument("--out", default="results/rerun_independent")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    todo = list(DESIGNS) if a.design == "all" else [a.design]
    plan = {}
    for name in todo:
        cases, complete, reps = DESIGNS[name]
        topo = {k: v for k, v in topologies(10).items() if complete or k != "complete"}
        plan[name] = (topo, cases, reps, n_calls(topo, replicas=reps, rounds=a.rounds, cases=cases))
    total = sum(p[3] for p in plan.values())
    tin = rough_input_tokens()
    for name, (_, cases, reps, n) in plan.items():
        print(f"{name:10} cases={','.join(cases)} replicas={reps}  calls≤{n}")
    print(f"total calls≤{total}  ≈{tin} input tokens/call → rough cost "
          f"${total * (tin * 1 + 150 * 5) / 1e6:.0f} (Haiku 4.5 at $1/$5 per M, ~150 output tokens)")
    if a.dry_run:
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.base_url:
        b = OpenAICompatBackend(a.model, a.base_url, temperature=1.0, cache_dir=out / "cache")
    else:
        b = AnthropicBackend(a.model, cache_dir=out / "cache")   # same settings as the original
    pol = LLMPolicy(b, independent_agents=True)
    meta = {"model": a.model, "backend": b.name, "independent_agents": True, "rounds": a.rounds,
            "designs": {}}
    for name, (topo, cases, reps, _) in plan.items():
        t = time.time()
        f0 = pol.parse_failures
        run_grid(topo, pol, replicas=reps, rounds=a.rounds, out_path=out / f"{name}_trials.jsonl",
                 workers=a.workers, parallel_trials=a.parallel_trials, cases=cases, progress=False)
        meta["designs"][name] = {"cases": cases, "replicas": reps, "topologies": list(topo),
                                 "parse_failures": pol.parse_failures - f0,
                                 "seconds": round(time.time() - t)}
        print(f"{name}: {meta['designs'][name]}")
    meta.update({"calls": b.calls, "cache_hits": b.cache_hits,
                 "input_tokens": getattr(b, "input_tokens", None),
                 "output_tokens": getattr(b, "output_tokens", None)})
    if meta["input_tokens"]:
        meta["cost_usd"] = round(meta["input_tokens"] / 1e6 + 5 * meta["output_tokens"] / 1e6, 2)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("calls", "cache_hits", "input_tokens", "output_tokens")}))


if __name__ == "__main__":
    main()
