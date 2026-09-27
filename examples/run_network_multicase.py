"""Phase-2 multi-case network experiment (same design for each new case).

    PYTHONPATH=.. python run_network_multicase.py --cases inferior_mi lupus
"""
import argparse
import time
from pathlib import Path

from run_validation import topologies
from topoguard.validation import AnthropicBackend, LLMPolicy, n_calls, run_grid

ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="+", default=["inferior_mi", "lupus"])
ap.add_argument("--model", default="claude-haiku-4-5-20251001")
ap.add_argument("--replicas", type=int, default=4)
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--out", default="results/multicase")
ap.add_argument("--parallel-trials", type=int, default=16)
a = ap.parse_args()

topo = {k: v for k, v in topologies(10).items() if k != "complete"}
print("Expected calls (upper bound, before caching):",
      n_calls(topo, replicas=a.replicas, rounds=a.rounds, cases=a.cases))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
b = AnthropicBackend(a.model, cache_dir=out / "cache_network")
pol = LLMPolicy(b)
t = time.time()
run_grid(topo, pol, replicas=a.replicas, rounds=a.rounds, out_path=out / "network_trials.jsonl",
         workers=10, parallel_trials=a.parallel_trials, cases=tuple(a.cases), progress=False)
print(f"time {time.time() - t:.0f}s calls={b.calls} cache={b.cache_hits} "
      f"parse_failures={pol.parse_failures} in={b.input_tokens} out={b.output_tokens} "
      f"cost≈${b.input_tokens / 1e6 + 5 * b.output_tokens / 1e6:.2f}")
