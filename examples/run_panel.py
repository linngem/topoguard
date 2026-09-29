"""Model panel: re-runs the Phase-2 design (micro-experiment + network) with any model.

Same prompts, seeds, grid and parser as the Haiku 4.5 run, so results are directly comparable.
Each model gets its own folder `results/panel/<tag>/` with `micro.jsonl`, `network_trials.jsonl`
and `meta.json` (model, digest, sampling settings, commit, call counts). Calls are cached, so an
interrupted run resumes where it stopped.

    # Ollama (OpenAI-compatible endpoint); temperature fixed to 1.0 to match the Anthropic default
    PYTHONPATH=.. python run_panel.py --model gpt-oss:20b  --base-url http://localhost:11434/v1
    PYTHONPATH=.. python run_panel.py --model qwen3:30b    --base-url http://localhost:11434/v1
    PYTHONPATH=.. python run_panel.py --model gemma3:27b   --base-url http://localhost:11434/v1

    # Haiku 4.5 again, with independent agents (needed: the original network run shared samples)
    PYTHONPATH=.. python run_panel.py --backend anthropic --model claude-haiku-4-5-20251001 \
        --max-tokens 400 --stage network

    # only count calls, or run one stage
    PYTHONPATH=.. python run_panel.py --model gemma3:27b --base-url ... --dry-run
    PYTHONPATH=.. python run_panel.py --model gemma3:27b --base-url ... --stage micro
"""
import argparse
import datetime as dt
import json
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

from run_validation import topologies
from topoguard.validation import (CASES, AnthropicBackend, LLMPolicy, OpenAICompatBackend, micro,
                                  n_calls, run_grid)

NETWORK_CASES = ["pneumonia", "inferior_mi", "lupus"]   # the three Phase-2 network cases


def ollama_digest(base_url: str | None, model: str) -> str | None:
    """Exact weights used (Ollama tags are mutable: the digest is what makes a run reproducible)."""
    if not base_url or "11434" not in base_url:
        return None
    try:
        root = base_url.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(f"{root}/api/tags", timeout=5) as r:
            for m in json.load(r).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    return f"{m.get('digest')} ({m.get('details', {}).get('quantization_level')})"
    except Exception:  # noqa: BLE001 — metadata only, never blocks the run
        return None
    return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--tag", help="folder name (default: derived from the model)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="same value for every model; 1.0 = Anthropic default used for Haiku")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="reasoning models need room before the JSON; truncations are reported")
    ap.add_argument("--extra-body", default=None,
                    help='JSON passed to the API, e.g. \'{"reasoning_effort": "low"}\'')
    ap.add_argument("--stage", choices=["micro", "network", "all"], default="all")
    ap.add_argument("--micro-reps", type=int, default=4)
    ap.add_argument("--network-cases", nargs="+", default=NETWORK_CASES)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4, help="concurrent calls (≈ OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--out", default="results/panel")
    ap.add_argument("--shared-samples", action="store_true",
                    help="legacy behaviour: agents with an identical prompt share one cached answer")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tag = a.tag or a.model.replace("/", "_").replace(":", "_")
    out = Path(a.out) / tag
    topo = {k: v for k, v in topologies(10).items() if k != "complete"}
    n_micro = len(micro.jobs(list(CASES), a.micro_reps))
    n_net = n_calls(topo, replicas=a.replicas, rounds=a.rounds, cases=a.network_cases)
    print(f"[{tag}] expected calls: micro={n_micro}  network≤{n_net}  total≤{n_micro + n_net}")
    if a.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(a.extra_body) if a.extra_body else None
    if a.backend == "openai":
        b = OpenAICompatBackend(a.model, a.base_url, temperature=a.temperature,
                                max_tokens=a.max_tokens, cache_dir=out / "cache", extra_body=extra)
    else:
        b = AnthropicBackend(a.model, temperature=a.temperature, max_tokens=a.max_tokens,
                             cache_dir=out / "cache", extra_kwargs=extra)

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"model": a.model, "backend": b.name, "temperature": a.temperature,
                 "max_tokens": a.max_tokens, "extra_body": extra,
                 "weights": ollama_digest(a.base_url if a.backend == "openai" else None, a.model),
                 "commit": git_commit(), "python": platform.python_version(),
                 "micro_reps": a.micro_reps, "network_cases": a.network_cases,
                 "replicas": a.replicas, "rounds": a.rounds,
                 "independent_agents": not a.shared_samples})

    if a.stage in ("micro", "all"):
        t = time.time()
        micro.run(b, reps=a.micro_reps, out_path=out / "micro.jsonl", workers=a.workers)
        recs = micro.load(out / "micro.jsonl")
        meta["micro"] = {"responses": len(recs),
                         "parse_failures": sum(r["claims"] is None for r in recs),
                         "seconds": round(time.time() - t)}
        print(f"[{tag}] micro done in {meta['micro']['seconds']}s, "
              f"parse failures {meta['micro']['parse_failures']}/{len(recs)}")

    if a.stage in ("network", "all"):
        pol = LLMPolicy(b, independent_agents=not a.shared_samples)
        t = time.time()
        run_grid(topo, pol, replicas=a.replicas, rounds=a.rounds,
                 out_path=out / "network_trials.jsonl", workers=a.workers,
                 parallel_trials=max(1, a.workers // 2), cases=tuple(a.network_cases),
                 progress=True)
        meta["network"] = {"parse_failures": pol.parse_failures, "seconds": round(time.time() - t)}

    meta.update({"calls": b.calls, "cache_hits": b.cache_hits,
                 "input_tokens": getattr(b, "input_tokens", None),
                 "output_tokens": getattr(b, "output_tokens", None),
                 "truncated": getattr(b, "truncated", None),
                 "finished": dt.datetime.now().isoformat(timespec="seconds")})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if meta.get("truncated"):
        print(f"[{tag}] WARNING: {meta['truncated']} responses hit max_tokens — raise --max-tokens "
              "or lower reasoning effort, otherwise parse failures bias the adoption rates")


if __name__ == "__main__":
    main()
