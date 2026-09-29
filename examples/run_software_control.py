"""Out-of-domain control: the micro-experiment on two software-incident cases.

Question: are plausibility filtering and conformity properties of LLM agents in general, or
specific to clinical reasoning? The two cases (topoguard.validation.cases.SOFTWARE_CASES) have
the same structure as the clinical ones (a subtle true cause and three false causes of graded a
priori plausibility), are also in Spanish, and only the domain phrase of the agent's system
prompt changes.

Stages (run in order; answers are cached, so re-running resumes):
  plausibility  the same panel as phase 2 rates every candidate cause
                (Claude Sonnet 5: defaults, 300 tokens; Claude Opus 5.5: adaptive thinking at low
                effort, 2,000 tokens; 3 replicates each)                      48 calls
  micro         Haiku 4.5 agents, phase-2 grid, 4 replicates                  472 calls

    cd examples
    PYTHONPATH=.. python run_software_control.py --dry-run
    PYTHONPATH=.. python run_software_control.py
    PYTHONPATH=.. python analyze_software_control.py
"""
import argparse
import json
import time
from pathlib import Path

from topoguard.validation import SOFTWARE_CASES, AnthropicBackend, micro
from topoguard.validation.plausibility import rate

OUT = Path("results/software")
PANEL = {  # same settings as the phase-2 panel (docs/CASES_AND_PROMPTS.md)
    "sonnet5": dict(model="claude-sonnet-5", max_tokens=300),
    "opus5.5": dict(model="claude-opus-5-5", max_tokens=2000,
                    extra_kwargs={"thinking": {"type": "adaptive"},
                                  "output_config": {"effort": "low"}}),
}
AGENT = "claude-haiku-4-5-20251001"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["plausibility", "micro", "all"], default="all")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cases = list(SOFTWARE_CASES)
    n_micro = len(micro.jobs(cases, a.reps))
    print(f"cases: {cases}\nplausibility calls: {len(cases) * 4 * len(PANEL) * 3}"
          f"   micro calls: {n_micro}")
    if a.dry_run:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"cases": cases, "agent": AGENT, "panel": {k: v["model"] for k, v in PANEL.items()}}

    if a.stage in ("plausibility", "all"):
        backends = {name: AnthropicBackend(cache_dir=OUT / "cache_panel", **cfg)
                    for name, cfg in PANEL.items()}
        rate(backends, reps=3, out_csv=OUT / "plausibility.csv", cases=cases)
        meta["plausibility_calls"] = sum(b.calls for b in backends.values())
        print("plausibility ✓", OUT / "plausibility.csv")

    if a.stage in ("micro", "all"):
        b = AnthropicBackend(AGENT, max_tokens=400, cache_dir=OUT / "cache_micro")
        t = time.time()
        micro.run(b, cases=cases, reps=a.reps, out_path=OUT / "micro.jsonl", workers=16)
        recs = micro.load(OUT / "micro.jsonl")
        meta["micro"] = {"responses": len(recs),
                         "parse_failures": sum(r["claims"] is None for r in recs),
                         "calls": b.calls, "seconds": round(time.time() - t),
                         "input_tokens": b.input_tokens, "output_tokens": b.output_tokens}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
