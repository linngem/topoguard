# Original network runs (shared samples) — superseded

These are the first Haiku 4.5 network runs (pneumonia; inferior MI and lupus), kept for
transparency. The response cache was keyed on the prompt only, so agents that received an
identical prompt in the same round (the leaves of the star, 85 % of its agent-rounds) shared one
sampled answer instead of drawing their own.

Both designs were re-run with independent agents; the re-run is the official data
(`../anthropic_claude-haiku-4-5-20251001/trials.jsonl`, `../multicase/network_trials.jsonl`,
metadata in `../network_rerun_meta.json`). No conclusion changed.

- `comparison/report.md`: original vs re-run, per claim and per topology, with 95 % CIs
  (`cd examples && PYTHONPATH=.. python compare_rerun.py`).
- `*_comparison.csv`, `*_fits.json`, `network_pred*`: the analysis outputs computed from these
  original runs.
