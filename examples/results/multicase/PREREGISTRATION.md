# Analysis plan fixed before running Haiku (27 Sep 2026, ~17:2x CEST)

*English translation of [`PREREGISTRO.md`](PREREGISTRO.md), which is kept unchanged as the
original record.*

- Plausibility: mean of the panel (Sonnet 5 + Opus 5.5, 3 replicates) in `plausibility.csv`
  (sha256 prefix `ebf2f8e89b89eadc`), used on the log-odds scale.
- Micro-experiment (Haiku 4.5, 12 cases, 4 replicates): n ∈ {2, 4, 8}, m according to
  `ADOPT_GRID`; retention with n = 4, m ∈ {0, 2, 4}.
- Question 1 (fraction vs count): compare the logistic models `fraction`, `count` and `both` by
  leave-one-case-out log-loss. The lowest log-loss wins.
- Question 2 (plausibility): plausibility coefficient with a 95 % CI from a case bootstrap.
- Question 3 (network): with the winning micro-experiment rule, WITHOUT fitting anything on
  network data, predict the reach in the pneumonia (existing data), inferior MI and lupus cases.
  Metrics: Spearman and MAE per finding type. Compare with the fraction rule.

## Changes after the analysis was run (none affect the results)

- **Identifiers translated to English** for publication: case ids (`neumonia` → `pneumonia`,
  `iam_inferior` → `inferior_mi`, …), topology names (`estrella` → `star`, `anillo` → `ring`,
  `completa` → `complete`) and model names (`fraccion` → `fraction`, `recuento` → `count`,
  `ambos` → `both`). Only the `case` column of `plausibility.csv` changed; the original file is
  kept as `plausibility_original_es.csv` and still matches the hash above. All results were
  re-computed after the migration and are identical.
- **Bootstrap resamples increased from 300 to 2,000** so the confidence intervals do not depend
  on the order of case names. Conclusions are unchanged.
- **Network data re-run with independent agents (29 Sep 2026).** The response cache was keyed on
  the prompt only, so agents receiving an identical prompt in the same round (the leaves of the
  star: 85 % of its agent-rounds) shared one sampled answer. Both network designs (pneumonia;
  inferior MI and lupus) were re-run with the same grid, prompts and seeds, each agent drawing
  its own answer, and Q3 was re-computed on the new data with nothing else changed. The
  pre-registered conclusions hold: `both` rule, Spearman truth 0.68 → 0.62, plausible error
  0.42 → 0.43; lupus star-hub contamination 80 % → 77 %; the k = 2 gate still contained every
  error. The original network files and the full comparison are kept in
  `examples/results/original_shared_samples/`.
