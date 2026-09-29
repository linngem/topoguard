# Phase 3 analysis plan — fixed before running any open-weight model (29 Sep 2026)

Phase 3 asks whether the Phase-2 findings generalise beyond Claude Haiku 4.5, and whether errors
are less correlated across model families than across repeated calls to the same model — the
empirical basis for counting *independence groups* rather than modules in the k-of-n gate.

## Models

| Family | Model (Ollama, local) | Role |
|---|---|---|
| Claude (Anthropic) | Claude Haiku 4.5 — existing Phase-2 data | reference |
| gpt-oss (OpenAI) | gpt-oss 20B | agent |
| Qwen3 (Alibaba) | Qwen3, ~30B class | agent |
| Gemma (Google) | Gemma, ~27B class | agent |
| Gemma (Google) | MedGemma, ~27B class | agent; same family as Gemma |

The exact tag, weights digest and quantisation of each model are recorded in its `meta.json` at
run time. If a model cannot be run (hardware, availability), it is replaced by one from a family
not yet in the panel, and the substitution is logged below before its data are analysed.

## Design (identical for every open-weight model)

- Same prompts, seeds, grids and parser as Phase 2 (`run_panel.py`).
- **Micro-experiment**: 12 cases, 4 replicates, `ADOPT_GRID` and `KEEP_GRID` unchanged (2,832
  responses per model).
- **Network**: pneumonia, inferior MI and lupus; 4 topologies (star, ring, small-world,
  clustered scale-free); 2 error types × 2 entry points × 2 gates (none, local k = 2);
  3 replicates; 4 rounds; every agent an independent draw.
- Sampling: temperature 1.0 for every model (the Anthropic default used for Haiku);
  `max_tokens` 1,024; `<think>` blocks removed before parsing.
- Plausibility: the Phase-2 panel scores (`../multicase/plausibility.csv`, sha256 prefix
  `ebf2f8e89b89eadc`), log-odds scale. Not re-rated.
- Haiku 4.5 enters with its Phase-2 micro-experiment and its independent-agent network re-run.

## Quality rule (fixed in advance)

A model whose micro-experiment parse failures **or** truncations exceed 5 % of responses is
re-run once with a larger `max_tokens` (or lower reasoning effort). If it still exceeds 5 %, it
is reported descriptively and excluded from the confirmatory tests below.

## Hypotheses and tests (`analyze_panel.py`, B = 2,000 case-bootstrap resamples)

**H1 — plausibility generalises.** For every model, the plausibility coefficient of the `both`
logistic rule is positive, with the lower bound of its 95 % CI above 0.

**H2 — conformity to plausible errors generalises.** For every model, adoption of the high-
plausibility error under unanimity (m = n) exceeds adoption of the low-plausibility error under
unanimity. The *level* of conformity per model is reported descriptively (no hypothesis).

**H3 — independence groups (primary).** On identical items, the residual error-adoption
correlation (after each model's own `both` rule) is lower **between families** than **within a
model** across replicates. Supported if the lower bound of the 95 % CI of
(mean within-model ρ − mean between-family ρ) is above 0. The same-family pair (Gemma ×
MedGemma) is reported separately and does not enter the test.
Families: `--family "<gemma tag>=gemma,<medgemma tag>=gemma"`.

**H4 — the firewall generalises.** For every model, in ≥ 95 % of gated (k = 2) network trials
the error does not leave its source agent.

**Secondary (Q3 per model).** The network is predicted from each model's own micro-experiment
rule with nothing fitted on network data; Spearman and MAE per finding type, as in Phase 2.

No multiplicity correction is applied: H3 is the single primary test; H1, H2 and H4 are
per-model replications and are reported model by model.

## Changes after the analysis was run

*(none yet)*
