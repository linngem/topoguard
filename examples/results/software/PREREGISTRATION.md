# Out-of-domain control (software incidents) — plan fixed before any run (29 Sep 2026)

**Question.** Are plausibility filtering and conformity to plausible errors general properties of
LLM agents, or specific to clinical reasoning?

**Stimuli.** Two software-incident cases (`SOFTWARE_CASES` in `topoguard/validation/cases.py`):
`session_logout` (web sessions expire after about one hour) and `duplicate_orders` (a nightly
import duplicates some orders). Same structure as the clinical cases: an evident base finding,
a subtle true cause, and three false causes with a priori high, intermediate and low
plausibility. Written in Spanish like the clinical stimuli, so language does not differ. Only
the domain phrase of the agent's system prompt changes ("diagnóstico de incidencias de software"
instead of "diagnóstico clínico"); the clinical prompts are byte-identical to phases 1–3.

**Design** (`run_software_control.py`). Micro-experiment only, phase-2 grid (n ∈ {2, 4, 8},
`ADOPT_GRID`, `KEEP_GRID`), 4 replicates, Claude Haiku 4.5 (400 output tokens, provider-default
temperature): 472 responses. Plausibility rated by the phase-2 panel with the same settings
(Sonnet 5, Opus 5.5 with adaptive thinking at low effort; 3 replicates each), with a
software-engineer framing of the same question.

**Analysis** (`analyze_software_control.py`).
- *Manipulation check.* Panel plausibility is expected to follow truth > high > low in each case;
  if it does not, the measured plausibility (not the a priori label) is used, as in phase 2.
- **S1 (primary).** In both cases, adoption of the high-plausibility error under unanimity
  (m = n, n ∈ {4, 8}) exceeds adoption of the low-plausibility error.
- **S2.** The phase-2 `both` rule fitted on the 12 clinical cases, applied without refitting,
  has lower log-loss on the software adoption data than the clinical social-pressure-only rule
  (plausibility transfers across domains).
- **S3 (descriptive).** Coefficients of the `both` rule fitted on the software data alone, and
  adoption curves next to the clinical ones (fig. 10). With two cases there are no case-level
  confidence intervals; this is a control, not a replication.

**Quality rule.** More than 5 % parse failures → reported and interpreted with caution.

## Changes after the analysis was run

*(none yet)*
