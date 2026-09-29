# Two-source error experiment — analysis plan fixed before running any model (29 Sep 2026)

**Why.** With a single error source, a local k = 2 gate cannot pass the error by construction:
no agent hears it from two neighbours unless another agent produces it spontaneously. The 100 %
containment reported in phases 1–2 therefore mainly shows that agents almost never generate the
planted error on their own. This experiment tests the gate where it can fail.

**Design** (`run_two_sources.py`). The same false datum is given to two agents, placed in the
worst case for the gate: the pair with the most common neighbours (`pair_seeds`; star: two
leaves sharing the hub; ring: two nodes at distance 2; small-world and clustered scale-free:
3 common neighbours; complete: 8). Cases: pneumonia, inferior MI, lupus. Topologies: star, ring,
small-world, clustered scale-free, complete. Plausible and implausible error; no gate and local
k = 2; 4 replicates; 4 rounds; 3 truth seeds as before (never a source); independent agents.
Haiku 4.5 first (provider-default temperature, 400 output tokens, as in phases 1–2); the same
design is then run on the Phase-3 panel.

**Primary estimands** (per model, per topology, pooled over the 3 cases; 95 % CI by bootstrap
over trials), from the final round:
1. **Escape rate under the gate**: share of gated trials in which at least one non-source agent
   asserts the error.
2. **Second-order spread under the gate**: share of gated trials in which the error reaches an
   agent that does not neighbour both sources, i.e. it passes the gate again downstream.
3. **Reach** with and without the gate (share of the 10 agents; 0.20 = sources only), and the
   reach of the true finding (cost of the gate).

**Expectations stated in advance** (descriptive, no formal test):
- Without a gate, a plausible error from two sources reaches more agents than the same error
  from one source (phases 1–2, Haiku, same case and topology).
- Under the gate, the plausible error escapes in a non-zero share of trials where the sources
  share neighbours; second-order spread is rarer than first-order escape.
- The implausible error escapes in ≤ 10 % of gated trials even with two sources.

## Prospective test of the network-damped rule (Haiku 4.5)

The Phase-2 micro-experiment rule over-predicts how far errors travel in the network. A single
network-damping parameter (logit of adoption lowered by λ, retention unchanged) was fitted
post hoc on the single-source network data: **λ = 1.2**, frozen in
`../multicase/posthoc_prediction.json` (`posthoc_prediction.py`) before any two-source data
existed. On those data it halved the mean absolute error for plausible errors (13.6 → 6.4 pp;
6.6 pp leave-one-case-out) without changing rank correlations.

On the two-source networks, with nothing fitted on them, `analyze_two_sources.py` compares the
undamped rule (λ = 0) and the damped rule (λ = 1.2):
- **Expected:** lower mean absolute error for the plausible error with the damped rule.
- **Not expected:** a change in Spearman correlation (a uniform logit shift barely reorders
  conditions); it is reported for completeness.
If the damped rule does not reduce the error on these new data, the damping is reported as
not replicated and the undamped rule remains the main result.

## Changes after the analysis was run

*(none yet)*
