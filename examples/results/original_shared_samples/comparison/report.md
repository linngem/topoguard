# Haiku 4.5 network: shared samples (original) vs independent agents (re-run)

## Exposure of the original run

Share of agent-rounds whose prompt coincided with another agent's (→ shared answer):

- star: 85%
- small_world: 0%
- scale_free_cluster: 0%
- ring: 0%
- complete: 0%

## Claims of the article

| check | original | re-run |
|---|---|---|
| max final reach, plausible error at star hub, no gate | 80% | 77% |
| gated trials (k = 2) where the error stayed in its source | 100% (n=188) | 100% (n=188) |
| Q3 micro→network error_plausible: Spearman / MAE | 0.42 / 0.130 | 0.43 / 0.136 |
| Q3 micro→network error_implausible: Spearman / MAE | 0.73 / 0.049 | 0.74 / 0.051 |
| Q3 micro→network truth: Spearman / MAE | 0.68 / 0.068 | 0.62 / 0.072 |

## Final reach by topology (pooled over cases and positions)

| topology | error | gate k | shared prompts (old) | error reach old → new (Δ, 95 % CI) | truth reach old → new (Δ, 95 % CI) |
|---|---|---|---|---|---|
| complete | implausible | 1 | 0% | 0.10 → 0.10 (+0.00, [+0.00, +0.00]) | 1.00 → 1.00 (+0.00, [+0.00, +0.00]) |
| complete | implausible | 2 | 0% | 0.10 → 0.10 (+0.00, [+0.00, +0.00]) | 1.00 → 1.00 (+0.00, [+0.00, +0.00]) |
| complete | plausible | 1 | 0% | 0.32 → 0.40 (+0.08, [-0.18, +0.33]) | 1.00 → 1.00 (+0.00, [+0.00, +0.00]) |
| complete | plausible | 2 | 0% | 0.10 → 0.10 (+0.00, [+0.00, +0.00]) | 1.00 → 1.00 (+0.00, [+0.00, +0.00]) |
| ring | implausible | 1 | 0% | 0.06 → 0.05 (-0.00, [-0.03, +0.02]) | 0.84 → 0.84 (+0.00, [-0.16, +0.16]) |
| ring | implausible | 2 | 0% | 0.05 → 0.05 (+0.01, [-0.02, +0.04]) | 0.81 → 0.81 (+0.00, [-0.19, +0.19]) |
| ring | plausible | 1 | 0% | 0.19 → 0.18 (-0.01, [-0.08, +0.06]) | 0.80 → 0.79 (-0.01, [-0.20, +0.17]) |
| ring | plausible | 2 | 0% | 0.10 → 0.10 (+0.00, [+0.00, +0.00]) | 0.79 → 0.79 (+0.00, [-0.17, +0.18]) |
| scale_free_cluster | implausible | 1 | 0% | 0.06 → 0.06 (+0.00, [-0.03, +0.04]) | 0.97 → 0.96 (-0.01, [-0.05, +0.04]) |
| scale_free_cluster | implausible | 2 | 0% | 0.06 → 0.05 (-0.02, [-0.05, +0.01]) | 0.96 → 0.96 (+0.00, [-0.05, +0.07]) |
| scale_free_cluster | plausible | 1 | 0% | 0.18 → 0.17 (-0.01, [-0.07, +0.05]) | 0.97 → 0.95 (-0.02, [-0.08, +0.04]) |
| scale_free_cluster | plausible | 2 | 0% | 0.10 → 0.10 (+0.00, [+0.00, +0.00]) | 0.91 → 0.88 (-0.03, [-0.13, +0.08]) |
| small_world | implausible | 1 | 0% | 0.06 → 0.05 (-0.00, [-0.03, +0.02]) | 1.00 → 1.00 (+0.00, [+0.00, +0.00]) |
| small_world | implausible | 2 | 0% | 0.06 → 0.05 (-0.00, [-0.03, +0.02]) | 1.00 → 1.00 (+0.00, [+0.00, +0.00]) |
| small_world | plausible | 1 | 0% | 0.26 → 0.18 (-0.08, [-0.20, +0.04]) | 1.00 → 1.00 (+0.00, [+0.00, +0.00]) |
| small_world | plausible | 2 | 0% | 0.10 → 0.10 (+0.00, [+0.00, +0.00]) | 1.00 → 1.00 (+0.00, [+0.00, +0.00]) |
| star | implausible | 1 | 85% | 0.08 → 0.09 (+0.01, [-0.04, +0.05]) | 0.98 → 0.99 (+0.01, [-0.01, +0.04]) |
| star | implausible | 2 | 85% | 0.05 → 0.05 (-0.00, [-0.03, +0.02]) | 0.79 → 0.79 (-0.00, [-0.17, +0.15]) |
| star | plausible | 1 | 85% | 0.28 → 0.29 (+0.01, [-0.16, +0.17]) | 0.94 → 0.92 (-0.01, [-0.10, +0.07]) |
| star | plausible | 2 | 85% | 0.10 → 0.10 (+0.00, [+0.00, +0.00]) | 0.81 → 0.81 (+0.00, [-0.16, +0.16]) |

Ring never shared prompts, so its Δ is the re-sampling noise floor.
