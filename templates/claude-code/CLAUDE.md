# Corroboration protocol for critical findings

This project uses a **k-of-n** protocol so that findings asserted by a single agent are never
taken as established. It is based on the [topoguard](https://github.com/linngem/topoguard)
experiments: **plausible** errors spread among agents that can see each other, and a
corroboration gate with **independent** verifiers blocks them.

## When to apply it

Apply the protocol to any finding that will support a decision or a conclusion in the final
report: the cause of a bug, that something "is unused" and can be deleted, a vulnerability, a
behaviour change, a claim about an API or a version. It is not needed for trivial or easily
reversible tasks.

## How to apply it (you are the orchestrator)

1. **List the critical findings** with an identifier: C1, C2, C3…
2. **Launch the three verifiers in parallel**: `code-verifier`, `execution-verifier` and
   `docs-verifier`. Each uses a different model and a different evidence source; that is what
   makes them independent.
3. **Give them only the finding and the minimum context.** Never pass on what another verifier
   said, nor your own opinion on whether it is true: that reintroduces contagion.
4. **Do not pick the verdict yourself.** When they finish, run
   `python3 .claude/hooks/corroboration_gate.py --report` and use `.verification/DECISIONS.md`.
5. **Write the final report** citing each finding as `[C1]`, `[C2]`… according to its status:
   - `ACCEPTED` → you may present it as confirmed: `[C1]`.
   - `MINORITY` → **do not discard it**: present it as a hypothesis for human review,
     `[C2] (minority)`. Unexpected truths often start this way.
   - `CONFLICT` → `[C3] (conflict)`, with the evidence on both sides.
   - `REJECTED` → do not present it as a finding.

The stop hook blocks the end of the session if you cite as confirmed something that is not, or
if verifications are missing.

## Design rules (from the experiments)

- Count **independent sources, not agents**: three agents of the same model reading the same
  thing count as one.
- The demand level *k* must be **lower than the number of verifiers** (k = 2 with 3).
- Do not chain agents without intermediate verification: in a chain, a single upstream voice is
  "unanimity" for the next agent.
- Use debates or *agent teams* to **explore** ideas, not to **validate** facts.
