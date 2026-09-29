# Claude Code template: k-of-n corroboration protocol

Turns Claude Code's default architecture (a **star**: the main agent launches sub-agents that
report back to it) into a star **with a gate**: the centre does not relay what it is told, it
only accepts what several independent sources confirm.

## Contents

| File | Purpose |
|---|---|
| `CLAUDE.md` | Protocol for the orchestrator: when and how to verify, how to cite findings |
| `.claude/agents/code-verifier.md` | Verifier that reads code (Sonnet) |
| `.claude/agents/execution-verifier.md` | Verifier that runs tests and commands (Haiku) |
| `.claude/agents/docs-verifier.md` | Verifier that checks documentation (Opus) |
| `.claude/hooks/block_peek.py` | Prevents a verifier from reading another's conclusions |
| `.claude/hooks/corroboration_gate.py` | Computes the k-of-n gate and blocks the end if the report overstates |
| `.claude/settings.json` | Registers the stop hook and the K and N_MIN parameters |

## Installation

Copy the contents of this folder into the root of your project. If you already have a
`CLAUDE.md` or `.claude/settings.json`, merge them with yours instead of overwriting.

```bash
cp -r templates/claude-code/.claude  /path/to/your/project/
cat templates/claude-code/CLAUDE.md >> /path/to/your/project/CLAUDE.md
echo ".verification/" >> /path/to/your/project/.gitignore
```

## Usage

Ask Claude something like *"Investigate why login fails; apply the corroboration protocol to
every candidate cause"*. The orchestrator lists findings (C1, C2…), launches the three
verifiers and writes the report according to `.verification/DECISIONS.md`.

The IDs C1, C2… are **not configured in advance**: the orchestrator assigns them at runtime to
whatever candidate findings it comes up with. The only settings fixed beforehand are `K`,
`N_MIN` and the verifier definitions.

### Example

Users are logged out after one hour although sessions should last 24 hours. The orchestrator
finds three candidate causes and gets these verdicts:

| Claim | code-verifier | execution-verifier | docs-verifier | Decision |
|---|---|---|---|---|
| C1 `SESSION_TTL = 3600` in a legacy config file | confirms | refutes (file not loaded) | inconclusive | MINORITY |
| C2 expiry compared in local time instead of UTC | confirms | confirms (reproduced in a test) | inconclusive | ACCEPTED |
| C3 load-balancer affinity timeout of 60 min | inconclusive | inconclusive | confirms | MINORITY |

The report states C2 as the cause and lists C1 and C3 as hypotheses for human review. C1 is the
typical *plausible error*: it fits the symptom perfectly, and a team of agents that could see
each other would likely have converged on it. See the
[worked example in the article](../../ARTICLE.md#worked-example-debugging-in-software-development).

## Where each design decision comes from

| Decision | Experimental result behind it |
|---|---|
| Verifiers that cannot see each other | With neighbours visible, a unanimous plausible error is adopted 94–100 % of the time |
| Different models and sources | With correlated errors (ρ = 0.2) no *k* meets the false-positive bound |
| k = 2 with 3 verifiers | The gate needs *k* lower than the number of sources; otherwise it also blocks the truth |
| Minorities are escalated, not deleted | The gate mainly slows down unexpected truths |
| The orchestrator does not give its opinion when delegating | In a star, what the centre says contaminated 57–77 % of the team |

## Limitations

- It is a protocol requested through instructions plus two hooks. The read hook blocks access
  to `.verification/`, but it cannot guarantee independence if the orchestrator leaks its own
  opinion when delegating.
- The experiments used clinical cases and Claude Haiku 4.5; the protocol has not been
  empirically validated on programming tasks.
- Three verifiers cost roughly three times the tokens: use it for critical findings.
