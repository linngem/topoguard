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

## Where each design decision comes from

| Decision | Experimental result behind it |
|---|---|
| Verifiers that cannot see each other | With neighbours visible, a unanimous plausible error is adopted 94–100 % of the time |
| Different models and sources | With correlated errors (ρ = 0.2) no *k* meets the false-positive bound |
| k = 2 with 3 verifiers | The gate needs *k* lower than the number of sources; otherwise it also blocks the truth |
| Minorities are escalated, not deleted | The gate mainly slows down unexpected truths |
| The orchestrator does not give its opinion when delegating | In a star, what the centre says contaminated 47–80 % of the team |

## Limitations

- It is a protocol requested through instructions plus two hooks. The read hook blocks access
  to `.verification/`, but it cannot guarantee independence if the orchestrator leaks its own
  opinion when delegating.
- The experiments used clinical cases and Claude Haiku 4.5; the protocol has not been
  empirically validated on programming tasks.
- Three verifiers cost roughly three times the tokens: use it for critical findings.
