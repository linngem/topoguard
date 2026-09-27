#!/usr/bin/env python3
"""k-of-n corroboration gate for Claude Code (the orchestrator's `Stop` hook).

Verifiers write one file per finding in `.verification/claims/`:
    <CLAIM_ID>__<agent>.json
    {"claim_id": "C1", "claim": "...", "verdict": "confirms|refutes|inconclusive",
     "evidence": "...", "agent": "code-verifier", "group": "code-reading"}

Rules (derived from the topoguard experiments):
- INDEPENDENCE GROUPS are counted, not agents.
- ACCEPTED : ≥ K groups confirm and < K refute.
- CONFLICT : ≥ K confirm and ≥ K refute → human review.
- MINORITY : between 1 and K-1 groups confirm → NOT discarded: escalated as a hypothesis.
- PENDING  : fewer than N_MIN verifiers have answered.

As a `Stop` hook it blocks (exit 2) if there are pending findings, or if the orchestrator's
last message cites as confirmed ([C-ID]) something that is not ACCEPTED.

Manual use:  python3 .claude/hooks/corroboration_gate.py --report
Parameters:  TOPOGUARD_K (default 2), TOPOGUARD_N_MIN (default 3)
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

K = int(os.environ.get("TOPOGUARD_K", "2"))
N_MIN = int(os.environ.get("TOPOGUARD_N_MIN", "3"))
ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
VDIR = ROOT / ".verification"
CITE = re.compile(r"\[C-?(\w+)\](?!\s*\((?:minority|conflict|pending)\))", re.I)


def load_claims() -> dict[str, list[dict]]:
    claims: dict[str, list[dict]] = defaultdict(list)
    for f in sorted((VDIR / "claims").glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cid = str(d.get("claim_id", f.stem.split("__")[0])).upper().lstrip("C-")
        claims[cid].append(d)
    return claims


def decide(verdicts: list[dict]) -> dict:
    conf = {v.get("group", v.get("agent")) for v in verdicts if v.get("verdict") == "confirms"}
    ref = {v.get("group", v.get("agent")) for v in verdicts if v.get("verdict") == "refutes"}
    if len(verdicts) < N_MIN:
        status = "PENDING"
    elif len(conf) >= K and len(ref) >= K:
        status = "CONFLICT"
    elif len(conf) >= K:
        status = "ACCEPTED"
    elif conf:
        status = "MINORITY"
    else:
        status = "REJECTED"
    return {"status": status, "confirm": sorted(conf), "refute": sorted(ref),
            "n_verifiers": len(verdicts),
            "claim": next((v.get("claim") for v in verdicts if v.get("claim")), "")}


def write_decisions(dec: dict) -> None:
    VDIR.mkdir(exist_ok=True)
    (VDIR / "decisions.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Corroboration decisions (K={K}, N_MIN={N_MIN})", "",
             "| Finding | Status | Confirm | Refute | Claim |", "|---|---|---|---|---|"]
    for cid, d in sorted(dec.items()):
        lines.append(f"| C{cid} | {d['status']} | {', '.join(d['confirm'])} | "
                     f"{', '.join(d['refute'])} | {d['claim']} |")
    (VDIR / "DECISIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    claims = load_claims()
    dec = {cid: decide(v) for cid, v in claims.items()}
    if claims:
        write_decisions(dec)

    if "--report" in sys.argv:
        print((VDIR / "DECISIONS.md").read_text(encoding="utf-8") if claims else "No findings.")
        return 0

    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        event = {}
    if event.get("stop_hook_active"):
        return 0                      # avoid loops: we already blocked once in this turn
    if not claims:
        return 0                      # the protocol is not in use in this session

    problems = []
    pending = [f"C{c}" for c, d in dec.items() if d["status"] == "PENDING"]
    if pending:
        problems.append(f"Missing verifications (minimum {N_MIN}) for: {', '.join(pending)}.")
    msg = event.get("last_assistant_message") or ""
    wrong = sorted({f"C{m.upper()}" for m in CITE.findall(msg)
                    if dec.get(m.upper(), {}).get("status") not in (None, "ACCEPTED")})
    if wrong:
        problems.append("Your report presents as confirmed findings that have NOT been "
                        f"corroborated: {', '.join(wrong)}. Mark them '(minority)', "
                        "'(conflict)' or '(pending)', or remove them.")
    if problems:
        print("Corroboration gate: " + " ".join(problems) +
              " See .verification/DECISIONS.md.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
