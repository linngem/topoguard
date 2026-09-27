#!/usr/bin/env python3
"""`PreToolUse` hook for verifiers: prevents reading the other verifiers' conclusions.

Independence is what makes the k-of-n gate work: in the experiments, when agents can see what
the others assert, a plausible error repeated by all is adopted 94–100 % of the time. This hook
blocks (exit 2) any read, search or command touching `.verification/`; writing one's own file
with the Write tool is still allowed.
"""
import json
import sys

try:
    event = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError):
    sys.exit(0)

tool = event.get("tool_name", "")
payload = json.dumps(event.get("tool_input", {}), ensure_ascii=False)

if tool in {"Read", "Grep", "Glob", "Bash", "NotebookRead"} and ".verification" in payload:
    print("Independence: you cannot look at other verifiers' conclusions "
          "(.verification/). Verify with your own source of evidence.", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
