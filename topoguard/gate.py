"""Workspace with a runtime k-of-n corroboration gate.

- Corroboration counts *independence groups*, not modules: two agents with the same base
  model reading the same source count as one.
- If a finding is corroborated both ways (present/absent) → CONFLICT, escalate to a human.
- Every decision is written to an audit log (JSON lines) for traceability."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable


class Status(str, Enum):
    ACCEPTED = "accepted"
    PENDING = "pending"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Module:
    name: str
    independence_group: str   # e.g. "imaging/modelA", "lab/rules", "history/modelB"


@dataclass(frozen=True)
class Finding:
    module: str
    key: str                  # concept; ideally a code (SNOMED CT, LOINC…)
    present: bool = True
    evidence: str = ""
    confidence: float = 1.0


@dataclass
class Decision:
    key: str
    status: Status
    present: bool | None
    support: dict[str, list[str]]           # group → modules supporting the winning polarity
    against: dict[str, list[str]] = field(default_factory=dict)


class CorroborationGate:
    def __init__(self, modules: Iterable[Module], k: int,
                 normalize: Callable[[str], str] = lambda s: s.strip().lower(),
                 min_confidence: float = 0.0, audit_path: str | Path | None = None):
        self.modules = {m.name: m for m in modules}
        n_groups = len({m.independence_group for m in self.modules.values()})
        if not 1 <= k <= n_groups:
            raise ValueError(f"k={k} but there are only {n_groups} independent groups")
        self.k = k
        self.normalize = normalize
        self.min_confidence = min_confidence
        self.audit_path = Path(audit_path) if audit_path else None
        self.workspace: dict[str, Decision] = {}

    def _groups(self, fs: list[Finding]) -> dict[str, list[str]]:
        g: dict[str, list[str]] = defaultdict(list)
        for f in fs:
            g[self.modules[f.module].independence_group].append(f.module)
        return dict(g)

    def evaluate(self, findings: Iterable[Finding]) -> list[Decision]:
        findings = list(findings)
        unknown = {f.module for f in findings} - self.modules.keys()
        if unknown:
            raise KeyError(f"Unregistered modules: {unknown}")
        by_key: dict[tuple[str, bool], list[Finding]] = defaultdict(list)
        for f in findings:
            if f.confidence >= self.min_confidence:
                by_key[(self.normalize(f.key), f.present)].append(f)

        decisions = []
        for key in sorted({k for k, _ in by_key}):
            pos, neg = self._groups(by_key.get((key, True), [])), self._groups(by_key.get((key, False), []))
            pos_ok, neg_ok = len(pos) >= self.k, len(neg) >= self.k
            if pos_ok and neg_ok:
                d = Decision(key, Status.CONFLICT, None, pos, neg)
            elif pos_ok or neg_ok:
                d = Decision(key, Status.ACCEPTED, pos_ok, pos if pos_ok else neg, neg if pos_ok else pos)
                self.workspace[key] = d
            else:
                best = pos if len(pos) >= len(neg) else neg
                d = Decision(key, Status.PENDING, None, best, {})
            decisions.append(d)
        self._audit(findings, decisions)
        return decisions

    def broadcast(self) -> dict[str, bool]:
        """The only thing downstream modules see: corroborated findings."""
        return {k: d.present for k, d in self.workspace.items() if d.status is Status.ACCEPTED}

    def _audit(self, findings: list[Finding], decisions: list[Decision]) -> None:
        if not self.audit_path:
            return
        payload = [asdict(f) for f in findings]
        rec = {
            "ts": time.time(),
            "k": self.k,
            "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "findings": payload,
            "decisions": [{**asdict(d), "status": d.status.value} for d in decisions],
        }
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
