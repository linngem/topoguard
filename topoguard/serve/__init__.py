"""Serving layer: a panel of heterogeneous LLM agents behind the k-of-n corroboration gate.

Framework-free core. `bento_service.py` wraps it as a BentoML service; any other server
(FastAPI, Ray Serve…) can call `Panel.assess` directly.

Design notes
- Each agent is an existing `validation.backends` backend, so any OpenAI-compatible endpoint
  (vLLM, Ollama, llama.cpp, a BentoML/OpenLLM deployment) or the Anthropic API can be mixed.
- Independence groups default to the backend name: two agents on the same base model count
  once, exactly as in `CorroborationGate`.
- One gate per request: the service is stateless, so it can run with several workers.
- No response cache by default: clinical inputs should not be written to disk. The audit log
  stores findings and a hash, never the vignette.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..gate import CorroborationGate, Decision, Finding, Module, Status
from ..reliability import design_gate, effective_n
from ..validation.backends import Backend
from ..validation.scenario import system_prompt

_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class PanelAgent:
    name: str
    backend: Backend
    group: str | None = None          # independence group; defaults to backend.name

    @property
    def independence_group(self) -> str:
        return self.group or self.backend.name


@dataclass
class AgentReport:
    agent: str
    group: str
    present: list[str] | None          # None = unparseable or failed
    reason: str = ""
    error: str | None = None


@dataclass
class Assessment:
    k: int
    n_agents: int
    n_groups: int
    answered_groups: int                # groups that returned a parseable report
    degraded: bool                      # answered_groups < k: nothing could be accepted
    accepted: dict[str, bool]           # what downstream consumers may act on
    decisions: list[dict[str, Any]]
    reports: list[AgentReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_report(text: str, vocab: list[str]) -> tuple[list[str] | None, str]:
    """Same JSON contract as the validated experiments: {"presentes": [...], "razon": "..."}."""
    m = _JSON.search(text or "")
    if not m:
        return None, ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    keys = d.get("presentes", d.get("present", []))
    if not isinstance(keys, list):
        return None, ""
    return sorted({k for k in keys if k in vocab}), str(d.get("razon", d.get("reason", "")))


def render_case(vignette: str, private: dict[str, str] | None, agent: str) -> str:
    lines = [f"Caso: {vignette}"]
    if private and private.get(agent):
        lines.append("Datos solo disponibles para ti: " + private[agent])
    lines.append("¿Qué hallazgos están presentes?")
    return "\n".join(lines)


def _decision_dict(d: Decision) -> dict[str, Any]:
    return {**asdict(d), "status": d.status.value}


class Panel:
    """n agents answer independently (star topology with the gate at the hub: no agent sees
    another's output before corroboration, which is the configuration that kept every error
    in its source agent in the experiments)."""

    def __init__(self, agents: list[PanelAgent], k: int, audit_path: str | Path | None = None,
                 min_confidence: float = 0.0):
        if len({a.name for a in agents}) != len(agents):
            raise ValueError("agent names must be unique")
        self.agents = agents
        self.modules = [Module(a.name, a.independence_group) for a in agents]
        self.n_groups = len({m.independence_group for m in self.modules})
        CorroborationGate(self.modules, k)          # validates k against the groups
        self.k = k
        self.audit_path = audit_path
        self.min_confidence = min_confidence

    @classmethod
    def with_designed_k(cls, agents: list[PanelAgent], *, p_fp: float, sens: float,
                        rho: float = 0.0, max_false_acceptance: float = 1e-3, **kw) -> "Panel":
        """Chooses k with `design_gate` over the number of independence groups."""
        n = len({a.independence_group for a in agents})
        d = design_gate(n, p_fp, sens, rho, max_false_acceptance)
        if d is None:
            raise ValueError(f"no k ≤ {n} meets FAR ≤ {max_false_acceptance} with rho={rho}; "
                             "add more independent groups (different base models / data)")
        return cls(agents, d.k, **kw)

    async def _ask(self, a: PanelAgent, system: str, user: str, vocab: list[str]) -> AgentReport:
        try:
            text = await asyncio.to_thread(a.backend.complete, system, user)
        except Exception as e:  # noqa: BLE001 — one failing agent must not break the panel
            return AgentReport(a.name, a.independence_group, None, error=f"{type(e).__name__}: {e}")
        keys, reason = parse_report(text, vocab)
        return AgentReport(a.name, a.independence_group, keys, reason,
                           None if keys is not None else "unparseable response")

    async def assess(self, vignette: str, vocab: list[str],
                     private: dict[str, str] | None = None) -> Assessment:
        if not vocab:
            raise ValueError("vocab must list the candidate finding keys")
        system = system_prompt(vocab)
        reports = await asyncio.gather(*(
            self._ask(a, system, render_case(vignette, private, a.name), vocab)
            for a in self.agents))

        findings = [Finding(r.agent, key, True, evidence=r.reason)
                    for r in reports if r.present for key in r.present]
        gate = CorroborationGate(self.modules, self.k, audit_path=self.audit_path,
                                 min_confidence=self.min_confidence)
        decisions = gate.evaluate(findings)
        answered = len({r.group for r in reports if r.present is not None})
        return Assessment(
            k=self.k, n_agents=len(self.agents), n_groups=self.n_groups,
            answered_groups=answered, degraded=answered < self.k,
            accepted=gate.broadcast(),
            decisions=[_decision_dict(d) for d in decisions],
            reports=list(reports))

    def describe(self, rho: float = 0.0) -> dict[str, Any]:
        return {"k": self.k, "n_agents": len(self.agents), "n_groups": self.n_groups,
                "effective_n": effective_n(self.n_groups, rho),
                "agents": [{"name": a.name, "group": a.independence_group,
                            "backend": a.backend.name} for a in self.agents]}


def backend_from_spec(spec: dict[str, Any]) -> Backend:
    """{"kind": "openai", "model": ..., "base_url": ...} | {"kind": "anthropic", "model": ...}.
    Response caching is off unless `cache_dir` is given explicitly."""
    from ..validation.backends import AnthropicBackend, OpenAICompatBackend
    spec = dict(spec)
    kind = spec.pop("kind")
    spec.setdefault("cache_dir", None)
    if kind == "openai":
        return OpenAICompatBackend(**spec)
    if kind == "anthropic":
        return AnthropicBackend(**spec)
    raise ValueError(f"unknown backend kind: {kind!r}")


def load_panel(config: str | Path | dict[str, Any]) -> Panel:
    """JSON config:
    {"k": 2 | null, "design": {"p_fp":..,"sens":..,"rho":..,"max_false_acceptance":..},
     "audit_path": "audit.jsonl",
     "agents": [{"name": "qwen", "group": "qwen-family",
                 "backend": {"kind": "openai", "model": "...", "base_url": "http://..."}}]}
    Give either `k` or `design`."""
    cfg = config if isinstance(config, dict) else json.loads(Path(config).read_text("utf-8"))
    agents = [PanelAgent(a["name"], backend_from_spec(a["backend"]), a.get("group"))
              for a in cfg["agents"]]
    kw = {"audit_path": cfg.get("audit_path"), "min_confidence": cfg.get("min_confidence", 0.0)}
    if cfg.get("k") is not None:
        return Panel(agents, int(cfg["k"]), **kw)
    return Panel.with_designed_k(agents, **cfg["design"], **kw)


__all__ = ["AgentReport", "Assessment", "Panel", "PanelAgent", "Status", "backend_from_spec",
           "load_panel", "parse_report", "render_case"]
