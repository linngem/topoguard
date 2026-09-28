"""BentoML service for a topoguard panel.

    pip install "topoguard[serve]"
    TOPOGUARD_PANEL=panel.json bentoml serve topoguard.serve.bento_service:TopoguardPanel

Endpoints (POST, JSON):
    /assess    {"vignette": str, "vocab": [keys], "private": {agent: str} | null}
    /describe  {}                       → agents, groups, k, effective n
    /design    {"n", "p_fp", "sens", "rho", "max_false_acceptance"} → smallest k

The service is stateless (one gate per request), so `workers` can be raised freely. Each
worker writes its own audit file (`<audit_path>.w<index>`) to avoid interleaved appends.
The LLM agents themselves are served elsewhere (vLLM, Ollama, a BentoML/OpenLLM deployment,
the Anthropic API); this service only orchestrates and gates.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bentoml

from . import load_panel
from ..reliability import design_gate


def _worker_audit(path: str | None) -> str | None:
    if not path:
        return None
    idx = getattr(bentoml.server_context, "worker_index", None) or os.getpid()
    p = Path(path)
    return str(p.with_name(f"{p.stem}.w{idx}{p.suffix}"))


@bentoml.service(
    workers=int(os.environ.get("TOPOGUARD_WORKERS", "1")),
    traffic={"timeout": int(os.environ.get("TOPOGUARD_TIMEOUT", "300")),
             "concurrency": int(os.environ.get("TOPOGUARD_CONCURRENCY", "16"))},
)
class TopoguardPanel:
    def __init__(self) -> None:
        self.panel = load_panel(os.environ.get("TOPOGUARD_PANEL", "panel.json"))
        self.panel.audit_path = _worker_audit(self.panel.audit_path)

    @bentoml.api
    async def assess(self, vignette: str, vocab: list[str],
                     private: dict[str, str] | None = None) -> dict[str, Any]:
        return (await self.panel.assess(vignette, vocab, private)).to_dict()

    @bentoml.api
    def describe(self, rho: float = 0.0) -> dict[str, Any]:
        return self.panel.describe(rho)

    @bentoml.api
    def design(self, n: int, p_fp: float, sens: float, rho: float = 0.0,
               max_false_acceptance: float = 1e-3) -> dict[str, Any]:
        d = design_gate(n, p_fp, sens, rho, max_false_acceptance)
        return asdict(d) if d else {"k": None, "reason": "no k meets the bound; add diversity"}
