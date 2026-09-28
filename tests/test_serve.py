"""Tests for topoguard.serve against a local fake OpenAI-compatible server (no network, no keys)."""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from topoguard.serve import Panel, PanelAgent, load_panel, parse_report
from topoguard.validation.backends import OpenAICompatBackend

VOCAB = ["neumonia_lli", "hiponatremia", "derrame_pericardico"]

# model → scripted reply
REPLIES = {
    "model-a": '{"presentes": ["neumonia_lli", "hiponatremia"], "razon": "consolidación + Na 128"}',
    "model-b": 'Respuesta: {"presentes": ["neumonia_lli", "hiponatremia"], "razon": "ok"}',
    "model-c": '{"presentes": ["neumonia_lli", "derrame_pericardico", "no_en_vocab"], "razon": "x"}',
    "model-broken": "no JSON here",
}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = REPLIES.get(body["model"], "{}")
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def base_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _agent(name, model, url, group=None):
    return PanelAgent(name, OpenAICompatBackend(model, base_url=url, api_key="x", cache_dir=None),
                      group)


def test_parse_report():
    assert parse_report(REPLIES["model-c"], VOCAB) == (["derrame_pericardico", "neumonia_lli"], "x")
    assert parse_report("garbage", VOCAB) == (None, "")


def test_panel_gates_by_independence_group(base_url, tmp_path):
    agents = [_agent("a1", "model-a", base_url),
              _agent("a2", "model-a", base_url),          # same model → same group
              _agent("b", "model-b", base_url),
              _agent("c", "model-c", base_url),
              _agent("broken", "model-broken", base_url)]
    panel = Panel(agents, k=2, audit_path=tmp_path / "audit.jsonl")
    assert panel.n_groups == 4

    res = asyncio.run(panel.assess("Varón de 70 años con tos y fiebre.", VOCAB))
    status = {d["key"]: d["status"] for d in res.decisions}
    assert status == {"neumonia_lli": "accepted", "hiponatremia": "accepted",
                      "derrame_pericardico": "pending"}          # single group → blocked
    assert res.accepted == {"neumonia_lli": True, "hiponatremia": True}
    hypo = next(d for d in res.decisions if d["key"] == "hiponatremia")
    assert set(hypo["support"]) == {agents[0].independence_group, agents[2].independence_group}
    broken = next(r for r in res.reports if r.agent == "broken")
    assert broken.present is None and broken.error
    assert res.answered_groups == 3 and not res.degraded     # broken agent excluded
    assert (tmp_path / "audit.jsonl").read_text().count("\n") == 1
    assert "Varón" not in (tmp_path / "audit.jsonl").read_text()   # vignette never logged


def test_duplicated_model_cannot_self_corroborate(base_url):
    panel = Panel([_agent("c1", "model-c", base_url), _agent("c2", "model-c", base_url),
                   _agent("b", "model-b", base_url)], k=2)
    res = asyncio.run(panel.assess("caso", VOCAB))
    status = {d["key"]: d["status"] for d in res.decisions}
    assert status["derrame_pericardico"] == "pending"    # two copies of one model = one vote


def test_k_larger_than_groups_rejected(base_url):
    with pytest.raises(ValueError):
        Panel([_agent("a1", "model-a", base_url), _agent("a2", "model-a", base_url)], k=2)


def test_load_panel_with_designed_k(base_url):
    cfg = {"design": {"p_fp": 0.05, "sens": 0.85, "rho": 0.0, "max_false_acceptance": 1e-2},
           "agents": [{"name": n, "backend": {"kind": "openai", "model": m, "base_url": base_url,
                                              "api_key": "x"}}
                      for n, m in [("a", "model-a"), ("b", "model-b"), ("c", "model-c")]]}
    panel = load_panel(cfg)
    assert panel.k == 2 and panel.n_groups == 3


def test_unreachable_agents_flag_degraded():
    dead = "http://127.0.0.1:9/v1"
    agents = [PanelAgent(n, OpenAICompatBackend(n, base_url=dead, api_key="x", cache_dir=None))
              for n in ("x", "y")]
    for a in agents:
        a.backend.client = a.backend.client.with_options(max_retries=0)
    import topoguard.validation.backends as b
    orig, b._retry = b._retry, lambda fn, **k: fn()
    try:
        res = asyncio.run(Panel(agents, k=2).assess("caso", VOCAB))
    finally:
        b._retry = orig
    assert res.degraded and res.accepted == {} and all(r.error for r in res.reports)
