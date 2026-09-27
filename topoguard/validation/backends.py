"""Interchangeable LLM backends. All expose `complete(system, user) -> str`.

- AnthropicBackend:     pip install anthropic;  export ANTHROPIC_API_KEY=...
                        (and ANTHROPIC_WORKSPACE_ID if your key is not scoped to a workspace)
- OpenAICompatBackend:  pip install openai;     works with OpenAI, vLLM, Ollama, llama.cpp server…
- On-disk cache keyed by hash(model, system, user, temperature, replica): re-running the
  analysis never pays for the same call twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    name: str

    def complete(self, system: str, user: str, *, replica: int = 0) -> str: ...


class _Cached:
    def __init__(self, cache_dir: str | Path | None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()

    def _key(self, *parts) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()

    def _cached(self, key: str, fn):
        path = self.cache_dir / f"{key}.txt" if self.cache_dir else None
        if path and path.exists():
            with self._lock:
                self.cache_hits += 1
            return path.read_text(encoding="utf-8")
        out = fn()
        with self._lock:
            self.calls += 1
        if path:
            path.write_text(out, encoding="utf-8")
        return out


def _retry(fn, tries: int = 5, base: float = 2.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — provider network / rate-limit errors
            status = getattr(e, "status_code", None)
            if i == tries - 1 or (status is not None and 400 <= status < 500 and status != 429):
                raise                     # client errors (except 429) are not retried
            time.sleep(base ** i)


class AnthropicBackend(_Cached):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", temperature: float | None = None,
                 max_tokens: int = 400, cache_dir: str | Path | None = ".llm_cache",
                 workspace_id: str | None = None, extra_kwargs: dict | None = None):
        super().__init__(cache_dir)
        self.extra_kwargs = extra_kwargs or {}
        import anthropic
        ws = workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        self.client = anthropic.Anthropic(
            default_headers={"anthropic-workspace-id": ws} if ws else None)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"anthropic:{model}"
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, user: str, *, replica: int = 0) -> str:
        def call():
            extra = {"temperature": self.temperature} if self.temperature is not None else None
            r = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system,
                messages=[{"role": "user", "content": user}], extra_body=extra,
                **self.extra_kwargs)
            with self._lock:
                self.input_tokens += r.usage.input_tokens
                self.output_tokens += r.usage.output_tokens
            return "".join(b.text for b in r.content if b.type == "text")
        return self._cached(self._key(self.name, system, user, self.temperature, replica),
                            lambda: _retry(call))


class OpenAICompatBackend(_Cached):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 400,
                 cache_dir: str | Path | None = ".llm_cache"):
        super().__init__(cache_dir)
        import openai
        self.client = openai.OpenAI(base_url=base_url,
                                    api_key=api_key or os.environ.get("OPENAI_API_KEY", "local"))
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"openai:{model}@{base_url or 'api.openai.com'}"

    def complete(self, system: str, user: str, *, replica: int = 0) -> str:
        def call():
            r = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return r.choices[0].message.content or ""
        return self._cached(self._key(self.name, system, user, self.temperature, replica),
                            lambda: _retry(call))
