"""Interchangeable LLM backends. All expose `complete(system, user) -> str`.

- AnthropicBackend:     pip install anthropic;  export ANTHROPIC_API_KEY=...
                        (and ANTHROPIC_WORKSPACE_ID if your key is not scoped to a workspace)
- OpenAICompatBackend:  pip install openai;     works with OpenAI, vLLM, Ollama, llama.cpp server…
- On-disk cache keyed by hash(model, system, user, temperature, replica[, salt]): re-running
  the analysis never pays for the same call twice. `salt` (the agent id in the network
  experiment) makes agents that receive an identical prompt draw independent samples instead of
  sharing one cached answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    name: str

    def complete(self, system: str, user: str, *, replica: int = 0,
                 salt: str | None = None) -> str: ...


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
            # atomic write: concurrent threads with the same key used to interleave their
            # writes, and a reader could see a half-written file (→ spurious parse failures)
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(out, encoding="utf-8")
            os.replace(tmp, path)
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

    def complete(self, system: str, user: str, *, replica: int = 0,
                 salt: str | None = None) -> str:
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
        parts = (self.name, system, user, self.temperature, replica)
        if salt is not None:
            parts += (salt,)
        return self._cached(self._key(*parts), lambda: _retry(call))


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str) -> str:
    """Drops <think>…</think> blocks (Qwen3, DeepSeek-R1…). Needed because `parse` takes the
    outermost {...} span, and braces inside the reasoning would corrupt it."""
    text = _THINK.sub("", text)
    if "<think>" in text:                  # reasoning truncated by max_tokens: nothing usable
        text = text.split("<think>")[0]
    return text.strip()


class OpenAICompatBackend(_Cached):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 400,
                 cache_dir: str | Path | None = ".llm_cache", extra_body: dict | None = None):
        super().__init__(cache_dir)
        import openai
        self.client = openai.OpenAI(base_url=base_url,
                                    api_key=api_key or os.environ.get("OPENAI_API_KEY", "local"))
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.extra_body = extra_body or None
        self.name = f"openai:{model}@{base_url or 'api.openai.com'}"
        self.input_tokens = 0
        self.output_tokens = 0
        self.truncated = 0

    def complete(self, system: str, user: str, *, replica: int = 0,
                 salt: str | None = None) -> str:
        def call():
            r = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                extra_body=self.extra_body)
            with self._lock:
                if r.usage:
                    self.input_tokens += r.usage.prompt_tokens or 0
                    self.output_tokens += r.usage.completion_tokens or 0
                if r.choices[0].finish_reason == "length":
                    self.truncated += 1
            return strip_reasoning(r.choices[0].message.content or "")
        # extra_body only enters the key when set, so existing caches stay valid
        parts = (self.name, system, user, self.temperature, replica)
        if self.extra_body:
            parts += (json.dumps(self.extra_body, sort_keys=True),)
        if salt is not None:
            parts += (salt,)
        return self._cached(self._key(*parts), lambda: _retry(call))
