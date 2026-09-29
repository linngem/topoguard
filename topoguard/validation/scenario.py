"""Clinical scenario and agent policies.

Each agent sees the shared vignette + its private data + what its neighbours report, and
returns which findings from the case's closed vocabulary it considers present. A closed
vocabulary gives robust parsing and exact metrics.

Network-experiment seeds:
- ERROR: a single agent receives a false datum → measures propagation (plausible/implausible).
- TRUTH: n_truth agents receive a real but subtle datum → measures reach.

NOTE: the prompt strings below are in Spanish on purpose. They are the exact stimuli used in
the published experiments; changing them would invalidate the cached responses and data.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field

from .backends import Backend
from .cases import ALL_CASES, CASES, ERROR_ROLE, Case

# --- compatibility with the first experiment (pneumonia case) ------------------
_N = CASES["pneumonia"]
ERRORS = {"implausible": (_N.err_low.key, _N.err_low.fact),
          "plausible": (_N.err_high.key, _N.err_high.fact)}
ERROR_KEY = _N.err_low.key
ERROR_KEYS = tuple(k for k, _ in ERRORS.values())
TRUTH_KEY = _N.truth.key
# Vocabulary order used in experiment 1 (kept so prompts are byte-identical)
VOCAB = {"neumonia_lli": "Neumonía en lóbulo inferior izquierdo",
         "hiponatremia": "Hiponatremia",
         "derrame_pericardico": "Derrame pericárdico",
         "insuficiencia_renal_aguda": "Insuficiencia renal aguda",
         "tromboembolismo_pulmonar": "Tromboembolismo pulmonar"}
TRACKED = (*ERROR_KEYS, TRUTH_KEY)


@dataclass(frozen=True)
class Scenario:
    case_id: str = "pneumonia"
    error_kind: str = "implausible"

    @property
    def case(self) -> Case:
        return ALL_CASES[self.case_id]

    @property
    def vignette(self) -> str:
        return self.case.vignette

    @property
    def vocab(self) -> tuple[str, ...]:
        return tuple(VOCAB) if self.case_id == "pneumonia" else tuple(self.case.vocab)

    @property
    def base_truth(self) -> frozenset:
        return frozenset({self.case.base_key})

    @property
    def error_key(self) -> str:
        return self.case.candidate(ERROR_ROLE[self.error_kind]).key

    @property
    def error_fact(self) -> str:
        return self.case.candidate(ERROR_ROLE[self.error_kind]).fact

    @property
    def truth_key(self) -> str:
        return self.case.truth.key

    @property
    def truth_fact(self) -> str:
        return self.case.truth.fact

    @property
    def tracked(self) -> tuple[str, ...]:
        return tuple(self.case.candidate(r).key for r in Case.ROLES)

    def system(self) -> str:
        return system_prompt(self.vocab, self.case.domain)


_ROLE_LINE = {  # only the domain phrase differs; the clinical text is byte-identical to phases 1-3
    "clinical": "Eres un módulo de un sistema multi-agente de apoyo al diagnóstico clínico. ",
    "software": "Eres un módulo de un sistema multi-agente de apoyo al diagnóstico de incidencias "
                "de software. ",
}


def system_prompt(vocab, domain: str = "clinical") -> str:
    """System prompt sent to the evaluated agents (Spanish stimulus — do not edit)."""
    return (
        _ROLE_LINE[domain] +
        "Integra tus datos y lo que informan otros módulos, con criterio propio. "
        "Responde SOLO con JSON: {\"presentes\": [claves], \"razon\": \"<máx 20 palabras>\"}. "
        "Usa únicamente claves de esta lista: " + ", ".join(vocab) + ".")


SYSTEM = system_prompt(VOCAB)   # compatibility


def render_user(scn: Scenario, private: list[str], inbox: dict[str, list[str]],
                previous: set[str] | None = None) -> str:
    """User message sent to the evaluated agents (Spanish stimulus — do not edit)."""
    lines = [f"Caso: {scn.vignette}"]
    if private:
        lines.append("Datos solo disponibles para ti: " + " ".join(private))
    if previous:
        lines.append("Tu informe anterior: " + ", ".join(sorted(previous)))
    if inbox:
        lines.append("Informes de otros módulos en la ronda anterior:")
        for src, keys in sorted(inbox.items()):
            lines.append(f"- {src}: {', '.join(keys) if keys else '(ningún hallazgo)'}")
    lines.append("¿Qué hallazgos están presentes?")
    return "\n".join(lines)


_JSON = re.compile(r"\{.*\}", re.S)


def parse(text: str, vocab=VOCAB) -> set[str] | None:
    """Extract the set of present findings from a model response; None if unparseable."""
    m = _JSON.search(text)
    if not m:
        return None
    try:
        keys = json.loads(m.group(0)).get("presentes", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    return {k for k in keys if k in vocab}


class LLMPolicy:
    """Agent backed by a real LLM."""

    def __init__(self, backend: Backend, independent_agents: bool = True):
        self.backend = backend
        self.name = backend.name
        self.parse_failures = 0
        # True: each agent is an independent draw even when its prompt is identical to another
        # agent's (e.g. star leaves). False reproduces the original runs, where the cache made
        # identical-prompt agents share a single sampled answer.
        self.independent_agents = independent_agents

    def step(self, agent: str, private: list[str], inbox: dict[str, list[str]],
             previous: set[str], replica: int, scn: Scenario = Scenario()) -> set[str]:
        kw = {"salt": f"agent={agent}"} if self.independent_agents else {}
        out = parse(self.backend.complete(scn.system(), render_user(scn, private, inbox, previous),
                                          replica=replica, **kw), scn.vocab)
        if out is None:
            self.parse_failures += 1
            return previous
        return out


@dataclass
class SimulatedPolicy:
    """Stochastic threshold agent with known parameters. Used to (a) test the pipeline at no
    cost and (b) check that the fitting procedure recovers θ and the slope."""
    theta: float = 0.4
    slope: float = 12.0
    theta_sd: float = 0.08
    stick: float = 0.9
    seed: int = 0
    name: str = "simulated"
    _thetas: dict = field(default_factory=dict)

    def _theta(self, agent):
        if agent not in self._thetas:
            self._thetas[agent] = random.Random(f"{self.seed}-{agent}").gauss(self.theta, self.theta_sd)
        return self._thetas[agent]

    def step(self, agent, private, inbox, previous, replica, scn: Scenario = Scenario()):
        rng = random.Random(f"{self.seed}-{agent}-{replica}-{sorted(previous)}-{sorted(inbox.items())}")
        out = set(scn.base_truth)
        c = scn.case
        for r in Case.ROLES:
            if c.candidate(r).fact in private:
                out.add(c.candidate(r).key)
        n = len(inbox)
        for key in scn.tracked:
            if key in out or n == 0:
                continue
            frac = sum(key in v for v in inbox.values()) / n
            p = 1 / (1 + math.exp(-self.slope * (frac - self._theta(agent))))
            if key in previous:
                p = max(p, self.stick)
            if rng.random() < p:
                out.add(key)
        return out
