"""Heterogeneous-panel experiment: does model diversity make the k-of-n gate safer?

Setting (no network: the gate is the only channel, as in `topoguard.serve`):
- Every agent reads the case vignette plus the real-but-subtle datum (`truth.fact`).
- In each scenario ONE agent (the *source*) additionally receives a false datum
  (`err_high` = plausible, `err_low` = implausible) — a hallucination entering the panel.
- The gate accepts a finding only if ≥ k independence groups assert it.

Calls are collected once per (case, condition, model, replica) and panels are assembled
offline, so many panel designs and every k are compared at no extra cost:
- condition "clean": vignette + truth datum (what every non-source agent sees);
- condition "inject:<error_kind>": the same plus the false datum (what the source sees).

Measured per panel and k:
- FAR (injected): the injected false finding is accepted;
- FAR (spontaneous): any other false finding is accepted;
- sensitivity: the subtle truth (and the base finding) are accepted.
Plus, from the clean condition, the error matrix (case×replica × model) → ρ with
`reliability.estimate_rho`, and the pairwise error correlation between models (does it
follow model lineage, e.g. Gemma ↔ MedGemma?).

Prompts are the exact validated stimuli (`scenario.system_prompt` / `render_user`).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from ..gate import CorroborationGate, Finding, Module, Status
from ..reliability import estimate_rho, false_acceptance
from .backends import Backend
from .cases import CASES, ERROR_ROLE, Case
from .scenario import Scenario, parse, render_user

ERROR_ROLES = ("err_high", "err_mid", "err_low")


# ------------------------------------------------------------------ design
@dataclass(frozen=True)
class ModelSpec:
    name: str           # short id used in the results ("qwen", "medgemma"…)
    family: str         # lineage → independence group ("gemma" for gemma4 and medgemma)


@dataclass(frozen=True)
class PanelSpec:
    """A panel: agents = (label, model name, replica offset, independence group)."""
    name: str
    agents: tuple[tuple[str, str, int, str], ...]

    @property
    def n_groups(self) -> int:
        return len({g for *_, g in self.agents})


def heterogeneous(models: list[ModelSpec], name: str = "hetero_family",
                  by_family: bool = True) -> PanelSpec:
    """One agent per model. by_family=False counts every model as its own group (naive)."""
    return PanelSpec(name, tuple((m.name, m.name, 0, m.family if by_family else m.name)
                                 for m in models))


def homogeneous(model: ModelSpec, n: int, naive: bool = True) -> PanelSpec:
    """n replicas of one model. naive=True gives each replica its own group, i.e. what a
    system that does not know they share a base model would do."""
    return PanelSpec(f"homo_{model.name}_x{n}" + ("" if naive else "_grouped"),
                     tuple((f"{model.name}#{j}", model.name, j,
                            f"{model.name}#{j}" if naive else model.family) for j in range(n)))


def conditions(error_kinds: Iterable[str]) -> list[str]:
    return ["clean"] + [f"inject:{k}" for k in error_kinds]


def private_data(case: Case, condition: str) -> list[str]:
    facts = [case.truth.fact]
    if condition.startswith("inject:"):
        facts.append(case.candidate(ERROR_ROLE[condition.split(":", 1)[1]]).fact)
    return facts


def n_calls(n_cases: int, n_models: int, error_kinds: Iterable[str], replicas: int) -> int:
    return n_cases * n_models * len(conditions(error_kinds)) * replicas


# ------------------------------------------------------------------ collection
def collect(backends: dict[str, Backend], *, cases: Iterable[str] = tuple(CASES),
            error_kinds: Iterable[str] = ("plausible", "implausible"), replicas: int = 3,
            out_path: str | Path = "panel_responses.jsonl", workers: dict[str, int] | None = None,
            progress: bool = True) -> Path:
    """Runs model by model (keeps a local server from swapping models in and out of VRAM).
    Appends to out_path and skips jobs already present, so an interrupted run resumes."""
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                done.add((r["model"], r["case"], r["condition"], r["replica"]))
    lock = threading.Lock()
    conds = conditions(error_kinds)
    with out_path.open("a", encoding="utf-8") as fh:
        for model, backend in backends.items():
            jobs = [(cid, c, rep) for cid in cases for c in conds for rep in range(replicas)
                    if (model, cid, c, rep) not in done]
            if progress:
                print(f"[{model}] {len(jobs)} calls", flush=True)

            def run(job, backend=backend, model=model):
                cid, cond, rep = job
                scn = Scenario(cid)
                user = render_user(scn, private_data(scn.case, cond), {})
                try:
                    raw = backend.complete(scn.system(), user, replica=rep)
                    err = None
                except Exception as e:  # noqa: BLE001 — record and continue
                    raw, err = "", f"{type(e).__name__}: {e}"
                keys = parse(raw, scn.vocab)
                rec = {"model": model, "case": cid, "condition": cond, "replica": rep,
                       "present": sorted(keys) if keys is not None else None,
                       "raw": raw, "error": err}
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()

            with ThreadPoolExecutor((workers or {}).get(model, 4)) as pool:
                list(pool.map(run, jobs))
    return out_path


def load_responses(path: str | Path) -> dict[tuple, list[str] | None]:
    """(model, case, condition, replica) → present keys (None = unparseable). Last wins."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            out[(r["model"], r["case"], r["condition"], r["replica"])] = r["present"]
    return out


# ------------------------------------------------------------------ evaluation
@dataclass
class PanelResult:
    panel: str
    k: int
    n_groups: int
    error_kind: str
    scenarios: int
    far_injected: float
    far_spontaneous: float
    sens_truth: float
    sens_base: float
    source_adopted: float
    parse_fail: float


def _replicas(resp: dict) -> int:
    return 1 + max(r for *_, r in resp)


def evaluate(panel: PanelSpec, resp: dict, *, error_kinds=("plausible", "implausible"),
             cases: Iterable[str] = tuple(CASES), ks: Iterable[int] | None = None) -> list[PanelResult]:
    R = _replicas(resp)
    slots = [(m, off % R) for _, m, off, _ in panel.agents]
    if len(set(slots)) != len(slots):
        raise ValueError(f"{panel.name}: needs ≥ {max(o for *_, o, _ in panel.agents) + 1} "
                         f"replicas (have {R}); two agents would reuse the same response")
    modules = [Module(label, group) for label, _, _, group in panel.agents]
    ks = list(ks or range(1, panel.n_groups + 1))
    out = []
    for ek in error_kinds:
        role = ERROR_ROLE[ek]
        tally = {k: defaultdict(float) for k in ks}
        n = 0
        for cid in cases:
            case = CASES[cid]
            inj = case.candidate(role).key
            others = {case.candidate(r).key for r in ERROR_ROLES} - {inj}
            for block in range(R):
                for s_idx in range(len(panel.agents)):
                    findings, fails, adopted, missing = [], 0, 0.0, False
                    for i, (label, model, off, _) in enumerate(panel.agents):
                        cond = f"inject:{ek}" if i == s_idx else "clean"
                        key = (model, cid, cond, (block + off) % R)
                        if key not in resp:
                            missing = True
                            break
                        keys = resp[key]
                        if keys is None:
                            fails += 1
                            continue
                        if i == s_idx:
                            adopted = float(inj in keys)
                        findings += [Finding(label, k) for k in keys]
                    if missing:
                        continue
                    n += 1
                    for k in ks:
                        gate = CorroborationGate(modules, k)
                        acc = {d.key for d in gate.evaluate(findings)
                               if d.status is Status.ACCEPTED and d.present}
                        t = tally[k]
                        t["far_inj"] += inj in acc
                        t["far_spont"] += bool(acc & others)
                        t["truth"] += case.truth.key in acc
                        t["base"] += case.base_key in acc
                        t["adopt"] += adopted
                        t["fail"] += fails / len(panel.agents)
        for k in ks:
            t = tally[k]
            d = max(n, 1)
            out.append(PanelResult(panel.name, k, panel.n_groups, ek, n, t["far_inj"] / d,
                                   t["far_spont"] / d, t["truth"] / d, t["base"] / d,
                                   t["adopt"] / d, t["fail"] / d))
    return out


def error_matrix(resp: dict, models: list[str], kind: str = "fp",
                 cases: Iterable[str] = tuple(CASES)) -> tuple[np.ndarray, list[tuple]]:
    """Binary (case×replica) × model matrix from the clean condition.
    kind="fp": asserted any false finding; kind="fn": missed the subtle truth."""
    R = _replicas(resp)
    rows, idx = [], []
    for cid in cases:
        case = CASES[cid]
        false = {case.candidate(r).key for r in ERROR_ROLES}
        for rep in range(R):
            row = []
            for m in models:
                keys = resp.get((m, cid, "clean", rep))
                if keys is None:
                    row.append(np.nan)
                elif kind == "fp":
                    row.append(float(bool(set(keys) & false)))
                else:
                    row.append(float(case.truth.key not in keys))
            if all(math.isnan(v) for v in row):
                continue                     # not collected (subset of cases / replicas)
            rows.append(row)
            idx.append((cid, rep))
    return np.array(rows, float).reshape(-1, len(models)), idx


def _pair_stats(E: np.ndarray, names: list[str], fam: dict[str, str]):
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            e = E[:, [i, j]]
            e = e[~np.isnan(e).any(1)]
            pairs[f"{names[i]}~{names[j]}"] = (
                float(np.corrcoef(e.T)[0, 1]) if len(e) > 2 and (e.std(0) > 0).all() else float("nan"))
    same = lambda p: fam[p.split("~")[0]] == fam[p.split("~")[1]]   # noqa: E731
    within = [v for p, v in pairs.items() if same(p) and not math.isnan(v)]
    across = [v for p, v in pairs.items() if not same(p) and not math.isnan(v)]
    return (pairs, float(np.mean(within)) if within else float("nan"),
            float(np.mean(across)) if across else float("nan"))


def rho_report(resp: dict, models: list[ModelSpec], kind: str = "fp", n_boot: int = 1000,
               seed: int = 0) -> dict:
    """ρ over all models and per pair (NaN rows dropped pairwise), with a cluster bootstrap
    over cases (replicas of one case are not independent) for ρ_all and for
    ρ_within_family − ρ_across_family. Few cases → wide intervals: read them before the
    point estimates."""
    names = [m.name for m in models]
    fam = {m.name: m.family for m in models}
    E, idx = error_matrix(resp, names, kind)
    pairs, within, across = _pair_stats(E, names, fam)
    full = E[~np.isnan(E).any(1)]
    rates = {n: float(np.nanmean(E[:, i])) if np.isfinite(E[:, i]).any() else float("nan")
             for i, n in enumerate(names)}
    cases = sorted({c for c, _ in idx})
    rows_of = {c: [i for i, (cc, _) in enumerate(idx) if cc == c] for c in cases}
    rng = np.random.default_rng(seed)
    boot_all, boot_diff = [], []
    for _ in range(n_boot):
        pick = [i for c in rng.choice(cases, len(cases)) for i in rows_of[c]]
        Eb = E[pick]
        fb = Eb[~np.isnan(Eb).any(1)]
        if len(fb) > 2:
            boot_all.append(estimate_rho(fb))
        _, w, a = _pair_stats(Eb, names, fam)
        if not (math.isnan(w) or math.isnan(a)):
            boot_diff.append(w - a)
    ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b else None  # noqa: E731
    return {"kind": kind, "rows": int(len(full)), "cases": len(cases), "rates": rates,
            "rho_all": estimate_rho(full) if len(full) else float("nan"),
            "rho_all_ci95": ci(boot_all),
            "rho_within_family": within, "rho_across_family": across,
            "within_minus_across_ci95": ci(boot_diff),
            "pairs": pairs}


def predicted_far_injected(adopt: float, p_spont: float, n_groups: int, k: int,
                           rho: float) -> float:
    """Prior from `reliability`: the source adopts with prob `adopt`; the remaining n_groups-1
    groups must supply ≥ k-1 spontaneous assertions of the same key (beta-binomial, ρ)."""
    if k <= 1:
        return adopt
    return adopt * false_acceptance(n_groups - 1, k - 1, p_spont, rho)


# ------------------------------------------------------------------ simulation backend
class SimulatedPanelBackend:
    """Cost-free stand-in for an LLM, used to test the pipeline end to end. Spontaneous false
    findings are drawn from a Gaussian copula with case, family, model and replica
    components, so models of one family share errors (ρ_within > ρ_across), replicas of one
    model share even more, and the analysis can be checked against known structure. Responds in the same JSON contract as the real agents."""

    def __init__(self, name: str, family: str, *, p_fp: float = 0.06, sens: float = 0.8,
                 adopt: float = 0.9, w_case: float = 0.3, w_family: float = 0.5,
                 w_model: float = 0.5, seed: int = 0):
        self.name, self.family = f"sim:{name}", family
        self.p_fp, self.sens, self.adopt = p_fp, sens, adopt
        self.w_case, self.w_family, self.w_model, self.seed = w_case, w_family, w_model, seed
        self.calls = self.cache_hits = 0

    def _g(self, *parts) -> float:
        return random.Random("|".join(map(str, (self.seed, *parts)))).gauss(0, 1)

    def complete(self, system: str, user: str, *, replica: int = 0) -> str:
        self.calls += 1
        case = next(c for c in CASES.values() if c.vignette in user)
        out = {case.base_key}
        h = hashlib.sha256(user.encode()).hexdigest()[:12]
        rng = random.Random(f"{self.seed}|{self.name}|{case.id}|{replica}|{h}")
        if case.truth.fact in user and rng.random() < self.sens:
            out.add(case.truth.key)
        wc, wf, wm = self.w_case, self.w_family, self.w_model
        wr = math.sqrt(max(1e-9, 1 - wc ** 2 - wf ** 2 - wm ** 2))
        for r in ERROR_ROLES:
            c = case.candidate(r)
            if c.fact in user:
                if rng.random() < self.adopt:
                    out.add(c.key)
                continue
            # case, family and model components persist across replicas (a model errs on
            # the same case again); only the last term is sampling noise
            z = (wc * self._g("case", case.id, c.key)
                 + wf * self._g("fam", self.family, case.id, c.key)
                 + wm * self._g("mod", self.name, case.id, c.key)
                 + wr * self._g("rep", self.name, case.id, c.key, replica))
            if 0.5 * (1 + math.erf(z / math.sqrt(2))) < self.p_fp:
                out.add(c.key)
        return json.dumps({"presentes": sorted(out), "razon": "sim"}, ensure_ascii=False)
