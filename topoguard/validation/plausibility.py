"""A priori plausibility of each candidate finding, scored by an independent panel.

The panel must NOT be the evaluated model (Haiku). Sonnet 5 and Opus 5.5 are used here; they
belong to the same model family, which is a declared limitation. The score only enters the
analysis: it can be replaced by a clinician's rating without re-running any experiment call.

NOTE: `SYSTEM` and `prompt()` are in Spanish on purpose — they are the exact stimuli used.
"""
from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .cases import CASES, LEGACY_CASE_IDS, Case

SYSTEM = ("Eres un médico internista experto. Estimas probabilidades clínicas con criterio, "
          "sin información adicional a la que se te da.")


def prompt(case: Case, label: str) -> str:
    """Panel user prompt (Spanish stimulus — do not edit)."""
    return (f"Caso: {case.vignette}\n\n"
            f"Sin disponer de más pruebas que las descritas, ¿qué probabilidad (0-100) estimas "
            f"de que este paciente presente además: {label}?\n"
            "Responde SOLO con JSON: {\"p\": <entero 0-100>}")


_P = re.compile(r'"p"\s*:\s*(\d+(?:\.\d+)?)')


def rate(backends: dict, reps: int = 3, out_csv: str | Path = "plausibility.csv",
         workers: int = 16) -> Path:
    """Asks every panel model, `reps` times, for the probability of every candidate finding."""
    jobs = [(cid, role, name, rep) for cid, c in CASES.items() for role in Case.ROLES
            for name in backends for rep in range(reps)]

    def run(job):
        cid, role, name, rep = job
        c = CASES[cid]
        cand = c.candidate(role)
        txt = backends[name].complete(SYSTEM, prompt(c, cand.label), replica=rep)
        m = _P.search(txt)
        return {"case": cid, "role": role, "key": cand.key, "label": cand.label, "rater": name,
                "rep": rep, "p": float(m.group(1)) if m else float("nan")}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(run, jobs))
    out_csv = Path(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out_csv


def load_scores(path: str | Path) -> dict[tuple[str, str], float]:
    """(case, key) → mean panel plausibility in [0, 1]."""
    acc: dict = {}
    with Path(path).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["p"] != "nan":
                cid = LEGACY_CASE_IDS.get(r["case"], r["case"])
                acc.setdefault((cid, r["key"]), []).append(float(r["p"]) / 100)
    return {k: float(np.mean(v)) for k, v in acc.items()}
