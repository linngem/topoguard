"""Generates docs/CASES_AND_PROMPTS.md from the code and data (i.e. what was actually sent).

    cd examples && PYTHONPATH=.. python make_appendix.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from topoguard.validation import CASES, Scenario, render_user
from topoguard.validation import plausibility as P
from topoguard.validation.cases import LEGACY_CASE_IDS
from topoguard.validation.micro import ADOPT_GRID, KEEP_GRID, inbox_for

ROOT = Path(__file__).resolve().parents[1]
RES = Path("results/multicase")
ROLE = {"truth": "Truth (real, subtle)", "err_high": "Error — high a priori plausibility",
        "err_mid": "Error — medium a priori plausibility", "err_low": "Error — low a priori plausibility"}

scores = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(RES / "plausibility.csv", encoding="utf-8")):
    if r["p"] != "nan":
        scores[(LEGACY_CASE_IDS.get(r["case"], r["case"]), r["key"])][r["rater"]].append(float(r["p"]))

L = ["# Appendix: clinical cases and prompts",
     "",
     "Generated automatically from the code (`topoguard/validation/cases.py`, `scenario.py`,",
     "`micro.py`, `plausibility.py`). The texts are exactly those sent to the models.",
     "",
     "**The experiments were run in Spanish.** Every stimulus is shown in its original Spanish",
     "(what the models received) followed by an English translation for readers. Finding keys",
     "such as `hiponatremia` are part of the stimulus and are kept as they were.",
     "",
     "> **Synthetic** cases for AI-systems research. They are not validated clinical material and",
     "> must not be used for clinical decisions.",
     "",
     "## Models and parameters",
     "",
     "| Role | Model | Parameters |",
     "|---|---|---|",
     "| Agents under test (network and micro-experiment) | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | default temperature, `max_tokens` 400 |",
     "| Plausibility panel (never acted as an agent) | Claude Sonnet 5 (`claude-sonnet-5`) | defaults, `max_tokens` 300, 3 replicates |",
     "| Plausibility panel (never acted as an agent) | Claude Opus 5.5 (`claude-opus-5-5`) | adaptive thinking at low effort, `max_tokens` 2000, 3 replicates |",
     "",
     "## 1. The 12 cases",
     "",
     "Each case has a base finding (evident from the vignette) and four candidates. In the network",
     "experiment the *private datum* of the truth is given to 3 agents and that of the error to 1.",
     "*Plausibility* is the panel's mean estimated probability (Sonnet 5 / Opus 5.5).",
     ""]
for i, (cid, c) in enumerate(CASES.items(), 1):
    L += [f"### {i}. {c.base_label_en} — {c.specialty} (`{cid}`)", "",
          f"**Vignette (sent, Spanish):** {c.vignette}", "",
          f"**Vignette (English):** {c.vignette_en}", "",
          f"**Base finding:** `{c.base_key}` — {c.base_label_en}", "",
          "| Role | Finding (`key`) | Private datum in the network (Spanish → English) | Plausibility (mean · Sonnet / Opus) |",
          "|---|---|---|---|"]
    for role in c.ROLES:
        cand = c.candidate(role)
        s = scores[(cid, cand.key)]
        allv = [v for vs in s.values() for v in vs]
        L.append(f"| {ROLE[role]} | {cand.label_en} (`{cand.key}`) | {cand.fact} → *{cand.fact_en}* | "
                 f"{np.mean(allv)/100:.2f} · {np.mean(s['sonnet5'])/100:.2f} / {np.mean(s['opus5.5'])/100:.2f} |")
    L.append("")

pn = Scenario("pneumonia", "plausible")
lup = Scenario("lupus", "plausible")
ex_inbox = {"m0": ["neumonia_lli"], "m4": ["hiponatremia", "neumonia_lli"],
            "m7": ["neumonia_lli", "tromboembolismo_pulmonar"]}
EN_SYSTEM = ("You are a module of a multi-agent clinical diagnostic support system. Integrate your "
             "data and what other modules report, using your own judgement. Answer ONLY with JSON: "
             "{\"presentes\": [keys], \"razon\": \"<max 20 words>\"}. Use only keys from this list: …")
EN_USER = ("Case: <vignette>\nData available only to you: <private datum>\nYour previous report: <keys>\n"
           "Reports from other modules in the previous round:\n- m0: <keys>\n- …\n"
           "Which findings are present?")
EN_PANEL_SYS = ("You are an expert internist. You estimate clinical probabilities with judgement, "
                "without any information beyond what you are given.")
EN_PANEL_USER = ("Case: <vignette>\n\nWithout any tests beyond those described, what probability "
                 "(0-100) do you estimate that this patient also has: <finding>?\n"
                 "Answer ONLY with JSON: {\"p\": <integer 0-100>}")

L += ["## 2. Prompts for the agents under test (Haiku 4.5)", "",
      "### 2.1 System prompt",
      "",
      "Identical in every case; only the list of keys (the case vocabulary) changes. Example, lupus case:",
      "", "```text", lup.system(), "```", "",
      "English translation:", "", "```text", EN_SYSTEM, "```", "",
      "### 2.2 User message — network experiment",
      "",
      "Round 0, agent receiving the false datum (pneumonia case, plausible error):", "",
      "```text", render_user(pn, [pn.error_fact], {}, None), "```", "",
      "Round r ≥ 1, agent without private data seeing its neighbours' reports from the previous",
      "round (with the local gate *k* = 2, only findings asserted by ≥ 2 neighbours are shown):", "",
      "```text", render_user(pn, [], ex_inbox, {"neumonia_lli"}), "```", "",
      "Structure, in English:", "", "```text", EN_USER, "```", "",
      "### 2.3 User message — micro-experiment",
      "",
      f"A single isolated agent; *n* ∈ {sorted(ADOPT_GRID)} neighbours, *m* of whom assert the candidate.",
      "Retention condition: the agent's previous report already included the candidate",
      f"(*n* = {list(KEEP_GRID)[0]}, *m* ∈ {list(KEEP_GRID.values())[0]}).",
      "Example: lupus case, *n* = 4, *m* = 2, candidate = antiphospholipid syndrome:", "",
      "```text",
      render_user(lup, [], inbox_for("lupus", "sindrome_antifosfolipido", 4, 2, "err_high", 0),
                  {lup.case.base_key}),
      "```", "",
      "Expected response format:", "",
      "```json", '{"presentes": ["lupus_eritematoso_sistemico", "hipocomplementemia"], "razon": "..."}',
      "```", "",
      "## 3. Plausibility panel prompts (Sonnet 5 and Opus 5.5)", "",
      "### 3.1 System prompt", "", "```text", P.SYSTEM, "```", "",
      "English translation:", "", "```text", EN_PANEL_SYS, "```", "",
      "### 3.2 User message (example: lupus, antiphospholipid syndrome)", "",
      "```text", P.prompt(lup.case, "Síndrome antifosfolípido"), "```", "",
      "English translation:", "", "```text", EN_PANEL_USER, "```", "",
      "The panel does **not** see the private data or Haiku's responses: only the vignette and the",
      "name of the finding. Its score is only used in the analysis and can be replaced by a",
      "clinician's rating without re-running any call.", ""]

out = ROOT / "docs" / "CASES_AND_PROMPTS.md"
out.write_text("\n".join(L), encoding="utf-8")
print("✓", out)
