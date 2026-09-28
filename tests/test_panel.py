"""Tests for the heterogeneous-panel experiment (simulated models, no calls)."""
from __future__ import annotations

import json

import pytest

from topoguard.validation.cases import CASES
from topoguard.validation.panel import (ModelSpec, PanelSpec, SimulatedPanelBackend, collect,
                                        evaluate, heterogeneous, homogeneous, load_responses,
                                        n_calls, predicted_far_injected, rho_report)

MODELS = [ModelSpec("a", "fa"), ModelSpec("b", "fb"), ModelSpec("g1", "gem"), ModelSpec("g2", "gem")]
CASE_IDS = ("pneumonia", "lupus", "anca_vasculitis")


@pytest.fixture(scope="module")
def resp(tmp_path_factory):
    out = tmp_path_factory.mktemp("panel") / "r.jsonl"
    backends = {m.name: SimulatedPanelBackend(m.name, m.family, p_fp=0.3) for m in MODELS}
    collect(backends, cases=CASE_IDS, replicas=3, out_path=out, progress=False)
    return out


def test_collect_counts_and_resumes(resp):
    lines = resp.read_text().splitlines()
    assert len(lines) == n_calls(len(CASE_IDS), len(MODELS), ("plausible", "implausible"), 3)
    backends = {m.name: SimulatedPanelBackend(m.name, m.family) for m in MODELS}
    collect(backends, cases=CASE_IDS, replicas=3, out_path=resp, progress=False)
    assert len(resp.read_text().splitlines()) == len(lines)          # nothing re-run
    assert all(b.calls == 0 for b in backends.values())


def test_prompts_are_the_validated_stimuli(resp):
    r = json.loads(resp.read_text().splitlines()[0])
    assert set(r) == {"model", "case", "condition", "replica", "present", "raw", "error"}
    assert r["present"] is not None


def test_family_grouping():
    fam = heterogeneous(MODELS)
    naive = heterogeneous(MODELS, "naive", by_family=False)
    assert fam.n_groups == 3 and naive.n_groups == 4
    assert homogeneous(MODELS[0], 3).n_groups == 3
    assert homogeneous(MODELS[0], 3, naive=False).n_groups == 1


def test_gate_blocks_single_source_errors(resp):
    r = load_responses(resp)
    res = {(x.k, x.error_kind): x for x in evaluate(heterogeneous(MODELS), r, cases=CASE_IDS)}
    k1, k2 = res[(1, "implausible")], res[(2, "implausible")]
    assert k1.scenarios == len(CASE_IDS) * 3 * len(MODELS)
    assert k1.far_injected == pytest.approx(k1.source_adopted, abs=0.2)   # k=1: no protection
    assert k2.far_injected < k1.far_injected
    assert k2.sens_base > 0.8


def test_family_counting_is_stricter_than_naive(resp):
    r = load_responses(resp)
    fam = {x.k: x for x in evaluate(heterogeneous(MODELS), r, cases=CASE_IDS,
                                    error_kinds=("plausible",))}
    naive = {x.k: x for x in evaluate(heterogeneous(MODELS, "n", by_family=False), r,
                                      cases=CASE_IDS, error_kinds=("plausible",))}
    # g1+g2 agreeing is one vote when grouped by family, two votes when counted naively
    assert fam[2].far_injected <= naive[2].far_injected


def test_reused_replica_rejected(resp):
    r = load_responses(resp)
    with pytest.raises(ValueError):
        evaluate(PanelSpec("bad", (("x", "a", 0, "x"), ("y", "a", 3, "y"))), r, cases=CASE_IDS)


def test_rho_report_structure(resp):
    rep = rho_report(load_responses(resp), MODELS, "fp", n_boot=50)
    assert rep["cases"] == len(CASE_IDS) and rep["rows"] == len(CASE_IDS) * 3
    assert set(rep["pairs"]) >= {"g1~g2", "a~b"}
    assert 0 <= rep["rho_all"] < 1


def test_predicted_far():
    assert predicted_far_injected(0.9, 0.05, 4, 1, 0.0) == 0.9
    # needing ≥2 of the other 3 groups: correlation makes co-errors more likely
    assert predicted_far_injected(0.9, 0.05, 4, 3, 0.0) < predicted_far_injected(0.9, 0.05, 4, 3, 0.5)


def test_all_cases_present():
    assert len(CASES) == 12
