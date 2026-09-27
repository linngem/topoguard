"""Demo: 1) pick a topology and threshold without any LLM call; 2) size k; 3) runtime gate."""
import networkx as nx

import topoguard as tg

N = 12
topologies = {
    "star": nx.star_graph(N - 1),
    "small-world (WS)": nx.connected_watts_strogatz_graph(N, 4, 0.2, seed=1),
    "clustered scale-free (Holme-Kim)": nx.powerlaw_cluster_graph(N, 2, 0.8, seed=1),
    "complete": nx.complete_graph(N),
}

print("1) TOPOLOGY — no LLM calls")
print(f"{'topology':34} {'C':>5} {'R_g':>7} {'BW hub':>7} {'BW periph':>9} {'θ*':>5} {'err@θ*':>7}")
for name, G in topologies.items():
    r = tg.topology_report(G)
    best = tg.best_threshold(tg.firewall_curve(G), min_truth=0.9)
    th = f"{best.theta:.2f}" if best else "—"
    er = f"{best.error_reach:.2f}" if best else "—"
    print(f"{name:34} {r.clustering:5.2f} {r.kirchhoff:7.1f} {r.bandwidth_hub:7.2f} "
          f"{r.bandwidth_periphery:9.2f} {th:>5} {er:>7}")

print("\n2) SIZING k — false-acceptance bound ≤ 1e-3")
for rho in (0.0, 0.2, 0.4):
    d = tg.design_gate(n=5, p_fp=0.05, sens=0.85, rho=rho, max_false_acceptance=1e-3)
    if d:
        print(f"ρ={rho:.1f}: k={d.k}  FAR={d.false_acceptance:.1e}  sensitivity={d.true_acceptance:.2f}  "
              f"effective_n={d.effective_n:.1f}")
    else:
        print(f"ρ={rho:.1f}: no k meets the bound → more model/data diversity is needed")

print("\n3) RUNTIME GATE (k=2 independent groups)")
modules = [
    tg.Module("xray_vision", "imaging/modelA"),
    tg.Module("xray_report", "imaging/modelA"),     # same group: does not count twice
    tg.Module("labs", "lab/rules"),
    tg.Module("history", "history/modelB"),
]
gate = tg.CorroborationGate(modules, k=2, audit_path="audit.jsonl")
findings = [
    tg.Finding("xray_vision", "LLL consolidation", evidence="left lower lobe opacity"),
    tg.Finding("xray_report", "lll consolidation"),
    tg.Finding("labs", "LLL consolidation", evidence="CRP 180, leukocytosis"),
    tg.Finding("xray_vision", "Pleural effusion"),
    tg.Finding("xray_report", "pleural effusion"),      # a single group → PENDING
    tg.Finding("history", "Penicillin allergy"),
    tg.Finding("labs", "penicillin allergy", present=False),
]
for d in gate.evaluate(findings):
    print(f"{d.status.value:9} {d.key:22} groups={list(d.support)}")
print("broadcast:", gate.broadcast())
