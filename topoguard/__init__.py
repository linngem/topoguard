"""topoguard — topology design and a corroboration firewall for multi-agent LLM systems,
based on consensus and contagion dynamics on networks."""
from .contagion import FirewallPoint, best_threshold, firewall_curve, ltm_cascade
from .gate import CorroborationGate, Decision, Finding, Module, Status
from .reliability import (GateDesign, design_gate, effective_n, estimate_rho,
                          false_acceptance, true_acceptance)
from .spectral import (TopologyReport, bandwidth, collective_response, kirchhoff_index,
                       normalized_spectrum, topology_report)

__all__ = [n for n in dir() if not n.startswith("_")]
__version__ = "0.1.0"
