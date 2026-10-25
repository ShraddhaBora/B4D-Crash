"""
agent/ — Agent37 reasoning layer for Before the Crash.

Receives structured evidence from the ML risk-engine and produces
evidence-grounded safety intelligence in a machine-readable format.

Data flow:
    risk_engine outputs (conflict_events.csv / hotspot_summary.csv)
        → risk_adapter.py  (converts to RiskAnalysisInput)
        → agent37.py       (reasoning / interpretation)
        → AgentAnalysisOutput (validated Pydantic model)

The ML risk scores produced by src/safety/risk_score.py are authoritative.
Agent37 adds interpretation, pattern detection, and recommendations;
it never recalculates or silently overrides ML-produced metrics.
"""

from .agent37 import Agent37
from .agent37_cloud import Agent37CloudClient
from .schemas import RiskAnalysisInput, AgentAnalysisOutput

__all__ = ["Agent37", "Agent37CloudClient", "RiskAnalysisInput", "AgentAnalysisOutput"]

