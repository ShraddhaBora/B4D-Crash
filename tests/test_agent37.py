"""
tests/test_agent37.py — Unit and integration tests for Agent37 reasoning layer.

Tests enforce all core architectural rules:
  1. Authoritative ML risk scores are preserved without alteration.
  2. Every finding and recommendation is strictly grounded in input evidence.
  3. Recommendation language is cautious and assessment-oriented.
  4. Limitations and uncertainty statements are explicitly present.
  5. The deterministic rule engine guarantees robust fallback.
  6. The risk adapter correctly handles both JSON and CSV contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agent import Agent37, RiskAnalysisInput, AgentAnalysisOutput
from agent.prompts import extract_json_from_response
from agent.risk_adapter import from_json, from_csv
from agent.schemas import (
    CameraInfo,
    ConflictEventInput,
    HotspotInput,
    RiskLevel,
    SeverityLevel,
    TrafficStats,
)


@pytest.fixture
def dummy_input() -> RiskAnalysisInput:
    """Load the realistic deterministic dummy output."""
    path = Path("data/dummy_risk_output.json")
    assert path.exists(), "data/dummy_risk_output.json must exist"
    return from_json(path)


def test_dummy_data_schema(dummy_input: RiskAnalysisInput):
    """Ensure dummy_risk_output.json validates against RiskAnalysisInput."""
    assert dummy_input.analysis_id == "ANALYSIS_DUMMY_SESSION_01"
    assert len(dummy_input.conflict_events) == 8
    assert len(dummy_input.hotspots) == 3
    assert dummy_input.peak_risk_score == 0.85
    assert dummy_input.high_severity_count == 3


def test_ml_risk_score_preservation(dummy_input: RiskAnalysisInput):
    """Agent37 must preserve the exact ML peak risk score without modification."""
    agent = Agent37(deterministic_only=True)
    out: AgentAnalysisOutput = agent.analyse(dummy_input)

    assert out.overall_assessment.ml_peak_risk_score == dummy_input.peak_risk_score
    assert out.overall_assessment.ml_peak_risk_score == 0.85


def test_evidence_traceability(dummy_input: RiskAnalysisInput):
    """All findings, patterns, prioritized hotspots, and recs must cite real input event IDs."""
    agent = Agent37(deterministic_only=True)
    out: AgentAnalysisOutput = agent.analyse(dummy_input)

    valid_event_ids = {e.event_id for e in dummy_input.conflict_events}

    # 1. Findings
    assert len(out.key_findings) > 0
    for finding in out.key_findings:
        assert len(finding.evidence_event_ids) >= 1
        for eid in finding.evidence_event_ids:
            assert eid in valid_event_ids, f"Finding {finding.finding_id} cites non-existent event {eid}"

    # 2. Patterns
    assert len(out.patterns) > 0
    for pattern in out.patterns:
        assert len(pattern.evidence_event_ids) >= 2
        for eid in pattern.evidence_event_ids:
            assert eid in valid_event_ids, f"Pattern {pattern.pattern_id} cites non-existent event {eid}"

    # 3. Prioritized hotspots
    assert len(out.prioritized_hotspots) > 0
    for hotspot in out.prioritized_hotspots:
        assert len(hotspot.evidence_event_ids) >= 1
        for eid in hotspot.evidence_event_ids:
            assert eid in valid_event_ids, f"Hotspot {hotspot.zone_id} cites non-existent event {eid}"

    # 4. Recommendations
    assert len(out.recommendations) > 0
    for rec in out.recommendations:
        assert len(rec.evidence_event_ids) >= 1
        for eid in rec.evidence_event_ids:
            assert eid in valid_event_ids, f"Recommendation {rec.recommendation_id} cites non-existent event {eid}"


def test_cautious_recommendation_language(dummy_input: RiskAnalysisInput):
    """Recommendations must use cautious assessment language, not assert certainty."""
    agent = Agent37(deterministic_only=True)
    out: AgentAnalysisOutput = agent.analyse(dummy_input)

    banned_phrases = [
        "is definitely needed",
        "is definitely required",
        "will cause a crash",
        "signal timing is wrong",
        "fatal crash",
    ]

    for rec in out.recommendations:
        text = f"{rec.action} {rec.rationale}".lower()
        for banned in banned_phrases:
            assert banned not in text, f"Found banned definitive phrase '{banned}' in recommendation {rec.recommendation_id}"


def test_limitations_and_uncertainty(dummy_input: RiskAnalysisInput):
    """Output must explicitly include at least one limitation / uncertainty statement."""
    agent = Agent37(deterministic_only=True)
    out: AgentAnalysisOutput = agent.analyse(dummy_input)

    assert len(out.limitations) >= 1
    # Check that lack of camera calibration is flagged
    assert any("calibration" in lim.lower() for lim in out.limitations)


def test_hotspot_ml_score_preservation(dummy_input: RiskAnalysisInput):
    """Hotspot prioritization must preserve ML avg_risk_score from the input."""
    agent = Agent37(deterministic_only=True)
    out: AgentAnalysisOutput = agent.analyse(dummy_input)

    input_hotspot_scores = {h.zone_id: h.avg_risk_score for h in dummy_input.hotspots}

    for prioritized in out.prioritized_hotspots:
        expected = input_hotspot_scores[prioritized.zone_id]
        assert prioritized.ml_avg_risk_score == pytest.approx(expected, abs=1e-4)


def test_empty_events_handling():
    """Agent37 must handle empty conflict events gracefully without crashing."""
    empty_input = RiskAnalysisInput(
        analysis_id="EMPTY_SESSION",
        conflict_events=[],
        hotspots=[],
    )
    agent = Agent37(deterministic_only=True)
    out = agent.analyse(empty_input)

    assert out.overall_assessment.risk_level == RiskLevel.LOW
    assert out.overall_assessment.ml_peak_risk_score == 0.0
    assert len(out.key_findings) == 0
    assert len(out.patterns) == 0
    assert len(out.recommendations) == 0
    assert len(out.limitations) >= 1


def test_from_csv_loader():
    """Test loading from real/dry-run pipeline CSV outputs."""
    events_csv = "outputs/dry_run/events/conflict_events.csv"
    hotspots_csv = "outputs/dry_run/reports/hotspot_summary.csv"
    if not (Path(events_csv).exists() and Path(hotspots_csv).exists()):
        pytest.skip("Dry run CSVs not present")

    inp = from_csv(events_csv, hotspots_csv)
    assert len(inp.conflict_events) == 8
    assert len(inp.hotspots) == 2
    assert inp.peak_risk_score == 0.35

    agent = Agent37(deterministic_only=True)
    out = agent.analyse(inp)
    assert out.overall_assessment.ml_peak_risk_score == 0.35
    assert len(out.limitations) >= 1


def test_json_extractor_resilience():
    """Test extract_json_from_response handles markdown code fences, preambles, and whitespace."""
    # Clean json
    res = extract_json_from_response('{"foo": "bar"}')
    assert res == {"foo": "bar"}

    # Markdown block
    res = extract_json_from_response('```json\n{"score": 0.85}\n```')
    assert res == {"score": 0.85}

    # With conversational preamble and postscript
    raw = 'Here is the analysis:\n```json\n{"findings": ["F1"]}\n```\nHope this helps!'
    res = extract_json_from_response(raw)
    assert res == {"findings": ["F1"]}

    # Raw block without fences
    raw2 = 'Sure thing! {"zone": "R1C2"} is the result.'
    res2 = extract_json_from_response(raw2)
    assert res2 == {"zone": "R1C2"}


def test_agent37_cloud_client_and_provider():
    """Test Agent37CloudClient initialization and Agent37 provider selection."""
    from agent.agent37_cloud import Agent37CloudClient

    client = Agent37CloudClient(api_key="sk_live_test123")
    assert client.api_key == "sk_live_test123"
    headers = client._headers()
    assert headers["Authorization"] == "Bearer sk_live_test123"
    assert headers["Content-Type"] == "application/json"

    # Agent37 with cloud provider degrades safely to deterministic if offline
    agent = Agent37(agent37_api_key="sk_live_test123", provider="agent37_cloud")
    assert agent.provider == "agent37_cloud"
    assert agent.agent37_cloud_client.api_key == "sk_live_test123"

