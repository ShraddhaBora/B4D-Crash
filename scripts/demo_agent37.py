#!/usr/bin/env python3
"""
scripts/demo_agent37.py — End-to-end demonstration runner for Agent37.

Demonstrates:
  1. Loading structured evidence (from JSON or pipeline CSVs).
  2. Executing Agent37 reasoning (LLM inference with HuggingFace, or deterministic fallback).
  3. Verifying authoritative ML risk score preservation.
  4. Verifying evidence grounding (every finding/recommendation references valid input event IDs).
  5. Inspecting structured safety intelligence output and saving to JSON.

Usage:
  # Using realistic dummy JSON output:
  python scripts/demo_agent37.py --input data/dummy_risk_output.json

  # Running in deterministic-only mode (offline / reproducible):
  python scripts/demo_agent37.py --input data/dummy_risk_output.json --deterministic

  # Running against pipeline CSV outputs:
  python scripts/demo_agent37.py --events-csv outputs/dry_run/events/conflict_events.csv \
                                 --hotspots-csv outputs/dry_run/reports/hotspot_summary.csv

  # With HuggingFace token:
  python scripts/demo_agent37.py --hf-token <YOUR_HF_TOKEN>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent import Agent37, RiskAnalysisInput, AgentAnalysisOutput
from agent.risk_adapter import from_json, from_csv


def _load_dotenv_if_needed() -> None:
    """Look for .env in current directory or project root and populate os.environ if missing."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass
            break


_load_dotenv_if_needed()


def run_demo(
    input_json: Path | None = None,
    events_csv: Path | None = None,
    hotspots_csv: Path | None = None,
    output_path: Path | None = None,
    deterministic: bool = False,
    agent37_api_key: str | None = None,
    instance_id: str | None = None,
    hf_token: str | None = None,
    model: str = "default",
) -> AgentAnalysisOutput:
    _load_dotenv_if_needed()
    print("=" * 80)
    print("  Agent37 — Safety Intelligence Reasoning Layer Demonstration")
    print("=" * 80)

    # 1. Load input evidence
    if events_csv and hotspots_csv:
        print(f"\n[1/4] Loading pipeline CSV evidence:")
        print(f"      Events CSV:   {events_csv}")
        print(f"      Hotspots CSV: {hotspots_csv}")
        risk_input = from_csv(events_csv, hotspots_csv)
    elif input_json:
        print(f"\n[1/4] Loading structured risk output JSON: {input_json}")
        risk_input = from_json(input_json)
    else:
        default_json = ROOT / "data" / "dummy_risk_output.json"
        print(f"\n[1/4] Loading default dummy risk output JSON: {default_json}")
        risk_input = from_json(default_json)

    print(f"      - Analysis ID:        {risk_input.analysis_id}")
    print(f"      - Total Events:       {len(risk_input.conflict_events)}")
    print(f"      - Total Hotspots:     {len(risk_input.hotspots)}")
    print(f"      - ML Peak Risk Score: {risk_input.peak_risk_score:.3f}")
    print(f"      - High Severity (<0.7): {risk_input.high_severity_count}")
    print(f"      - Conflict Types:     {dict(risk_input.conflict_type_counts)}")

    # 2. Initialize Agent37
    print("\n[2/4] Initializing Agent37...")
    agent37_key = agent37_api_key or os.environ.get("AGENT37_API_KEY")
    agent37_inst = instance_id or os.environ.get("AGENT37_INSTANCE_ID")
    hf_tok = hf_token or os.environ.get("HF_TOKEN")

    if deterministic:
        mode_str = "Deterministic Rule Engine (Forced offline mode)"
        agent = Agent37(deterministic_only=True)
    elif agent37_key:
        inst_label = f" [Instance: {agent37_inst}]" if agent37_inst else " [Auto-provision sandbox]"
        mode_str = f"Agent37 Cloud Sandbox (agent37.com){inst_label}"
        agent = Agent37(
            agent37_api_key=agent37_key,
            agent37_instance_id=agent37_inst,
            provider="agent37_cloud",
            model=model if model != "default" else None,
        )
    elif hf_tok:
        model_name = model if model != "default" else "meta-llama/Llama-3.1-8B-Instruct"
        mode_str = f"HuggingFace LLM ({model_name})"
        agent = Agent37(hf_token=hf_tok, model=model_name, deterministic_only=False)
    else:
        mode_str = "Deterministic Rule Engine (Fallback mode — set AGENT37_API_KEY to use cloud sandbox)"
        print("      [NOTE] Neither AGENT37_API_KEY nor HF_TOKEN supplied; defaulting to deterministic rule engine.")
        agent = Agent37(deterministic_only=True)

    print(f"      - Mode: {mode_str}")

    # 3. Perform Analysis
    print("\n[3/4] Running Agent37 safety reasoning...")
    result: AgentAnalysisOutput = agent.analyse(risk_input)

    # 4. Verification and Consistency Checks
    print("\n[4/4] Verifying Architectural Constraints:")

    # Check 1: ML Risk Score Preservation
    ml_peak_input = risk_input.peak_risk_score
    agent_peak_output = result.overall_assessment.ml_peak_risk_score
    score_match = abs(ml_peak_input - agent_peak_output) < 1e-4
    print(f"      [Check 1] ML Risk Score Preservation: {'PASSED [OK]' if score_match else 'FAILED [X]'}")
    print(f"                Input ML peak: {ml_peak_input:.3f} | Output ML peak: {agent_peak_output:.3f}")
    assert score_match, f"ML risk score altered! {ml_peak_input} != {agent_peak_output}"

    # Check 2: Evidence Grounding
    valid_eids = {e.event_id for e in risk_input.conflict_events}
    all_cited_eids = []
    for f in result.key_findings:
        all_cited_eids.extend(f.evidence_event_ids)
    for p in result.patterns:
        all_cited_eids.extend(p.evidence_event_ids)
    for h in result.prioritized_hotspots:
        all_cited_eids.extend(h.evidence_event_ids)
    for r in result.recommendations:
        all_cited_eids.extend(r.evidence_event_ids)

    invalid_eids = [eid for eid in all_cited_eids if eid not in valid_eids]
    grounding_passed = len(invalid_eids) == 0
    print(f"      [Check 2] Evidence Grounding (Traceability): {'PASSED [OK]' if grounding_passed else 'FAILED [X]'}")
    print(f"                Total citations: {len(all_cited_eids)} | Invalid citations: {len(invalid_eids)}")
    assert grounding_passed, f"Found ungrounded event citations: {invalid_eids}"

    # Check 3: Honest Limitations
    lim_passed = len(result.limitations) >= 1
    print(f"      [Check 3] Honest Uncertainty / Limitations: {'PASSED [OK]' if lim_passed else 'FAILED [X]'}")
    print(f"                Limitation statements count: {len(result.limitations)}")
    assert lim_passed, "No limitations provided in analysis output"

    # Check 4: Agent37 Cloud Token Telemetry
    if getattr(agent, "provider", "") == "agent37_cloud" and getattr(agent, "agent37_instance_id", None):
        try:
            usage_info = agent.agent37_cloud_client.get_usage(agent.agent37_instance_id)
            llm_stats = usage_info.get("by_integration", {}).get("llm", {})
            total_toks = llm_stats.get("input_tokens", 0) + llm_stats.get("output_tokens", 0)
            print(f"      [Check 4] Agent37 Cloud Token Telemetry (agent37.com):")
            print(f"                Instance ID:    {agent.agent37_instance_id}")
            print(f"                Platform Calls: {llm_stats.get('calls', 0)}")
            print(f"                Tokens Used:    {total_toks} ({llm_stats.get('input_tokens', 0)} in / {llm_stats.get('output_tokens', 0)} out)")
            print(f"                Billed Cost:    ${llm_stats.get('cost_micros', 0) / 1e6:.6f}")
            print(f"                Dashboard URL:  https://agent37.com/dashboard/cloud/usage")
        except Exception as exc:
            print(f"      [Check 4] Agent37 Cloud Telemetry note: {exc}")

    # Display Report Summary
    print("\n" + "-" * 80)
    print("                    STRUCTURED SAFETY INTELLIGENCE REPORT")
    print("-" * 80)
    print(f"Model Used:         {result.model_used}")
    print(f"Analysis Timestamp: {result.generated_at}")
    print(f"Overall Risk Level: {result.overall_assessment.risk_level.value}")
    print(f"Agent Confidence:   {result.overall_assessment.agent_confidence:.2f}")
    print(f"Executive Summary:  {result.overall_assessment.summary}\n")

    print(f"--- Key Findings ({len(result.key_findings)}) ---")
    for f in result.key_findings:
        print(f"  [{f.finding_id}] ({f.severity.value}) {f.finding}")
        print(f"         Reasoning: {f.reasoning}")
        print(f"         Evidence:  {f.evidence_event_ids}\n")

    print(f"--- Recurring Conflict Patterns ({len(result.patterns)}) ---")
    for p in result.patterns:
        print(f"  [{p.pattern_id}] {p.pattern_type} (Freq: {p.frequency}, Severity: {p.severity.value})")
        print(f"         Description: {p.description}")
        print(f"         Evidence:    {p.evidence_event_ids}\n")

    print(f"--- Prioritized Hotspots ({len(result.prioritized_hotspots)}) ---")
    for h in result.prioritized_hotspots:
        print(f"  Priority {h.priority}: Zone '{h.zone_id}' — Risk: {h.risk_level.value} (ML Avg Risk: {h.ml_avg_risk_score:.3f})")
        print(f"         Reason:   {h.reason}")
        print(f"         Evidence: {h.evidence_event_ids}\n")

    print(f"--- Evidence-Backed Recommendations ({len(result.recommendations)}) ---")
    for r in result.recommendations:
        print(f"  [{r.recommendation_id}] [{r.priority.value}] {r.category.value}")
        print(f"         Action:    {r.action}")
        print(f"         Rationale: {r.rationale}")
        print(f"         Evidence:  {r.evidence_event_ids}\n")

    print(f"--- Limitations and Uncertainty Statements ({len(result.limitations)}) ---")
    for idx, lim in enumerate(result.limitations, 1):
        print(f"  {idx}. {lim}")

    print("-" * 80)

    # Save output if requested
    out_target = output_path or (ROOT / "outputs" / "reports" / "agent37_safety_intelligence.json")
    out_target.parent.mkdir(parents=True, exist_ok=True)
    with open(out_target, "w", encoding="utf-8") as fh:
        fh.write(result.model_dump_json(indent=2))
    print(f"\n[OK] Machine-readable safety intelligence JSON saved to:\n     {out_target}")
    print("=" * 80 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Agent37 Safety Intelligence Demo Runner")
    parser.add_argument("--input", "-i", type=Path, default=ROOT / "data" / "dummy_risk_output.json",
                        help="Path to input risk JSON (default: data/dummy_risk_output.json)")
    parser.add_argument("--events-csv", type=Path, default=None,
                        help="Path to conflict_events.csv (alternative to JSON)")
    parser.add_argument("--hotspots-csv", type=Path, default=None,
                        help="Path to hotspot_summary.csv (alternative to JSON)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path for generated JSON report")
    parser.add_argument("--deterministic", "-d", action="store_true",
                        help="Force deterministic rule engine fallback mode")
    parser.add_argument("--agent37-api-key", type=str, default=None,
                        help="Agent37 Cloud API key from agent37.com")
    parser.add_argument("--instance-id", type=str, default=None,
                        help="Agent37 sandbox instance ID (optional)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token for LLM reasoning")
    parser.add_argument("--model", type=str, default="default",
                        help="Model ID (e.g. meta-llama/Llama-3.1-8B-Instruct or agent37 default)")
    args = parser.parse_args()

    run_demo(
        input_json=args.input,
        events_csv=args.events_csv,
        hotspots_csv=args.hotspots_csv,
        output_path=args.output,
        deterministic=args.deterministic,
        agent37_api_key=args.agent37_api_key,
        instance_id=args.instance_id,
        hf_token=args.hf_token,
        model=args.model,
    )


if __name__ == "__main__":
    main()
