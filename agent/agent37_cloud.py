"""
agent/agent37_cloud.py — Client for the Agent37 Cloud Sandbox platform (agent37.com).

Connects to https://api.agent37.com/v1 to:
  1. Manage isolated cloud sandbox instances.
  2. Execute reasoning / agent tasks in the persistent sandbox environment.
  3. Route through Agent37's OpenAI-compatible gateway.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DEFAULT_AGENT37_API_BASE = "https://api.agent37.com/v1"
DEFAULT_TIMEOUT = 150


class Agent37CloudClient:
    """
    Client for interacting with the Agent37 Cloud Sandbox platform (agent37.com).

    Authentication requires an API key from https://agent37.com/dashboard/cloud/api-keys.
    Set the AGENT37_API_KEY environment variable or pass api_key to the constructor.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_AGENT37_API_BASE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("AGENT37_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            log.warning(
                "Agent37CloudClient: No AGENT37_API_KEY found. "
                "Get your key from https://agent37.com/dashboard/cloud/api-keys"
            )

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform an HTTP request to the Agent37 Cloud API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        body = json.dumps(data).encode("utf-8") if data is not None else None

        req = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_text = resp.read().decode("utf-8")
                return json.loads(resp_text) if resp_text else {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Agent37 Cloud API HTTP {e.code} on {method} {url}: {err_body[:400]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Agent37 Cloud API network error connecting to {url}: {e.reason}"
            ) from e

    # ──────────────────────────────────────────────────────────────────────────
    # Sandbox Instance Management
    # ──────────────────────────────────────────────────────────────────────────

    def create_instance(
        self,
        template: str = "default",
        auto_sleep: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new persistent agent sandbox instance on Agent37 Cloud.

        Args:
            template:   Template name (e.g. 'default', 'agent37-opencode', 'agent37-hermes')
            auto_sleep: Whether the instance should automatically sleep when idle.
            metadata:   Optional metadata dictionary.

        Returns:
            Dict containing instance details (e.g. 'id', 'status', 'gateway_url').
        """
        payload: Dict[str, Any] = {
            "template": template,
            "auto_sleep": auto_sleep,
        }
        if metadata:
            payload["metadata"] = metadata

        log.info("Creating Agent37 cloud sandbox instance (template=%s)...", template)
        res = self._request("POST", "/instances", data=payload)
        instance_id = res.get("id")
        if instance_id:
            try:
                self.set_instance_budget(instance_id, 5000000)
            except Exception as e:
                log.debug("Auto-setting budget on new instance note: %s", e)
        return res

    def list_instances(self) -> List[Dict[str, Any]]:
        """List all agent instances in your Agent37 workspace."""
        res = self._request("GET", "/instances")
        if isinstance(res, list):
            return res
        return res.get("data", res.get("instances", []))

    def get_instance(self, instance_id: str) -> Dict[str, Any]:
        """Get status and details of a specific sandbox instance."""
        return self._request("GET", f"/instances/{instance_id}")

    def delete_instance(self, instance_id: str) -> Dict[str, Any]:
        """Delete an instance to terminate its storage/compute billing."""
        return self._request("DELETE", f"/instances/{instance_id}")

    # ──────────────────────────────────────────────────────────────────────────
    # Budget & Usage Management
    # ──────────────────────────────────────────────────────────────────────────

    def get_instance_budget(self, instance_id: str) -> Dict[str, Any]:
        """Get the monthly budget cap and remaining micros for an instance."""
        return self._request("GET", f"/instances/{instance_id}/budget")

    def set_instance_budget(self, instance_id: str, monthly_cap_micros: int = 5000000) -> Dict[str, Any]:
        """
        Set the monthly cap for an instance (5000000 micros = $5.00).
        Required for the sandbox instance to invoke AI models via Agent37 LLM proxy.
        """
        return self._request(
            "PATCH",
            f"/instances/{instance_id}/budget",
            data={"monthly_cap_micros": monthly_cap_micros},
        )

    def get_usage(self, instance_id: str) -> Dict[str, Any]:
        """
        Get token and cost usage registered for this instance on Agent37 Cloud.
        Reflects model calls made by the sandbox agent under the dashboard Usage tab.
        """
        return self._request("GET", f"/instances/{instance_id}/usage")

    # ──────────────────────────────────────────────────────────────────────────
    # Sandbox Command Execution (Remote Container Exec)
    # ──────────────────────────────────────────────────────────────────────────

    def exec_command(self, instance_id: str, command: str) -> Dict[str, Any]:
        """
        Execute a shell command inside the isolated Agent37 sandbox instance.

        POST /v1/instances/{instance_id}/exec
        Payload: {"command": command}

        Returns:
            Dict containing 'exit_code', 'stdout', 'stderr', 'truncated'.
        """
        return self._request(
            "POST",
            f"/instances/{instance_id}/exec",
            data={"command": command},
        )

    def run_agent_in_sandbox(self, instance_id: str, input_json_str: str) -> str:
        """
        Execute Agent37 safety reasoning inside the isolated Agent37 sandbox.

        Uses the Agent37 LLM reasoning proxy within the sandbox container (consuming
        platform model tokens that register under Agent37 dashboard Usage), validates
        authoritative ML risk metrics, and includes deterministic fallback inside
        the container for resilient offline execution.
        """
        import base64

        # Ensure instance has an active budget cap so LLM calls are authorized
        try:
            b = self.get_instance_budget(instance_id)
            if b.get("monthly_cap_micros", 0) <= 0:
                self.set_instance_budget(instance_id, 5000000)
        except Exception as exc:
            log.debug("Agent37 budget check notice: %s", exc)

        py_script = f'''
import json, sys, os, urllib.request, urllib.error, re

input_data = json.loads({json.dumps(input_json_str)})
events = input_data.get("conflict_events", [])
hotspots = input_data.get("hotspots", [])
peak_risk = float(input_data.get("peak_risk_score", 0.0))
analysis_id = input_data.get("analysis_id", "A37-SANDBOX")

# 1. Attempt reasoning via Agent37 LLM proxy inside sandbox
llm_output = None
llm_err = None

proxy_url = (os.environ.get("AGENT37_LLM_PROXY_URL") or "https://api.agent37.com/llm/v1").rstrip("/") + "/chat/completions"
token = os.environ.get("AGENT37_MANAGED_TOKEN") or os.environ.get("AGENT37_STARTER_TOKEN") or ""

system_prompt = """You are Agent37, an expert AI traffic safety intelligence and risk reasoning agent.
You are running inside an isolated Agent37 cloud sandbox container.
Your task: analyse near-miss conflict events and hotspot zones produced by the upstream ML pipeline.

CRITICAL RULES:
1. Output valid JSON ONLY.
2. NEVER recalculate or alter numerical risk metrics (ttc_min, pet_min, severity, risk_score). Peak risk score must match input exactly.
3. Every finding, pattern, and recommendation MUST cite valid evidence_event_ids from the input.
4. Output schema MUST contain overall_assessment, key_findings, patterns, prioritized_hotspots, recommendations, and limitations.
"""

user_prompt = "Analyse the following near-miss conflict evidence from Before the Crash:\\n" + json.dumps(input_data, indent=2) + "\\n\\nProduce your complete AgentAnalysisOutput JSON following all rules strictly."

if token:
    try:
        payload = {{
            "model": "default",
            "messages": [
                {{"role": "system", "content": system_prompt}},
                {{"role": "user", "content": user_prompt}}
            ],
            "max_tokens": 1200,
            "temperature": 0.1
        }}
        req = urllib.request.Request(
            proxy_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={{"Content-Type": "application/json", "Authorization": f"Bearer {{token}}"}},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            raw_text = resp_body.get("choices", [{{}}])[0].get("message", {{}}).get("content", "")
            
            # Extract JSON block
            start_i = raw_text.find(chr(123))
            end_i = raw_text.rfind(chr(125))
            if start_i != -1 and end_i > start_i:
                parsed = json.loads(raw_text[start_i:end_i + 1])
                if "overall_assessment" in parsed and "key_findings" in parsed:
                    llm_output = parsed
    except Exception as exc:
        llm_err = str(exc)

# 2. If LLM succeeded, validate and preserve ML peak risk score
if llm_output is not None:
    llm_output.setdefault("overall_assessment", {{}})
    llm_output["overall_assessment"]["ml_peak_risk_score"] = peak_risk
    lims = llm_output.get("limitations", [])
    lims.append(f"Executed inside Agent37 cloud sandbox instance {instance_id} via LLM reasoning.")
    llm_output["limitations"] = lims
    final_out = llm_output
else:
    # 3. Deterministic rule engine fallback inside sandbox
    peds = [e.get("event_id") for e in events if "PED" in e.get("conflict_type", "")]
    highs = [e.get("event_id") for e in events if float(e.get("severity", 0)) >= 0.7]
    brakes = [e.get("event_id") for e in events if any("brak" in str(r).lower() for r in e.get("reasons", []))]
    motos = [e.get("event_id") for e in events if "MOTO" in e.get("conflict_type", "")]

    findings = []
    if peds:
        findings.append({{
            "finding_id": "F001",
            "finding": f"{{len(peds)}} conflict events involve vulnerable pedestrian-vehicle interactions.",
            "severity": "HIGH" if highs else "MEDIUM",
            "evidence_event_ids": peds,
            "reasoning": "Pedestrians are vulnerable road users; interactions carry elevated injury risk."
        }})
    if motos:
        findings.append({{
            "finding_id": f"F{{len(findings)+1:03d}}",
            "finding": f"{{len(motos)}} conflict events involve motorcycles.",
            "severity": "MEDIUM",
            "evidence_event_ids": motos,
            "reasoning": "Motorcycles have reduced visibility and elevated injury risk."
        }})
    if brakes:
        findings.append({{
            "finding_id": f"F{{len(findings)+1:03d}}",
            "finding": f"{{len(brakes)}} events include evasive hard-braking or sudden deceleration.",
            "severity": "MEDIUM",
            "evidence_event_ids": brakes,
            "reasoning": "Repeated hard-braking indicates unexpected stopping demand or sight-distance issues."
        }})

    patterns = []
    if len(peds) >= 2:
        patterns.append({{
            "pattern_id": "P001",
            "pattern_type": "RECURRING_PED_VEHICLE_ZONE",
            "description": f"Recurring pedestrian conflicts concentrated in zone ({{len(peds)}} events).",
            "frequency": len(peds),
            "severity": "HIGH",
            "evidence_event_ids": peds
        }})
    if len(brakes) >= 2:
        patterns.append({{
            "pattern_id": f"P{{len(patterns)+1:03d}}",
            "pattern_type": "REPEATED_EVASIVE_BRAKING",
            "description": f"Evasive deceleration detected across {{len(brakes)}} events.",
            "frequency": len(brakes),
            "severity": "MEDIUM",
            "evidence_event_ids": brakes
        }})

    prioritized = []
    for idx, h in enumerate(sorted(hotspots, key=lambda x: (x.get("high_risk_count", 0), x.get("avg_risk_score", 0)), reverse=True), 1):
        eids = h.get("event_ids") or [e.get("event_id") for e in events if e.get("zone_id") == h.get("zone_id")]
        if eids:
            avg_r = float(h.get("avg_risk_score", 0.0))
            prioritized.append({{
                "zone_id": h.get("zone_id", "UNKNOWN"),
                "priority": idx,
                "risk_level": "HIGH" if avg_r >= 0.65 else ("MEDIUM" if avg_r >= 0.4 else "LOW"),
                "reason": f"Zone '{{h.get('zone_id')}}' recorded {{h.get('event_count', 0)}} events with ML avg risk {{avg_r:.2f}}.",
                "evidence_event_ids": eids[:10],
                "ml_avg_risk_score": avg_r
            }})

    recs = []
    if peds:
        recs.append({{
            "recommendation_id": "R001",
            "category": "PEDESTRIAN_CROSSING",
            "priority": "URGENT" if peak_risk >= 0.8 else "HIGH",
            "action": "Prioritize this location for traffic-engineering assessment of pedestrian crossing placement, visibility, and signal phasing.",
            "rationale": f"{{len(peds)}} pedestrian conflicts detected. Vulnerable road users require crossing protection.",
            "evidence_event_ids": peds,
            "finding_ids": ["F001"]
        }})
    if brakes:
        recs.append({{
            "recommendation_id": f"R{{len(recs)+1:03d}}",
            "category": "VISIBILITY_SIGNAGE",
            "priority": "MEDIUM",
            "action": "Investigate whether sight-distance deficiencies, advance warning signage, or queue spill-back contribute to evasive braking.",
            "rationale": f"{{len(brakes)}} events required evasive braking, suggesting unanticipated hazards.",
            "evidence_event_ids": brakes,
            "finding_ids": ["F002"] if len(findings) > 1 else []
        }})

    rlevel = "CRITICAL" if peak_risk >= 0.85 else ("HIGH" if peak_risk >= 0.65 else ("MEDIUM" if peak_risk >= 0.4 else "LOW"))
    lims = [
        "All spatial metrics are computed in image-space pixels. Camera calibration was not active; physical-unit values are unavailable.",
        "TTC and PET estimates assume linear motion over the prediction horizon; non-linear turning movements may alter real values.",
        "Conflict rates are not normalised by traffic exposure (volume data). Higher event counts may reflect higher traffic density.",
        f"Executed inside Agent37 cloud sandbox instance {instance_id} (fallback reason: {{llm_err or 'rule_mode'}})."
    ]

    final_out = {{
        "overall_assessment": {{
            "risk_level": rlevel,
            "ml_peak_risk_score": peak_risk,
            "agent_confidence": 0.85,
            "summary": f"Agent37 sandbox analysis of {{len(events)}} conflict events (ML peak risk: {{peak_risk:.2f}}). High-severity events and vulnerable road user interactions warrant prioritization for engineering assessment."
        }},
        "key_findings": findings,
        "patterns": patterns,
        "prioritized_hotspots": prioritized,
        "recommendations": recs,
        "limitations": lims
    }}

print("__AGENT37_OUT_START__" + json.dumps(final_out) + "__AGENT37_OUT_END__")
'''
        b64_script = base64.b64encode(py_script.encode("utf-8")).decode("ascii")
        cmd = f"echo {b64_script} | base64 -d | python3"

        res = self.exec_command(instance_id, cmd)
        if res.get("exit_code") != 0:
            raise RuntimeError(f"Sandbox execution failed (exit {res.get('exit_code')}): {res.get('stderr')}")

        stdout = res.get("stdout", "")
        if "__AGENT37_OUT_START__" in stdout and "__AGENT37_OUT_END__" in stdout:
            return stdout.split("__AGENT37_OUT_START__")[1].split("__AGENT37_OUT_END__")[0].strip()
        return stdout

