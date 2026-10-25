"""
agent/agent37.py — Agent37: LLM-powered reasoning layer for Before the Crash.

Architecture:
    RiskAnalysisInput
        ↓
    Agent37.analyse()
        ├── calls HuggingFace Inference API (Llama 3.1 8B by default)
        ├── validates JSON response with Pydantic
        └── falls back to deterministic rule engine on any failure
        ↓
    AgentAnalysisOutput

The HF Inference API is free-tier with no local GPU requirement.
Set HF_TOKEN environment variable or pass hf_token= to the constructor.

Model selection is configurable. Default: meta-llama/Llama-3.1-8B-Instruct
(the best free reasoning model available on HF Inference API as of 2025).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .prompts import (
    AGENT37_SYSTEM_PROMPT,
    build_analysis_prompt,
    extract_json_from_response,
)
from .recommendation_rules import run_deterministic_analysis
from .schemas import (
    AgentAnalysisOutput,
    OverallAssessment,
    RiskAnalysisInput,
    RiskLevel,
)

log = logging.getLogger(__name__)


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


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

HF_API_BASE     = "https://api-inference.huggingface.co/models"
DEFAULT_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
REQUEST_TIMEOUT = 90   # seconds — HF free tier can be slow on cold start
MAX_NEW_TOKENS  = 2048


# ──────────────────────────────────────────────────────────────────────────────
# Agent37
# ──────────────────────────────────────────────────────────────────────────────

class Agent37:
    """
    Agent37 — LLM reasoning layer for the Before the Crash pipeline.

    Usage:
        agent = Agent37()                          # reads HF_TOKEN from env
        agent = Agent37(hf_token="hf_...")         # explicit token
        agent = Agent37(deterministic_only=True)   # skip LLM, rule engine only

        output: AgentAnalysisOutput = agent.analyse(risk_input)

    The agent attempts LLM analysis first. If the call fails (network error,
    rate limit, invalid JSON, Pydantic validation failure) it falls back to
    the deterministic rule engine and logs a warning. This guarantees output
    even during offline demos.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        agent37_api_key: Optional[str] = None,
        agent37_instance_id: Optional[str] = None,
        provider: str = "auto",
        model: Optional[str] = None,
        deterministic_only: bool = False,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.1,
    ):
        """
        Args:
            hf_token:           HuggingFace API token. Falls back to HF_TOKEN env var.
            agent37_api_key:    Agent37 Cloud API key from agent37.com. Falls back to AGENT37_API_KEY.
            agent37_instance_id: Optional Agent37 instance ID from agent37.com/dashboard/cloud.
            provider:           'auto', 'agent37_cloud', 'huggingface', or 'deterministic'.
            model:              Model identifier or default for provider.
            deterministic_only: If True, skip remote LLM calls entirely.
            max_new_tokens:     Max tokens the agent may generate.
            temperature:        Sampling temperature. Keep low for structured JSON.
        """
        _load_dotenv_if_needed()
        self.agent37_api_key = agent37_api_key or os.environ.get("AGENT37_API_KEY", "")
        self.agent37_instance_id = agent37_instance_id or os.environ.get("AGENT37_INSTANCE_ID", "")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.deterministic_only = deterministic_only
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Resolve provider
        if deterministic_only or provider == "deterministic":
            self.provider = "deterministic"
            self.deterministic_only = True
        elif provider == "agent37_cloud" or (provider == "auto" and self.agent37_api_key):
            self.provider = "agent37_cloud"
            self.model = model or "default"
            from .agent37_cloud import Agent37CloudClient
            self.agent37_cloud_client = Agent37CloudClient(api_key=self.agent37_api_key)
        elif provider == "huggingface" or (provider == "auto" and self.hf_token):
            self.provider = "huggingface"
            self.model = model or DEFAULT_MODEL
        else:
            log.warning(
                "Agent37: Neither AGENT37_API_KEY nor HF_TOKEN found. "
                "Defaulting to deterministic rule engine."
            )
            self.provider = "deterministic"
            self.deterministic_only = True
            self.model = model or "deterministic"

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def analyse(self, inp: RiskAnalysisInput) -> AgentAnalysisOutput:
        """
        Analyse a RiskAnalysisInput and return a validated AgentAnalysisOutput.

        Always returns a result — LLM / sandbox failures degrade to the rule engine.
        """
        if self.deterministic_only or self.provider == "deterministic":
            log.info("Agent37: deterministic-only mode.")
            return run_deterministic_analysis(inp, model_label="deterministic-fallback")

        # Attempt sandbox / LLM analysis
        try:
            return self._llm_analyse(inp)
        except Exception as exc:
            log.warning(
                "Agent37: %s call failed (%s: %s). "
                "Falling back to deterministic rule engine.",
                self.provider, type(exc).__name__, exc,
            )
            result = run_deterministic_analysis(
                inp, model_label=f"deterministic-fallback ({self.provider} failed)"
            )
            result = result.model_copy(update={
                "model_used": f"deterministic-fallback ({self.provider} failed: {type(exc).__name__})"
            })
            return result

    # ──────────────────────────────────────────────────────────────────────────
    # LLM / Sandbox call
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_analyse(self, inp: RiskAnalysisInput) -> AgentAnalysisOutput:
        """Execute reasoning via Agent37 cloud sandbox or HuggingFace API."""
        input_json = inp.model_dump_json(
            indent=2,
            exclude={"llm_raw_response"},
        )
        user_msg = build_analysis_prompt(input_json)

        if self.provider == "agent37_cloud":
            raw_response = self._call_agent37_cloud(input_json)
        else:
            raw_response = self._call_hf_api(user_msg)

        log.debug("Agent37 raw response (first 500 chars): %s", raw_response[:500])

        # Extract JSON from response
        parsed_dict = extract_json_from_response(raw_response)

        # Enforce ML risk score preservation — never let LLM change it
        ml_peak = inp.peak_risk_score
        if "overall_assessment" in parsed_dict:
            parsed_dict["overall_assessment"]["ml_peak_risk_score"] = ml_peak

        # Add required top-level provenance fields
        parsed_dict.setdefault("analysis_id",       inp.analysis_id)
        parsed_dict.setdefault("agent_version",     "37.0")
        parsed_dict.setdefault(
            "generated_at",
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        parsed_dict["model_used"]        = self.model
        parsed_dict["input_event_count"] = len(inp.conflict_events)
        parsed_dict["input_hotspot_count"] = len(inp.hotspots)
        parsed_dict["llm_raw_response"]  = raw_response

        # Pydantic validation — raises ValidationError on schema mismatch
        try:
            output = AgentAnalysisOutput.model_validate(parsed_dict)
        except Exception as val_err:
            log.warning(
                "Agent37: LLM JSON failed Pydantic validation: %s. "
                "Attempting partial merge with deterministic output.",
                val_err,
            )
            output = self._merge_with_fallback(inp, parsed_dict, raw_response)

        return output

    def _call_agent37_cloud(self, input_json_str: str) -> str:
        """Execute reasoning directly inside the Agent37 Cloud Sandbox container (agent37.com)."""
        instance_id = self.agent37_instance_id
        if not instance_id:
            # Auto-discover or provision sandbox instance
            instances = self.agent37_cloud_client.list_instances()
            if instances:
                instance_id = instances[0].get("id")
            else:
                inst = self.agent37_cloud_client.create_instance(template="agent37-opencode", auto_sleep=True)
                instance_id = inst.get("id")
            self.agent37_instance_id = instance_id

        log.info("Agent37: executing safety reasoning inside cloud sandbox instance '%s'...", instance_id)
        return self.agent37_cloud_client.run_agent_in_sandbox(instance_id, input_json_str)

    def _call_hf_api(self, user_message: str) -> str:
        """
        Call the HuggingFace text-generation Inference API.

        Attempts:
          1. huggingface_hub.InferenceClient (if installed and available)
          2. Standard library urllib (zero external dependency)
        """
        messages = [
            {"role": "system", "content": AGENT37_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        # 1. Try official InferenceClient if available
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=self.hf_token or None, timeout=REQUEST_TIMEOUT)
            res = client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            content = res.choices[0].message.content
            if content:
                return content
        except ImportError:
            pass
        except Exception as e:
            log.info("InferenceClient attempt encountered %s: %s; trying direct HTTP.", type(e).__name__, e)

        # 2. Urllib fallback
        # Try modern router URL first, then classic models endpoint
        endpoints = [
            "https://router.huggingface.co/hf-inference/v1/chat/completions",
            f"{HF_API_BASE}/{self.model}/v1/chat/completions",
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens":  self.max_new_tokens,
            "temperature": self.temperature,
            "stream":      False,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type":  "application/json",
        }

        last_error = None
        for url in endpoints:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    resp_body = resp.read().decode("utf-8")
                resp_json = json.loads(resp_body)
                if "choices" in resp_json and resp_json["choices"]:
                    return resp_json["choices"][0]["message"]["content"]
                if "generated_text" in resp_json:
                    return resp_json["generated_text"]
                if isinstance(resp_json, list) and resp_json and "generated_text" in resp_json[0]:
                    return resp_json[0]["generated_text"]
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HF API ({url}) returned HTTP {e.code}: {err_body[:300]}")
            except urllib.error.URLError as e:
                last_error = RuntimeError(f"HF API ({url}) network error: {e.reason}")
            except Exception as e:
                last_error = e

        if last_error:
            raise last_error
        raise RuntimeError("Failed to query Hugging Face API across all endpoints.")

    def _merge_with_fallback(
        self,
        inp: RiskAnalysisInput,
        partial: Dict[str, Any],
        raw_response: str,
    ) -> AgentAnalysisOutput:
        """
        When LLM JSON partially validates, use the deterministic engine as the
        base and overlay any valid LLM fields on top.

        This ensures we always produce a valid output even when the model
        produces partially malformed JSON.
        """
        base = run_deterministic_analysis(
            inp, model_label=f"{self.model} (partial-merge)"
        )

        # Try to use LLM summary if it provided one
        if (
            "overall_assessment" in partial
            and isinstance(partial["overall_assessment"], dict)
            and "summary" in partial["overall_assessment"]
            and isinstance(partial["overall_assessment"]["summary"], str)
            and len(partial["overall_assessment"]["summary"]) > 10
        ):
            base = base.model_copy(update={
                "overall_assessment": base.overall_assessment.model_copy(update={
                    "summary": partial["overall_assessment"]["summary"],
                }),
                "llm_raw_response": raw_response,
                "model_used": f"{self.model} (partial-merge)",
            })

        return base
