// frontend/src/data/dummyAgentData.js
// Source: outputs/reports/agent37_safety_intelligence.json & agent37_dummy_session_output.json

export const dummyAgentData = {
  "analysis_id": "ANALYSIS_DUMMY_SESSION_01",
  "agent_version": "37.0",
  "generated_at": "2026-10-24T23:25:10.494879+00:00",
  "model_used": "Agent37 Cloud Sandbox (gVisor isolated)",
  "overall_assessment": {
    "risk_level": "CRITICAL",
    "ml_peak_risk_score": 0.85,
    "agent_confidence": 0.85,
    "summary": "Agent37 sandbox analysis of 8 conflict events (ML peak risk: 0.85). High-severity events and vulnerable road user interactions warrant prioritization for engineering assessment."
  },
  "key_findings": [
    {
      "finding_id": "F001",
      "finding": "3 conflict events involve vulnerable pedestrian-vehicle interactions.",
      "severity": "HIGH",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002",
        "EVT_008"
      ],
      "reasoning": "Pedestrians are vulnerable road users; interactions carry elevated injury risk. 2 events exhibit ML severity ≥ 0.70 with minimal time-to-collision."
    },
    {
      "finding_id": "F002",
      "finding": "2 conflict events involve motorcycles.",
      "severity": "MEDIUM",
      "evidence_event_ids": [
        "EVT_004",
        "EVT_005"
      ],
      "reasoning": "Motorcycles have reduced visibility and elevated injury risk. Merging and turning dynamics contribute to proximity conflicts."
    },
    {
      "finding_id": "F003",
      "finding": "4 events include evasive hard-braking or sudden deceleration.",
      "severity": "MEDIUM",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002",
        "EVT_003",
        "EVT_006"
      ],
      "reasoning": "Repeated hard-braking indicates unexpected stopping demand, sight-distance issues, or late driver anticipation."
    },
    {
      "finding_id": "F004",
      "finding": "7 events have TTC below the configured safety threshold (3.0s).",
      "severity": "MEDIUM",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002",
        "EVT_003",
        "EVT_004",
        "EVT_005",
        "EVT_006",
        "EVT_007"
      ],
      "reasoning": "Minimum observed TTC reached 1.15s in Zone R1C2, leaving negligible driver reaction buffer."
    }
  ],
  "patterns": [
    {
      "pattern_id": "P001",
      "pattern_type": "RECURRING_PED_VEHICLE_ZONE",
      "description": "Recurring pedestrian conflicts concentrated in zone 'R1C2' (2 of 3 pedestrian-vehicle events).",
      "frequency": 3,
      "severity": "HIGH",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002",
        "EVT_008"
      ]
    },
    {
      "pattern_id": "P002",
      "pattern_type": "REPEATED_EVASIVE_BRAKING",
      "description": "Evasive deceleration detected across 4 events, indicating repeated unanticipated stop demands.",
      "frequency": 4,
      "severity": "MEDIUM",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002",
        "EVT_003",
        "EVT_006"
      ]
    },
    {
      "pattern_id": "P003",
      "pattern_type": "HIGH_SEVERITY_CLUSTER",
      "description": "Zone 'R1C2' contains 2 high-severity events (ML avg risk: 0.727).",
      "frequency": 2,
      "severity": "HIGH",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002"
      ]
    }
  ],
  "prioritized_hotspots": [
    {
      "zone_id": "R1C2",
      "priority": 1,
      "risk_level": "HIGH",
      "reason": "Zone 'R1C2' recorded 3 events (2 high-severity) with ML avg risk 0.727. Dominant conflict: VEH_PED. Min TTC: 1.15s.",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002",
        "EVT_003"
      ],
      "ml_avg_risk_score": 0.727
    },
    {
      "zone_id": "R2C3",
      "priority": 2,
      "risk_level": "MEDIUM",
      "reason": "Zone 'R2C3' recorded 3 events (2 high-severity) with ML avg risk 0.600. Dominant conflict: VEH_MOTO. Min TTC: 1.35s.",
      "evidence_event_ids": [
        "EVT_004",
        "EVT_005",
        "EVT_006"
      ],
      "ml_avg_risk_score": 0.600
    },
    {
      "zone_id": "R0C1",
      "priority": 3,
      "risk_level": "LOW",
      "reason": "Zone 'R0C1' recorded 2 events (0 high-severity) with ML avg risk 0.190. Dominant conflict: VEH_VEH. Min TTC: 2.85s.",
      "evidence_event_ids": [
        "EVT_007",
        "EVT_008"
      ],
      "ml_avg_risk_score": 0.190
    }
  ],
  "recommendations": [
    {
      "recommendation_id": "R001",
      "category": "PEDESTRIAN_CROSSING",
      "priority": "URGENT",
      "action": "Prioritize this location for traffic-engineering assessment of pedestrian crossing placement, visibility, and signal phasing.",
      "rationale": "3 pedestrian conflicts detected. Vulnerable road users require crossing protection and adequate lead-pedestrian intervals.",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002",
        "EVT_008"
      ],
      "finding_ids": [
        "F001",
        "F004"
      ]
    },
    {
      "recommendation_id": "R002",
      "category": "VISIBILITY_SIGNAGE",
      "priority": "MEDIUM",
      "action": "Investigate whether sight-distance deficiencies, advance warning signage, or queue spill-back contribute to evasive braking.",
      "rationale": "4 events required evasive braking, suggesting unanticipated hazards or poor forward sight distance.",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002",
        "EVT_003",
        "EVT_006"
      ],
      "finding_ids": [
        "F003"
      ]
    },
    {
      "recommendation_id": "R003",
      "category": "TRAFFIC_CALMING",
      "priority": "HIGH",
      "action": "Evaluate traffic-calming measures in zone R1C2 such as kerb extensions, high-visibility crosswalk markings, or speed table adjustments.",
      "rationale": "2 high-severity pedestrian conflicts show rapid deceleration from crossing vehicles.",
      "evidence_event_ids": [
        "EVT_001",
        "EVT_002"
      ],
      "finding_ids": [
        "F001"
      ]
    },
    {
      "recommendation_id": "R004",
      "category": "MOTORCYCLE_SAFETY",
      "priority": "MEDIUM",
      "action": "Investigate lane-splitting, turning visibility, and motorcycle merging dynamics at zone R2C3.",
      "rationale": "2 motorcycle-involved encounters with trucks and cars with significant swerving maneuvers.",
      "evidence_event_ids": [
        "EVT_004",
        "EVT_005"
      ],
      "finding_ids": [
        "F002"
      ]
    }
  ],
  "limitations": [
    "All spatial metrics are computed in image-space pixels. Camera calibration was not active; physical-unit values (meters, km/h) are unavailable.",
    "TTC and PET estimates assume linear motion over the prediction horizon; non-linear turning movements may alter real values.",
    "Conflict rates are not normalised by traffic exposure (volume data). Higher event counts may reflect higher traffic density rather than intrinsic structural danger.",
    "Observations cover a 60-second video sample. Short observation periods cannot account for hourly or diurnal traffic fluctuation.",
    "Analysis is based on observed near-miss interactions rather than confirmed crashes. Recommendations require civil and traffic engineering validation before field deployment."
  ],
  "input_event_count": 8,
  "input_hotspot_count": 3
};
