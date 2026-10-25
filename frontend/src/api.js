// frontend/src/api.js
// Provides uniform API access to either the live FastAPI backend (http://localhost:8000)
// or the embedded offline demo dataset.

import { dummyRiskData } from './data/dummyRiskData.js';
import { dummyAgentData } from './data/dummyAgentData.js';

const API_BASE_URL = 'http://localhost:8000/api';

class ApiService {
  constructor() {
    this.mode = 'auto'; // 'auto', 'live', 'demo'
    this.isLiveAvailable = false;
    this.simulationState = 'none'; // 'none', 'empty_events', 'empty_hotspots', 'api_error', 'no_report'
  }

  setSimulationState(state) {
    this.simulationState = state;
  }

  setMode(mode) {
    this.mode = mode;
  }

  async checkHealth() {
    try {
      const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        const data = await res.json();
        this.isLiveAvailable = data.status === 'healthy';
        return { isLive: true, data };
      }
    } catch {
      this.isLiveAvailable = false;
    }
    return { isLive: false, data: null };
  }

  shouldUseLive() {
    if (this.mode === 'live') return true;
    if (this.mode === 'demo') return false;
    return this.isLiveAvailable;
  }

  async getAnalysis() {
    if (this.simulationState === 'api_error') {
      throw new Error('API connection refused (503 Service Unavailable)');
    }

    if (this.shouldUseLive()) {
      try {
        const res = await fetch(`${API_BASE_URL}/analysis/latest`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (err) {
        console.warn('Live API request failed, falling back to demo data:', err);
      }
    }

    // Demo Data Mode
    const data = { ...dummyRiskData };
    return {
      analysis_id: data.analysis_id,
      session_id: data.session_id,
      analysis_ts: data.analysis_ts,
      video_path: data.video_path,
      camera: data.camera,
      traffic_stats: data.traffic_stats,
      ttc_threshold_s: data.ttc_threshold_s,
      pet_threshold_s: data.pet_threshold_s,
      total_event_count: data.total_event_count,
      high_severity_count: data.high_severity_count,
      peak_risk_score: data.peak_risk_score,
      avg_risk_score: data.avg_risk_score,
      conflict_type_counts: data.conflict_type_counts,
      overall_assessment: dummyAgentData.overall_assessment,
      agent_version: dummyAgentData.agent_version,
      generated_at: dummyAgentData.generated_at
    };
  }

  async getHotspots() {
    if (this.simulationState === 'api_error') {
      throw new Error('Failed to load hotspots: 503 Service Unavailable');
    }
    if (this.simulationState === 'empty_hotspots') {
      return [];
    }

    if (this.shouldUseLive()) {
      try {
        const res = await fetch(`${API_BASE_URL}/hotspots`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (err) {
        console.warn('Live API hotspots failed, using fallback:', err);
      }
    }

    // Demo enrich
    const prioritized = {};
    for (const p of dummyAgentData.prioritized_hotspots) {
      prioritized[p.zone_id] = p;
    }

    return dummyRiskData.hotspots.map(h => {
      const p = prioritized[h.zone_id] || {};
      const zoneEvents = new Set(h.event_ids || []);
      const relatedFindings = dummyAgentData.key_findings
        .filter(f => f.evidence_event_ids.some(eid => zoneEvents.has(eid)))
        .map(f => f.finding_id);
      const relatedRecommendations = dummyAgentData.recommendations
        .filter(r => r.evidence_event_ids.some(eid => zoneEvents.has(eid)))
        .map(r => r.recommendation_id);

      return {
        ...h,
        priority: p.priority ?? 99,
        risk_level: p.risk_level || 'LOW',
        reason: p.reason || '',
        related_finding_ids: relatedFindings,
        related_recommendation_ids: relatedRecommendations
      };
    }).sort((a, b) => (a.priority || 99) - (b.priority || 99));
  }

  async getEvents(filters = {}) {
    if (this.simulationState === 'api_error') {
      throw new Error('Failed to load conflict events: 503 Service Unavailable');
    }
    if (this.simulationState === 'empty_events') {
      return [];
    }

    let events = [];
    if (this.shouldUseLive()) {
      try {
        const params = new URLSearchParams();
        if (filters.zone_id) params.set('zone_id', filters.zone_id);
        if (filters.conflict_type) params.set('conflict_type', filters.conflict_type);
        if (filters.min_risk !== undefined && filters.min_risk !== null) params.set('min_risk', filters.min_risk);
        if (filters.min_severity !== undefined && filters.min_severity !== null) params.set('min_severity', filters.min_severity);

        const res = await fetch(`${API_BASE_URL}/events?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        events = await res.json();
      } catch (err) {
        console.warn('Live API events failed, using demo data:', err);
      }
    }

    if (!events.length) {
      events = dummyRiskData.conflict_events.map(e => {
        const clipName = e.evidence_clip_path ? e.evidence_clip_path.split('/').pop() : null;
        // In demo mode, EVT_001 through EVT_006 have video clips
        const available = ['EVT_001', 'EVT_002', 'EVT_003', 'EVT_004', 'EVT_005', 'EVT_006'].includes(e.event_id);
        return {
          ...e,
          evidence_available: available,
          evidence_url: available ? `/clips/${clipName}` : null
        };
      });

      if (filters.zone_id) {
        events = events.filter(e => e.zone_id === filters.zone_id);
      }
      if (filters.conflict_type) {
        events = events.filter(e => e.conflict_type === filters.conflict_type);
      }
      if (filters.min_risk !== undefined && filters.min_risk !== null) {
        events = events.filter(e => e.risk_score >= filters.min_risk);
      }
      if (filters.min_severity !== undefined && filters.min_severity !== null) {
        events = events.filter(e => e.severity >= filters.min_severity);
      }
    }

    return events;
  }

  async getEventDetail(eventId) {
    if (this.simulationState === 'api_error') {
      throw new Error(`Failed to load event ${eventId}: 503 Service Unavailable`);
    }

    if (this.shouldUseLive()) {
      try {
        const res = await fetch(`${API_BASE_URL}/events/${eventId}`);
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn('Live API event detail failed:', err);
      }
    }

    // Demo Data
    const ev = dummyRiskData.conflict_events.find(e => e.event_id === eventId);
    if (!ev) throw new Error(`Event ${eventId} not found`);

    const clipName = ev.evidence_clip_path ? ev.evidence_clip_path.split('/').pop() : null;
    const available = ['EVT_001', 'EVT_002', 'EVT_003', 'EVT_004', 'EVT_005', 'EVT_006'].includes(ev.event_id);

    return {
      ...ev,
      evidence_available: available,
      evidence_url: available ? `/clips/${clipName}` : null,
      related_findings: dummyAgentData.key_findings.filter(f => f.evidence_event_ids.includes(eventId)),
      related_recommendations: dummyAgentData.recommendations.filter(r => r.evidence_event_ids.includes(eventId)),
      related_patterns: dummyAgentData.patterns.filter(p => p.evidence_event_ids.includes(eventId))
    };
  }

  async getAgentReport() {
    if (this.simulationState === 'api_error' || this.simulationState === 'no_report') {
      throw new Error('Agent37 Report unavailable');
    }

    if (this.shouldUseLive()) {
      try {
        const res = await fetch(`${API_BASE_URL}/agent-report`);
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn('Live API agent report failed, using fallback:', err);
      }
    }

    return dummyAgentData;
  }

  async getTraceability() {
    if (this.shouldUseLive()) {
      try {
        const res = await fetch(`${API_BASE_URL}/traceability`);
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn('Live API traceability failed, using local graph:', err);
      }
    }

    const events = {};
    for (const e of dummyRiskData.conflict_events) events[e.event_id] = e;

    const hotspots = {};
    for (const h of dummyRiskData.hotspots) hotspots[h.zone_id] = h;

    const findings = {};
    for (const f of dummyAgentData.key_findings) findings[f.finding_id] = f;

    const recommendations = {};
    for (const r of dummyAgentData.recommendations) recommendations[r.recommendation_id] = r;

    const chains = [];
    for (const rec of dummyAgentData.recommendations) {
      for (const eid of rec.evidence_event_ids) {
        const ev = events[eid] || {};
        const zid = ev.zone_id;
        const hotspot = hotspots[zid] || {};

        chains.push({
          recommendation_id: rec.recommendation_id,
          recommendation_action: rec.action,
          recommendation_priority: rec.priority,
          recommendation_category: rec.category,
          finding_ids: rec.finding_ids || [],
          event_id: eid,
          zone_id: zid,
          hotspot_priority: zid === 'R1C2' ? 1 : zid === 'R2C3' ? 2 : 3,
          conflict_type: ev.conflict_type,
          risk_score: ev.risk_score,
          severity: ev.severity,
          ttc_min: ev.ttc_min,
          pet_min: ev.pet_min,
          reasons: ev.reasons || []
        });
      }
    }

    return {
      analysis_id: dummyRiskData.analysis_id,
      chains,
      nodes: {
        hotspots,
        events,
        findings,
        recommendations,
        patterns: Object.fromEntries(dummyAgentData.patterns.map(p => [p.pattern_id, p]))
      }
    };
  }
}

export const api = new ApiService();
