// frontend/src/views/agentIntelligenceView.js
// Page 5 — Agent37 Safety Intelligence Report for B4DCrash

import { api } from '../api.js';
import { store } from '../state/store.js';
import { renderEmptyState, renderErrorState, renderLoadingSkeleton } from '../components/emptyState.js';

export async function renderAgentIntelligenceView() {
  const container = document.createElement('div');
  container.className = 'view-container agent-intelligence-view';

  container.innerHTML = renderLoadingSkeleton({ title: 'Synthesizing Agent37 Structured Safety Intelligence...' });

  try {
    const report = await api.getAgentReport();
    const assessment = report.overall_assessment || {};

    const html = `
      <!-- Header -->
      <div class="view-header">
        <div class="view-header-main">
          <div class="view-eyebrow">
            <span class="badge-tag live-beacon"><span class="beacon-dot"></span> DECISION SUPPORT LAYER</span>
            <span class="meta-tag">Agent Version: <strong>${report.agent_version || '37.0'}</strong></span>
            <span class="meta-tag">Runtime: <strong>${report.model_used || 'Agent37 Cloud Sandbox (gVisor)'}</strong></span>
          </div>
          <h1 class="view-title">Agent37 Safety Intelligence</h1>
          <p class="view-subtitle">
            Evidence-grounded synthesis connecting physical conflict metrics to traffic engineering recommendations.
          </p>
        </div>

        <div class="view-header-actions">
          <button id="btn-agent-trace" class="btn btn-primary glow-btn">
            Traceability Graph →
          </button>
        </div>
      </div>

      <!-- Section 1: Top-Level Assessment Hero -->
      <div class="card glass-panel agent-hero-card kpi-critical-glow">
        <div class="ahc-top-bar">
          <div class="ahc-title-group">
            <div class="agent-brand-row">
              <span class="agent-monogram-badge">A37</span>
              <span class="agent-name">AGENT37 REASONING ENGINE</span>
              <span class="badge badge-muted badge-xs">CONTAINER ISOLATED</span>
            </div>
            <h2 class="ahc-main-title">Executive Safety Synthesis</h2>
          </div>

          <div class="ahc-metrics-badges">
            <div class="assessment-badge-box">
              <span class="abb-label">OVERALL RISK</span>
              <span class="badge badge-critical text-base">${assessment.risk_level || 'CRITICAL'}</span>
            </div>
            <div class="assessment-badge-box">
              <span class="abb-label">PEAK ML RISK</span>
              <span class="font-mono text-critical font-bold text-xl">${(assessment.ml_peak_risk_score ?? 0.85).toFixed(2)}</span>
            </div>
            <div class="assessment-badge-box">
              <span class="abb-label">CONFIDENCE</span>
              <span class="font-mono text-cyan font-bold text-xl">${Math.round((assessment.agent_confidence ?? 0.85) * 100)}%</span>
            </div>
          </div>
        </div>

        <div class="ahc-narrative">
          <p class="ahc-summary-text">
            "${assessment.summary || 'Agent37 sandbox analysis of 8 conflict events (ML peak risk: 0.85). High-severity events and vulnerable road user interactions warrant prioritization for engineering assessment.'}"
          </p>
        </div>
      </div>

      <!-- Section 2: Key Findings Grid -->
      <div class="section-container">
        <div class="section-header-row">
          <div>
            <div class="card-eyebrow text-cyan">EVIDENCE-BACKED OBSERVATIONS</div>
            <h3 class="section-title">Key Safety Findings (${report.key_findings?.length || 0})</h3>
          </div>
        </div>

        <div class="findings-full-grid">
          ${(report.key_findings || []).map(f => `
            <div class="card glass-panel finding-full-card">
              <div class="ffc-header">
                <span class="font-mono text-cyan font-bold text-base">${f.finding_id}</span>
                <span class="badge ${f.severity === 'HIGH' ? 'badge-critical' : 'badge-warning'} badge-sm">
                  ${f.severity} SEVERITY
                </span>
              </div>
              <h4 class="ffc-title">${f.finding}</h4>
              <p class="ffc-reasoning text-muted">${f.reasoning}</p>
              
              <div class="ffc-evidence">
                <span class="text-xs text-muted font-semibold">GROUNDING EVIDENCE (${f.evidence_event_ids?.length || 0} EVENTS):</span>
                <div class="event-chips-row">
                  ${(f.evidence_event_ids || []).map(eid => `
                    <button class="event-chip btn-event-jump" data-event-id="${eid}">
                      ${eid}
                    </button>
                  `).join('')}
                </div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>

      <!-- Section 3: Recurring Conflict Patterns -->
      <div class="section-container">
        <div class="section-header-row">
          <div>
            <div class="card-eyebrow text-orange">TEMPOROSPATIAL CLUSTERING</div>
            <h3 class="section-title">Recurring Conflict Patterns (${report.patterns?.length || 0})</h3>
          </div>
        </div>

        <div class="patterns-grid">
          ${(report.patterns || []).map(p => `
            <div class="card glass-panel pattern-card">
              <div class="pattern-header">
                <div class="ph-left">
                  <span class="font-mono text-orange text-sm font-semibold">${p.pattern_id}</span>
                  <span class="pattern-type-tag">${p.pattern_type}</span>
                </div>
                <div class="ph-right">
                  <span class="badge ${p.severity === 'HIGH' ? 'badge-critical' : 'badge-warning'} badge-xs">
                    FREQUENCY: ${p.frequency} EVENTS
                  </span>
                </div>
              </div>
              <p class="pattern-description">${p.description}</p>
              
              <div class="pattern-evidence">
                <span class="text-xs text-muted font-mono">SUPPORTING EVENTS:</span>
                <div class="event-chips-row">
                  ${(p.evidence_event_ids || []).map(eid => `
                    <button class="event-chip btn-event-jump" data-event-id="${eid}">
                      ${eid}
                    </button>
                  `).join('')}
                </div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>

      <!-- Section 4: Prioritized Hotspots Ranking -->
      <div class="section-container">
        <div class="section-header-row">
          <div>
            <div class="card-eyebrow text-cyan">RISK HIERARCHY</div>
            <h3 class="section-title">Prioritized Spatial Hotspots</h3>
          </div>
        </div>

        <div class="card glass-panel p-0 overflow-hidden">
          <div class="table-responsive">
            <table class="data-table">
              <thead>
                <tr>
                  <th>PRIORITY</th>
                  <th>ZONE ID</th>
                  <th>RISK LEVEL</th>
                  <th>ML AVG RISK</th>
                  <th>PRIORITIZATION RATIONALE</th>
                  <th>SUPPORTING EVIDENCE</th>
                  <th>ACTION</th>
                </tr>
              </thead>
              <tbody>
                ${(report.prioritized_hotspots || []).map(h => `
                  <tr>
                    <td class="font-bold text-center">
                      <span class="badge ${h.priority === 1 ? 'badge-critical' : h.priority === 2 ? 'badge-warning' : 'badge-success'}">
                        #${h.priority}
                      </span>
                    </td>
                    <td class="font-mono font-bold text-cyan">${h.zone_id}</td>
                    <td>
                      <span class="badge ${h.risk_level === 'HIGH' ? 'badge-critical' : h.risk_level === 'MEDIUM' ? 'badge-warning' : 'badge-success'} badge-xs">
                        ${h.risk_level}
                      </span>
                    </td>
                    <td class="font-mono font-bold">${h.ml_avg_risk_score.toFixed(3)}</td>
                    <td class="text-sm">${h.reason}</td>
                    <td>
                      <div class="event-chips-row">
                        ${(h.evidence_event_ids || []).map(eid => `
                          <button class="event-chip btn-event-jump" data-event-id="${eid}">
                            ${eid}
                          </button>
                        `).join('')}
                      </div>
                    </td>
                    <td>
                      <button class="btn btn-outline-cyan btn-xs btn-jump-zone" data-zone-id="${h.zone_id}">
                        View Map →
                      </button>
                    </td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Section 5: Engineering Recommendations -->
      <div class="section-container">
        <div class="section-header-row">
          <div>
            <div class="card-eyebrow text-critical">ACTIONABLE INTERVENTIONS</div>
            <h3 class="section-title">Traffic-Engineering Recommendations</h3>
          </div>
        </div>

        <!-- Prominent Engineering Advisory Banner -->
        <div class="engineering-disclaimer-banner glass-panel">
          <div class="edb-text">
            <strong>CIVIL & TRAFFIC ENGINEERING ASSESSMENT NOTICE:</strong>
            These recommendations represent proposed diagnostic actions generated from video conflict metrics. They are decision-support suggestions for qualified traffic engineers, not automatically validated or binding engineering mandates.
          </div>
        </div>

        <div class="recommendations-list">
          ${(report.recommendations || []).map(rec => `
            <div class="card glass-panel recommendation-full-card">
              <div class="rec-top-row">
                <div class="rtr-left">
                  <span class="font-mono text-orange font-bold text-base">${rec.recommendation_id}</span>
                  <span class="badge ${rec.priority === 'URGENT' ? 'badge-critical' : rec.priority === 'HIGH' ? 'badge-orange' : 'badge-warning'}">
                    ${rec.priority} PRIORITY
                  </span>
                  <span class="rec-category-badge">${rec.category.replace(/_/g, ' ')}</span>
                </div>
                <div class="rtr-right">
                  <span class="rec-status-tag">PROPOSED ASSESSMENT</span>
                </div>
              </div>

              <div class="rec-action-body">
                <h4 class="rec-action-headline font-semibold text-white">
                  "${rec.action}"
                </h4>
                <p class="rec-rationale text-muted">
                  <strong>Rationale:</strong> ${rec.rationale}
                </p>
              </div>

              <div class="rec-footer-evidence">
                <div class="rfe-col">
                  <span class="text-xs text-muted font-semibold">EVIDENCE GROUNDING (${rec.evidence_event_ids?.length || 0} EVENTS):</span>
                  <div class="event-chips-row">
                    ${(rec.evidence_event_ids || []).map(eid => `
                      <button class="event-chip btn-event-jump" data-event-id="${eid}">
                        ${eid}
                      </button>
                    `).join('')}
                  </div>
                </div>

                ${rec.finding_ids?.length ? `
                  <div class="rfe-col">
                    <span class="text-xs text-muted font-semibold">SUPPORTING FINDINGS:</span>
                    <div class="event-chips-row">
                      ${rec.finding_ids.map(fid => `
                        <span class="badge badge-muted badge-xs">${fid}</span>
                      `).join('')}
                    </div>
                  </div>
                ` : ''}
              </div>
            </div>
          `).join('')}
        </div>
      </div>

      <!-- Section 6: Limitations & System Uncertainty -->
      <div class="section-container mb-4">
        <div class="section-header-row">
          <div>
            <div class="card-eyebrow text-muted">SYSTEM BOUNDARIES</div>
            <h3 class="section-title">Limitations & Uncertainty Disclosure</h3>
          </div>
        </div>

        <div class="limitations-grid">
          ${(report.limitations || []).map((lim, idx) => `
            <div class="card glass-panel limitation-card">
              <div class="lim-header">
                <span class="lim-number font-mono text-muted">#0${idx + 1}</span>
                <span class="badge badge-muted badge-xs">UNCERTAINTY DISCLOSURE</span>
              </div>
              <p class="lim-text">${lim}</p>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    container.innerHTML = html;

    // Attach click listeners
    container.querySelector('#btn-agent-trace')?.addEventListener('click', () => {
      store.navigateTo('traceability');
    });

    container.querySelectorAll('.btn-event-jump').forEach(btn => {
      btn.addEventListener('click', () => {
        const eid = btn.getAttribute('data-event-id');
        if (eid) store.selectEvent(eid);
      });
    });

    container.querySelectorAll('.btn-jump-zone').forEach(btn => {
      btn.addEventListener('click', () => {
        const zid = btn.getAttribute('data-zone-id');
        if (zid) store.selectHotspot(zid);
      });
    });

  } catch (err) {
    console.error('Error rendering Agent37 Intelligence View:', err);
    container.innerHTML = renderErrorState({
      title: 'Failed to Retrieve Agent37 Safety Intelligence Report',
      message: err.message,
      retryCallback: () => store.notify()
    });
  }

  return container;
}
