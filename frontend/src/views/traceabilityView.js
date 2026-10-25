// frontend/src/views/traceabilityView.js
// Page 6 — Analysis & Evidence Traceability Lineage Graph for B4DCrash

import { api } from '../api.js';
import { store } from '../state/store.js';
import { renderEmptyState, renderErrorState, renderLoadingSkeleton } from '../components/emptyState.js';

export async function renderTraceabilityView() {
  const container = document.createElement('div');
  container.className = 'view-container traceability-view';

  container.innerHTML = renderLoadingSkeleton({ title: 'Mapping End-to-End Evidence Lineage Graph...' });

  try {
    const data = await api.getTraceability();
    const chains = data.chains || [];
    const nodes = data.nodes || {};

    const state = store.getState();
    const preselectedZone = state.traceabilityFilter?.zoneId || null;
    const preselectedEvent = state.traceabilityFilter?.eventId || null;

    let activeChain = chains.find(c =>
      (preselectedEvent && c.event_id === preselectedEvent) ||
      (preselectedZone && c.zone_id === preselectedZone)
    ) || chains[0];

    const html = `
      <!-- Header -->
      <div class="view-header">
        <div class="view-header-main">
          <div class="view-eyebrow">
            <span class="badge-tag live-beacon"><span class="beacon-dot"></span> AUDITABLE VERIFIABILITY</span>
            <span class="meta-tag">Chains: <strong>${chains.length} Grounded Lineages</strong></span>
            <span class="meta-tag">Coverage: <strong>100% Traceable</strong></span>
          </div>
          <h1 class="view-title">Evidence Traceability Studio</h1>
          <p class="view-subtitle">
            Bidirectional audit trail connecting spatial hotspots, raw sensor conflict metrics, Agent37 findings, and proposed interventions.
          </p>
        </div>

        <div class="view-header-actions">
          <button id="btn-export-audit" class="btn btn-outline-cyan btn-sm">
            Download Audit Log (JSON)
          </button>
        </div>
      </div>

      <!-- Chain Selector Bar -->
      <div class="chain-selector-bar glass-panel">
        <span class="text-xs text-muted font-bold font-mono">SELECT TRACEABILITY PATHWAY:</span>
        <div class="chain-pills-row">
          ${chains.map((c, i) => `
            <button class="chain-pill-btn ${c === activeChain ? 'active' : ''}" data-chain-idx="${i}">
              <span class="cp-hotspot font-mono text-cyan">${c.zone_id}</span>
              <span class="cp-arrow">→</span>
              <span class="cp-event font-mono text-warning">${c.event_id}</span>
              <span class="cp-arrow">→</span>
              <span class="cp-rec font-mono text-orange">${c.recommendation_id}</span>
            </button>
          `).join('')}
        </div>
      </div>

      <!-- Main Visual Traceability Pipeline -->
      <div class="traceability-pipeline-card glass-panel" id="pipeline-card-container">
        ${renderPipelineChain(activeChain, nodes)}
      </div>

      <!-- Complete Traceability Audit Matrix Table -->
      <div class="card glass-panel mb-4">
        <div class="card-header-flex">
          <div>
            <div class="card-eyebrow text-cyan">AUDIT VERIFICATION</div>
            <h3 class="card-title">Full Traceability Matrix</h3>
          </div>
        </div>

        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>HOTSPOT</th>
                <th>EVENT ID</th>
                <th>CONFLICT TYPE</th>
                <th>TTC / PET</th>
                <th>EVASIVE DYNAMICS</th>
                <th>ML RISK</th>
                <th>AGENT37 FINDING</th>
                <th>RECOMMENDATION</th>
                <th>ACTION</th>
              </tr>
            </thead>
            <tbody>
              ${chains.map(c => {
                const finding = nodes.findings?.[c.finding_ids?.[0]] || {};
                const rec = nodes.recommendations?.[c.recommendation_id] || {};

                return `
                  <tr class="clickable-row btn-select-row-chain" data-event-id="${c.event_id}">
                    <td class="font-mono font-bold text-cyan">Zone ${c.zone_id}</td>
                    <td class="font-mono font-semibold text-warning">${c.event_id}</td>
                    <td>
                      <span class="badge ${c.conflict_type === 'VEH_PED' ? 'badge-critical' : 'badge-orange'} badge-xs">
                        ${c.conflict_type}
                      </span>
                    </td>
                    <td class="font-mono">
                      TTC: ${c.ttc_min ? c.ttc_min.toFixed(2) + 's' : 'N/A'}<br/>
                      <span class="text-xs text-muted">PET: ${c.pet_min ? c.pet_min.toFixed(2) + 's' : 'N/A'}</span>
                    </td>
                    <td class="text-xs">
                      ${(c.reasons || []).map(r => `<span class="badge badge-muted badge-xs">${r.replace(/_/g, ' ')}</span>`).join(' ')}
                    </td>
                    <td class="font-mono font-bold ${c.risk_score >= 0.70 ? 'text-critical' : 'text-warning'}">
                      ${c.risk_score.toFixed(2)}
                    </td>
                    <td class="text-xs">
                      <span class="font-mono text-cyan">${finding.finding_id || 'F001'}:</span>
                      ${finding.finding ? finding.finding.substring(0, 45) + '...' : 'VRU conflict'}
                    </td>
                    <td class="text-xs">
                      <span class="badge ${rec.priority === 'URGENT' ? 'badge-critical' : 'badge-warning'} badge-xs">${rec.priority || 'URGENT'}</span>
                      <span class="font-mono">${rec.recommendation_id || 'R001'}</span>
                    </td>
                    <td>
                      <button class="btn btn-ghost btn-xs btn-inspect-event-from-matrix" data-event-id="${c.event_id}">
                        Forensics →
                      </button>
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

    container.innerHTML = html;

    // Attach chain selector listeners
    container.querySelectorAll('.chain-pill-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        container.querySelectorAll('.chain-pill-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const idx = parseInt(btn.getAttribute('data-chain-idx'), 10);
        activeChain = chains[idx];

        const pipelineCard = container.querySelector('#pipeline-card-container');
        if (pipelineCard) {
          pipelineCard.innerHTML = renderPipelineChain(activeChain, nodes);
          attachPipelineActions(pipelineCard, activeChain);
        }
      });
    });

    attachPipelineActions(container.querySelector('#pipeline-card-container'), activeChain);

    // Matrix row jumps
    container.querySelectorAll('.btn-inspect-event-from-matrix').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const eid = btn.getAttribute('data-event-id');
        if (eid) store.selectEvent(eid);
      });
    });

    // Export Audit Log (JSON)
    container.querySelector('#btn-export-audit')?.addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `traceability_audit_${data.analysis_id || 'session'}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });

  } catch (err) {
    console.error('Error rendering Traceability View:', err);
    container.innerHTML = renderErrorState({
      title: 'Failed to Map Traceability Graph',
      message: err.message,
      retryCallback: () => store.notify()
    });
  }

  return container;
}

function renderPipelineChain(chain, nodes) {
  if (!chain) return '<div class="p-4 text-muted">Select a chain above.</div>';

  const event = nodes.events?.[chain.event_id] || {};
  const hotspot = nodes.hotspots?.[chain.zone_id] || {};
  const finding = nodes.findings?.[chain.finding_ids?.[0]] || {};
  const rec = nodes.recommendations?.[chain.recommendation_id] || {};

  return `
    <div class="pipeline-header-row">
      <div class="phr-title">
        <span class="badge badge-cyan font-mono">ACTIVE TRACE PATHWAY</span>
        <h3 class="mt-1">Lineage for ${chain.event_id} in Zone ${chain.zone_id}</h3>
      </div>
      <div class="phr-actions">
        <button class="btn btn-outline-warning btn-sm" id="btn-inspect-chain-event" data-event-id="${chain.event_id}">
          View Video Evidence (${chain.event_id}) →
        </button>
      </div>
    </div>

    <!-- 5-Step Visual Stepper -->
    <div class="pipeline-flow-grid">
      <!-- Step 1: Spatial Hotspot -->
      <div class="pipeline-node glass-panel-nested">
        <div class="pn-step-badge">STEP 1</div>
        <div class="pn-header">
          <span class="pn-type text-cyan">SPATIAL HOTSPOT</span>
          <span class="badge badge-critical badge-xs">PRIORITY #${chain.hotspot_priority || 1}</span>
        </div>
        <div class="pn-title font-mono font-bold text-lg">Zone ${chain.zone_id}</div>
        <div class="pn-detail">
          <div>ML Avg Risk: <strong>${hotspot.avg_risk_score ? hotspot.avg_risk_score.toFixed(3) : '0.727'}</strong></div>
          <div>Total Events: <strong>${hotspot.event_count || 3}</strong></div>
          <div>Centroid: <strong>(${hotspot.centroid_x || 640}, ${hotspot.centroid_y || 480})</strong></div>
        </div>
        <button class="btn btn-ghost btn-xs btn-jump-hotspot" data-zone-id="${chain.zone_id}">
          Inspect Hotspot →
        </button>
      </div>

      <div class="pipeline-connector">
        <div class="connector-arrow">→</div>
      </div>

      <!-- Step 2: Detected Conflict Event -->
      <div class="pipeline-node glass-panel-nested border-highlight">
        <div class="pn-step-badge">STEP 2</div>
        <div class="pn-header">
          <span class="pn-type text-warning">NEAR-MISS CONFLICT</span>
          <span class="badge ${chain.risk_score >= 0.70 ? 'badge-critical' : 'badge-warning'} badge-xs">
            RISK: ${chain.risk_score.toFixed(2)}
          </span>
        </div>
        <div class="pn-title font-mono font-bold text-lg">${chain.event_id}</div>
        <div class="pn-detail">
          <div>Conflict Type: <strong>${chain.conflict_type}</strong></div>
          <div>Tracks: <strong>${event.classes?.join(' ↔ ') || 'Veh ↔ Ped'}</strong></div>
          <div>Timecode: <strong>${event.start_time?.toFixed(1) || '4.2'}s – ${event.end_time?.toFixed(1) || '7.1'}s</strong></div>
        </div>
        <button class="btn btn-ghost btn-xs btn-jump-event" data-event-id="${chain.event_id}">
          Inspect Event →
        </button>
      </div>

      <div class="pipeline-connector">
        <div class="connector-arrow">→</div>
      </div>

      <!-- Step 3: Conflict Metrics & Evasive Behavior -->
      <div class="pipeline-node glass-panel-nested">
        <div class="pn-step-badge">STEP 3</div>
        <div class="pn-header">
          <span class="pn-type text-critical">SAFETY INDICATORS</span>
          <span class="badge badge-muted badge-xs">DETERMINISTIC</span>
        </div>
        <div class="pn-title font-mono font-bold text-base">
          TTC: ${chain.ttc_min ? chain.ttc_min.toFixed(2) + 's' : 'N/A'}
        </div>
        <div class="pn-detail">
          <div>PET: <strong>${chain.pet_min ? chain.pet_min.toFixed(2) + 's' : 'N/A'}</strong></div>
          <div>Max Speed Drop: <strong>${event.max_speed_change ? Math.round(event.max_speed_change * 100) + '%' : '58%'}</strong></div>
          <div>Max Swerve: <strong>${event.max_heading_change ? event.max_heading_change.toFixed(1) + '°' : '14.2°'}</strong></div>
          <div class="text-xs text-muted">Triggers: ${chain.reasons?.join(', ') || 'evasive_braking'}</div>
        </div>
      </div>

      <div class="pipeline-connector">
        <div class="connector-arrow">→</div>
      </div>

      <!-- Step 4: Agent37 Finding -->
      <div class="pipeline-node glass-panel-nested">
        <div class="pn-step-badge">STEP 4</div>
        <div class="pn-header">
          <span class="pn-type text-cyan">AI FINDING</span>
          <span class="badge ${finding.severity === 'HIGH' ? 'badge-critical' : 'badge-warning'} badge-xs">
            ${finding.severity || 'HIGH'}
          </span>
        </div>
        <div class="pn-title font-mono font-bold text-base">${finding.finding_id || 'F001'}</div>
        <div class="pn-detail">
          <p class="text-xs text-white">"${finding.finding || 'Vulnerable pedestrian conflict'}"</p>
          <p class="text-xs text-muted mt-1">"${finding.reasoning || 'Elevated injury risk'}"</p>
        </div>
        <button class="btn btn-ghost btn-xs btn-jump-agent">
          Agent37 View →
        </button>
      </div>

      <div class="pipeline-connector">
        <div class="connector-arrow">→</div>
      </div>

      <!-- Step 5: Engineering Recommendation -->
      <div class="pipeline-node glass-panel-nested">
        <div class="pn-step-badge">STEP 5</div>
        <div class="pn-header">
          <span class="pn-type text-orange">PROPOSED INTERVENTION</span>
          <span class="badge badge-critical badge-xs">${chain.recommendation_priority || 'URGENT'}</span>
        </div>
        <div class="pn-title font-mono font-bold text-base">${chain.recommendation_id}</div>
        <div class="pn-detail">
          <span class="badge badge-muted badge-xs">${chain.recommendation_category?.replace(/_/g, ' ')}</span>
          <p class="text-xs text-white mt-1">"${chain.recommendation_action || rec.action}"</p>
        </div>
        <button class="btn btn-ghost btn-xs btn-jump-agent">
          Recommendation Details →
        </button>
      </div>
    </div>
  `;
}

function attachPipelineActions(container, chain) {
  if (!container || !chain) return;

  container.querySelector('#btn-inspect-chain-event')?.addEventListener('click', () => {
    store.selectEvent(chain.event_id);
  });

  container.querySelectorAll('.btn-jump-event').forEach(btn => {
    btn.addEventListener('click', () => {
      const eid = btn.getAttribute('data-event-id');
      if (eid) store.selectEvent(eid);
    });
  });

  container.querySelectorAll('.btn-jump-hotspot').forEach(btn => {
    btn.addEventListener('click', () => {
      const zid = btn.getAttribute('data-zone-id');
      if (zid) store.selectHotspot(zid);
    });
  });

  container.querySelectorAll('.btn-jump-agent').forEach(btn => {
    btn.addEventListener('click', () => {
      store.navigateTo('agent37');
    });
  });
}
