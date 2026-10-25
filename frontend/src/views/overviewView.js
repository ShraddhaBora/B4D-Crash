// frontend/src/views/overviewView.js
// Page 1 — Command Center / Overview for B4DCrash

import { api } from '../api.js';
import { store } from '../state/store.js';
import { renderErrorState, renderLoadingSkeleton } from '../components/emptyState.js';

export async function renderOverviewView() {
  const container = document.createElement('div');
  container.className = 'view-container overview-view';

  container.innerHTML = renderLoadingSkeleton({ title: 'Loading Road-Safety Telemetry & Agent37 Intelligence...' });

  try {
    const [analysis, hotspots, events, agentReport] = await Promise.all([
      api.getAnalysis(),
      api.getHotspots(),
      api.getEvents(),
      api.getAgentReport()
    ]);

    const assessment = analysis.overall_assessment || {};
    const topHotspot = hotspots[0] || null;
    const topRecommendation = agentReport.recommendations?.find(r => r.priority === 'URGENT') || agentReport.recommendations?.[0];
    const camera = analysis.camera || {};
    const trafficStats = analysis.traffic_stats || {};

    const html = `
      <!-- Page Header -->
      <div class="view-header">
        <div class="view-header-main">
          <div class="view-eyebrow">
            <span class="badge-tag live-beacon"><span class="beacon-dot"></span> SESSION ACTIVE</span>
            <span class="meta-tag">ID: <strong>${analysis.analysis_id || 'ANALYSIS_DUMMY_SESSION_01'}</strong></span>
            <span class="meta-tag">Camera: <strong>${camera.camera_id || 'CAM_INTERSECTION_04'}</strong></span>
          </div>
          <h1 class="view-title">B4DCrash Command Center</h1>
          <p class="view-subtitle">
            Pre-collision conflict detection, spatial clustering, and Agent37 safety intelligence.
          </p>
        </div>

        <div class="view-header-actions">
          <button id="btn-quick-trace" class="btn btn-primary glow-btn">
            Trace Evidence Lineage →
          </button>
        </div>
      </div>

      <!-- Core KPI Matrix -->
      <div class="kpi-grid">
        <!-- Overall Risk Card -->
        <div class="kpi-card glass-panel kpi-critical-glow">
          <div class="kpi-header">
            <span class="kpi-label">OVERALL RISK LEVEL</span>
            <span class="badge badge-critical">${assessment.risk_level || 'CRITICAL'}</span>
          </div>
          <div class="kpi-value-row">
            <div class="kpi-hero-number text-critical">${(assessment.ml_peak_risk_score ?? analysis.peak_risk_score ?? 0.85).toFixed(2)}</div>
            <div class="kpi-meta-sub">
              <div class="font-medium text-white">Peak ML Risk</div>
              <div class="text-muted">Threshold: ≥ 0.70 Critical</div>
            </div>
          </div>
          <div class="kpi-footer">
            <span class="text-cyan font-mono">Agent37 Confidence: ${Math.round((assessment.agent_confidence ?? 0.85) * 100)}%</span>
            <span class="kpi-hint">Audited Output</span>
          </div>
        </div>

        <!-- Detected Conflicts -->
        <div class="kpi-card glass-panel clickable-card" id="kpi-card-events">
          <div class="kpi-header">
            <span class="kpi-label">CONFLICT INTERACTIONS</span>
            <span class="badge badge-warning">${analysis.high_severity_count || 3} HIGH SEVERITY</span>
          </div>
          <div class="kpi-value-row">
            <div class="kpi-hero-number text-warning">${analysis.total_event_count || events.length}</div>
            <div class="kpi-meta-sub">
              <div class="font-medium text-white">Detected Near-Misses</div>
              <div class="text-muted">60.0s window</div>
            </div>
          </div>
          <div class="kpi-footer">
            <span class="link-text">Inspect all events →</span>
          </div>
        </div>

        <!-- Hotspots Identified -->
        <div class="kpi-card glass-panel clickable-card" id="kpi-card-hotspots">
          <div class="kpi-header">
            <span class="kpi-label">SPATIAL HOTSPOTS</span>
            <span class="badge badge-orange">${hotspots.length} ACTIVE ZONES</span>
          </div>
          <div class="kpi-value-row">
            <div class="kpi-hero-number text-orange">${topHotspot ? topHotspot.zone_id : 'R1C2'}</div>
            <div class="kpi-meta-sub">
              <div class="font-medium text-white">Highest Risk Zone</div>
              <div class="text-muted">Avg Risk: ${topHotspot ? topHotspot.avg_risk_score.toFixed(3) : '0.727'}</div>
            </div>
          </div>
          <div class="kpi-footer">
            <span class="link-text">View spatial map →</span>
          </div>
        </div>

        <!-- Mode Breakdown -->
        <div class="kpi-card glass-panel">
          <div class="kpi-header">
            <span class="kpi-label">CONFLICT DISTRIBUTION</span>
            <span class="badge badge-muted">ROAD USER</span>
          </div>
          <div class="distribution-bars">
            <div class="dist-row">
              <span class="dist-name">Pedestrian-Vehicle</span>
              <span class="dist-val font-mono">${analysis.conflict_type_counts?.VEH_PED || 3}</span>
              <div class="progress-bar-bg"><div class="progress-bar-fill fill-critical" style="width: 37.5%"></div></div>
            </div>
            <div class="dist-row">
              <span class="dist-name">Vehicle-Vehicle</span>
              <span class="dist-val font-mono">${analysis.conflict_type_counts?.VEH_VEH || 3}</span>
              <div class="progress-bar-bg"><div class="progress-bar-fill fill-warning" style="width: 37.5%"></div></div>
            </div>
            <div class="dist-row">
              <span class="dist-name">Motorcycle-Vehicle</span>
              <span class="dist-val font-mono">${analysis.conflict_type_counts?.VEH_MOTO || 2}</span>
              <div class="progress-bar-bg"><div class="progress-bar-fill fill-orange" style="width: 25%"></div></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Executive Assessment Banner -->
      <div class="agent-narrative-banner glass-panel">
        <div class="agent-avatar-col">
          <div class="agent-monogram-badge">A37</div>
          <div class="agent-version-tag">v${agentReport.agent_version || '37.0'}</div>
        </div>
        <div class="agent-narrative-content">
          <div class="agent-narrative-header">
            <h4>Agent37 Executive Safety Assessment</h4>
            <span class="agent-timestamp">${new Date(agentReport.generated_at || analysis.analysis_ts).toLocaleString()}</span>
          </div>
          <p class="agent-narrative-text">
            "${assessment.summary || 'Agent37 sandbox analysis of 8 conflict events (ML peak risk: 0.85). High-severity events and vulnerable road user interactions warrant prioritization for engineering assessment.'}"
          </p>
          <div class="agent-limitations-notice">
            <span>Spatial metrics derived from image-space coordinates. Traffic volume normalization recommended before civil intervention.</span>
          </div>
        </div>
      </div>

      <!-- Two-Column Spotlight Grid -->
      <div class="grid-2col">
        <!-- Priority Hotspot Spotlight -->
        <div class="card glass-panel spotlight-card">
          <div class="card-header-flex">
            <div>
              <div class="card-eyebrow text-orange">PRIORITY #1 HOTSPOT</div>
              <h3 class="card-title">Zone ${topHotspot ? topHotspot.zone_id : 'R1C2'}</h3>
            </div>
            <button id="btn-spotlight-hotspot" class="btn btn-outline-orange btn-sm">
              Inspect Hotspot →
            </button>
          </div>

          <div class="spotlight-body">
            <p class="spotlight-reason">
              ${topHotspot?.reason || "Zone 'R1C2' recorded 3 events with ML avg risk 0.727. Dominant conflict: VEH_PED. Minimum observed TTC: 1.15s."}
            </p>

            <div class="metrics-pill-grid">
              <div class="metric-pill">
                <span class="pill-label">Total Events</span>
                <span class="pill-value font-mono">${topHotspot?.event_count || 3}</span>
              </div>
              <div class="metric-pill">
                <span class="pill-label">High-Risk</span>
                <span class="pill-value font-mono text-critical">${topHotspot?.high_risk_count || 2}</span>
              </div>
              <div class="metric-pill">
                <span class="pill-label">Min TTC</span>
                <span class="pill-value font-mono text-warning">${topHotspot?.min_ttc ? topHotspot.min_ttc.toFixed(2) + 's' : '1.15s'}</span>
              </div>
              <div class="metric-pill">
                <span class="pill-label">Min PET</span>
                <span class="pill-value font-mono text-cyan">${topHotspot?.min_pet ? topHotspot.min_pet.toFixed(2) + 's' : '1.30s'}</span>
              </div>
            </div>

            <div class="events-contributing">
              <span class="sub-label">Contributing Evidence Events:</span>
              <div class="event-chips-row">
                ${(topHotspot?.event_ids || ['EVT_001', 'EVT_002', 'EVT_003']).map(eid => `
                  <button class="event-chip btn-event-jump" data-event-id="${eid}">
                    ${eid}
                  </button>
                `).join('')}
              </div>
            </div>
          </div>
        </div>

        <!-- Priority Recommendation Spotlight -->
        <div class="card glass-panel spotlight-card">
          <div class="card-header-flex">
            <div>
              <div class="card-eyebrow text-critical">URGENT INTERVENTION</div>
              <h3 class="card-title">${topRecommendation?.category ? topRecommendation.category.replace(/_/g, ' ') : 'PEDESTRIAN CROSSING'}</h3>
            </div>
            <button id="btn-spotlight-agent" class="btn btn-outline-cyan btn-sm">
              Intelligence Report →
            </button>
          </div>

          <div class="spotlight-body">
            <div class="rec-disclaimer-tag">PROPOSED ACTION FOR TRAFFIC-ENGINEERING ASSESSMENT</div>
            <p class="rec-action-text font-medium">
              "${topRecommendation?.action || 'Prioritize this location for traffic-engineering assessment of pedestrian crossing placement, visibility, and signal phasing.'}"
            </p>
            <p class="rec-rationale-text text-muted">
              <strong>Rationale:</strong> ${topRecommendation?.rationale || '3 pedestrian conflicts detected. Vulnerable road users require crossing protection.'}
            </p>

            <div class="events-contributing">
              <span class="sub-label">Grounding Evidence (${topRecommendation?.evidence_event_ids?.length || 3} events):</span>
              <div class="event-chips-row">
                ${(topRecommendation?.evidence_event_ids || ['EVT_001', 'EVT_002', 'EVT_008']).map(eid => `
                  <button class="event-chip btn-event-jump" data-event-id="${eid}">
                    ${eid}
                  </button>
                `).join('')}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Agent37 Key Findings Section -->
      <div class="card glass-panel">
        <div class="card-header-flex">
          <div>
            <div class="card-eyebrow text-cyan">STRUCTURED INTELLIGENCE</div>
            <h3 class="card-title">Key Safety Findings (${agentReport.key_findings?.length || 4})</h3>
          </div>
          <button id="btn-view-all-findings" class="btn btn-ghost btn-sm">
            View All Findings & Patterns →
          </button>
        </div>

        <div class="findings-grid">
          ${(agentReport.key_findings || []).slice(0, 3).map(f => `
            <div class="finding-card glass-panel-nested">
              <div class="finding-header">
                <span class="finding-id">${f.finding_id}</span>
                <span class="badge ${f.severity === 'HIGH' ? 'badge-critical' : 'badge-warning'}">${f.severity} SEVERITY</span>
              </div>
              <p class="finding-text">${f.finding}</p>
              <p class="finding-reasoning text-muted">${f.reasoning}</p>
              <div class="finding-footer">
                <span class="text-xs text-muted">Evidence:</span>
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

      <!-- Telemetry Footer Bar -->
      <div class="telemetry-bar glass-panel">
        <div class="telemetry-item">
          <span class="tel-label">CAMERA LOCATION</span>
          <span class="tel-val">${camera.location_name || 'Market St & 5th Ave'}</span>
        </div>
        <div class="telemetry-item">
          <span class="tel-label">SENSOR SPECS</span>
          <span class="tel-val">${camera.mount_height_m || 5.5}m mount • ${camera.frame_width_px || 1920}x${camera.frame_height_px || 1080}px</span>
        </div>
        <div class="telemetry-item">
          <span class="tel-label">DETECTIONS</span>
          <span class="tel-val font-mono">${trafficStats.total_detections || 1420}</span>
        </div>
        <div class="telemetry-item">
          <span class="tel-label">UNIQUE TRACKS</span>
          <span class="tel-val font-mono">${trafficStats.unique_track_count || 38}</span>
        </div>
        <div class="telemetry-item">
          <span class="tel-label">CALIBRATION</span>
          <span class="tel-val ${camera.calibration_active ? 'text-success' : 'text-warning'}">
            ${camera.calibration_active ? 'ACTIVE' : 'INACTIVE (IMAGE-SPACE)'}
          </span>
        </div>
      </div>
    `;

    container.innerHTML = html;

    // Attach event listeners
    container.querySelector('#btn-quick-trace')?.addEventListener('click', () => {
      store.navigateTo('traceability');
    });

    container.querySelector('#kpi-card-events')?.addEventListener('click', () => {
      store.navigateTo('events');
    });

    container.querySelector('#kpi-card-hotspots')?.addEventListener('click', () => {
      store.navigateTo('hotspots');
    });

    container.querySelector('#btn-spotlight-hotspot')?.addEventListener('click', () => {
      store.selectHotspot(topHotspot?.zone_id || 'R1C2');
    });

    container.querySelector('#btn-spotlight-agent')?.addEventListener('click', () => {
      store.navigateTo('agent37');
    });

    container.querySelector('#btn-view-all-findings')?.addEventListener('click', () => {
      store.navigateTo('agent37');
    });

    // Jump to specific event details from evidence chips
    container.querySelectorAll('.btn-event-jump').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const eid = btn.getAttribute('data-event-id');
        if (eid) store.selectEvent(eid);
      });
    });

  } catch (err) {
    console.error('Error rendering Overview:', err);
    container.innerHTML = renderErrorState({
      title: 'Failed to Load Command Center Telemetry',
      message: err.message,
      retryCallback: () => store.notify()
    });
  }

  return container;
}
