// frontend/src/views/eventDetailsView.js
// Page 4 — Detailed Event Investigation Screen for B4DCrash

import { api } from '../api.js';
import { store } from '../state/store.js';
import { renderEmptyState, renderErrorState, renderLoadingSkeleton } from '../components/emptyState.js';

export async function renderEventDetailsView() {
  const container = document.createElement('div');
  container.className = 'view-container event-details-view';

  container.innerHTML = renderLoadingSkeleton({ title: 'Loading Forensics for Selected Conflict Event...' });

  const state = store.getState();
  const eventId = state.selectedEventId || 'EVT_001';

  try {
    const [event, allEvents, analysis] = await Promise.all([
      api.getEventDetail(eventId),
      api.getEvents(),
      api.getAnalysis()
    ]);

    const currentIndex = allEvents.findIndex(e => e.event_id === eventId);
    const prevEvent = currentIndex > 0 ? allEvents[currentIndex - 1] : null;
    const nextEvent = currentIndex < allEvents.length - 1 ? allEvents[currentIndex + 1] : null;

    const riskBadgeClass = event.risk_score >= 0.70 ? 'badge-critical' : event.risk_score >= 0.40 ? 'badge-warning' : 'badge-success';

    const html = `
      <!-- Navigation & Breadcrumb -->
      <div class="details-top-bar">
        <button id="btn-back-to-events" class="btn btn-ghost btn-sm">
          ← Back to Events List
        </button>

        <!-- Quick Event Stepper -->
        <div class="event-stepper">
          <button id="btn-prev-event" class="btn btn-ghost btn-xs" ${!prevEvent ? 'disabled' : ''}>
            ← Prev (${prevEvent ? prevEvent.event_id : 'Start'})
          </button>
          <select id="event-selector-dropdown" class="form-select form-select-sm font-mono">
            ${allEvents.map(e => `
              <option value="${e.event_id}" ${e.event_id === eventId ? 'selected' : ''}>
                ${e.event_id} — ${e.conflict_type} (Risk: ${e.risk_score.toFixed(2)})
              </option>
            `).join('')}
          </select>
          <button id="btn-next-event" class="btn btn-ghost btn-xs" ${!nextEvent ? 'disabled' : ''}>
            Next (${nextEvent ? nextEvent.event_id : 'End'}) →
          </button>
        </div>
      </div>

      <!-- Event Header Banner -->
      <div class="event-hero-header glass-panel">
        <div class="eh-left">
          <div class="eh-eyebrow">
            <span class="badge ${riskBadgeClass} font-mono">ML RISK: ${event.risk_score.toFixed(2)}</span>
            <span class="badge ${event.conflict_type === 'VEH_PED' ? 'badge-critical' : 'badge-orange'} font-mono">
              ${event.conflict_type}
            </span>
            <span class="badge badge-muted font-mono">ZONE ${event.zone_id || 'R1C2'}</span>
          </div>
          <h1 class="event-hero-title font-mono">${event.event_id}</h1>
          <p class="event-hero-subtitle">
            Observed interaction between <strong>${event.classes?.join(' and ') || 'road users'}</strong> at timestamp <strong>${event.start_time.toFixed(1)}s – ${event.end_time.toFixed(1)}s</strong>.
          </p>
        </div>

        <div class="eh-right">
          <button id="btn-trace-this-event" class="btn btn-primary btn-sm">
            Trace in Evidence Graph →
          </button>
        </div>
      </div>

      <!-- Main Layout: Evidence & Trajectory (Left) + Forensics & Agent37 (Right) -->
      <div class="event-investigation-grid">
        <!-- Left Column: Visual Media & Trajectory Visualizer -->
        <div class="investigation-col-media">
          <!-- Evidence Media Card -->
          <div class="card glass-panel">
            <div class="card-header-flex">
              <div>
                <div class="card-eyebrow text-cyan">EVIDENCE RECORDER</div>
                <h3 class="card-title">Video Clip & Telemetry</h3>
              </div>
              <div class="evidence-tag-badge">
                <span class="badge ${event.evidence_available ? 'badge-success' : 'badge-warning'}">
                  ${event.evidence_available ? 'VERIFIED VIDEO CLIP' : 'COORDINATE LOG ONLY'}
                </span>
              </div>
            </div>

            <!-- Media Container -->
            <div class="evidence-media-container">
              ${event.evidence_available && event.evidence_url ? `
                <div class="video-player-wrapper">
                  <video
                    id="event-evidence-video"
                    src="${event.evidence_url}"
                    controls
                    autoplay
                    loop
                    muted
                    playsinline
                    class="evidence-video-element"
                  ></video>
                  <div class="video-overlay-telemetry font-mono">
                    <span>${event.event_id} • PEAK ${event.peak_conflict_time ? event.peak_conflict_time.toFixed(1) + 's' : 'N/A'}</span>
                    <span>TTC: ${event.ttc_min ? event.ttc_min.toFixed(2) + 's' : 'N/A'}</span>
                  </div>
                </div>
              ` : renderEmptyState({
                title: 'Evidence Clip Not Recorded for this Event',
                message: 'This event was captured via coordinate telemetry only; raw video segment was not cached by the video ingestion buffer.',
                icon: 'video',
                badge: 'CLIP UNAVAILABLE'
              })}
            </div>

            <!-- Trajectory Motion Visualizer -->
            <div class="trajectory-visualizer-section">
              <div class="card-header-flex">
                <div>
                  <div class="card-eyebrow text-cyan">TRAJECTORY DYNAMICS</div>
                  <h4 class="card-title text-base">Reconstructed Path Simulation</h4>
                </div>
                <div class="trajectory-controls">
                  <button id="btn-replay-trajectory" class="btn btn-ghost btn-xs">
                    Replay Simulation
                  </button>
                </div>
              </div>

              <div class="trajectory-canvas-wrapper">
                <canvas id="trajectory-canvas" width="600" height="280" class="trajectory-canvas"></canvas>
                <div class="trajectory-legend font-mono text-xs">
                  <span class="legend-track1">● Track #${event.track_ids?.[0]} (${event.classes?.[0] || 'Vehicle'})</span>
                  <span class="legend-track2">▲ Track #${event.track_ids?.[1]} (${event.classes?.[1] || 'Pedestrian'})</span>
                  <span class="legend-conflict">★ Conflict Point</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Column: Safety Metrics, Behavior & Agent37 Context -->
        <div class="investigation-col-forensics">
          <!-- Safety Metrics Card -->
          <div class="card glass-panel">
            <div class="card-header-flex">
              <div>
                <div class="card-eyebrow text-critical">METRIC VALIDATION</div>
                <h3 class="card-title">Deterministic Safety Indicators</h3>
              </div>
            </div>

            <div class="metrics-forensic-grid">
              <!-- TTC Card -->
              <div class="metric-card-forensic glass-panel-nested ${event.ttc_min !== null && event.ttc_min < 2.0 ? 'border-critical' : ''}">
                <div class="mcf-label">MIN TIME-TO-COLLISION (TTC)</div>
                <div class="mcf-value font-mono ${event.ttc_min !== null && event.ttc_min < 2.0 ? 'text-critical' : 'text-warning'}">
                  ${event.ttc_min !== null ? event.ttc_min.toFixed(2) + 's' : 'N/A'}
                </div>
                <div class="mcf-sub text-xs text-muted">
                  Config Limit: <strong>${analysis.ttc_threshold_s || 3.0}s</strong>
                  ${event.ttc_min !== null ? `(${Math.round(((analysis.ttc_threshold_s || 3.0) - event.ttc_min) / (analysis.ttc_threshold_s || 3.0) * 100)}% below safe limit)` : ''}
                </div>
              </div>

              <!-- PET Card -->
              <div class="metric-card-forensic glass-panel-nested">
                <div class="mcf-label">POST-ENCROACHMENT TIME (PET)</div>
                <div class="mcf-value font-mono text-cyan">
                  ${event.pet_min !== null ? event.pet_min.toFixed(2) + 's' : '<span class="text-muted">Not Applicable</span>'}
                </div>
                <div class="mcf-sub text-xs text-muted">
                  Config Threshold: <strong>${analysis.pet_threshold_s || 2.0}s</strong>
                </div>
              </div>

              <!-- Severity Score -->
              <div class="metric-card-forensic glass-panel-nested">
                <div class="mcf-label">SEVERITY INDEX</div>
                <div class="mcf-value font-mono text-warning">
                  ${event.severity.toFixed(2)}
                </div>
                <div class="mcf-sub text-xs text-muted">
                  Normalized [0.0 - 1.0]
                </div>
              </div>

              <!-- Risk Score -->
              <div class="metric-card-forensic glass-panel-nested">
                <div class="mcf-label">ML RISK SCORE</div>
                <div class="mcf-value font-mono text-critical">
                  ${event.risk_score.toFixed(2)}
                </div>
                <div class="mcf-sub text-xs text-muted">
                  Confidence: ${Math.round((event.confidence || 0.88) * 100)}%
                </div>
              </div>
            </div>

            <!-- Evasive Behavior Dynamics -->
            <div class="behavior-section">
              <div class="section-label">EVASIVE BEHAVIOR DYNAMICS</div>
              <div class="behavior-pills-row">
                <div class="behavior-metric-pill glass-panel-nested">
                  <span class="bmp-label">Max Speed Drop</span>
                  <span class="bmp-val font-mono text-warning">
                    ${event.max_speed_change ? Math.round(event.max_speed_change * 100) + '%' : '0%'}
                  </span>
                </div>
                <div class="behavior-metric-pill glass-panel-nested">
                  <span class="bmp-label">Max Heading Swerve</span>
                  <span class="bmp-val font-mono text-cyan">
                    ${event.max_heading_change ? event.max_heading_change.toFixed(1) + '°' : '0.0°'}
                  </span>
                </div>
              </div>

              <div class="reasons-list">
                <span class="text-xs text-muted font-mono">TRIGGER CONDITIONS:</span>
                <div class="reasons-chips-row">
                  ${(event.reasons || []).map(r => `
                    <span class="badge badge-warning badge-sm">${r.replace(/_/g, ' ')}</span>
                  `).join('')}
                </div>
              </div>
            </div>
          </div>

          <!-- Agent37 Interpretation Card -->
          <div class="card glass-panel">
            <div class="card-header-flex">
              <div>
                <div class="card-eyebrow text-cyan">AGENT37 CONTEXT</div>
                <h3 class="card-title">Intelligence Grounding</h3>
              </div>
            </div>

            <!-- Findings referencing this event -->
            <div class="agent-crosslinks">
              <span class="sub-label">REFERENCED IN FINDINGS:</span>
              ${(event.related_findings || []).length > 0 ? (event.related_findings || []).map(f => `
                <div class="crosslink-card glass-panel-nested">
                  <div class="cl-header">
                    <span class="font-mono text-cyan font-bold">${f.finding_id}</span>
                    <span class="badge ${f.severity === 'HIGH' ? 'badge-critical' : 'badge-warning'} badge-xs">${f.severity} SEVERITY</span>
                  </div>
                  <p class="cl-text">${f.finding}</p>
                  <p class="cl-reasoning text-xs text-muted">"${f.reasoning}"</p>
                </div>
              `).join('') : '<div class="text-xs text-muted mt-1">No explicit finding isolated for this event.</div>'}
            </div>

            <!-- Recommendations referencing this event -->
            <div class="agent-crosslinks">
              <span class="sub-label">GROUNDS RECOMMENDATIONS:</span>
              ${(event.related_recommendations || []).length > 0 ? (event.related_recommendations || []).map(r => `
                <div class="crosslink-card glass-panel-nested">
                  <div class="cl-header">
                    <span class="font-mono text-orange font-bold">${r.recommendation_id}</span>
                    <span class="badge badge-critical badge-xs">${r.priority}</span>
                  </div>
                  <p class="cl-text">${r.action}</p>
                  <button class="btn btn-ghost btn-xs btn-jump-rec" data-rec-id="${r.recommendation_id}">
                    Inspect in Agent37 Report →
                  </button>
                </div>
              `).join('') : '<div class="text-xs text-muted mt-1">Part of aggregate session context.</div>'}
            </div>

            <!-- Hotspot Link -->
            <div class="hotspot-link-box glass-panel-nested">
              <div class="hl-content">
                <span class="text-xs text-muted">OCCURS IN HOTSPOT:</span>
                <span class="font-mono font-bold text-cyan">Zone ${event.zone_id || 'R1C2'}</span>
              </div>
              <button id="btn-jump-hotspot" class="btn btn-outline-cyan btn-xs">
                Inspect Zone →
              </button>
            </div>
          </div>
        </div>
      </div>
    `;

    container.innerHTML = html;

    // Attach Stepper & Navigation handlers
    container.querySelector('#btn-back-to-events')?.addEventListener('click', () => {
      store.navigateTo('events');
    });

    container.querySelector('#btn-prev-event')?.addEventListener('click', () => {
      if (prevEvent) store.selectEvent(prevEvent.event_id);
    });

    container.querySelector('#btn-next-event')?.addEventListener('click', () => {
      if (nextEvent) store.selectEvent(nextEvent.event_id);
    });

    container.querySelector('#event-selector-dropdown')?.addEventListener('change', (e) => {
      store.selectEvent(e.target.value);
    });

    container.querySelector('#btn-trace-this-event')?.addEventListener('click', () => {
      store.setState({
        currentView: 'traceability',
        traceabilityFilter: { eventId: event.event_id }
      });
    });

    container.querySelector('#btn-jump-hotspot')?.addEventListener('click', () => {
      store.selectHotspot(event.zone_id || 'R1C2');
    });

    container.querySelectorAll('.btn-jump-rec').forEach(btn => {
      btn.addEventListener('click', () => {
        const rid = btn.getAttribute('data-rec-id');
        store.navigateTo('agent37', { recommendationId: rid });
      });
    });

    // Initialize Trajectory Canvas Animation
    setTimeout(() => {
      initTrajectoryAnimation(container.querySelector('#trajectory-canvas'), event);
    }, 50);

    container.querySelector('#btn-replay-trajectory')?.addEventListener('click', () => {
      initTrajectoryAnimation(container.querySelector('#trajectory-canvas'), event);
    });

  } catch (err) {
    console.error('Error rendering Event Details:', err);
    container.innerHTML = renderErrorState({
      title: `Failed to Load Event Details for ${eventId}`,
      message: err.message,
      retryCallback: () => store.notify()
    });
  }

  return container;
}

// Canvas-based trajectory animation showing the interaction between the two tracks
function initTrajectoryAnimation(canvas, event) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;

  let t = 0;
  let animId = null;

  const isPed = event.classes?.includes('pedestrian');
  const isMoto = event.classes?.includes('motorcycle');

  const track1Start = { x: 60, y: 140 };
  const track1End = { x: 540, y: 140 };

  const track2Start = isPed ? { x: 300, y: 40 } : isMoto ? { x: 120, y: 220 } : { x: 300, y: 240 };
  const track2End = isPed ? { x: 300, y: 240 } : isMoto ? { x: 480, y: 150 } : { x: 300, y: 40 };

  const conflictPoint = { x: 300, y: 140 };

  function renderFrame() {
    ctx.clearRect(0, 0, w, h);

    // Background road grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Lane lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    ctx.moveTo(0, 140);
    ctx.lineTo(w, 140);
    ctx.stroke();
    ctx.setLineDash([]);

    // Crosswalk stripes if pedestrian involved
    if (isPed) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
      for (let y = 50; y < 230; y += 20) {
        ctx.fillRect(280, y, 40, 10);
      }
    }

    // Trajectory paths
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(track1Start.x, track1Start.y);
    ctx.lineTo(track1End.x, track1End.y);
    ctx.stroke();

    ctx.strokeStyle = isPed ? 'rgba(239, 68, 68, 0.4)' : 'rgba(249, 115, 22, 0.4)';
    ctx.beginPath();
    ctx.moveTo(track2Start.x, track2Start.y);
    ctx.lineTo(track2End.x, track2End.y);
    ctx.stroke();

    // Conflict proximity circle
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(conflictPoint.x, conflictPoint.y, 35, 0, Math.PI * 2);
    ctx.stroke();

    // Current positions at time fraction t
    const p1x = track1Start.x + (track1End.x - track1Start.x) * t;
    const p1y = track1Start.y + (track1End.y - track1Start.y) * t;

    const p2x = track2Start.x + (track2End.x - track2Start.x) * t;
    const p2y = track2Start.y + (track2End.y - track2Start.y) * t;

    // Track 1
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(p1x, p1y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = '10px monospace';
    ctx.fillText(`T#${event.track_ids?.[0] || '12'}`, p1x - 12, p1y - 12);

    // Track 2
    ctx.fillStyle = isPed ? '#ef4444' : '#f97316';
    ctx.beginPath();
    ctx.arc(p2x, p2y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(`T#${event.track_ids?.[1] || '18'}`, p2x + 8, p2y + 4);

    // Conflict proximity alert
    const dist = Math.hypot(p1x - p2x, p1y - p2y);
    if (dist < 50) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
      ctx.font = 'bold 11px monospace';
      ctx.fillText(`MIN TTC: ${event.ttc_min ? event.ttc_min.toFixed(2) + 's' : 'CRITICAL'}`, conflictPoint.x - 45, conflictPoint.y - 45);

      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p1x, p1y);
      ctx.lineTo(p2x, p2y);
      ctx.stroke();
    }

    t += 0.008;
    if (t > 1) t = 0;

    animId = requestAnimationFrame(renderFrame);
  }

  renderFrame();
}
