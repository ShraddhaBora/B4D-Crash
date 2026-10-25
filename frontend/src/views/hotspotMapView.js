// frontend/src/views/hotspotMapView.js
// Page 2 — Risk / Hotspot Spatial Map & Zone Forensics

import { api } from '../api.js';
import { store } from '../state/store.js';
import { renderEmptyState, renderErrorState, renderLoadingSkeleton } from '../components/emptyState.js';

export async function renderHotspotMapView() {
  const container = document.createElement('div');
  container.className = 'view-container hotspot-view';

  container.innerHTML = renderLoadingSkeleton({ title: 'Rendering Spatial Zone Matrix & Hotspots...' });

  try {
    const [hotspots, analysis, agentReport] = await Promise.all([
      api.getHotspots(),
      api.getAnalysis(),
      api.getAgentReport()
    ]);

    if (!hotspots || hotspots.length === 0) {
      container.innerHTML = renderEmptyState({
        title: 'No Conflict Hotspots Identified',
        message: 'No spatial clustering met the risk-threshold criteria for this session.',
        icon: 'mapPin',
        badge: 'HOTSPOTS: ZERO'
      });
      return container;
    }

    const state = store.getState();
    let activeZoneId = state.selectedZoneId || hotspots[0].zone_id;
    let selectedHotspot = hotspots.find(h => h.zone_id === activeZoneId) || hotspots[0];

    const camera = analysis.camera || {};

    const html = `
      <!-- View Header -->
      <div class="view-header">
        <div class="view-header-main">
          <div class="view-eyebrow">
            <span class="badge-tag"><span class="beacon-dot"></span> SPATIAL AGGREGATION</span>
            <span class="meta-tag">Zones: <strong>${hotspots.length} Prioritized</strong></span>
            <span class="meta-tag">Camera: <strong>${camera.camera_id || 'CAM_INTERSECTION_04'}</strong></span>
          </div>
          <h1 class="view-title">Spatial Hotspot Analysis</h1>
          <p class="view-subtitle">
            Camera-space coordinate clustering and physical geographic site intelligence.
          </p>
        </div>

        <!-- Mode Toggle: Camera Sensor Grid vs GIS Map -->
        <div class="view-header-actions">
          <div class="toggle-pill-group">
            <button id="toggle-camera-view" class="toggle-pill active">
              Camera Sensor Grid
            </button>
            <button id="toggle-gis-view" class="toggle-pill">
              GIS Satellite Map
            </button>
          </div>
        </div>
      </div>

      <!-- Main Layout: Spatial Viewport (Left) + Hotspot Drawer (Right) -->
      <div class="spatial-layout-grid">
        <!-- Visual Viewport Card -->
        <div class="card glass-panel spatial-canvas-card">
          <div class="viewport-meta-header">
            <div class="viewport-title-group">
              <span class="text-xs uppercase text-muted font-mono" id="viewport-mode-label">SENSOR MATRIX: 1920x1080 SENSOR GRID</span>
              <div class="viewport-location-name">${camera.location_name || 'Market St & 5th Ave Intersection (Northbound)'}</div>
            </div>
            <div class="viewport-badges">
              <span class="badge ${camera.calibration_active ? 'badge-success' : 'badge-warning'}">
                ${camera.calibration_active ? 'HOMOGRAPHY ACTIVE' : 'CALIBRATION INACTIVE (PIXEL UNITS)'}
              </span>
            </div>
          </div>

          <!-- Spatial Viewport Container -->
          <div class="viewport-viewport-wrapper">
            <!-- CAMERA GRID PERSPECTIVE (Default) -->
            <div id="camera-viewport-display" class="spatial-display-container">
              <img src="/assets/camera_view.jpg" alt="Traffic Intersection Camera View" class="camera-backdrop-image" />
              <div class="camera-grid-overlay">
                <!-- Zone Overlays mapped directly to camera coordinates -->
                <!-- Zone R0C1 (Top Center-Left) -->
                <div class="zone-polygon zone-r0c1 ${activeZoneId === 'R0C1' ? 'zone-active' : ''}" data-zone-id="R0C1">
                  <div class="zone-tag">
                    <span class="zone-code">R0C1</span>
                    <span class="badge badge-success badge-xs">LOW RISK (0.19)</span>
                  </div>
                  <div class="zone-pulse"></div>
                  <div class="zone-events-badge">2 Events</div>
                </div>

                <!-- Zone R1C2 (Center Right Crosswalk) -->
                <div class="zone-polygon zone-r1c2 ${activeZoneId === 'R1C2' ? 'zone-active' : ''}" data-zone-id="R1C2">
                  <div class="zone-tag">
                    <span class="zone-code">R1C2</span>
                    <span class="badge badge-critical badge-xs">#1 CRITICAL (0.73)</span>
                  </div>
                  <div class="zone-pulse pulse-critical"></div>
                  <div class="zone-events-badge badge-critical">3 Events • 2 High-Sev</div>
                </div>

                <!-- Zone R2C3 (Bottom Right Merging) -->
                <div class="zone-polygon zone-r2c3 ${activeZoneId === 'R2C3' ? 'zone-active' : ''}" data-zone-id="R2C3">
                  <div class="zone-tag">
                    <span class="zone-code">R2C3</span>
                    <span class="badge badge-orange badge-xs">#2 MEDIUM (0.60)</span>
                  </div>
                  <div class="zone-pulse pulse-warning"></div>
                  <div class="zone-events-badge badge-orange">3 Events • Moto</div>
                </div>
              </div>
            </div>

            <!-- GEOGRAPHIC GIS MAP (Alternative View) -->
            <div id="gis-viewport-display" class="spatial-display-container hidden">
              <img src="/assets/gis_map.jpg" alt="San Francisco GIS Road Safety Map" class="gis-backdrop-image" />
              <div class="gis-overlay-info glass-panel">
                <div class="gis-coord-row">
                  <span class="gis-label">COORDINATES:</span>
                  <span class="gis-val font-mono">${camera.location_lat || 37.7833}° N, ${Math.abs(camera.location_lon || -122.4089)}° W</span>
                </div>
                <div class="gis-coord-row">
                  <span class="gis-label">LOCATION:</span>
                  <span class="gis-val">Downtown SF Urban Core (Market & 5th)</span>
                </div>
                <div class="gis-notice text-xs text-muted">
                  Precise geographic mapping requires site-specific homography calibration matrices. All spatial conflict metrics currently computed in pixel sensor space.
                </div>
              </div>
            </div>
          </div>

          <!-- Hotspot Selection Row below viewport -->
          <div class="hotspot-selector-bar">
            <span class="text-xs text-muted">SELECT ZONE:</span>
            <div class="hotspot-pills-row">
              ${hotspots.map(h => `
                <button class="hotspot-pill-btn ${h.zone_id === activeZoneId ? 'active' : ''}" data-zone-id="${h.zone_id}">
                  <span class="pill-priority">#${h.priority}</span>
                  <span class="pill-zid">${h.zone_id}</span>
                  <span class="badge ${h.risk_level === 'HIGH' ? 'badge-critical' : h.risk_level === 'MEDIUM' ? 'badge-warning' : 'badge-success'} badge-xs">
                    ${h.avg_risk_score.toFixed(2)}
                  </span>
                </button>
              `).join('')}
            </div>
          </div>
        </div>

        <!-- Hotspot Forensic Drawer / Detail Panel (Right) -->
        <div class="card glass-panel hotspot-detail-drawer" id="hotspot-drawer-panel">
          ${renderDrawerContent(selectedHotspot, agentReport)}
        </div>
      </div>
    `;

    container.innerHTML = html;

    // View switcher buttons
    const cameraBtn = container.querySelector('#toggle-camera-view');
    const gisBtn = container.querySelector('#toggle-gis-view');
    const cameraDisplay = container.querySelector('#camera-viewport-display');
    const gisDisplay = container.querySelector('#gis-viewport-display');
    const modeLabel = container.querySelector('#viewport-mode-label');

    cameraBtn?.addEventListener('click', () => {
      cameraBtn.classList.add('active');
      gisBtn?.classList.remove('active');
      cameraDisplay?.classList.remove('hidden');
      gisDisplay?.classList.add('hidden');
      if (modeLabel) modeLabel.textContent = 'SENSOR MATRIX: 1920x1080 SENSOR GRID';
    });

    gisBtn?.addEventListener('click', () => {
      gisBtn.classList.add('active');
      cameraBtn?.classList.remove('active');
      gisDisplay?.classList.remove('hidden');
      cameraDisplay?.classList.add('hidden');
      if (modeLabel) modeLabel.textContent = `GEOGRAPHIC GPS: ${camera.location_lat}° N, ${camera.location_lon}° W`;
    });

    // Zone polygon click handlers
    const attachZoneClickListeners = () => {
      container.querySelectorAll('.zone-polygon, .hotspot-pill-btn').forEach(el => {
        el.addEventListener('click', () => {
          const zid = el.getAttribute('data-zone-id');
          if (zid && zid !== activeZoneId) {
            activeZoneId = zid;
            store.setState({ selectedZoneId: zid });
            selectedHotspot = hotspots.find(h => h.zone_id === zid) || hotspots[0];

            // Update UI active styles
            container.querySelectorAll('.zone-polygon').forEach(z => z.classList.remove('zone-active'));
            container.querySelector(`.zone-polygon[data-zone-id="${zid}"]`)?.classList.add('zone-active');

            container.querySelectorAll('.hotspot-pill-btn').forEach(b => b.classList.remove('active'));
            container.querySelector(`.hotspot-pill-btn[data-zone-id="${zid}"]`)?.classList.add('active');

            // Re-render drawer content
            const drawer = container.querySelector('#hotspot-drawer-panel');
            if (drawer) {
              drawer.innerHTML = renderDrawerContent(selectedHotspot, agentReport);
              attachDrawerActions(drawer, selectedHotspot);
            }
          }
        });
      });
    };

    attachZoneClickListeners();
    attachDrawerActions(container.querySelector('#hotspot-drawer-panel'), selectedHotspot);

  } catch (err) {
    console.error('Error rendering Hotspot Map:', err);
    container.innerHTML = renderErrorState({
      title: 'Failed to Render Spatial Hotspot Matrix',
      message: err.message,
      retryCallback: () => store.notify()
    });
  }

  return container;
}

function renderDrawerContent(hotspot, agentReport) {
  if (!hotspot) return '<div class="p-4 text-muted">Select a hotspot to view details.</div>';

  const riskBadgeClass = hotspot.risk_level === 'HIGH' ? 'badge-critical' : hotspot.risk_level === 'MEDIUM' ? 'badge-warning' : 'badge-success';

  return `
    <div class="drawer-header">
      <div>
        <div class="drawer-eyebrow">
          <span class="badge ${riskBadgeClass}">PRIORITY #${hotspot.priority || 1} • ${hotspot.risk_level} RISK</span>
        </div>
        <h2 class="drawer-title">Zone ${hotspot.zone_id}</h2>
      </div>
      <div class="drawer-score-box">
        <div class="score-value font-mono">${hotspot.avg_risk_score.toFixed(3)}</div>
        <div class="score-label">ML AVG RISK</div>
      </div>
    </div>

    <!-- Prioritization Reason -->
    <div class="drawer-section">
      <div class="section-label">AGENT37 PRIORITIZATION RATIONALE</div>
      <p class="drawer-reason-text">
        "${hotspot.reason || `Zone '${hotspot.zone_id}' recorded ${hotspot.event_count} events with ML avg risk ${hotspot.avg_risk_score.toFixed(2)}.`}"
      </p>
    </div>

    <!-- Core Metrics Grid -->
    <div class="drawer-metrics-grid">
      <div class="drawer-metric-item glass-panel-nested">
        <span class="dm-label">Event Count</span>
        <span class="dm-value font-mono">${hotspot.event_count}</span>
      </div>
      <div class="drawer-metric-item glass-panel-nested">
        <span class="dm-label">High-Risk Events</span>
        <span class="dm-value font-mono text-critical">${hotspot.high_risk_count}</span>
      </div>
      <div class="drawer-metric-item glass-panel-nested">
        <span class="dm-label">Minimum TTC</span>
        <span class="dm-value font-mono text-warning">${hotspot.min_ttc ? hotspot.min_ttc.toFixed(2) + 's' : 'N/A'}</span>
      </div>
      <div class="drawer-metric-item glass-panel-nested">
        <span class="dm-label">Minimum PET</span>
        <span class="dm-value font-mono text-cyan">${hotspot.min_pet ? hotspot.min_pet.toFixed(2) + 's' : 'N/A'}</span>
      </div>
    </div>

    <!-- Conflict & Road User Distribution -->
    <div class="drawer-section">
      <div class="section-label">CONFLICT COMPOSITION</div>
      <div class="distribution-tags-row">
        ${Object.entries(hotspot.conflict_type_dist || {}).map(([type, count]) => `
          <div class="dist-badge">
            <span class="badge-type">${type}</span>
            <span class="badge-count font-mono">${count}</span>
          </div>
        `).join('')}
      </div>

      <div class="section-label mt-3">ROAD USERS INVOLVED</div>
      <div class="distribution-tags-row">
        ${Object.entries(hotspot.road_user_dist || {}).map(([user, count]) => `
          <div class="dist-badge">
            <span class="badge-type">${user}</span>
            <span class="badge-count font-mono">${count}</span>
          </div>
        `).join('')}
      </div>
    </div>

    <!-- Contributing Events -->
    <div class="drawer-section">
      <div class="section-label">CONTRIBUTING EVIDENCE EVENTS (${hotspot.event_ids?.length || 0})</div>
      <div class="contributing-events-list">
        ${(hotspot.event_ids || []).map(eid => `
          <div class="contributing-event-card glass-panel-nested clickable-item btn-event-jump" data-event-id="${eid}">
            <div class="cec-left">
              <span class="font-mono text-cyan font-semibold">${eid}</span>
            </div>
            <div class="cec-right">
              <span class="text-xs link-text">Inspect Event →</span>
            </div>
          </div>
        `).join('')}
      </div>
    </div>

    <!-- Drawer Actions -->
    <div class="drawer-actions">
      <button class="btn btn-primary btn-block" id="btn-filter-zone-events">
        Explore All Events in Zone ${hotspot.zone_id} →
      </button>
      <button class="btn btn-outline-cyan btn-block mt-2" id="btn-jump-traceability">
        Trace Zone Evidence Lineage →
      </button>
    </div>
  `;
}

function attachDrawerActions(drawer, hotspot) {
  if (!drawer || !hotspot) return;

  drawer.querySelectorAll('.btn-event-jump').forEach(btn => {
    btn.addEventListener('click', () => {
      const eid = btn.getAttribute('data-event-id');
      if (eid) store.selectEvent(eid);
    });
  });

  drawer.querySelector('#btn-filter-zone-events')?.addEventListener('click', () => {
    store.filterEventsByZone(hotspot.zone_id);
  });

  drawer.querySelector('#btn-jump-traceability')?.addEventListener('click', () => {
    store.setState({
      currentView: 'traceability',
      traceabilityFilter: { zoneId: hotspot.zone_id }
    });
  });
}
