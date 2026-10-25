// frontend/src/views/eventsView.js
// Page 3 — Conflict Events Exploration Interface

import { api } from '../api.js';
import { store } from '../state/store.js';
import { icons } from '../components/icons.js';
import { renderEmptyState, renderErrorState, renderLoadingSkeleton } from '../components/emptyState.js';

export async function renderEventsView() {
  const container = document.createElement('div');
  container.className = 'view-container events-view';

  container.innerHTML = renderLoadingSkeleton({ title: 'Loading Conflict Events Database...' });

  try {
    const events = await api.getEvents();
    const state = store.getState();
    const currentFilters = state.filters;

    // Apply client-side filters
    let filteredEvents = [...events];

    if (currentFilters.searchQuery.trim()) {
      const q = currentFilters.searchQuery.toLowerCase();
      filteredEvents = filteredEvents.filter(e =>
        e.event_id.toLowerCase().includes(q) ||
        e.conflict_type.toLowerCase().includes(q) ||
        (e.zone_id && e.zone_id.toLowerCase().includes(q)) ||
        (e.classes && e.classes.some(c => c.toLowerCase().includes(q))) ||
        (e.reasons && e.reasons.some(r => r.toLowerCase().includes(q))) ||
        (e.track_ids && e.track_ids.some(t => t.toString().includes(q)))
      );
    }

    if (currentFilters.conflictType !== 'ALL') {
      filteredEvents = filteredEvents.filter(e => e.conflict_type === currentFilters.conflictType);
    }

    if (currentFilters.zoneId !== 'ALL') {
      filteredEvents = filteredEvents.filter(e => e.zone_id === currentFilters.zoneId);
    }

    if (currentFilters.severity !== 'ALL') {
      if (currentFilters.severity === 'HIGH') {
        filteredEvents = filteredEvents.filter(e => e.severity >= 0.70);
      } else if (currentFilters.severity === 'MEDIUM') {
        filteredEvents = filteredEvents.filter(e => e.severity >= 0.40 && e.severity < 0.70);
      } else if (currentFilters.severity === 'LOW') {
        filteredEvents = filteredEvents.filter(e => e.severity < 0.40);
      }
    }

    if (currentFilters.minRisk > 0) {
      filteredEvents = filteredEvents.filter(e => e.risk_score >= currentFilters.minRisk);
    }

    // Default sort by Risk Score descending
    filteredEvents.sort((a, b) => b.risk_score - a.risk_score);

    const html = `
      <!-- Header -->
      <div class="view-header">
        <div class="view-header-main">
          <div class="view-eyebrow">
            <span class="badge-tag"><span class="beacon-dot"></span> CONFLICT ENGINE</span>
            <span class="meta-tag">Total: <strong>${events.length} Events</strong></span>
            <span class="meta-tag">Showing: <strong>${filteredEvents.length} Matched</strong></span>
          </div>
          <h1 class="view-title">Traffic Conflict Events</h1>
          <p class="view-subtitle">
            Near-miss interactions detected via Time-To-Collision (TTC), Post-Encroachment Time (PET), and evasive dynamics.
          </p>
        </div>

        <!-- View Switcher -->
        <div class="view-header-actions">
          <div class="toggle-pill-group">
            <button id="view-mode-cards" class="toggle-pill ${state.eventsViewMode === 'cards' ? 'active' : ''}">
              Cards
            </button>
            <button id="view-mode-table" class="toggle-pill ${state.eventsViewMode === 'table' ? 'active' : ''}">
              Table
            </button>
          </div>
        </div>
      </div>

      <!-- Filter Controls Toolbar -->
      <div class="filter-toolbar glass-panel">
        <div class="filter-grid">
          <!-- Search Bar -->
          <div class="filter-col search-col">
            <label class="filter-label">SEARCH</label>
            <div class="search-input-wrapper">
              ${icons.search(16)}
              <input
                type="text"
                id="events-search-input"
                class="form-control"
                placeholder="Event ID, tracks, vehicle class, reasons..."
                value="${currentFilters.searchQuery}"
              />
            </div>
          </div>

          <!-- Conflict Type Filter -->
          <div class="filter-col">
            <label class="filter-label">CONFLICT TYPE</label>
            <select id="filter-conflict-type" class="form-select">
              <option value="ALL" ${currentFilters.conflictType === 'ALL' ? 'selected' : ''}>All Types</option>
              <option value="VEH_PED" ${currentFilters.conflictType === 'VEH_PED' ? 'selected' : ''}>Pedestrian-Vehicle</option>
              <option value="VEH_VEH" ${currentFilters.conflictType === 'VEH_VEH' ? 'selected' : ''}>Vehicle-Vehicle</option>
              <option value="VEH_MOTO" ${currentFilters.conflictType === 'VEH_MOTO' ? 'selected' : ''}>Motorcycle-Vehicle</option>
            </select>
          </div>

          <!-- Zone Filter -->
          <div class="filter-col">
            <label class="filter-label">SPATIAL ZONE</label>
            <select id="filter-zone" class="form-select">
              <option value="ALL" ${currentFilters.zoneId === 'ALL' ? 'selected' : ''}>All Zones</option>
              <option value="R1C2" ${currentFilters.zoneId === 'R1C2' ? 'selected' : ''}>Zone R1C2 (#1 Critical)</option>
              <option value="R2C3" ${currentFilters.zoneId === 'R2C3' ? 'selected' : ''}>Zone R2C3 (#2 Medium)</option>
              <option value="R0C1" ${currentFilters.zoneId === 'R0C1' ? 'selected' : ''}>Zone R0C1 (#3 Low)</option>
            </select>
          </div>

          <!-- Severity Filter -->
          <div class="filter-col">
            <label class="filter-label">SEVERITY LEVEL</label>
            <select id="filter-severity" class="form-select">
              <option value="ALL" ${currentFilters.severity === 'ALL' ? 'selected' : ''}>All Severities</option>
              <option value="HIGH" ${currentFilters.severity === 'HIGH' ? 'selected' : ''}>High (≥ 0.70)</option>
              <option value="MEDIUM" ${currentFilters.severity === 'MEDIUM' ? 'selected' : ''}>Medium (0.40 - 0.69)</option>
              <option value="LOW" ${currentFilters.severity === 'LOW' ? 'selected' : ''}>Low (< 0.40)</option>
            </select>
          </div>

          <!-- Min Risk Slider -->
          <div class="filter-col slider-col">
            <div class="slider-header">
              <label class="filter-label">MIN ML RISK</label>
              <span class="font-mono text-cyan" id="min-risk-display">${currentFilters.minRisk.toFixed(2)}</span>
            </div>
            <input
              type="range"
              id="filter-min-risk"
              class="form-range"
              min="0.0"
              max="0.90"
              step="0.05"
              value="${currentFilters.minRisk}"
            />
          </div>

          <!-- Reset Button -->
          <div class="filter-col reset-col">
            <label class="filter-label">&nbsp;</label>
            <button id="btn-reset-filters" class="btn btn-ghost btn-sm" title="Clear all filters">
              Reset Filters
            </button>
          </div>
        </div>
      </div>

      <!-- Events List / Grid Area -->
      <div class="events-content-area">
        ${filteredEvents.length === 0 ? renderEmptyState({
          title: 'No Matching Conflict Events',
          message: 'Try adjusting your filters, search keyword, or minimum risk score.',
          icon: 'search',
          actionLabel: 'Reset Filters',
          actionCallback: () => {
            store.resetFilters();
          }
        }) : state.eventsViewMode === 'cards' ? renderEventCards(filteredEvents) : renderEventTable(filteredEvents)}
      </div>
    `;

    container.innerHTML = html;

    // Attach filter change listeners
    const searchInput = container.querySelector('#events-search-input');
    searchInput?.addEventListener('input', (e) => {
      store.setFilters({ searchQuery: e.target.value });
    });

    container.querySelector('#filter-conflict-type')?.addEventListener('change', (e) => {
      store.setFilters({ conflictType: e.target.value });
    });

    container.querySelector('#filter-zone')?.addEventListener('change', (e) => {
      store.setFilters({ zoneId: e.target.value });
    });

    container.querySelector('#filter-severity')?.addEventListener('change', (e) => {
      store.setFilters({ severity: e.target.value });
    });

    const riskSlider = container.querySelector('#filter-min-risk');
    const riskDisplay = container.querySelector('#min-risk-display');
    riskSlider?.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (riskDisplay) riskDisplay.textContent = val.toFixed(2);
      store.setFilters({ minRisk: val });
    });

    container.querySelector('#btn-reset-filters')?.addEventListener('click', () => {
      store.resetFilters();
    });

    // View mode toggle
    container.querySelector('#view-mode-cards')?.addEventListener('click', () => {
      store.setState({ eventsViewMode: 'cards' });
    });

    container.querySelector('#view-mode-table')?.addEventListener('click', () => {
      store.setState({ eventsViewMode: 'table' });
    });

    // Click to jump to event details
    container.querySelectorAll('.btn-inspect-event').forEach(el => {
      el.addEventListener('click', () => {
        const eid = el.getAttribute('data-event-id');
        if (eid) store.selectEvent(eid);
      });
    });

  } catch (err) {
    console.error('Error rendering Events View:', err);
    container.innerHTML = renderErrorState({
      title: 'Failed to Retrieve Conflict Events',
      message: err.message,
      retryCallback: () => store.notify()
    });
  }

  return container;
}

function renderEventCards(events) {
  return `
    <div class="events-card-grid">
      ${events.map(ev => {
        const riskLevelClass = ev.risk_score >= 0.70 ? 'badge-critical' : ev.risk_score >= 0.40 ? 'badge-warning' : 'badge-success';
        const ttcCritical = ev.ttc_min !== null && ev.ttc_min < 2.0;

        return `
          <div class="event-card glass-panel clickable-card btn-inspect-event" data-event-id="${ev.event_id}">
            <div class="event-card-header">
              <div class="ech-left">
                <span class="event-id-tag font-mono">${ev.event_id}</span>
                <span class="badge ${ev.conflict_type === 'VEH_PED' ? 'badge-critical' : ev.conflict_type === 'VEH_MOTO' ? 'badge-orange' : 'badge-cyan'} badge-xs">
                  ${ev.conflict_type}
                </span>
              </div>
              <div class="ech-right">
                <span class="badge ${riskLevelClass}">RISK: ${ev.risk_score.toFixed(2)}</span>
              </div>
            </div>

            <!-- Road Users Involved -->
            <div class="event-tracks-row">
              <span class="text-xs text-muted">TRACKS:</span>
              <span class="font-mono text-sm">${(ev.classes || []).map((c, i) => `#${ev.track_ids?.[i] || '?'} (${c})`).join(' ↔ ')}</span>
            </div>

            <!-- Key Safety Indicators -->
            <div class="event-metrics-row">
              <div class="em-item">
                <span class="em-label">MIN TTC</span>
                <span class="em-val font-mono ${ttcCritical ? 'text-critical font-bold' : ''}">
                  ${ev.ttc_min !== null ? ev.ttc_min.toFixed(2) + 's' : '<span class="text-muted">N/A</span>'}
                </span>
              </div>
              <div class="em-item">
                <span class="em-label">MIN PET</span>
                <span class="em-val font-mono">
                  ${ev.pet_min !== null ? ev.pet_min.toFixed(2) + 's' : '<span class="text-muted">N/A</span>'}
                </span>
              </div>
              <div class="em-item">
                <span class="em-label">SEVERITY</span>
                <span class="em-val font-mono text-warning">${ev.severity.toFixed(2)}</span>
              </div>
              <div class="em-item">
                <span class="em-label">ZONE</span>
                <span class="em-val font-mono text-cyan">${ev.zone_id || 'R1C2'}</span>
              </div>
            </div>

            <!-- Evasive Action Indicators -->
            <div class="evasive-tags-row">
              ${(ev.reasons || []).map(r => `
                <span class="reason-tag">${r.replace(/_/g, ' ')}</span>
              `).join('')}
            </div>

            <!-- Card Footer -->
            <div class="event-card-footer">
              <div class="evidence-status-pill ${ev.evidence_available ? 'text-success' : 'text-muted'}">
                <span>${ev.evidence_available ? 'HD Video Available' : 'Sensor Coordinate Log'}</span>
              </div>
              <span class="link-text text-sm">
                Inspect Forensics →
              </span>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function renderEventTable(events) {
  return `
    <div class="table-responsive glass-panel">
      <table class="data-table">
        <thead>
          <tr>
            <th>EVENT ID</th>
            <th>CONFLICT TYPE</th>
            <th>INVOLVED TRACKS</th>
            <th>ZONE</th>
            <th>RISK SCORE</th>
            <th>SEVERITY</th>
            <th>MIN TTC</th>
            <th>MIN PET</th>
            <th>TIMECODE</th>
            <th>EVIDENCE</th>
            <th>ACTION</th>
          </tr>
        </thead>
        <tbody>
          ${events.map(ev => `
            <tr class="clickable-row btn-inspect-event" data-event-id="${ev.event_id}">
              <td class="font-mono text-cyan font-semibold">${ev.event_id}</td>
              <td>
                <span class="badge ${ev.conflict_type === 'VEH_PED' ? 'badge-critical' : ev.conflict_type === 'VEH_MOTO' ? 'badge-orange' : 'badge-cyan'} badge-xs">
                  ${ev.conflict_type}
                </span>
              </td>
              <td class="text-sm font-mono">${(ev.classes || []).map((c, i) => `#${ev.track_ids?.[i]} (${c})`).join(' vs ')}</td>
              <td class="font-mono">${ev.zone_id || 'R1C2'}</td>
              <td class="font-mono font-bold ${ev.risk_score >= 0.70 ? 'text-critical' : ev.risk_score >= 0.40 ? 'text-warning' : 'text-success'}">
                ${ev.risk_score.toFixed(2)}
              </td>
              <td class="font-mono">${ev.severity.toFixed(2)}</td>
              <td class="font-mono ${ev.ttc_min !== null && ev.ttc_min < 2.0 ? 'text-critical font-bold' : ''}">
                ${ev.ttc_min !== null ? ev.ttc_min.toFixed(2) + 's' : '<span class="text-muted">N/A</span>'}
              </td>
              <td class="font-mono">
                ${ev.pet_min !== null ? ev.pet_min.toFixed(2) + 's' : '<span class="text-muted">N/A</span>'}
              </td>
              <td class="font-mono text-xs text-muted">${ev.start_time.toFixed(1)}s – ${ev.end_time.toFixed(1)}s</td>
              <td>
                <span class="badge ${ev.evidence_available ? 'badge-success' : 'badge-muted'} badge-xs">
                  ${ev.evidence_available ? 'VIDEO CLIP' : 'COORDINATES'}
                </span>
              </td>
              <td>
                <button class="btn btn-ghost btn-xs">Inspect →</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}
