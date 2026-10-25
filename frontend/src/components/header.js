// frontend/src/components/header.js
// Clean, balanced application header for B4DCrash

import { store } from '../state/store.js';
import { api } from '../api.js';
import { icons } from './icons.js';

export function renderHeader() {
  const state = store.getState();
  const header = document.createElement('header');
  header.className = 'app-header glass-panel';

  header.innerHTML = `
    <div class="header-left">
      <div class="brand-group clickable" id="brand-logo-btn">
        <div class="brand-badge-box">B4D</div>
        <div class="brand-text-group">
          <div class="brand-title">B4DCrash</div>
          <div class="brand-subtitle">PRE-COLLISION SAFETY INTELLIGENCE</div>
        </div>
      </div>
    </div>

    <div class="header-center">
      <div class="telemetry-chip font-mono">
        <span class="tc-dot live-beacon-dot"></span>
        <span class="tc-label">SESSION:</span>
        <span class="tc-val">URBAN_INTERSECTION_01</span>
      </div>
      <div class="telemetry-chip font-mono">
        <span class="tc-label">PEAK RISK:</span>
        <span class="tc-val text-critical font-bold">0.85 (CRITICAL)</span>
      </div>
    </div>

    <div class="header-right">
      <!-- Scenario Simulator for State Auditing -->
      <div class="scenario-simulator-group">
        <label class="sim-label">SIMULATE:</label>
        <select id="scenario-selector" class="form-select form-select-xs font-mono">
          <option value="none" ${state.simulationState === 'none' ? 'selected' : ''}>Normal (Loaded)</option>
          <option value="empty_events" ${state.simulationState === 'empty_events' ? 'selected' : ''}>Empty Events</option>
          <option value="empty_hotspots" ${state.simulationState === 'empty_hotspots' ? 'selected' : ''}>Empty Hotspots</option>
          <option value="api_error" ${state.simulationState === 'api_error' ? 'selected' : ''}>API Unavailable (503)</option>
          <option value="no_report" ${state.simulationState === 'no_report' ? 'selected' : ''}>No Agent Report</option>
        </select>
      </div>

      <!-- Mode Toggle (Live vs Demo) -->
      <button class="api-mode-badge ${state.isLiveApi ? 'mode-live' : 'mode-demo'}" id="api-mode-toggle" title="Toggle between Live API and Demo Data">
        <span class="mode-beacon"></span>
        <span class="mode-text">${state.isLiveApi ? 'LIVE API (8000)' : 'DEMO MODE'}</span>
      </button>
    </div>
  `;

  // Attach logo click
  header.querySelector('#brand-logo-btn')?.addEventListener('click', () => {
    store.navigateTo('overview');
  });

  // Attach scenario simulator
  header.querySelector('#scenario-selector')?.addEventListener('change', (e) => {
    const simState = e.target.value;
    api.setSimulationState(simState);
    store.setSimulation(simState);
  });

  // Attach mode toggle
  header.querySelector('#api-mode-toggle')?.addEventListener('click', async () => {
    const nextMode = state.isLiveApi ? 'demo' : 'live';
    api.setMode(nextMode);
    const health = await api.checkHealth();
    store.setLiveStatus(health.isLive && nextMode === 'live');
  });

  return header;
}
