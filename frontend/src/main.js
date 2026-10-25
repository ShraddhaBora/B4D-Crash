// frontend/src/main.js
// Application bootstrap, router, and view orchestrator.

import './style.css';
import { store } from './state/store.js';
import { api } from './api.js';
import { renderHeader } from './components/header.js';
import { renderSidebar } from './components/sidebar.js';

import { renderOverviewView } from './views/overviewView.js';
import { renderHotspotMapView } from './views/hotspotMapView.js';
import { renderEventsView } from './views/eventsView.js';
import { renderEventDetailsView } from './views/eventDetailsView.js';
import { renderAgentIntelligenceView } from './views/agentIntelligenceView.js';
import { renderTraceabilityView } from './views/traceabilityView.js';

class Application {
  constructor() {
    this.root = document.getElementById('app');
    this.contentContainer = null;
    this.currentViewInstance = null;
  }

  async init() {
    // 1. Initial health check of the backend API
    const health = await api.checkHealth();
    store.setLiveStatus(health.isLive);

    // 2. Hash routing map
    const hashMap = {
      '#overview': 'overview',
      '#hotspots': 'hotspots',
      '#events': 'events',
      '#forensics': 'event-detail',
      '#event-detail': 'event-detail',
      '#intelligence': 'agent37',
      '#agent37': 'agent37',
      '#traceability': 'traceability'
    };

    const initialHash = window.location.hash;
    if (initialHash && hashMap[initialHash]) {
      store.setState({ currentView: hashMap[initialHash] });
    }

    window.addEventListener('hashchange', () => {
      const h = window.location.hash;
      if (h && hashMap[h]) {
        if (store.getState().currentView !== hashMap[h]) {
          store.setState({ currentView: hashMap[h] });
        }
      }
    });

    // 3. Render static layout shell
    this.renderLayout();

    // 4. Subscribe to reactive state updates
    store.subscribe((state) => {
      // Keep hash in sync
      const viewToHash = {
        'overview': '#overview',
        'hotspots': '#hotspots',
        'events': '#events',
        'event-detail': '#forensics',
        'agent37': '#intelligence',
        'traceability': '#traceability'
      };
      if (viewToHash[state.currentView] && window.location.hash !== viewToHash[state.currentView]) {
        window.location.hash = viewToHash[state.currentView];
      }
      this.handleStateChange(state);
    });

    // 5. Render initial view
    await this.renderCurrentView();
  }

  renderLayout() {
    this.root.innerHTML = '';

    // Header
    const header = renderHeader();
    this.root.appendChild(header);

    // Body with Sidebar and Content Container
    const body = document.createElement('div');
    body.className = 'app-body';

    const sidebar = renderSidebar();
    body.appendChild(sidebar);

    this.contentContainer = document.createElement('main');
    this.contentContainer.className = 'app-content';
    this.contentContainer.id = 'main-content-viewport';
    body.appendChild(this.contentContainer);

    this.root.appendChild(body);
  }

  async renderCurrentView() {
    if (!this.contentContainer) return;

    const state = store.getState();
    this.contentContainer.innerHTML = '';

    let viewEl = null;
    switch (state.currentView) {
      case 'overview':
        viewEl = await renderOverviewView();
        break;
      case 'hotspots':
        viewEl = await renderHotspotMapView();
        break;
      case 'events':
        viewEl = await renderEventsView();
        break;
      case 'event-detail':
        viewEl = await renderEventDetailsView();
        break;
      case 'agent37':
        viewEl = await renderAgentIntelligenceView();
        break;
      case 'traceability':
        viewEl = await renderTraceabilityView();
        break;
      default:
        viewEl = await renderOverviewView();
    }

    if (viewEl) {
      viewEl.classList.add('animate-fade-in');
      this.contentContainer.appendChild(viewEl);
    }
  }

  handleStateChange(state) {
    // Re-render layout header & sidebar to update active indicators
    const currentHeader = this.root.querySelector('.app-header');
    if (currentHeader) {
      const newHeader = renderHeader();
      currentHeader.replaceWith(newHeader);
    }

    const currentSidebar = this.root.querySelector('.app-sidebar');
    if (currentSidebar) {
      const newSidebar = renderSidebar();
      currentSidebar.replaceWith(newSidebar);
    }

    this.renderCurrentView();
  }
}

// Bootstrap application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new Application();
  app.init().catch(err => console.error('Failed to initialize application:', err));
});
