// frontend/src/state/store.js
// Reactive state manager for the Before the Crash dashboard.

class Store {
  constructor() {
    this.state = {
      currentView: 'overview', // 'overview', 'hotspots', 'events', 'event-detail', 'agent37', 'traceability'
      selectedEventId: 'EVT_001',
      selectedZoneId: 'R1C2',
      selectedFindingId: null,
      selectedRecommendationId: null,
      traceabilityFilter: {
        zoneId: null,
        eventId: null,
        findingId: null,
        recommendationId: null
      },
      filters: {
        searchQuery: '',
        conflictType: 'ALL',
        zoneId: 'ALL',
        severity: 'ALL',
        minRisk: 0.0
      },
      eventsViewMode: 'cards', // 'cards' | 'table'
      isLiveApi: false,
      simulationState: 'none', // 'none', 'empty_events', 'empty_hotspots', 'api_error'
      lastError: null,
      isLoading: false
    };

    this.listeners = new Set();
  }

  getState() {
    return this.state;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify() {
    for (const listener of this.listeners) {
      try {
        listener(this.state);
      } catch (err) {
        console.error('Store listener error:', err);
      }
    }
  }

  setState(partial) {
    this.state = { ...this.state, ...partial };
    this.notify();
  }

  // Navigation actions
  navigateTo(view, params = {}) {
    const update = { currentView: view };
    if (params.eventId) update.selectedEventId = params.eventId;
    if (params.zoneId) update.selectedZoneId = params.zoneId;
    if (params.findingId) update.selectedFindingId = params.findingId;
    if (params.recommendationId) update.selectedRecommendationId = params.recommendationId;
    this.setState(update);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  selectEvent(eventId, navigate = true) {
    const update = { selectedEventId: eventId };
    if (navigate) update.currentView = 'event-detail';
    this.setState(update);
    if (navigate) window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  selectHotspot(zoneId, navigate = true) {
    const update = { selectedZoneId: zoneId };
    if (navigate) update.currentView = 'hotspots';
    this.setState(update);
    if (navigate) window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  filterEventsByZone(zoneId) {
    this.setState({
      currentView: 'events',
      filters: {
        ...this.state.filters,
        zoneId: zoneId
      }
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  filterEventsByType(type) {
    this.setState({
      currentView: 'events',
      filters: {
        ...this.state.filters,
        conflictType: type
      }
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  setFilters(partialFilters) {
    this.setState({
      filters: { ...this.state.filters, ...partialFilters }
    });
  }

  resetFilters() {
    this.setState({
      filters: {
        searchQuery: '',
        conflictType: 'ALL',
        zoneId: 'ALL',
        severity: 'ALL',
        minRisk: 0.0
      }
    });
  }

  setSimulation(simState) {
    this.setState({ simulationState: simState });
  }

  setLiveStatus(isLive) {
    this.setState({ isLiveApi: isLive });
  }
}

export const store = new Store();
