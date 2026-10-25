// frontend/src/components/sidebar.js
// Navigation Sidebar with clean spacing and guided walkthrough

import { store } from '../state/store.js';
import { icons } from './icons.js';

export function renderSidebar() {
  const state = store.getState();
  const sidebar = document.createElement('aside');
  sidebar.className = 'app-sidebar glass-panel';

  const navItems = [
    {
      id: 'overview',
      label: 'Command Center',
      sub: 'Session Overview & KPIs',
      icon: icons.activity(18),
      badge: 'LIVE'
    },
    {
      id: 'hotspots',
      label: 'Spatial Hotspots',
      sub: 'Camera Grid & GIS Map',
      icon: icons.mapPin(18),
      badge: '3 ZONES'
    },
    {
      id: 'events',
      label: 'Conflict Events',
      sub: 'TTC / PET Encounters',
      icon: icons.shieldAlert(18),
      badge: '8 DETECTED'
    },
    {
      id: 'event-detail',
      label: 'Event Forensics',
      sub: 'Video Clip & Trajectory',
      icon: icons.video(18),
      badge: state.selectedEventId || 'EVT_001'
    },
    {
      id: 'agent37',
      label: 'Agent37 Intelligence',
      sub: 'Findings & Interventions',
      icon: icons.brain(18),
      badge: 'v37.0'
    },
    {
      id: 'traceability',
      label: 'Evidence Traceability',
      sub: 'End-to-End Lineage',
      icon: icons.gitMerge(18),
      badge: 'AUDIT'
    }
  ];

  sidebar.innerHTML = `
    <div class="sidebar-nav-container">
      <div class="nav-section-title">NAVIGATION</div>
      <nav class="sidebar-nav">
        ${navItems.map(item => `
          <button class="nav-item-btn ${state.currentView === item.id ? 'active' : ''}" data-nav-id="${item.id}">
            <div class="nav-item-icon">${item.icon}</div>
            <div class="nav-item-text">
              <span class="nav-item-title">${item.label}</span>
              <span class="nav-item-sub">${item.sub}</span>
            </div>
            ${item.badge ? `<span class="nav-item-badge">${item.badge}</span>` : ''}
          </button>
        `).join('')}
      </nav>

      <!-- Guided Walkthrough -->
      <div class="sidebar-demo-box glass-panel-nested">
        <div class="demo-box-header">
          <span class="demo-box-title">HACKATHON WALKTHROUGH</span>
        </div>
        <p class="demo-box-desc">
          Guided 6-step walkthrough demonstrating the complete near-miss to intervention flow.
        </p>
        <button id="btn-start-demo-walkthrough" class="btn btn-outline-cyan btn-sm btn-block mt-2">
          Start Walkthrough
        </button>
      </div>

      <!-- Specs Footer -->
      <div class="sidebar-footer-info font-mono">
        <div>PIPELINE: YOLOv8 + ByteTrack</div>
        <div>METRICS: TTC + PET + Kinematics</div>
        <div>ENGINE: Agent37 Cloud Sandbox</div>
      </div>
    </div>
  `;

  // Attach navigation click listeners
  sidebar.querySelectorAll('.nav-item-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const navId = btn.getAttribute('data-nav-id');
      if (navId) store.navigateTo(navId);
    });
  });

  // Attach Hackathon Demo Walkthrough
  sidebar.querySelector('#btn-start-demo-walkthrough')?.addEventListener('click', () => {
    startDemoWalkthrough();
  });

  return sidebar;
}

// 6-Step Guided Narrative
function startDemoWalkthrough() {
  const steps = [
    {
      view: 'overview',
      title: 'Step 1/6: Command Center & Overall Situation',
      msg: 'Observe the session overview: Critical Risk Level (0.85 peak ML risk), 8 detected conflict interactions, and 3 spatial hotspots.',
      action: () => store.navigateTo('overview')
    },
    {
      view: 'hotspots',
      title: 'Step 2/6: Identify Highest-Priority Hotspot',
      msg: 'Navigating to Hotspot Map: Zone R1C2 is prioritized #1 with 3 conflict events, 2 high-severity encounters, and minimum TTC of 1.15s.',
      action: () => store.selectHotspot('R1C2')
    },
    {
      view: 'events',
      title: 'Step 3/6: Inspect Conflicts in Zone R1C2',
      msg: 'Filter database to Zone R1C2: Observe the dominant pedestrian-vehicle conflict pattern and low TTC metrics.',
      action: () => store.filterEventsByZone('R1C2')
    },
    {
      view: 'event-detail',
      title: 'Step 4/6: Forensic Deep-Dive into EVT_001',
      msg: 'Inspect EVT_001: TTC 1.15s, 58% evasive speed drop, play the HD evidence clip, and inspect the trajectory collision course.',
      action: () => store.selectEvent('EVT_001')
    },
    {
      view: 'agent37',
      title: 'Step 5/6: Agent37 AI Safety Intelligence & Interventions',
      msg: 'Examine Agent37 Report: Key Finding F001, Urgent Recommendation R001 (Pedestrian Crossing Assessment), and explicit system limitations.',
      action: () => store.navigateTo('agent37')
    },
    {
      view: 'traceability',
      title: 'Step 6/6: End-to-End Evidence Traceability',
      msg: 'Traceability Lineage: Verify that Recommendation R001 traces back through Finding F001, Event EVT_001, to Hotspot Zone R1C2!',
      action: () => store.navigateTo('traceability', { eventId: 'EVT_001', zoneId: 'R1C2' })
    }
  ];

  let currentStep = 0;

  function showModal() {
    const existing = document.getElementById('demo-guide-modal');
    if (existing) existing.remove();

    const step = steps[currentStep];
    step.action();

    const modal = document.createElement('div');
    modal.id = 'demo-guide-modal';
    modal.className = 'demo-modal-toast glass-panel animate-slide-up';
    modal.innerHTML = `
      <div class="dmt-header">
        <span class="badge badge-cyan font-mono">${step.title}</span>
        <button id="dmt-close-btn" class="btn btn-ghost btn-xs">✕</button>
      </div>
      <p class="dmt-text">${step.msg}</p>
      <div class="dmt-actions">
        ${currentStep > 0 ? `<button id="dmt-prev-btn" class="btn btn-ghost btn-xs">Back</button>` : ''}
        <button id="dmt-next-btn" class="btn btn-primary btn-xs">
          ${currentStep < steps.length - 1 ? 'Next Step →' : 'Finish ✓'}
        </button>
      </div>
    `;

    document.body.appendChild(modal);

    modal.querySelector('#dmt-close-btn')?.addEventListener('click', () => modal.remove());
    modal.querySelector('#dmt-prev-btn')?.addEventListener('click', () => {
      if (currentStep > 0) {
        currentStep--;
        showModal();
      }
    });
    modal.querySelector('#dmt-next-btn')?.addEventListener('click', () => {
      if (currentStep < steps.length - 1) {
        currentStep++;
        showModal();
      } else {
        modal.remove();
      }
    });
  }

  showModal();
}
