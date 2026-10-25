// frontend/src/components/emptyState.js
// Reusable empty, error, and loading state components.

import { icons } from './icons.js';

export function renderEmptyState({
  title = 'No Data Available',
  message = 'There are no records matching your current criteria.',
  icon = 'info',
  actionLabel = null,
  actionCallback = null,
  badge = 'STATUS: EMPTY'
}) {
  const iconSvg = icons[icon] ? icons[icon](36) : icons.info(36);
  const containerId = 'empty-' + Math.random().toString(36).substring(2, 9);

  setTimeout(() => {
    if (actionLabel && actionCallback) {
      const btn = document.getElementById(`${containerId}-btn`);
      if (btn) btn.addEventListener('click', actionCallback);
    }
  }, 10);

  return `
    <div id="${containerId}" class="empty-state-card glass-panel animate-fade-in">
      <div class="empty-state-badge">${badge}</div>
      <div class="empty-state-icon">${iconSvg}</div>
      <h3 class="empty-state-title">${title}</h3>
      <p class="empty-state-message">${message}</p>
      ${actionLabel ? `
        <button id="${containerId}-btn" class="btn btn-primary btn-sm mt-3">
          ${actionLabel}
        </button>
      ` : ''}
    </div>
  `;
}

export function renderErrorState({
  title = 'System Alert / Service Disconnected',
  message = 'Unable to communicate with the road-safety analysis pipeline.',
  details = null,
  retryCallback = null
}) {
  const btnId = 'retry-' + Math.random().toString(36).substring(2, 9);

  if (retryCallback) {
    setTimeout(() => {
      const btn = document.getElementById(btnId);
      if (btn) btn.addEventListener('click', retryCallback);
    }, 10);
  }

  return `
    <div class="error-banner glass-panel border-danger animate-fade-in">
      <div class="error-icon text-danger">${icons.alertTriangle(28)}</div>
      <div class="error-content">
        <h4 class="error-title text-danger">${title}</h4>
        <p class="error-message">${message}</p>
        ${details ? `<pre class="error-details">${details}</pre>` : ''}
      </div>
      ${retryCallback ? `
        <button id="${btnId}" class="btn btn-outline-danger btn-sm">
          ${icons.refresh(14)} Reconnect
        </button>
      ` : ''}
    </div>
  `;
}

export function renderLoadingSkeleton({ title = 'Processing Telemetry...' }) {
  return `
    <div class="loading-skeleton-container glass-panel animate-pulse">
      <div class="loading-spinner"></div>
      <div class="loading-label">${title}</div>
      <div class="skeleton-bar" style="width: 70%; height: 16px; margin-top: 12px;"></div>
      <div class="skeleton-bar" style="width: 90%; height: 12px; margin-top: 8px;"></div>
      <div class="skeleton-bar" style="width: 50%; height: 12px; margin-top: 8px;"></div>
    </div>
  `;
}
