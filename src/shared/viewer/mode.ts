// Run mode. The app defaults to operating on real server-received data; the
// self-contained auto-demo (auto route, synthetic telemetry/detections/splats)
// is opt-in via ?demo. Single source of truth so CONTROL and STATUS agree.

export function isDemo(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).has('demo');
}
