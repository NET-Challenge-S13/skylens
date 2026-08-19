// Gateway configuration. Everything is env-driven so the demo launcher can bring
// the component up in either mode without editing files.

import process from 'node:process';
import type { LinkMode } from '../shared/protocol.ts';

function str(name: string, fallback: string): string {
  const v = process.env[name];
  return v === undefined || v === '' ? fallback : v;
}

function num(name: string, fallback: number): number {
  const v = process.env[name];
  if (v === undefined || v === '') return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export interface GatewayConfig {
  mode: LinkMode;
  port: number;
  host: string;
  /** Base ws:// URL of the proxy. Paths are derived from it. */
  proxyUrl: string;
  /** Where relayed uplink traffic goes. */
  proxyIngressUrl: string;
  /** Where hole-punch signalling is brokered with the proxy. */
  proxySignalUrl: string;
  /** Max frames held while the upstream is down, then drop-oldest. */
  maxQueue: number;
  /** Frames older than this are dropped even if the queue has room —
   *  stale telemetry is worse than no telemetry. */
  maxQueueAgeMs: number;
  reconnectMs: number;
  pingMs: number;
  statusMs: number;
}

function joinPath(base: string, path: string): string {
  return base.replace(/\/+$/, '') + path;
}

export function loadConfig(): GatewayConfig {
  const rawMode = str('SKYLENS_GATEWAY_MODE', 'relay').toLowerCase();
  const mode: LinkMode = rawMode === 'webrtc' ? 'webrtc' : 'relay';
  if (rawMode !== 'relay' && rawMode !== 'webrtc') {
    console.warn(`[gateway] unknown SKYLENS_GATEWAY_MODE="${rawMode}", falling back to relay`);
  }
  const proxyUrl = str('SKYLENS_PROXY_URL', 'ws://127.0.0.1:8082');
  return {
    mode,
    port: num('SKYLENS_GATEWAY_PORT', 8081),
    host: str('SKYLENS_GATEWAY_HOST', '0.0.0.0'),
    proxyUrl,
    proxyIngressUrl: str('SKYLENS_PROXY_INGRESS_URL', joinPath(proxyUrl, '/ingress')),
    proxySignalUrl: str('SKYLENS_PROXY_SIGNAL_URL', joinPath(proxyUrl, '/signal')),
    maxQueue: num('SKYLENS_GATEWAY_QUEUE', 256),
    maxQueueAgeMs: num('SKYLENS_GATEWAY_QUEUE_AGE_MS', 5000),
    reconnectMs: num('SKYLENS_GATEWAY_RECONNECT_MS', 1000),
    pingMs: num('SKYLENS_GATEWAY_PING_MS', 2000),
    statusMs: num('SKYLENS_GATEWAY_STATUS_MS', 1000),
  };
}

/** WebSocket path the drone connects to, per mode. */
export function dronePath(mode: LinkMode): string {
  return mode === 'webrtc' ? '/signal' : '/uplink';
}
