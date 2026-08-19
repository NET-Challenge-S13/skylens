// Proxy configuration. The interesting knob is SKYLENS_CORE_ENDPOINTS: a comma
// separated, PRIORITY ORDERED list. Index 0 is the preferred path; the rest are
// standbys kept warm for failover (COMPONENTS.md §3.3 "이중화·다중화").

import process from 'node:process';

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

function bool(name: string, fallback: boolean): boolean {
  const v = process.env[name];
  if (v === undefined || v === '') return fallback;
  return v !== '0' && v.toLowerCase() !== 'false' && v.toLowerCase() !== 'no';
}

export interface ProxyConfig {
  port: number;
  host: string;
  /** Priority-ordered core WebSocket endpoints. At least one. */
  coreEndpoints: string[];
  /** Endpoint advertised to a drone as its direct (post hole-punch) target. */
  publicDirectUrl: string;
  /** Return to a higher-priority path as soon as it is healthy again. */
  failback: boolean;
  /** How often each path is probed. */
  healthIntervalMs: number;
  /** No pong within this window ⇒ the path is unhealthy. */
  healthTimeoutMs: number;
  reconnectMs: number;
  /** Boot-time window in which a better-priority path may still be dialling. */
  startupGraceMs: number;
  maxQueue: number;
  maxQueueAgeMs: number;
  statusMs: number;
}

export function loadConfig(): ProxyConfig {
  const port = num('SKYLENS_PROXY_PORT', 8082);
  const endpoints = str(
    'SKYLENS_CORE_ENDPOINTS',
    // Primary is the core's documented address; the second is the standby line.
    'ws://127.0.0.1:8080/uplink,ws://127.0.0.1:8180/uplink',
  )
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  return {
    port,
    host: str('SKYLENS_PROXY_HOST', '0.0.0.0'),
    coreEndpoints: endpoints.length > 0 ? endpoints : ['ws://127.0.0.1:8080/uplink'],
    publicDirectUrl: str('SKYLENS_PROXY_PUBLIC_URL', `ws://127.0.0.1:${port}/direct`),
    failback: bool('SKYLENS_PROXY_FAILBACK', true),
    healthIntervalMs: num('SKYLENS_PROXY_HEALTH_INTERVAL_MS', 1000),
    healthTimeoutMs: num('SKYLENS_PROXY_HEALTH_TIMEOUT_MS', 3000),
    reconnectMs: num('SKYLENS_PROXY_RECONNECT_MS', 1000),
    startupGraceMs: num('SKYLENS_PROXY_STARTUP_GRACE_MS', 500),
    maxQueue: num('SKYLENS_PROXY_QUEUE', 512),
    maxQueueAgeMs: num('SKYLENS_PROXY_QUEUE_AGE_MS', 5000),
    statusMs: num('SKYLENS_PROXY_STATUS_MS', 1000),
  };
}
