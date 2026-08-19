/// <reference types="node" />
// Client-server runtime configuration. Everything is env-overridable so the
// demo launcher (demo) can start the component on other ports without
// editing code.

import { RELAY_DEFAULT_PORT, RELAY_PEER_PATH, RELAY_STREAM_PATH } from '../relayProtocol.ts';

export type ServeMode = 'dev' | 'prod';

export interface ClientServerConfig {
  /** HTTP + WS listen port. */
  port: number;
  host: string;
  /** Upstream core viewer socket. */
  coreUrl: string;
  /** dev = reverse-proxy the Vite dev server; prod = serve the built dist/. */
  mode: ServeMode;
  /** Vite dev server origin (dev mode only). */
  viteUrl: string;
  /** Built asset root (prod mode only). */
  distDir: string;
  /** Board WebSocket path. */
  streamPath: string;
  /** PeerJS signalling mount path. */
  peerPath: string;
  /** Reconnect backoff bounds for the upstream link, ms. */
  backoffMinMs: number;
  backoffMaxMs: number;
}

function num(name: string, fallback: number): number {
  const raw = process.env[name];
  const n = raw ? Number(raw) : NaN;
  return Number.isFinite(n) ? n : fallback;
}

function str(name: string, fallback: string): string {
  const raw = process.env[name];
  return raw && raw.trim() ? raw.trim() : fallback;
}

export function loadConfig(): ClientServerConfig {
  const mode = str('SKYLENS_CLIENT_MODE', 'dev') === 'prod' ? 'prod' : 'dev';
  return {
    port: num('SKYLENS_CLIENT_PORT', RELAY_DEFAULT_PORT),
    host: str('SKYLENS_CLIENT_HOST', '0.0.0.0'),
    coreUrl: str('SKYLENS_CORE_WS', 'ws://localhost:8080/viewer'),
    mode,
    viteUrl: str('SKYLENS_VITE_URL', 'http://localhost:5173'),
    distDir: str('SKYLENS_CLIENT_DIST', 'dist'),
    streamPath: RELAY_STREAM_PATH,
    peerPath: RELAY_PEER_PATH,
    backoffMinMs: num('SKYLENS_CLIENT_BACKOFF_MIN', 500),
    backoffMaxMs: num('SKYLENS_CLIENT_BACKOFF_MAX', 8000),
  };
}
