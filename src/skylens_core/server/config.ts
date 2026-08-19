// Core configuration. Everything is env-driven so `demo` can bring the
// component up without editing files.
//
// Naming note: SKYLENS_CORE_ENDPOINTS belongs to the proxy (it lists the core
// addresses it dials). Anything the core itself owns is SKYLENS_CORE_*.

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

/** How the control tower UI is served (COMPONENTS.md §3.4). */
export type WebMode = 'dev' | 'prod' | 'off';

export interface CoreConfig {
  host: string;
  port: number;
  /** ws path the proxy (or a drone in dev) pushes Envelope<UplinkMessage> to. */
  uplinkPath: string;
  /** ws path viewers subscribe on (see distributor.ts for the WebRTC seam). */
  viewerPath: string;

  /** Base URL of skylens_model's FastAPI app. */
  modelUrl: string;
  /** Poll interval for GET /jobs/{id}. */
  modelPollMs: number;
  /** Give up on a single job after this long. */
  modelJobTimeoutMs: number;
  /** Backoff before a job that failed on transport is re-queued. */
  modelRetryMs: number;
  /** Attempts per job before it is abandoned (the ladder moves on). */
  modelMaxAttempts: number;

  /** Prebuilt assets instead of training. Scheduling is IDENTICAL either way. */
  demo: boolean;

  /** Route arc-length that makes one reconstruction segment, in meters. */
  segmentMeters: number;
  /** Where this core is, `lat,lon[,alt]`. Empty falls back to siteFallback. */
  site: string | undefined;
  /** Ask a geo-IP service when `site` is unset. OFF by default: it locates the
   *  ISP's public IP, not the machine — see site.ts. */
  siteLookup: boolean;
  siteLookupUrl: string;
  siteLookupTimeoutMs: number;
  /** Used when neither the config nor a lookup answers. */
  siteFallback: { lat: number; lon: number; alt: number };
  /** Recon jobs the core keeps in flight at once. >= 2 is what makes the
   *  refinement of segment k overlap the first level of segment k+1. */
  reconConcurrency: number;
  /** Detection lane runs beside the recon lane and never blocks it. */
  detectConcurrency: number;
  /** Training steps per ladder level, ascending (보고서 표 9). */
  levelSteps: number[];
  /** Run detection jobs at all. */
  detect: boolean;

  /** "태스크 지정 완료" dwell before the awaiting-drone screen. */
  assignedHoldMs: number;
  /** Countdown shown while waiting for the drone (데모 시나리오 4: 10초). */
  droneEtaSeconds: number;

  /** ServerStatus / MissionStatus push period. */
  statusMs: number;
  /** Telemetry samples kept per drone. */
  telemetryHistory: number;

  webMode: WebMode;
  /** dev: the Vite dev server the core reverse-proxies to. */
  webTarget: string;
  /** prod: the built directory served statically. */
  webDist: string;
}

export function loadConfig(): CoreConfig {
  const rawWeb = str('SKYLENS_CORE_WEB_MODE', 'dev').toLowerCase();
  const webMode: WebMode = rawWeb === 'prod' || rawWeb === 'off' ? rawWeb : 'dev';
  if (rawWeb !== webMode) {
    console.warn(`[core] unknown SKYLENS_CORE_WEB_MODE="${rawWeb}", falling back to dev`);
  }

  const levels = str('SKYLENS_CORE_LEVEL_STEPS', '1000,7000,30000')
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0)
    .sort((a, b) => a - b);

  return {
    host: str('SKYLENS_CORE_HOST', '0.0.0.0'),
    port: num('SKYLENS_CORE_PORT', 8080),
    uplinkPath: str('SKYLENS_CORE_UPLINK_PATH', '/uplink'),
    viewerPath: str('SKYLENS_CORE_VIEWER_PATH', '/viewer'),

    modelUrl: str('SKYLENS_MODEL_URL', 'http://localhost:8100').replace(/\/+$/, ''),
    modelPollMs: num('SKYLENS_CORE_MODEL_POLL_MS', 500),
    modelJobTimeoutMs: num('SKYLENS_CORE_MODEL_JOB_TIMEOUT_MS', 15 * 60_000),
    modelRetryMs: num('SKYLENS_CORE_MODEL_RETRY_MS', 3000),
    modelMaxAttempts: num('SKYLENS_CORE_MODEL_MAX_ATTEMPTS', 20),

    demo: bool('SKYLENS_DEMO', false),

    segmentMeters: num('SKYLENS_CORE_SEGMENT_METERS', 40),
    site: process.env.SKYLENS_CORE_SITE,
    siteLookup: bool('SKYLENS_CORE_SITE_LOOKUP', false),
    siteLookupUrl: str(
      'SKYLENS_CORE_SITE_LOOKUP_URL',
      'http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country',
    ),
    siteLookupTimeoutMs: num('SKYLENS_CORE_SITE_LOOKUP_MS', 2500),
    // Daejeon — the demo operating area (src/test/geography.spec.ts pins the
    // anchor, the tower's default map and the drone waypoints together).
    siteFallback: {
      lat: num('SKYLENS_CORE_SITE_LAT', 36.3685),
      lon: num('SKYLENS_CORE_SITE_LON', 127.3475),
      alt: num('SKYLENS_CORE_SITE_ALT', 30),
    },
    reconConcurrency: Math.max(1, num('SKYLENS_CORE_RECON_CONCURRENCY', 2)),
    detectConcurrency: Math.max(1, num('SKYLENS_CORE_DETECT_CONCURRENCY', 1)),
    levelSteps: levels.length > 0 ? levels : [1000, 7000, 30000],
    detect: bool('SKYLENS_CORE_DETECT', true),

    assignedHoldMs: num('SKYLENS_CORE_ASSIGNED_HOLD_MS', 2000),
    droneEtaSeconds: num('SKYLENS_CORE_DRONE_ETA_SEC', 10),

    statusMs: num('SKYLENS_CORE_STATUS_MS', 1000),
    telemetryHistory: num('SKYLENS_CORE_TELEMETRY_HISTORY', 600),

    webMode,
    webTarget: str('SKYLENS_CORE_WEB_TARGET', 'http://127.0.0.1:5173').replace(/\/+$/, ''),
    webDist: str('SKYLENS_CORE_WEB_DIST', 'dist'),
  };
}
