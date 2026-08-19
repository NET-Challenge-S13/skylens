// Drone runtime configuration.
//
// Resolved from a flat string map so the SAME code path works in every shell:
//   - Tauri / Node  -> process.env
//   - browser page  -> the query string (?demo&drone=2&gateway=ws://…)
//   - CLI           -> argv flags (--demo, --gateway=…)
//
// No DOM, no node: imports here — this module is imported by the pure core.

import type { DroneStation, LinkMode } from '../../shared/protocol.ts';
import type { Gps } from '../../shared/geo.ts';

export interface DroneConfig {
  droneId: number;
  /** Free-form model/firmware string shown on the operator panel. */
  model: string;
  /** relay = gateway forwards media; webrtc = gateway only brokers hole punching. */
  mode: LinkMode;
  /** Gateway signalling/uplink socket. */
  gatewayUrl: string;
  /** Replace the camera with the res/static/video footage and simulate the flight. */
  demo: boolean;

  /** Telemetry emissions per second. */
  telemetryHz: number;
  /** Cruise ground speed along the assigned route, m/s. */
  cruiseSpeed: number;
  /** Transit speed while flying from base to the first waypoint, m/s (demo). */
  transitSpeed: number;
  /**
   * How long the drone takes to reach the site after a route is assigned, ms.
   * COMPONENTS.md §5.2 step 4 — the 10 s "드론 연결 대기".
   */
  arrivalMs: number;
  /**
   * Which station this aircraft holds. It is also its NAME on the operator's
   * screens (LEFT / CENTER / RIGHT) and, in demo mode, which of the recorded
   * passes it plays — the footage was flown as a left, centre and right pass,
   * so the station maps onto real material.
   *
   * Every drone flies the same assigned route and offsets itself into its
   * station, so no aircraft depends on another.
   */
  station: DroneStation;
  /** Whether this aircraft transmits video. All three do by default: the
   *  operator can switch the camera panel between them. */
  capture: boolean;
  /** Slices the route is cut into per one-way traverse. One VideoSegment each. */
  slicesPerLeg: number;
  /** Battery drain, percent per minute of flight. */
  batteryDrainPerMin: number;
  /** Seconds of no stick input before a manually flown drone rejoins the route. */
  manualIdleReturn: number;

  /** Where the drone sits before it is dispatched. Null = derive from waypoint 0. */
  home: Gps | null;
  /** Ground distance from waypoint 0 to the derived home position, m. */
  homeOffsetM: number;

  /** Assign a built-in route at startup instead of waiting for the core. */
  autoRoute: boolean;
  /** Reconnect backoff bounds, ms. */
  reconnectMinMs: number;
  reconnectMaxMs: number;
  /** Announce DroneHello only once the drone is on station, not at connect. */
  helloOnArrival: boolean;
}

/** Gateway origin the demo launcher brings up (COMPONENTS.md §7). */
export const GATEWAY_ORIGIN = 'ws://127.0.0.1:8081';

/**
 * Path the gateway serves per mode. Mirrors `dronePath()` in
 * src/skylens_gateway/config.ts — the gateway rejects an upgrade on any other
 * path, so getting this wrong is an instant disconnect rather than a soft fail.
 */
export function dronePathFor(mode: LinkMode): string {
  return mode === 'webrtc' ? '/signal' : '/uplink';
}

/** Full URL for a mode, and the place where a bare origin gets its path. */
export function gatewayUrlFor(mode: LinkMode, origin = GATEWAY_ORIGIN): string {
  const trimmed = origin.replace(/\/+$/, '');
  // A URL that already names a path is taken as-is; a bare origin gets the
  // mode's path appended, so `--gateway=ws://10.0.0.4:8081` just works.
  const afterScheme = trimmed.slice(trimmed.indexOf('://') + 3);
  return afterScheme.includes('/') ? trimmed : trimmed + dronePathFor(mode);
}

export const DEFAULT_CONFIG: DroneConfig = {
  droneId: 1,
  model: 'SkyLens D1 (H.265 / 5G)',
  mode: 'relay',
  gatewayUrl: gatewayUrlFor('relay'),
  demo: false,

  telemetryHz: 5,
  cruiseSpeed: 12,
  transitSpeed: 25,
  arrivalMs: 10_000,
  station: 'center',
  capture: true,
  slicesPerLeg: 4,
  batteryDrainPerMin: 3.5,
  manualIdleReturn: 2.0,

  home: null,
  homeOffsetM: 250,

  autoRoute: false,
  reconnectMinMs: 500,
  reconnectMaxMs: 8_000,
  helloOnArrival: false,
};

/**
 * Waypoints used by --autoroute / the standalone browser page.
 *
 * `alt` is GPS altitude — above sea level, not above the ground. The ground
 * here (대전 유성) is about 58 m, so these sit ~60 m over it. Altitudes near 60
 * would put the aircraft ON the terrain: that is what made the fleet look like
 * it was flying somewhere other than the planned line, because it was rendered
 * inside the hills instead of over them.
 */
export const DEMO_ROUTE: Gps[] = [
  { lat: 36.3680, lon: 127.3460, alt: 120 },
  { lat: 36.3686, lon: 127.3468, alt: 126 },
  { lat: 36.3692, lon: 127.3476, alt: 126 },
  { lat: 36.3695, lon: 127.3487, alt: 118 },
];

/** Parse a station name, or null when the value is absent or not one of ours. */
function stationOf(v: string | undefined): DroneStation | null {
  if (v === 'left' || v === 'center' || v === 'right') return v;
  if (v !== undefined) console.warn(`[drone] unknown station "${v}", falling back`);
  return null;
}

function truthy(v: string | undefined): boolean {
  if (v === undefined) return false;
  const s = v.trim().toLowerCase();
  return s === '' || s === '1' || s === 'true' || s === 'on' || s === 'yes';
}

function num(v: string | undefined, fallback: number): number {
  if (v === undefined) return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Turn `--demo --gateway=ws://x --drone 3` into the same flat map that
 * process.env / the query string produce, using SKYLENS_* keys.
 */
export function envFromArgv(argv: readonly string[]): Record<string, string> {
  const alias: Record<string, string> = {
    demo: 'SKYLENS_DEMO',
    gateway: 'SKYLENS_GATEWAY_URL',
    drone: 'SKYLENS_DRONE_ID',
    id: 'SKYLENS_DRONE_ID',
    mode: 'SKYLENS_LINK_MODE',
    model: 'SKYLENS_DRONE_MODEL',
    hz: 'SKYLENS_DRONE_TELEMETRY_HZ',
    speed: 'SKYLENS_DRONE_SPEED',
    station: 'SKYLENS_DRONE_STATION',
    capture: 'SKYLENS_DRONE_CAPTURE',
    slices: 'SKYLENS_DRONE_SLICES',
    arrival: 'SKYLENS_DRONE_ARRIVAL_MS',
    battery: 'SKYLENS_DRONE_BATTERY_DRAIN',
    autoroute: 'SKYLENS_DRONE_AUTOROUTE',
    'hello-on-arrival': 'SKYLENS_DRONE_HELLO_ON_ARRIVAL',
  };
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const body = arg.slice(2);
    const eq = body.indexOf('=');
    let key = eq >= 0 ? body.slice(0, eq) : body;
    let value = eq >= 0 ? body.slice(eq + 1) : '';
    if (eq < 0) {
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('--')) {
        value = next;
        i++;
      } else {
        value = '1';
      }
    }
    key = key.toLowerCase();
    out[alias[key] ?? key] = value;
  }
  return out;
}

/** Query string (`?demo&drone=2`) in the same SKYLENS_* shape. */
export function envFromQuery(search: string): Record<string, string> {
  const argv: string[] = [];
  const q = search.startsWith('?') ? search.slice(1) : search;
  for (const part of q.split('&')) {
    if (!part) continue;
    const [k, v] = part.split('=');
    argv.push(v === undefined ? `--${decodeURIComponent(k)}` : `--${decodeURIComponent(k)}=${decodeURIComponent(v)}`);
  }
  return envFromArgv(argv);
}

/** Later sources win, so callers pass [processEnv, argvFlags]. */
export function resolveConfig(...sources: Array<Record<string, string | undefined>>): DroneConfig {
  const env: Record<string, string | undefined> = {};
  for (const s of sources) for (const [k, v] of Object.entries(s)) if (v !== undefined) env[k] = v;

  const mode: LinkMode = env.SKYLENS_LINK_MODE === 'webrtc' ? 'webrtc' : 'relay';
  return {
    ...DEFAULT_CONFIG,
    droneId: num(env.SKYLENS_DRONE_ID, DEFAULT_CONFIG.droneId),
    model: env.SKYLENS_DRONE_MODEL ?? DEFAULT_CONFIG.model,
    mode,
    gatewayUrl: gatewayUrlFor(mode, env.SKYLENS_GATEWAY_URL ?? GATEWAY_ORIGIN),
    demo: truthy(env.SKYLENS_DEMO),
    telemetryHz: num(env.SKYLENS_DRONE_TELEMETRY_HZ, DEFAULT_CONFIG.telemetryHz),
    cruiseSpeed: num(env.SKYLENS_DRONE_SPEED, DEFAULT_CONFIG.cruiseSpeed),
    transitSpeed: num(env.SKYLENS_DRONE_TRANSIT_SPEED, DEFAULT_CONFIG.transitSpeed),
    arrivalMs: num(env.SKYLENS_DRONE_ARRIVAL_MS, DEFAULT_CONFIG.arrivalMs),
    station: stationOf(env.SKYLENS_DRONE_STATION) ?? DEFAULT_CONFIG.station,
    // Wingmen opt OUT explicitly; a lone drone films by default.
    capture: env.SKYLENS_DRONE_CAPTURE === undefined ? DEFAULT_CONFIG.capture : truthy(env.SKYLENS_DRONE_CAPTURE),
    slicesPerLeg: Math.max(1, Math.round(num(env.SKYLENS_DRONE_SLICES, DEFAULT_CONFIG.slicesPerLeg))),
    batteryDrainPerMin: num(env.SKYLENS_DRONE_BATTERY_DRAIN, DEFAULT_CONFIG.batteryDrainPerMin),
    manualIdleReturn: num(env.SKYLENS_DRONE_MANUAL_IDLE, DEFAULT_CONFIG.manualIdleReturn),
    homeOffsetM: num(env.SKYLENS_DRONE_HOME_OFFSET_M, DEFAULT_CONFIG.homeOffsetM),
    reconnectMinMs: num(env.SKYLENS_DRONE_RECONNECT_MIN_MS, DEFAULT_CONFIG.reconnectMinMs),
    reconnectMaxMs: num(env.SKYLENS_DRONE_RECONNECT_MAX_MS, DEFAULT_CONFIG.reconnectMaxMs),
    autoRoute: truthy(env.SKYLENS_DRONE_AUTOROUTE),
    helloOnArrival: truthy(env.SKYLENS_DRONE_HELLO_ON_ARRIVAL),
  };
}
