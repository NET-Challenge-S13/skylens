// Global constants for the SkyLens prototype.
// Tuning knobs for look, timing, and the two-viewer choreography (see PROJECT.md).

export const CONFIG = {
  // Real-world anchor: local ENU frame origin (see geo.ts). Default is a
  // plausible site; a real deployment sets this from the operation area.
  //
  // NOTE (control tower): the tower no longer uses this constant as its frame.
  // Its scene IS a real place (VWorld terrain), so the ENU origin is derived
  // from the loaded map bbox at runtime — see skylens_core/web/geoFrame.ts.
  geo: {
    anchor: { lat: 36.3685, lon: 127.3475, alt: 30 },
  },

  // Control tower (skylens_core/web) — the VWorld-only operator screen.
  control: {
    /**
     * Map preset loaded when the URL carries no `?map=` (terrainSource PRESETS).
     *
     * MUST cover `geo.anchor` above and the demo waypoints in
     * skylens_drone/core/config.ts — the tower renders real terrain at real
     * coordinates, so a mismatch puts the fleet outside the visible world.
     */
    defaultMap: 'daejeon' as 'seoul' | 'daejeon' | 'uljin' | 'uljinup' | 'gangneung',
    /** Building display option used before the operator picks one (COMPONENTS §4). */
    defaultDisplay: 'black' as 'points' | 'black' | 'aerial',
    /** localStorage key the settings panel persists the choice under. */
    settingsKey: 'skylens.control.settings.v1',
    /** Core viewer endpoint (protocol.ts ViewerMessage / ControlMessage). */
    corePort: 8080,
    corePath: '/viewer',
    /** Reconnect backoff bounds (ms) for coreLink. */
    reconnectMinMs: 800,
    reconnectMaxMs: 8000,
    /** Manual-control send rate (Hz) while keys are held. */
    manualSendHz: 10,
  },

  // Simulation clock
  clock: {
    /** Multiplier on real elapsed time; 1 = real-time playback. */
    speed: 1.0,
    /** Seconds of delay between a drone visiting a spot (viewer 1) and
     *  that spot being revealed in the reconstruction (viewer 2). §5.2 lag. */
    revealLagSeconds: 4.0,
  },

  // Progressive reveal (viewer 2)
  reveal: {
    /** Radius (world units) around each visited point that becomes visible. */
    radius: 6.0,
    /** Seconds for a freshly revealed chunk to fade 0 -> 1. */
    fadeSeconds: 1.2,
    /** Opacity of not-yet-scanned splats (0 = fully hidden). Kept >0 so the
     *  building is always faintly visible ("ghost") and brightens when scanned. */
    ghostOpacity: 0.16,
    /** Apply the progressive reveal MASK to the splat so it blooms in as drones
     *  scan. Override per-page with ?reveal=on / ?reveal=off. */
    splatMask: true,
  },

  // Drone path following / manual control (§4.2)
  drone: {
    /** Idle time (s) with no key input before a manually-flown drone returns to preset. */
    manualIdleReturn: 1.5,
    /** Manual translation speed (world units / s). */
    manualSpeed: 8.0,
    /** Manual altitude speed (world units / s). */
    manualAltitudeSpeed: 5.0,
    /** Manual yaw (turn) rate for left/right keys (radians / s). ~55°/s. */
    manualYawRate: 0.95,
    /** Return-tween duration (s) back onto the preset path. */
    returnTween: 1.2,
  },

  // Camera sync state machine (§8.3)
  camera: {
    /** FOCUSING / RETURNING tween duration (s). */
    tweenSeconds: 1.8,
    /** Distance (world units) the status camera holds from a focused detection. */
    focusDistance: 12.0,
    /** Smoothing factor for the SYNCED follow (0..1 per frame at 60fps). */
    syncLerp: 0.08,
  },

  // Real Gaussian-splat asset — the SINGLE SOURCE both viewers derive from
  // (CONTROL = low-fi point cloud of the splat, STATUS = full splat render). It is a
  // public sample (TEST asset) loaded at runtime from a CDN, NOT committed. The
  // fit transform is auto-computed from the splat's own bounds (see
  // sceneSource.ts). Swap `url` for your own capture later.
  splat: {
    enabled: true,
    // A clean outdoor scene (Mip-NeRF360 "garden"): a wooden table on a grass
    // lawn with a tree — verified to render clean + upright (needs the 180° flip,
    // the standard antimatter15 convention). Internet-photo "building" splats
    // were dominated by sky/foreground noise and abandoned.
    url: 'https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/garden/garden-7k.splat',
    /** Lighter scene used by e2e tests (~8.6 MB, loads fast). */
    urlLight: 'https://huggingface.co/datasets/stpete2/splat/resolve/main/church.splat',
    /** Our OWN capture (res/static/demo, not committed), cheapest level. Both
     *  viewers derive the shared point cloud + fit transform from it while the
     *  real geometry streams in segment by segment (see delayPattern). */
    demoPreview: '/res/static/demo/step00250_light.ply',
  },

  // Delay-pattern reconstruction stream (interim report Ⅱ-3-다).
  //
  // The pipeline is online: a flight SEGMENT is reconstructed as soon as its
  // frames land, at a low training-step LEVEL first, and refined afterwards —
  // while the NEXT segment starts its own low level. Staggering the levels
  // across segments is what puts a usable shape on the board seconds after the
  // drone passes instead of hours after the flight ends.
  delayPattern: {
    enabled: true,
    /** Segment × level assets + their cut geometry (split_segments.py). */
    manifest: '/res/static/demo/segments.json',
    /** Seconds before the first segment's level 1 lands. */
    firstSegmentDelay: 1.5,
    /** Seconds between consecutive segments being captured (drone moving on). */
    segmentPeriod: 7.0,
    /** Seconds from a segment being captured to each of its levels landing.
     *  Later entries deliberately overrun segmentPeriod: that overlap IS the
     *  delay pattern — segment k is still refining when k+1's level 1 arrives. */
    levelDelays: [0, 4, 12, 22],
  },

  // Palette — viewer 1 is deliberately low-fi / cold; viewer 2 warmer/real.
  color: {
    controlBg: 0x0a1128,
    controlPoint: 0x3fd0e0,
    controlPointFar: 0x2a4a8a,
    droneCore: 0x9fe8ff,
    droneTrail: 0x4a90d9,
    statusBg: 0x0d0f14,
    statusPoint: 0xc8c2b8,
    markerDanger: 0xff4d4d,
    markerPerson: 0x39d98a,
  },
} as const;

/** Drone id palette so viewer 1 can distinguish the swarm. */
export const DRONE_TINTS = [0x9fe8ff, 0xffd27f, 0xc79fff] as const;

/**
 * Map scenes span kilometers while the drone rig is sized for a tens-of-meters
 * splat scene. Shrinking ONLY the drone turns it into a dot, so instead the
 * rig, chase-camera offsets, and formation spread all shrink by this one factor
 * together. On screen that reads as the MAP being magnified: the drone keeps
 * its apparent size while terrain/buildings loom ~1/scale larger.
 * `?drone=<n>` overrides (smaller n = more map magnification).
 *
 * The control tower is ALWAYS a map scene now, so 0.15 is the default there;
 * `?splat`-based scenes (situation board) keep 1.0 via `?drone=1`.
 */
export function droneViewScale(): number {
  if (typeof window === 'undefined') return 0.15;
  const q = new URLSearchParams(window.location.search);
  const manual = parseFloat(q.get('drone') ?? '');
  if (Number.isFinite(manual) && manual > 0) return manual;
  return 0.15;
}
