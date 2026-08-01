// Global constants for the SkyLens prototype.
// Tuning knobs for look, timing, and the two-viewer choreography (see PROJECT.md).

export const CONFIG = {
  // Simulation clock
  sim: {
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
    /** Distance (world units) the recon camera holds from a focused detection. */
    focusDistance: 12.0,
    /** Smoothing factor for the SYNCED follow (0..1 per frame at 60fps). */
    syncLerp: 0.08,
  },

  // Real Gaussian-splat asset — the SINGLE SOURCE both viewers derive from
  // (SIM = low-fi point cloud of the splat, RECON = full splat render). It is a
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
  },

  // Palette — viewer 1 is deliberately low-fi / cold; viewer 2 warmer/real.
  color: {
    simBg: 0x0a1128,
    simPoint: 0x3fd0e0,
    simPointFar: 0x2a4a8a,
    droneCore: 0x9fe8ff,
    droneTrail: 0x4a90d9,
    reconBg: 0x0d0f14,
    reconPoint: 0xc8c2b8,
    markerDanger: 0xff4d4d,
    markerPerson: 0x39d98a,
  },
} as const;

/** Drone id palette so viewer 1 can distinguish the swarm. */
export const DRONE_TINTS = [0x9fe8ff, 0xffd27f, 0xc79fff] as const;
