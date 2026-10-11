// Placeholder 3D scene data — stands in for the real Gaussian splat reconstruction.
// PROJECT.md §2: both viewers consume ONE source; viewer 1 downsamples it into
// low-fi dots, viewer 2 renders it fully with progressive reveal.
//
// Textures / real .ply splats are deferred (per request). We procedurally build a
// point cloud that reads as a collapsed multi-storey structure with rubble, so the
// reveal choreography and coordinate frame are real and swappable for a splat later.

import * as THREE from 'three';

export interface SceneData {
  /** Full-resolution point positions (Float32, xyz interleaved). */
  positions: Float32Array;
  /** Per-point base color (Float32, rgb interleaved, 0..1). */
  colors: Float32Array;
  count: number;
  /** Axis-aligned bounds of the cloud. */
  bounds: THREE.Box3;
  /** Approximate ground plane Y (for marker grounding). */
  groundY: number;
}

// Deterministic PRNG so the scene is stable across reloads (no Math.random).
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const GROUND_Y = 0;

/**
 * Build the placeholder cloud. World frame: X right, Y up, Z forward.
 * Footprint roughly [-24..24] x [-24..24], height up to ~18.
 */
export function buildSceneData(pointCount = 120_000): SceneData {
  const rand = mulberry32(0x5ce7e);
  const pos = new Float32Array(pointCount * 3);
  const col = new Float32Array(pointCount * 3);
  const bounds = new THREE.Box3(
    new THREE.Vector3(Infinity, Infinity, Infinity),
    new THREE.Vector3(-Infinity, -Infinity, -Infinity),
  );

  const dust = new THREE.Color(0xb8b0a2);
  const concrete = new THREE.Color(0x8f8b83);
  const rebar = new THREE.Color(0x6b5a48);
  const scorch = new THREE.Color(0x40352c);
  const tmp = new THREE.Vector3();
  const c = new THREE.Color();

  // A few collapsed "slabs" (tilted boxes) + a rubble field + ground.
  const slabs = [
    { center: [-8, 5, -6], size: [14, 1.2, 12], tilt: 0.18 },
    { center: [9, 8, 3], size: [12, 1.0, 14], tilt: -0.25 },
    { center: [-2, 11, 8], size: [10, 1.0, 9], tilt: 0.32 },
    { center: [6, 3, -12], size: [9, 1.4, 8], tilt: 0.1 },
  ];

  for (let i = 0; i < pointCount; i++) {
    const r = rand();
    let x: number, y: number, z: number;

    if (r < 0.55) {
      // On a collapsed slab.
      const slab = slabs[(rand() * slabs.length) | 0];
      const [cx, cy, cz] = slab.center;
      const [sx, sy, sz] = slab.size;
      const lx = (rand() - 0.5) * sx;
      const ly = (rand() - 0.5) * sy;
      const lz = (rand() - 0.5) * sz;
      // Apply tilt about Z axis.
      const ct = Math.cos(slab.tilt);
      const st = Math.sin(slab.tilt);
      x = cx + lx * ct - ly * st;
      y = cy + lx * st + ly * ct;
      z = cz + lz;
      c.copy(rand() < 0.15 ? scorch : concrete);
    } else if (r < 0.85) {
      // Rubble mound near ground.
      const ang = rand() * Math.PI * 2;
      const rad = Math.pow(rand(), 0.6) * 22;
      x = Math.cos(ang) * rad;
      z = Math.sin(ang) * rad;
      const mound = Math.max(0, 6 - rad * 0.22) * rand();
      y = GROUND_Y + mound + rand() * 0.6;
      c.copy(rand() < 0.2 ? rebar : dust);
    } else {
      // Ground scatter.
      x = (rand() - 0.5) * 48;
      z = (rand() - 0.5) * 48;
      y = GROUND_Y + rand() * 0.25;
      c.copy(dust).multiplyScalar(0.8 + rand() * 0.2);
    }

    const j = i * 3;
    pos[j] = x;
    pos[j + 1] = y;
    pos[j + 2] = z;
    col[j] = c.r;
    col[j + 1] = c.g;
    col[j + 2] = c.b;
    bounds.expandByPoint(tmp.set(x, y, z));
  }

  return { positions: pos, colors: col, count: pointCount, bounds, groundY: GROUND_Y };
}
