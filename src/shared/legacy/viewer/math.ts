// Small math helpers shared across modules.
import * as THREE from 'three';
import type { Vec3, Waypoint } from './types';

export function toVector3(v: Vec3): THREE.Vector3 {
  return new THREE.Vector3(v[0], v[1], v[2]);
}

export const clamp = (x: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, x));

/** Smoothstep-style ease-in-out on [0,1]. */
export const easeInOut = (t: number): number => {
  const c = clamp(t, 0, 1);
  return c * c * (3 - 2 * c);
};

/** Frame-rate independent damping factor. `lambda` is the per-second rate. */
export const dampFactor = (lambda: number, dt: number): number =>
  1 - Math.exp(-lambda * dt);

/**
 * Sample a Catmull-Rom spline through waypoint positions at time `t` (seconds).
 * Clamps to the path ends. Also returns a forward tangent. §4.1
 */
export function samplePath(
  waypoints: Waypoint[],
  t: number,
): { pos: THREE.Vector3; forward: THREE.Vector3 } {
  const n = waypoints.length;
  if (n === 0) {
    return { pos: new THREE.Vector3(), forward: new THREE.Vector3(0, 0, 1) };
  }
  if (n === 1) {
    return {
      pos: toVector3(waypoints[0].pos),
      forward: new THREE.Vector3(0, 0, 1),
    };
  }

  const t0 = waypoints[0].t;
  const tEnd = waypoints[n - 1].t;
  const clamped = clamp(t, t0, tEnd);

  // Find the segment [i, i+1] containing `clamped`.
  let i = 0;
  while (i < n - 2 && waypoints[i + 1].t < clamped) i++;

  const p1 = waypoints[i];
  const p2 = waypoints[i + 1];
  const seg = Math.max(1e-6, p2.t - p1.t);
  const local = clamp((clamped - p1.t) / seg, 0, 1);

  const v0 = toVector3(waypoints[Math.max(0, i - 1)].pos);
  const v1 = toVector3(p1.pos);
  const v2 = toVector3(p2.pos);
  const v3 = toVector3(waypoints[Math.min(n - 1, i + 2)].pos);

  const pos = catmullRom(v0, v1, v2, v3, local);
  const ahead = catmullRom(v0, v1, v2, v3, clamp(local + 0.01, 0, 1));
  const forward = ahead.sub(pos);
  if (forward.lengthSq() < 1e-8) forward.set(0, 0, 1);
  forward.normalize();

  return { pos, forward };
}

function catmullRom(
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  p2: THREE.Vector3,
  p3: THREE.Vector3,
  t: number,
): THREE.Vector3 {
  const t2 = t * t;
  const t3 = t2 * t;
  const out = new THREE.Vector3();
  out.addScaledVector(p0, -0.5 * t3 + t2 - 0.5 * t);
  out.addScaledVector(p1, 1.5 * t3 - 2.5 * t2 + 1);
  out.addScaledVector(p2, -1.5 * t3 + 2 * t2 + 0.5 * t);
  out.addScaledVector(p3, 0.5 * t3 - 0.5 * t2);
  return out;
}

/** Total duration (seconds) of a waypoint list. */
export function pathDuration(waypoints: Waypoint[]): number {
  if (waypoints.length === 0) return 0;
  return waypoints[waypoints.length - 1].t - waypoints[0].t;
}
