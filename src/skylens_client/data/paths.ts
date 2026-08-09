// Drone paths (PROJECT.md §4.1). The scene comes from whatever splat is loaded,
// so the path is GENERATED to cover the actual scene footprint: a boustrophedon
// ("lawnmower") sweep over the whole XZ bounds at a fixed altitude.
//
// Swarm model: all three drones share this one full-scene sweep. The ACTIVE drone
// is the "leader" that flies it (or is steered); the other two cluster around the
// leader in formation (see pathFollower.ts). So the leader covers the whole scene
// and the swarm films it together — no gaps in the reveal.

import * as THREE from 'three';
import { CONFIG } from '../../skylens_core/config.ts';
import type { DronePath, Waypoint } from '../../skylens_core/types';

const DURATION = 30; // seconds for one full sweep
const ZONE_NAMES = ['리더 (조종)', '군집 #2', '군집 #3'];

export function buildDronePaths(bounds: THREE.Box3): DronePath[] {
  const size = new THREE.Vector3();
  bounds.getSize(size);
  const minX = bounds.min.x;
  const minZ = bounds.min.z;
  const maxZ = bounds.max.z;

  // Constant scan altitude above the scene top.
  const flightY = bounds.max.y + THREE.MathUtils.clamp(size.y * 0.15, 2, 8);
  // Line spacing < 2*radius so adjacent sweep lines' reveal discs overlap.
  const spacing = Math.max(2, CONFIG.reveal.radius * 1.6);
  const nLines = Math.max(2, Math.ceil(size.x / spacing));

  const raw: Array<[number, number]> = []; // [x, z] before timing
  for (let k = 0; k < nLines; k++) {
    const x = minX + ((k + 0.5) / nLines) * size.x;
    // Alternate sweep direction for a continuous lawnmower.
    if (k % 2 === 0) raw.push([x, minZ], [x, maxZ]);
    else raw.push([x, maxZ], [x, minZ]);
  }

  const n = raw.length;
  const waypoints: Waypoint[] = raw.map(([x, z], i): Waypoint => {
    const t = n > 1 ? (i / (n - 1)) * DURATION : 0;
    const next = raw[Math.min(n - 1, i + 1)];
    const dz = next[1] - z;
    return { t, pos: [x, flightY, z], look: [0, -1, dz >= 0 ? 0.3 : -0.3] };
  });

  // All drones reference the same full-scene sweep; only the leader flies it.
  return ZONE_NAMES.map((zone, d) => ({ id: d + 1, zone, waypoints }));
}
