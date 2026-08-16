// Progressive reveal field for viewer 2 (§5.2). Consumes the lagged
// `state.visited` trail from the drone module and blooms nearby points of the
// full point cloud into visibility over CONFIG.reveal.fadeSeconds.
//
// Semantics: an aerial scan reveals the vertical COLUMN beneath the drone, so
// proximity is measured horizontally (XZ) and ignores altitude. This makes
// reveal independent of how high the drones fly and of the scene's height —
// a point is revealed once a drone has passed over its ground footprint. Points
// are bucketed in a 2D (XZ) uniform grid so each new visited spot only tests
// points in nearby cells instead of the whole cloud.

import type { Visited } from '../../skylens_core/types';
import { CONFIG } from '../../skylens_core/config';
import { clamp } from '../../skylens_core/math';

const CELL_SIZE = CONFIG.reveal.radius * 2;

export class RevealField {
  /** Per-point reveal progress in [0,1], written into the geometry's aReveal attribute. */
  readonly progress: Float32Array;
  /** Per-point revealed bitmask — true once bloom has started (progress may still be < 1). */
  private readonly started: Uint8Array;
  private readonly positions: Float32Array;
  private readonly count: number;

  /** Uniform 2D grid: XZ cell key -> point indices, for fast spatial lookup. */
  private readonly grid = new Map<string, number[]>();

  /** How many entries of state.visited we've already consumed. */
  private consumedVisited = 0;

  constructor(positions: Float32Array, count: number) {
    this.positions = positions;
    this.count = count;
    this.progress = new Float32Array(count);
    this.started = new Uint8Array(count);
    this.buildGrid();
  }

  private cellKey(x: number, z: number): string {
    const cx = Math.floor(x / CELL_SIZE);
    const cz = Math.floor(z / CELL_SIZE);
    return `${cx}_${cz}`;
  }

  private buildGrid(): void {
    for (let i = 0; i < this.count; i++) {
      const j = i * 3;
      const key = this.cellKey(this.positions[j], this.positions[j + 2]);
      let bucket = this.grid.get(key);
      if (!bucket) {
        bucket = [];
        this.grid.set(key, bucket);
      }
      bucket.push(i);
    }
  }

  /**
   * Advance reveal state. `visitedLagged` is the trail already lagged upstream
   * (only entries with t <= now - lagSeconds), `dt` is frame delta. Returns true
   * if any progress values changed (so the caller marks the attribute dirty).
   */
  update(visitedLagged: Visited[], now: number, dt: number): boolean {
    let dirty = false;

    // Activate bloom for any newly-lagged-in visited spots (XZ column).
    if (this.consumedVisited < visitedLagged.length) {
      const radius = CONFIG.reveal.radius;
      const radiusSq = radius * radius;
      const cellRadius = Math.ceil(radius / CELL_SIZE);

      for (let vi = this.consumedVisited; vi < visitedLagged.length; vi++) {
        const spot = visitedLagged[vi];
        const sx = spot.pos[0];
        const sz = spot.pos[2];
        const cx = Math.floor(sx / CELL_SIZE);
        const cz = Math.floor(sz / CELL_SIZE);

        for (let gx = cx - cellRadius; gx <= cx + cellRadius; gx++) {
          for (let gz = cz - cellRadius; gz <= cz + cellRadius; gz++) {
            const bucket = this.grid.get(`${gx}_${gz}`);
            if (!bucket) continue;
            for (const pi of bucket) {
              if (this.started[pi]) continue;
              const j = pi * 3;
              const dx = this.positions[j] - sx;
              const dz = this.positions[j + 2] - sz;
              if (dx * dx + dz * dz <= radiusSq) {
                this.started[pi] = 1;
              }
            }
          }
        }
      }
      this.consumedVisited = visitedLagged.length;
      dirty = true;
    }

    // Fade any started-but-not-yet-full points toward 1.
    const fadeRate = dt / Math.max(1e-6, CONFIG.reveal.fadeSeconds);
    for (let i = 0; i < this.count; i++) {
      if (this.started[i] && this.progress[i] < 1) {
        this.progress[i] = clamp(this.progress[i] + fadeRate, 0, 1);
        dirty = true;
      }
    }

    void now;
    return dirty;
  }

  /** True if any point within `radius` (XZ) of `pos` has begun revealing. */
  isAreaRevealed(pos: [number, number, number], radius = CONFIG.reveal.radius): boolean {
    const sx = pos[0];
    const sz = pos[2];
    const radiusSq = radius * radius;
    const cellRadius = Math.ceil(radius / CELL_SIZE);
    const cx = Math.floor(sx / CELL_SIZE);
    const cz = Math.floor(sz / CELL_SIZE);

    for (let gx = cx - cellRadius; gx <= cx + cellRadius; gx++) {
      for (let gz = cz - cellRadius; gz <= cz + cellRadius; gz++) {
        const bucket = this.grid.get(`${gx}_${gz}`);
        if (!bucket) continue;
        for (const pi of bucket) {
          if (!this.started[pi]) continue;
          const j = pi * 3;
          const dx = this.positions[j] - sx;
          const dz = this.positions[j + 2] - sz;
          if (dx * dx + dz * dz <= radiusSq) return true;
        }
      }
    }
    return false;
  }
}
