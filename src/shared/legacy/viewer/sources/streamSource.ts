// Display-only world streamer (?map= + ?tex=sat, CONTROL): as the ACTIVE drone
// nears unloaded territory, fetch fine terrain + building prisms for grid
// cells around it — the "slippy map" feel. Never touches the deterministic
// SceneData contract: no points, no PRNG draws, and STATUS never runs this.
//
// Sliding window, not a bounded region: cells near the drone load in, cells
// that fall far behind get disposed and forgotten (retryable later if
// revisited) — so flight range is unbounded instead of hitting a hard wall.

import type { Bbox, TerrainContext, TerrainVisual } from './terrainSource.ts';
import { loadTerrainPatch } from './terrainSource.ts';
import { loadBuildings } from './buildingSource.ts';
import type { BuildingVisual } from './buildingSource.ts';

export interface StreamTarget {
  addStreamedTerrain(v: TerrainVisual): () => void;
  /** `cellTexture` is the cell's OWN aerial mosaic. A streamed cell is outside
   *  the core drape, so 실사 항공뷰 needs this or the imagery stops at the
   *  scene edge while the terrain under it keeps going. */
  addSurroundBuildings(v: BuildingVisual, cellTexture?: ImageBitmap): () => void;
}

/** Cells are half the core span → a 2×2 block of cells covers the core. */
const CELLS_PER_CORE_SPAN = 2;
/** Load cells whose center lies within this many world units of the drone
 *  (fog fades everything past ~80, so this covers what's actually visible). */
const LOAD_RADIUS = 34;
/** Evict cells once they fall this far behind — well past LOAD_RADIUS so a
 *  drone drifting near the boundary doesn't thrash load/evict every tick. */
const EVICT_RADIUS = LOAD_RADIUS * 2.2;
/** Concurrent cell loads — each is a DEM + satellite + WFS fetch bundle. */
const MAX_CONCURRENT = 2;
/** Streamed layers sit between the core surface (0) and the far ring (-0.08). */
const Y_OFFSET = -0.06;
/** How often to look for newly-needed / far-behind cells. */
const TICK_MS = 900;

interface CellDisposers {
  terrain?: () => void;
  buildings?: () => void;
}

export function startWorldStream(
  coreBbox: Bbox,
  ctx: TerrainContext,
  target: StreamTarget,
  getDronePos: () => { x: number; z: number } | null,
): () => void {
  const cellW = (coreBbox[2] - coreBbox[0]) / CELLS_PER_CORE_SPAN;
  const cellH = (coreBbox[3] - coreBbox[1]) / CELLS_PER_CORE_SPAN;
  /** Resident + in-flight cells. `null` = the core's own 2×2 block, which the
   *  core scene (not this module) rendered — never evicted, never disposed
   *  here. A present-but-empty object = a failed fetch, kept so we don't
   *  retry it every tick while still nearby (evicted like any other cell). */
  const cells = new Map<string, CellDisposers | null>();
  let active = 0;

  for (let i = 0; i < CELLS_PER_CORE_SPAN; i++) {
    for (let j = 0; j < CELLS_PER_CORE_SPAN; j++) cells.set(`${i},${j}`, null);
  }

  const cellBbox = (i: number, j: number): Bbox => [
    coreBbox[0] + i * cellW,
    coreBbox[1] + j * cellH,
    coreBbox[0] + (i + 1) * cellW,
    coreBbox[1] + (j + 1) * cellH,
  ];

  const cellCenterWorld = (i: number, j: number): { x: number; z: number } => {
    const lon = coreBbox[0] + (i + 0.5) * cellW;
    const lat = coreBbox[1] + (j + 0.5) * cellH;
    return {
      x: (lon - ctx.lon0) * ctx.mPerDegLon * ctx.s,
      z: -(lat - ctx.lat0) * ctx.mPerDegLat * ctx.s,
    };
  };

  async function loadCell(i: number, j: number): Promise<void> {
    active++;
    const key = `${i},${j}`;
    const box = cellBbox(i, j);
    const disposers: CellDisposers = {};
    try {
      const patch = await loadTerrainPatch(box, ctx, { yOffset: Y_OFFSET });
      if (patch.visual) disposers.terrain = target.addStreamedTerrain(patch.visual);
      const bld = await loadBuildings(box, patch.elevCtx, 0, {
        prismsOnly: true,
        onlyBbox: box,
        yOffset: Y_OFFSET,
        // Project this cell's drape onto its prisms so the aerial display
        // option covers streamed territory too.
        imagery: patch.imagery,
        // Streamed cells are display-only backdrop: if VWorld has no footprints
        // out here, stand-ins keep the skyline continuous instead of ending in
        // a bare plate. The core scene decides this for itself.
        allowSynthetic: true,
      });
      if (bld.visual) {
        disposers.buildings = target.addSurroundBuildings(
          bld.visual,
          patch.imagery?.texture,
        );
      }
      // The fetch can outlast the drone's stay nearby — if it's already past
      // EVICT_RADIUS by the time this lands, drop it immediately instead of
      // leaving orphaned geometry in the scene.
      const pos = getDronePos();
      const c = cellCenterWorld(i, j);
      if (pos && Math.hypot(c.x - pos.x, c.z - pos.z) > EVICT_RADIUS) {
        disposers.terrain?.();
        disposers.buildings?.();
        cells.delete(key);
        return;
      }
      cells.set(key, disposers);
    } catch (err) {
      // A dead cell stays claimed-but-empty (rare) — the stream keeps flowing
      // and can retry it once it's been evicted and revisited.
      console.warn(`[stream] cell ${i},${j} failed:`, err);
      cells.set(key, {});
    } finally {
      active--;
    }
  }

  /** Drop cells the drone has left far behind — geometry disposed, key freed
   *  for a future reload. Keeps memory bounded no matter how far/long the
   *  drone flies. */
  const evictFarCells = (pos: { x: number; z: number }): void => {
    for (const [key, disposers] of cells) {
      if (!disposers) continue; // core cell — not ours to evict
      const [i, j] = key.split(',').map(Number);
      const c = cellCenterWorld(i, j);
      if (Math.hypot(c.x - pos.x, c.z - pos.z) > EVICT_RADIUS) {
        disposers.terrain?.();
        disposers.buildings?.();
        cells.delete(key);
      }
    }
  };

  const tick = (): void => {
    const pos = getDronePos();
    if (!pos) return;

    evictFarCells(pos);

    if (active >= MAX_CONCURRENT) return;

    const droneLon = ctx.lon0 + pos.x / (ctx.s * ctx.mPerDegLon);
    const droneLat = ctx.lat0 - pos.z / (ctx.s * ctx.mPerDegLat);
    const iD = Math.floor((droneLon - coreBbox[0]) / cellW);
    const jD = Math.floor((droneLat - coreBbox[1]) / cellH);
    const worldCellMin = Math.min(cellW * ctx.mPerDegLon, cellH * ctx.mPerDegLat) * ctx.s;
    const reach = Math.ceil(LOAD_RADIUS / worldCellMin) + 1;

    const candidates: Array<{ i: number; j: number; d: number }> = [];
    for (let j = jD - reach; j <= jD + reach; j++) {
      for (let i = iD - reach; i <= iD + reach; i++) {
        if (cells.has(`${i},${j}`)) continue;
        const c = cellCenterWorld(i, j);
        const d = Math.hypot(c.x - pos.x, c.z - pos.z);
        if (d <= LOAD_RADIUS) candidates.push({ i, j, d });
      }
    }
    candidates.sort((a, b) => a.d - b.d);
    for (const { i, j } of candidates.slice(0, MAX_CONCURRENT - active)) {
      cells.set(`${i},${j}`, {}); // claim immediately so concurrent ticks don't double-fetch
      void loadCell(i, j);
    }
  };

  const id = setInterval(tick, TICK_MS);
  tick();
  return () => clearInterval(id);
}
