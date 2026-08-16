// Display-only world streamer (?map= + ?tex=sat, SIM): as the ACTIVE drone
// nears unloaded territory, fetch fine terrain + building prisms for grid
// cells around it — the "slippy map" feel. Never touches the deterministic
// SceneData contract: no points, no PRNG draws, and RECON never runs this.

import type { Bbox, TerrainContext, TerrainVisual } from './terrainSource.ts';
import { loadTerrainPatch } from './terrainSource.ts';
import { loadBuildings } from './buildingSource.ts';
import type { BuildingVisual } from './buildingSource.ts';

export interface StreamTarget {
  addStreamedTerrain(v: TerrainVisual): void;
  addSurroundBuildings(v: BuildingVisual): void;
}

/** Cells are half the core span → a 2×2 block of cells covers the core. */
const CELLS_PER_CORE_SPAN = 2;
/** Load cells whose center lies within this many world units of the drone
 *  (fog fades everything past ~80, so this covers what's actually visible). */
const LOAD_RADIUS = 34;
/** Concurrent cell loads — each is a DEM + satellite + WFS fetch bundle. */
const MAX_CONCURRENT = 2;
/** Hard cap on streamed cells (memory guard for very long manual flights). */
const MAX_CELLS = 60;
/** Streamed layers sit between the core surface (0) and the far ring (-0.08). */
const Y_OFFSET = -0.06;
/** How often to look for newly-needed cells. */
const TICK_MS = 900;

export function startWorldStream(
  coreBbox: Bbox,
  ctx: TerrainContext,
  target: StreamTarget,
  getDronePos: () => { x: number; z: number } | null,
): () => void {
  const cellW = (coreBbox[2] - coreBbox[0]) / CELLS_PER_CORE_SPAN;
  const cellH = (coreBbox[3] - coreBbox[1]) / CELLS_PER_CORE_SPAN;
  /** Loaded or in-flight cell keys ("i,j" on the core-anchored lattice). */
  const claimed = new Set<string>();
  let active = 0;
  let total = 0;
  let capWarned = false;

  // The core scene already renders its own 2×2 block of cells.
  for (let i = 0; i < CELLS_PER_CORE_SPAN; i++) {
    for (let j = 0; j < CELLS_PER_CORE_SPAN; j++) claimed.add(`${i},${j}`);
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
    total++;
    const box = cellBbox(i, j);
    try {
      const patch = await loadTerrainPatch(box, ctx, { yOffset: Y_OFFSET });
      if (patch.visual) target.addStreamedTerrain(patch.visual);
      const bld = await loadBuildings(box, patch.elevCtx, 0, {
        prismsOnly: true,
        onlyBbox: box,
        yOffset: Y_OFFSET,
      });
      if (bld.visual) target.addSurroundBuildings(bld.visual);
    } catch (err) {
      // A dead cell stays empty (rare) — the stream keeps flowing.
      console.warn(`[stream] cell ${i},${j} failed:`, err);
    } finally {
      active--;
    }
  }

  const tick = (): void => {
    if (active >= MAX_CONCURRENT) return;
    if (total >= MAX_CELLS) {
      if (!capWarned) {
        capWarned = true;
        console.warn(`[stream] cell cap reached (${MAX_CELLS}) — no further streaming`);
      }
      return;
    }
    const pos = getDronePos();
    if (!pos) return;

    const droneLon = ctx.lon0 + pos.x / (ctx.s * ctx.mPerDegLon);
    const droneLat = ctx.lat0 - pos.z / (ctx.s * ctx.mPerDegLat);
    const iD = Math.floor((droneLon - coreBbox[0]) / cellW);
    const jD = Math.floor((droneLat - coreBbox[1]) / cellH);
    const worldCellMin = Math.min(cellW * ctx.mPerDegLon, cellH * ctx.mPerDegLat) * ctx.s;
    const reach = Math.ceil(LOAD_RADIUS / worldCellMin) + 1;

    const candidates: Array<{ i: number; j: number; d: number }> = [];
    for (let j = jD - reach; j <= jD + reach; j++) {
      for (let i = iD - reach; i <= iD + reach; i++) {
        if (claimed.has(`${i},${j}`)) continue;
        const c = cellCenterWorld(i, j);
        const d = Math.hypot(c.x - pos.x, c.z - pos.z);
        if (d <= LOAD_RADIUS) candidates.push({ i, j, d });
      }
    }
    candidates.sort((a, b) => a.d - b.d);
    for (const { i, j } of candidates.slice(0, MAX_CONCURRENT - active)) {
      claimed.add(`${i},${j}`);
      void loadCell(i, j);
    }
  };

  const id = setInterval(tick, TICK_MS);
  tick();
  return () => clearInterval(id);
}
