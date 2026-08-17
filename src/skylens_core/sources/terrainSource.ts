// Real-world terrain scene source (?map=…) — DEM tiles → the same SceneData
// contract the splat pipeline produces, so SIM/RECON render it unchanged.
//
// Data: AWS Open Data "Terrain Tiles" (Mapzen terrarium PNGs, SRTM ~30 m in
// Korea). Public bucket, CORS-enabled, no API key:
//   https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
// Elevation per pixel = (R*256 + G + B/256) - 32768 meters.
//
// Why: disaster response isn't limited to one pre-captured splat scene. With a
// bbox anywhere in Korea (wildfire mountains, coasts, cities) the SAME viewers
// get a scene instantly. The pipeline is deterministic (fixed tiles + seeded
// PRNG), so both computers derive identical clouds (PROJECT.md §1).
//
//   ?map=uljin                          preset (2022 울진·삼척 산불 일대)
//   ?map=gangneung                      preset (2023 강릉 산불 일대)
//   ?map=<west>,<south>,<east>,<north>  custom bbox in lon/lat degrees

import * as THREE from 'three';
import type { SceneData } from './sceneData.ts';

/** Keep in sync with sceneSource.ts (not imported to avoid a module cycle). */
const TARGET_EXTENT = 44;
const TARGET_POINTS = 120_000;

/** Vertical exaggeration — real relief reads nearly flat at map scale. */
const EXAGGERATION = 1.5;
/** Native zoom for ~30 m SRTM data at Korean latitudes. */
const MAX_ZOOM = 12;
/** Tile budget per load (network + memory guard). */
const MAX_TILES = 12;

const TILE_URL = (z: number, x: number, y: number): string =>
  `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;

/** [west, south, east, north] in degrees. */
export type Bbox = [number, number, number, number];

const PRESETS: Record<string, Bbox> = {
  /** 2022 울진·삼척 산불 피해 지역 (응봉산 일대). */
  uljin: [129.22, 37.0, 129.36, 37.12],
  /** 2023 강릉 산불 지역 (경포 일대, 해안 포함). */
  gangneung: [128.83, 37.74, 128.93, 37.82],
  /** 울진읍 시가지 — 건물 레이어 시연용 (VWorld 실측 1,700동+, ~1.8 km). */
  uljinup: [129.39, 36.985, 129.41, 37.0],
  /** 기준 씬: 대전 유성 — 충남대 중심, 동쪽으로 카이스트 포함 (~3 km 도심). */
  daejeon: [127.33, 36.355, 127.365, 36.382],
};

/** Bare `?map` (no value) resolves to this preset. */
const DEFAULT_PRESET = 'daejeon';

/** Parse `?map=` into a bbox; null when terrain mode is not requested. */
export function resolveMapSpec(): Bbox | null {
  if (typeof window === 'undefined') return null;
  const q = new URLSearchParams(window.location.search).get('map');
  if (q == null) return null;
  if (q === '') return PRESETS[DEFAULT_PRESET];
  if (PRESETS[q]) return PRESETS[q];
  const parts = q.split(',').map(Number);
  if (parts.length === 4 && parts.every(Number.isFinite)) {
    const [w, s, e, n] = parts;
    if (w < e && s < n) return [w, s, e, n];
  }
  return null;
}

// Same deterministic PRNG as sceneData.ts — identical clouds on both computers.
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

// --- Web Mercator tile math ---
function lonToTileX(lon: number, z: number): number {
  return ((lon + 180) / 360) * 2 ** z;
}
function latToTileY(lat: number, z: number): number {
  const r = (lat * Math.PI) / 180;
  return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** z;
}
function tileYToLat(y: number, z: number): number {
  const n = Math.PI - (2 * Math.PI * y) / 2 ** z;
  return (180 / Math.PI) * Math.atan(Math.sinh(n));
}
function tileXToLon(x: number, z: number): number {
  return (x / 2 ** z) * 360 - 180;
}

/** Pick the deepest zoom whose tile count fits the budget. */
function pickZoom(bbox: Bbox): number {
  for (let z = MAX_ZOOM; z > 0; z--) {
    const cols = Math.floor(lonToTileX(bbox[2], z)) - Math.floor(lonToTileX(bbox[0], z)) + 1;
    const rows = Math.floor(latToTileY(bbox[1], z)) - Math.floor(latToTileY(bbox[3], z)) + 1;
    if (cols * rows <= MAX_TILES) return z;
  }
  return 1;
}

async function fetchTileImage(z: number, x: number, y: number): Promise<ImageBitmap> {
  // One retry — a single flaky tile must not take the whole scene down (demo!).
  let lastErr: unknown;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(TILE_URL(z, x, y));
      if (!res.ok) throw new Error(`terrain tile ${z}/${x}/${y}: HTTP ${res.status}`);
      return createImageBitmap(await res.blob());
    } catch (err) {
      lastErr = err;
      if (attempt === 0) await new Promise((r) => setTimeout(r, 300));
    }
  }
  throw lastErr;
}

/**
 * Repair SRTM voids / failed tiles in place. Invalid samples (NaN from a failed
 * tile, or absurd elevations from void pixels decoding to ~-32768 m) would wreck
 * minE/relief normalization and the color ramp, so: a few neighbor-averaging
 * sweeps for edges, then a global median fallback for anything left (a flat
 * patch instead of a broken scene).
 */
function sanitizeGrid(elev: Float32Array, width: number, height: number): void {
  const bad = (v: number) => !Number.isFinite(v) || v < -1000 || v > 9500;
  const sample: number[] = [];
  for (let i = 0; i < elev.length; i += 97) if (!bad(elev[i])) sample.push(elev[i]);
  sample.sort((a, b) => a - b);
  const fallback = sample.length ? sample[Math.floor(sample.length / 2)] : 0;

  for (let sweep = 0; sweep < 3; sweep++) {
    let fixed = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if (!bad(elev[i])) continue;
        let sum = 0;
        let n = 0;
        if (x > 0 && !bad(elev[i - 1])) { sum += elev[i - 1]; n++; }
        if (x < width - 1 && !bad(elev[i + 1])) { sum += elev[i + 1]; n++; }
        if (y > 0 && !bad(elev[i - width])) { sum += elev[i - width]; n++; }
        if (y < height - 1 && !bad(elev[i + width])) { sum += elev[i + width]; n++; }
        if (n > 0) { elev[i] = sum / n; fixed++; }
      }
    }
    if (fixed === 0) break;
  }
  for (let i = 0; i < elev.length; i++) if (bad(elev[i])) elev[i] = fallback;
}

interface ElevationGrid {
  /** Meters, row-major, width*height. */
  elev: Float32Array;
  width: number;
  height: number;
  /** Global mercator pixel coords of the grid's top-left corner. */
  originX: number;
  originY: number;
  zoom: number;
}

/** Fetch all tiles covering the bbox and merge them into one elevation grid. */
async function buildElevationGrid(
  bbox: Bbox,
  onProgress?: (percent: number) => void,
): Promise<ElevationGrid> {
  const zoom = pickZoom(bbox);
  const tx0 = Math.floor(lonToTileX(bbox[0], zoom));
  const tx1 = Math.floor(lonToTileX(bbox[2], zoom));
  const ty0 = Math.floor(latToTileY(bbox[3], zoom)); // north edge → smaller y
  const ty1 = Math.floor(latToTileY(bbox[1], zoom));

  const cols = tx1 - tx0 + 1;
  const rows = ty1 - ty0 + 1;
  const width = cols * 256;
  const height = rows * 256;
  // NaN = not loaded; sanitizeGrid() repairs whatever stays invalid.
  const elev = new Float32Array(width * height).fill(Number.NaN);

  if (zoom < MAX_ZOOM) {
    console.warn(
      `[terrain] bbox too large for native z${MAX_ZOOM} within ${MAX_TILES} tiles — using z${zoom} (lower detail)`,
    );
  }

  const jobs: Array<Promise<void>> = [];
  let done = 0;
  let failed = 0;
  for (let ty = ty0; ty <= ty1; ty++) {
    for (let tx = tx0; tx <= tx1; tx++) {
      jobs.push(
        fetchTileImage(zoom, tx, ty)
          .then((img) => {
            const canvas = new OffscreenCanvas(256, 256);
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('no 2d context');
            ctx.drawImage(img, 0, 0);
            const { data } = ctx.getImageData(0, 0, 256, 256);
            const ox = (tx - tx0) * 256;
            const oy = (ty - ty0) * 256;
            for (let py = 0; py < 256; py++) {
              const rowBase = (oy + py) * width + ox;
              for (let px = 0; px < 256; px++) {
                const s = (py * 256 + px) * 4;
                elev[rowBase + px] =
                  data[s] * 256 + data[s + 1] + data[s + 2] / 256 - 32768;
              }
            }
          })
          // A failed tile stays NaN and gets repaired — the scene must survive.
          .catch((err) => {
            failed++;
            console.warn(`[terrain] tile ${zoom}/${tx}/${ty} failed:`, err);
          })
          .finally(() => {
            done++;
            onProgress?.((done / jobs.length) * 100);
          }),
      );
    }
  }
  await Promise.all(jobs);
  if (failed === jobs.length) throw new Error('all terrain tiles failed');

  sanitizeGrid(elev, width, height);
  return { elev, width, height, originX: tx0 * 256, originY: ty0 * 256, zoom };
}

// --- Satellite imagery drape (P8) ---------------------------------------------
// VWorld WMTS satellite tiles → per-point real colors, so terrain scenes read
// as the actual place (fields/roads/river) instead of an abstract ramp.
// Proxy-only (no CORS on api.vworld.kr): /vworld/wmts/{z}/{y}/{x}.jpeg via
// vite.config.ts. Fails → hypsometric ramp fallback. `?tex=off` disables.

const SAT_URL = (z: number, y: number, x: number): string => `/vworld/wmts/${z}/${y}/${x}.jpeg`;
const SAT_MAX_ZOOM = 17;
// 160 tiles ≈ z15 on the uljin preset (~4.8 m/px). The close-in map camera
// (droneViewScale) magnifies the texture ~7×, so the old 28-tile budget (z13,
// ~19 m/px) reads as mush. One-time load cost: ~160 small JPEGs via the proxy.
const SAT_MAX_TILES = 160;

function imageryWanted(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('tex') !== 'off';
}

/** `?tex=cyber`: same satellite drape, graded into the SIM's cold tactical
 *  palette. The map deliberately stays dim so drones and mission overlays own
 *  the brightest cyan values. */
export function cyberOn(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('tex') === 'cyber';
}

// Luminance → restrained navy/teal ramp. This is a BACKGROUND palette: even
// the brightest satellite pixels remain below the UI/drone highlight range.
let CYBER_LUT: Uint8ClampedArray | null = null;
function cyberLut(): Uint8ClampedArray {
  if (CYBER_LUT) return CYBER_LUT;
  const stops: Array<[number, [number, number, number]]> = [
    [0.0, [4, 9, 18]],
    [0.38, [8, 27, 42]],
    [0.7, [15, 57, 72]],
    [0.9, [30, 87, 104]],
    [1.0, [58, 126, 142]],
  ];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    // Bake the midtone lift into the LUT so streamed cells need only one table
    // lookup per pixel instead of an expensive Math.pow in the hot loop.
    const t = Math.pow(i / 255, 0.88);
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k];
    const [t1, c1] = stops[k + 1];
    const f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
    lut[i * 3] = c0[0] + (c1[0] - c0[0]) * f;
    lut[i * 3 + 1] = c0[1] + (c1[1] - c0[1]) * f;
    lut[i * 3 + 2] = c0[2] + (c1[2] - c0[2]) * f;
  }
  CYBER_LUT = lut;
  return lut;
}

/** Regrade the mosaic in place — for the mesh TEXTURE only. The rgba array the
 *  shared point cloud samples is read before this runs, so point colors (and
 *  with them the SIM/RECON determinism contract) are untouched. The graticule
 *  aligns to GLOBAL tile-pixel coords so seams between streamed cells match. */
function applyCyberGrade(
  ctx: OffscreenCanvasRenderingContext2D,
  width: number,
  height: number,
  originX: number,
  originY: number,
  zoom: number,
  centerLat: number,
): void {
  const img = ctx.getImageData(0, 0, width, height);
  const d = img.data;
  const lut = cyberLut();
  for (let i = 0; i < d.length; i += 4) {
    const lum = Math.min(255, (d[i] * 54 + d[i + 1] * 183 + d[i + 2] * 19) >> 8);
    d[i] = lut[lum * 3];
    d[i + 1] = lut[lum * 3 + 1];
    d[i + 2] = lut[lum * 3 + 2];
  }
  ctx.putImageData(img, 0, 0);
  const grid = (step: number, style: string): void => {
    ctx.strokeStyle = style;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = (step - (originX % step)) % step; x < width; x += step) {
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
    }
    for (let y = (step - (originY % step)) % step; y < height; y += step) {
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(width, y + 0.5);
    }
    ctx.stroke();
  };
  // Keep the graticule stable in REAL distance as imagery zoom changes.
  // Web Mercator ground resolution at this latitude, metres per pixel.
  const metresPerPixel =
    (156543.03392 * Math.cos((centerLat * Math.PI) / 180)) / 2 ** zoom;
  const minorStep = Math.max(48, Math.round(250 / metresPerPixel));
  const majorStep = Math.max(minorStep * 2, Math.round(1000 / metresPerPixel));
  grid(minorStep, 'rgba(92, 190, 208, 0.035)');
  grid(majorStep, 'rgba(102, 210, 226, 0.085)');
}

interface ImageryGrid {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  originX: number;
  originY: number;
  zoom: number;
  /** Merged tile mosaic — used as the terrain mesh texture. */
  bitmap: ImageBitmap;
}

/** Fetch satellite tiles over the bbox at the deepest zoom within budget. */
async function buildImageryGrid(
  bbox: Bbox,
  maxTiles: number = SAT_MAX_TILES,
): Promise<ImageryGrid | null> {
  let zoom = 1;
  for (let z = SAT_MAX_ZOOM; z > 0; z--) {
    const cols = Math.floor(lonToTileX(bbox[2], z)) - Math.floor(lonToTileX(bbox[0], z)) + 1;
    const rows = Math.floor(latToTileY(bbox[1], z)) - Math.floor(latToTileY(bbox[3], z)) + 1;
    if (cols * rows <= maxTiles) {
      zoom = z;
      break;
    }
  }
  const tx0 = Math.floor(lonToTileX(bbox[0], zoom));
  const tx1 = Math.floor(lonToTileX(bbox[2], zoom));
  const ty0 = Math.floor(latToTileY(bbox[3], zoom));
  const ty1 = Math.floor(latToTileY(bbox[1], zoom));
  const cols = tx1 - tx0 + 1;
  const rows = ty1 - ty0 + 1;
  const width = cols * 256;
  const height = rows * 256;

  // One mosaic canvas — pixel reads for point colors AND the mesh texture.
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  let ok = 0;
  const jobs: Array<Promise<void>> = [];
  for (let ty = ty0; ty <= ty1; ty++) {
    for (let tx = tx0; tx <= tx1; tx++) {
      jobs.push(
        (async () => {
          const res = await fetch(SAT_URL(zoom, ty, tx));
          if (!res.ok) throw new Error(`sat tile HTTP ${res.status}`);
          const img = await createImageBitmap(await res.blob());
          ctx.drawImage(img, (tx - tx0) * 256, (ty - ty0) * 256);
          ok++;
        })().catch(() => undefined),
      );
    }
  }
  await Promise.all(jobs);
  if (ok === 0) return null;

  // Read pixels BEFORE transferToImageBitmap — the transfer empties the canvas.
  const rgba = new Uint8ClampedArray(ctx.getImageData(0, 0, width, height).data);
  // Cyber grade applies AFTER the rgba read: point colors stay real (contract),
  // only the mesh texture is restyled.
  if (cyberOn()) {
    applyCyberGrade(
      ctx,
      width,
      height,
      tx0 * 256,
      ty0 * 256,
      zoom,
      (bbox[1] + bbox[3]) / 2,
    );
  }
  const bitmap = canvas.transferToImageBitmap();
  return { rgba, width, height, originX: tx0 * 256, originY: ty0 * 256, zoom, bitmap };
}

/**
 * Shared geo→world transform so other layers (buildings) land EXACTLY on this
 * terrain: same origin, same scale, same vertical treatment.
 */
export interface TerrainContext {
  bbox: Bbox;
  lat0: number;
  lon0: number;
  mPerDegLat: number;
  mPerDegLon: number;
  /** Meters → world units. */
  s: number;
  minE: number;
  exaggeration: number;
  /** Terrain elevation in meters at lon/lat (bilinear); minE outside the grid. */
  elevationAt(lon: number, lat: number): number;
}

/**
 * Display-only textured surface for the SIM viewer (`?tex=sat`): the DEM as a
 * triangulated height mesh with the satellite mosaic as its texture. The point
 * cloud stays the single data contract (paths/detections/reveal) — this is a
 * rendering upgrade, not a data change.
 */
export interface TerrainVisual {
  positions: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  texture: ImageBitmap;
}

export interface TerrainScene {
  data: SceneData;
  ctx: TerrainContext;
  /** Present when the satellite mosaic loaded — enables the mesh view. */
  visual?: TerrainVisual;
  /** Decorative low-res terrain ring around the sim area (display-only), so
   *  the world doesn't visibly end at the scene edge. No points, no paths —
   *  pure backdrop. `?ring=off` disables. */
  surround?: TerrainVisual;
  /** Geo→world context over the RING's DEM — lets other display layers
   *  (ring buildings) snap to the backdrop terrain outside the core grid. */
  surroundCtx?: TerrainContext;
}

// --- Context ring ---------------------------------------------------------
/** Ring bbox = sim bbox span × RING_SPAN (centered). */
const RING_SPAN = 3;
/** Satellite tile budget for the ring — background quality is fine. */
const RING_SAT_TILES = 40;

function ringWanted(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('ring') !== 'off';
}

export function expandBbox(b: Bbox, f: number): Bbox {
  const cx = (b[0] + b[2]) / 2;
  const cy = (b[1] + b[3]) / 2;
  const hw = ((b[2] - b[0]) / 2) * f;
  const hh = ((b[3] - b[1]) / 2) * f;
  return [cx - hw, cy - hh, cx + hw, cy + hh];
}

/** Geo→world params shared by the core mesh and the ring so they align. */
interface GeoTransform {
  lon0: number;
  lat0: number;
  mPerDegLon: number;
  mPerDegLat: number;
  s: number;
  minE: number;
  exaggeration: number;
}

/** Bilinear elevation sampler over a grid, in the shared world frame. */
function makeElevationCtx(grid: ElevationGrid, bbox: Bbox, t: GeoTransform): TerrainContext {
  return {
    bbox,
    lat0: t.lat0,
    lon0: t.lon0,
    mPerDegLat: t.mPerDegLat,
    mPerDegLon: t.mPerDegLon,
    s: t.s,
    minE: t.minE,
    exaggeration: t.exaggeration,
    elevationAt(qLon: number, qLat: number): number {
      const gx = lonToTileX(qLon, grid.zoom) * 256 - grid.originX;
      const gy = latToTileY(qLat, grid.zoom) * 256 - grid.originY;
      const x0 = Math.floor(gx);
      const y0 = Math.floor(gy);
      if (x0 < 0 || y0 < 0 || x0 >= grid.width - 1 || y0 >= grid.height - 1) return t.minE;
      const fx = gx - x0;
      const fy = gy - y0;
      const i = y0 * grid.width + x0;
      const a = grid.elev[i];
      const b = grid.elev[i + 1];
      const c = grid.elev[i + grid.width];
      const d = grid.elev[i + grid.width + 1];
      return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
    },
  };
}

/**
 * Display-only terrain patch for the world streamer: DEM + imagery over one
 * cell, meshed in the SAME world frame as the core scene (pass the core
 * TerrainContext as `t`). Also returns an elevation context over the cell so
 * streamed buildings can snap to this patch.
 */
export interface TerrainPatch {
  visual: TerrainVisual | null;
  elevCtx: TerrainContext;
}

export async function loadTerrainPatch(
  cellBbox: Bbox,
  t: TerrainContext,
  opts?: { satTiles?: number; yOffset?: number; maxGrid?: number },
): Promise<TerrainPatch> {
  const [imagery, grid] = await Promise.all([
    buildImageryGrid(cellBbox, opts?.satTiles ?? 20).catch(() => null),
    buildElevationGrid(cellBbox),
  ]);
  const visual = imagery
    ? buildTerrainMesh(grid, imagery, cellBbox, t, opts?.yOffset ?? 0, opts?.maxGrid ?? 128)
    : null;
  return { visual, elevCtx: makeElevationCtx(grid, cellBbox, t) };
}

/** DEM grid + imagery mosaic → textured height mesh over `bbox`. */
function buildTerrainMesh(
  grid: ElevationGrid,
  imagery: ImageryGrid,
  bbox: Bbox,
  t: GeoTransform,
  yOffset: number,
  maxGrid: number,
): TerrainVisual {
  const { elev, width, zoom } = grid;
  const px0 = Math.max(0, Math.floor(lonToTileX(bbox[0], zoom) * 256 - grid.originX));
  const px1 = Math.min(width - 1, Math.ceil(lonToTileX(bbox[2], zoom) * 256 - grid.originX));
  const py0 = Math.max(0, Math.floor(latToTileY(bbox[3], zoom) * 256 - grid.originY));
  const py1 = Math.min(grid.height - 1, Math.ceil(latToTileY(bbox[1], zoom) * 256 - grid.originY));
  const spanX = px1 - px0;
  const spanY = py1 - py0;
  const step = Math.max(1, Math.ceil(Math.max(spanX, spanY) / maxGrid));
  const nx = Math.floor(spanX / step) + 1;
  const ny = Math.floor(spanY / step) + 1;
  const vPositions = new Float32Array(nx * ny * 3);
  const vUvs = new Float32Array(nx * ny * 2);
  const vIndices = new Uint32Array((nx - 1) * (ny - 1) * 6);

  for (let iy = 0; iy < ny; iy++) {
    const py = Math.min(py1, py0 + iy * step);
    for (let ix = 0; ix < nx; ix++) {
      const px = Math.min(px1, px0 + ix * step);
      const e = elev[py * width + px];
      const lon = tileXToLon((grid.originX + px + 0.5) / 256, zoom);
      const lat = tileYToLat((grid.originY + py + 0.5) / 256, zoom);
      const vi = iy * nx + ix;
      vPositions[vi * 3] = (lon - t.lon0) * t.mPerDegLon * t.s;
      vPositions[vi * 3 + 1] = (e - t.minE) * t.s * t.exaggeration + yOffset;
      vPositions[vi * 3 + 2] = -(lat - t.lat0) * t.mPerDegLat * t.s;
      // Mosaic pixel for this lon/lat (texture is used with flipY=false).
      const sx = lonToTileX(lon, imagery.zoom) * 256 - imagery.originX;
      const sy = latToTileY(lat, imagery.zoom) * 256 - imagery.originY;
      vUvs[vi * 2] = THREE.MathUtils.clamp(sx / imagery.width, 0, 1);
      vUvs[vi * 2 + 1] = THREE.MathUtils.clamp(sy / imagery.height, 0, 1);
    }
  }
  let ii = 0;
  for (let iy = 0; iy < ny - 1; iy++) {
    for (let ix = 0; ix < nx - 1; ix++) {
      const a = iy * nx + ix;
      const b = a + 1;
      const cIdx = a + nx;
      const d = cIdx + 1;
      vIndices[ii++] = a;
      vIndices[ii++] = cIdx;
      vIndices[ii++] = b;
      vIndices[ii++] = b;
      vIndices[ii++] = cIdx;
      vIndices[ii++] = d;
    }
  }
  return { positions: vPositions, uvs: vUvs, indices: vIndices, texture: imagery.bitmap };
}

/**
 * DEM grid → normalized SceneData point cloud. Frame matches the splat path:
 * X east, Z south, XZ centered at origin, lowest terrain at y=0, longest
 * dimension scaled to TARGET_EXTENT.
 */
export async function loadTerrainScene(
  bbox: Bbox,
  onProgress?: (percent: number) => void,
  targetPoints: number = TARGET_POINTS,
): Promise<TerrainScene> {
  // Satellite drape loads in parallel with the DEM; both are tile fetches.
  const wantImagery = imageryWanted();
  const imageryPromise: Promise<ImageryGrid | null> = wantImagery
    ? buildImageryGrid(bbox).catch(() => null)
    : Promise.resolve(null);
  // Context ring (background terrain around the sim area) — fetches kick off
  // in parallel too; a failed ring is silently skipped.
  const surBbox: Bbox | null = wantImagery && ringWanted() ? expandBbox(bbox, RING_SPAN) : null;
  const surGridPromise: Promise<ElevationGrid | null> = surBbox
    ? buildElevationGrid(surBbox).catch(() => null)
    : Promise.resolve(null);
  const surImageryPromise: Promise<ImageryGrid | null> = surBbox
    ? buildImageryGrid(surBbox, RING_SAT_TILES).catch(() => null)
    : Promise.resolve(null);
  const grid = await buildElevationGrid(bbox, onProgress);
  const imagery = await imageryPromise;
  if (wantImagery && !imagery) {
    console.warn('[terrain] satellite drape unavailable — falling back to hypsometric colors');
  }
  const { elev, width, zoom } = grid;
  const scale256 = 2 ** zoom * 256;

  // Grid pixel range covering the bbox (grid-local coordinates).
  const px0 = Math.max(0, Math.floor(lonToTileX(bbox[0], zoom) * 256 - grid.originX));
  const px1 = Math.min(width - 1, Math.ceil(lonToTileX(bbox[2], zoom) * 256 - grid.originX));
  const py0 = Math.max(0, Math.floor(latToTileY(bbox[3], zoom) * 256 - grid.originY));
  const py1 = Math.min(grid.height - 1, Math.ceil(latToTileY(bbox[1], zoom) * 256 - grid.originY));

  // Local meters around the bbox center (equirectangular is fine at city scale).
  const lat0 = (bbox[1] + bbox[3]) / 2;
  const lon0 = (bbox[0] + bbox[2]) / 2;
  const mPerDegLat = 111_320;
  const mPerDegLon = 111_320 * Math.cos((lat0 * Math.PI) / 180);

  // Elevation range over the bbox for normalization + coloring.
  let minE = Infinity;
  let maxE = -Infinity;
  for (let py = py0; py <= py1; py++) {
    for (let px = px0; px <= px1; px++) {
      const e = elev[py * width + px];
      if (e < minE) minE = e;
      if (e > maxE) maxE = e;
    }
  }
  const relief = Math.max(maxE - minE, 1);

  const widthM = (bbox[2] - bbox[0]) * mPerDegLon;
  const heightM = (bbox[3] - bbox[1]) * mPerDegLat;
  const s = TARGET_EXTENT / Math.max(widthM, heightM);
  /** Ground sample distance in meters (for jitter + slope shading). */
  const gsd = (360 / scale256) * mPerDegLon;

  // Keep-probability so we land near the point target regardless of bbox size.
  const totalPx = (px1 - px0 + 1) * (py1 - py0 + 1);
  const keepP = Math.min(1, targetPoints / totalPx);
  const rand = mulberry32(0x7e44a1);

  const positions = new Float32Array(Math.ceil(totalPx * keepP * 1.1) * 3 + 3);
  const colors = new Float32Array(positions.length);
  const bounds = new THREE.Box3(
    new THREE.Vector3(Infinity, Infinity, Infinity),
    new THREE.Vector3(-Infinity, -Infinity, -Infinity),
  );

  // Hypsometric ramp for the RECON reveal cloud (SIM applies its own gradient).
  const low = new THREE.Color(0x35543a); // valley forest green
  const mid = new THREE.Color(0x8a7a55); // ridge tan
  const high = new THREE.Color(0xd8d2c6); // peak light gray
  const c = new THREE.Color();
  const p = new THREE.Vector3();

  let j = 0;
  for (let py = py0; py <= py1; py++) {
    for (let px = px0; px <= px1; px++) {
      // Deterministic draws happen for EVERY pixel, so the kept set is stable.
      const draw = rand();
      const jx = rand() - 0.5;
      const jz = rand() - 0.5;
      if (draw > keepP) continue;
      if (j * 3 + 3 > positions.length) break;

      const e = elev[py * width + px];
      const lon = tileXToLon((grid.originX + px + 0.5) / 256, zoom);
      const lat = tileYToLat((grid.originY + py + 0.5) / 256, zoom);

      p.set(
        ((lon - lon0) * mPerDegLon + jx * gsd) * s,
        (e - minE) * s * EXAGGERATION,
        (-(lat - lat0) * mPerDegLat + jz * gsd) * s,
      );

      // Slope shading from the east neighbor (cheap, reads as hillshade).
      const eEast = elev[py * width + Math.min(px + 1, width - 1)];
      const shade = THREE.MathUtils.clamp(1 - (eEast - e) / gsd, 0.55, 1.25);

      if (imagery) {
        // Real satellite color at this lon/lat (nearest pixel) + a hint of relief.
        const sx = Math.min(
          imagery.width - 1,
          Math.max(0, Math.round(lonToTileX(lon, imagery.zoom) * 256 - imagery.originX)),
        );
        const sy = Math.min(
          imagery.height - 1,
          Math.max(0, Math.round(latToTileY(lat, imagery.zoom) * 256 - imagery.originY)),
        );
        const si = (sy * imagery.width + sx) * 4;
        // JPEG pixels are sRGB, but three treats vertex colors as LINEAR and
        // re-encodes on output — feed raw values and everything washes to white.
        // Convert to linear here so the photo colors survive the round trip.
        c.setRGB(
          imagery.rgba[si] / 255,
          imagery.rgba[si + 1] / 255,
          imagery.rgba[si + 2] / 255,
        ).convertSRGBToLinear();
        // Lift — photo colors sit dark in linear space; without a boost the
        // terrain reads as silhouettes next to the bright building points.
        c.multiplyScalar(1.15 + 0.4 * shade);
      } else {
        const t = (e - minE) / relief;
        if (t < 0.5) c.copy(low).lerp(mid, t * 2);
        else c.copy(mid).lerp(high, (t - 0.5) * 2);
        c.multiplyScalar(shade);
      }

      positions[j * 3] = p.x;
      positions[j * 3 + 1] = p.y;
      positions[j * 3 + 2] = p.z;
      colors[j * 3] = Math.min(c.r, 1);
      colors[j * 3 + 1] = Math.min(c.g, 1);
      colors[j * 3 + 2] = Math.min(c.b, 1);
      bounds.expandByPoint(p);
      j++;
    }
  }

  if (imagery) console.info(`[terrain] satellite drape active (z${imagery.zoom})`);

  // Textured height mesh (display-only). Vertex grid capped near 512² (the
  // close-in map camera makes coarser facets visible); UVs address the imagery
  // mosaic directly.
  const t: GeoTransform = { lon0, lat0, mPerDegLon, mPerDegLat, s, minE, exaggeration: EXAGGERATION };
  let visual: TerrainVisual | undefined;
  if (imagery) {
    visual = buildTerrainMesh(grid, imagery, bbox, t, 0, 512);
  }

  // Context ring: same transform, tucked slightly below the core surface so
  // the seam hides under the sharper sim-area terrain.
  let surround: TerrainVisual | undefined;
  let surroundCtx: TerrainContext | undefined;
  if (visual && surBbox) {
    const surGrid = await surGridPromise;
    const surImagery = await surImageryPromise;
    if (surGrid && surImagery) {
      surround = buildTerrainMesh(surGrid, surImagery, surBbox, t, -0.08, 256);
      surroundCtx = makeElevationCtx(surGrid, surBbox, t);
      console.info(
        `[terrain] context ring active (dem z${surGrid.zoom}, sat z${surImagery.zoom})`,
      );
    }
  }

  const data: SceneData = {
    positions: positions.subarray(0, j * 3),
    colors: colors.subarray(0, j * 3),
    count: j,
    bounds,
    groundY: 0,
  };

  const ctx: TerrainContext = {
    bbox,
    lat0,
    lon0,
    mPerDegLat,
    mPerDegLon,
    s,
    minE,
    exaggeration: EXAGGERATION,
    elevationAt(qLon: number, qLat: number): number {
      const gx = lonToTileX(qLon, zoom) * 256 - grid.originX;
      const gy = latToTileY(qLat, zoom) * 256 - grid.originY;
      const x0 = Math.floor(gx);
      const y0 = Math.floor(gy);
      if (x0 < 0 || y0 < 0 || x0 >= width - 1 || y0 >= grid.height - 1) return minE;
      const fx = gx - x0;
      const fy = gy - y0;
      const i = y0 * width + x0;
      const a = elev[i];
      const b = elev[i + 1];
      const cLow = elev[i + width];
      const d = elev[i + width + 1];
      return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + cLow * (1 - fx) * fy + d * fx * fy;
    },
  };

  return { data, ctx, visual, surround, surroundCtx };
}
