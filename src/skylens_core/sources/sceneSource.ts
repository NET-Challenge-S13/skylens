// Single source of truth for the 3D scene.
//
// PROJECT.md §1 core trick: BOTH viewers must show the SAME data, differing only
// in how they render it. So the real Gaussian splat is the source: we load it,
// extract its splat centers+colors into a downsampled point cloud, and auto-fit
// it into a normalized world frame. CONTROL renders that point cloud low-fi; STATUS
// renders the full splat (with the SAME transform) plus reveal over the same
// points. Both computers load the same URL and run the same deterministic
// pipeline, so their clouds are identical.
//
// If the splat is disabled (?splat=off) or fails to load, both sides fall back
// to the same deterministic procedural cloud — so they still match.

import * as THREE from 'three';
import { SplatLoader } from '@mkkellogg/gaussian-splats-3d';
import { CONFIG } from '../config.ts';
import { buildSceneData } from './sceneData.ts';
import type { SceneData } from './sceneData.ts';
import { loadTerrainScene, resolveMapSpec } from './terrainSource.ts';
import type { TerrainVisual, TerrainContext, Bbox } from './terrainSource.ts';
import { loadBuildings } from './buildingSource.ts';
import type { PointPatch, BuildingVisual } from './buildingSource.ts';

function query(name: string): string | null {
  if (typeof window === 'undefined') return null;
  return new URLSearchParams(window.location.search).get(name);
}

/**
 * Resolve which splat URL to load from `?splat=` + config. Both CONTROL and STATUS
 * call this so they load the same asset.
 *   off → null (procedural fallback) | light|nike → lighter sample
 *   demo → our own capture | cdn → public sample
 *   http(s)://… → custom | (absent) → the delay-pattern capture, else the sample
 *
 * Note this is only the asset both viewers derive the shared point cloud and fit
 * transform from. On STATUS the rendered geometry arrives from the server, segment
 * by segment (see resolveSegmentManifest).
 */
export function resolveSplatUrl(): string | null {
  if (!CONFIG.splat.enabled) return null;
  const q = query('splat');
  if (!q) return CONFIG.delayPattern.enabled ? CONFIG.splat.demoPreview : CONFIG.splat.url;
  if (q === 'off') return null;
  if (q === 'light' || q === 'nike') return CONFIG.splat.urlLight;
  if (q === 'demo') return CONFIG.splat.demoPreview;
  if (q === 'cdn') return CONFIG.splat.url;
  if (/^https?:\/\//.test(q)) return q;
  return CONFIG.splat.url;
}

/**
 * Manifest for the delay-pattern stream, or null when this page isn't showing
 * our own capture (the segment assets are cut from THAT scene, so they only line
 * up with it). `?delay=off` forces the single-scene stream.
 */
export function resolveSegmentManifest(): string | null {
  if (!CONFIG.delayPattern.enabled) return null;
  if (query('delay') === 'off') return null;
  if (resolveSplatUrl() !== CONFIG.splat.demoPreview) return null;
  return CONFIG.delayPattern.manifest;
}

/** Transform that places the splat identically to the derived point cloud. */
export interface SplatTransform {
  url: string;
  position: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
}

export interface LoadedScene {
  data: SceneData;
  /** Non-null when the real splat should be rendered (STATUS); null on fallback. */
  splat: SplatTransform | null;
  /** Textured DEM surface for map scenes (CONTROL `?tex=sat` renders it as a mesh). */
  terrainVisual?: TerrainVisual;
  /** Points [0, terrainPointCount) are terrain; the rest are buildings. */
  terrainPointCount?: number;
  /** Extruded building prisms for map scenes (CONTROL `?tex=sat` display-only). */
  buildingVisual?: BuildingVisual;
  /** Low-res backdrop terrain around the control area (display-only). */
  surroundVisual?: TerrainVisual;
  /** Present when the CONTROL should run the world streamer: drone-position-driven
   *  loading of surrounding terrain + building cells (display-only). */
  streamSeed?: { coreBbox: Bbox; ctx: TerrainContext };
}

/** Longest world-space dimension we fit the scene into (matches the old footprint). */
const TARGET_EXTENT = 44;
/** Points kept for the low-fi/reveal cloud (the full splat renders separately). */
const TARGET_POINTS = 120_000;
/** Orientation presets to stand the splat upright (SfM frames vary per asset). */
const UP_PRESETS: Record<string, THREE.Euler> = {
  none: new THREE.Euler(0, 0, 0),
  x180: new THREE.Euler(Math.PI, 0, 0),
  x90: new THREE.Euler(Math.PI / 2, 0, 0),
  'x-90': new THREE.Euler(-Math.PI / 2, 0, 0),
  z90: new THREE.Euler(0, 0, Math.PI / 2),
  'z-90': new THREE.Euler(0, 0, -Math.PI / 2),
  z180: new THREE.Euler(0, 0, Math.PI),
};

/**
 * Manual orientation override from ?up=<preset> or ?up=<x>,<y>,<z> euler degrees.
 * Returns null when not specified (→ automatic PCA leveling is used instead).
 */
function manualUpQuat(): THREE.Quaternion | null {
  if (typeof window === 'undefined') return null;
  const q = new URLSearchParams(window.location.search).get('up');
  if (!q) return null;
  if (UP_PRESETS[q]) return new THREE.Quaternion().setFromEuler(UP_PRESETS[q]);
  if (/^-?\d/.test(q)) {
    const [x, y, z] = q.split(',').map((n) => (parseFloat(n) || 0) * (Math.PI / 180));
    return new THREE.Quaternion().setFromEuler(new THREE.Euler(x, y, z));
  }
  return null;
}

/** Jacobi eigen-decomposition of a symmetric 3x3 matrix (row-major length 9). */
function jacobi3(m: number[]): { values: number[]; vectors: number[][] } {
  const a = m.slice();
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 24; sweep++) {
    // Largest off-diagonal.
    let p = 0, q = 1;
    let max = Math.abs(a[1]);
    if (Math.abs(a[2]) > max) { max = Math.abs(a[2]); p = 0; q = 2; }
    if (Math.abs(a[5]) > max) { max = Math.abs(a[5]); p = 1; q = 2; }
    if (max < 1e-12) break;
    const app = a[p * 3 + p], aqq = a[q * 3 + q], apq = a[p * 3 + q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const cs = Math.cos(phi), sn = Math.sin(phi);
    for (let k = 0; k < 3; k++) {
      const akp = a[k * 3 + p], akq = a[k * 3 + q];
      a[k * 3 + p] = cs * akp - sn * akq;
      a[k * 3 + q] = sn * akp + cs * akq;
    }
    for (let k = 0; k < 3; k++) {
      const apk = a[p * 3 + k], aqk = a[q * 3 + k];
      a[p * 3 + k] = cs * apk - sn * aqk;
      a[q * 3 + k] = sn * apk + cs * aqk;
    }
    for (let k = 0; k < 3; k++) {
      const vkp = v[k * 3 + p], vkq = v[k * 3 + q];
      v[k * 3 + p] = cs * vkp - sn * vkq;
      v[k * 3 + q] = sn * vkp + cs * vkq;
    }
  }
  return {
    values: [a[0], a[4], a[8]],
    vectors: [
      [v[0], v[3], v[6]],
      [v[1], v[4], v[7]],
      [v[2], v[5], v[8]],
    ],
  };
}

/**
 * Auto-level a splat: the vertical axis of a wide structure (building/ground
 * scene) has the SMALLEST spatial spread, so the smallest-variance principal
 * axis of the point set is "up". Align it to world +Y, choosing the sign so the
 * wider (ground) end sits at the bottom.
 */
function autoLevelQuat(samples: THREE.Vector3[]): THREE.Quaternion {
  const n = samples.length;
  if (n < 16) return new THREE.Quaternion();
  const mean = new THREE.Vector3();
  for (const s of samples) mean.add(s);
  mean.multiplyScalar(1 / n);

  let cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
  const d = new THREE.Vector3();
  for (const s of samples) {
    d.subVectors(s, mean);
    cxx += d.x * d.x; cyy += d.y * d.y; czz += d.z * d.z;
    cxy += d.x * d.y; cxz += d.x * d.z; cyz += d.y * d.z;
  }
  const { values, vectors } = jacobi3([cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz, czz]);
  // Smallest eigenvalue → up axis.
  let mi = 0;
  if (values[1] < values[mi]) mi = 1;
  if (values[2] < values[mi]) mi = 2;
  const up = new THREE.Vector3(vectors[mi][0], vectors[mi][1], vectors[mi][2]).normalize();

  // Sign: project onto up; the "ground" end has larger horizontal spread.
  const along: number[] = samples.map((s) => d.subVectors(s, mean).dot(up));
  along.sort((a, b) => a - b);
  const lo = along[Math.floor(0.1 * n)];
  const hi = along[Math.floor(0.9 * n)];
  let spreadLo = 0, spreadHi = 0, nLo = 0, nHi = 0;
  const tmp = new THREE.Vector3();
  for (const s of samples) {
    const t = tmp.subVectors(s, mean).dot(up);
    // Horizontal offset = distance from the up axis.
    const horiz = tmp.clone().addScaledVector(up, -t).length();
    if (t <= lo) { spreadLo += horiz; nLo++; }
    else if (t >= hi) { spreadHi += horiz; nHi++; }
  }
  const avgLo = nLo ? spreadLo / nLo : 0;
  const avgHi = nHi ? spreadHi / nHi : 0;
  // Want the wider end at the BOTTOM (−up before flip). If the high end is wider,
  // flip so it becomes the bottom.
  if (avgHi > avgLo) up.negate();

  return new THREE.Quaternion().setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
}

export interface LoadSceneOptions {
  url: string | null;
  onProgress?: (percent: number) => void;
}

/** Point budgets in map mode — together they respect the 120k contract. */
const TERRAIN_POINT_SHARE = 104_000;
// 24k ≈ 4 anchor+wall+roof points per building even in city scenes (~6k
// footprints in the daejeon preset) — below this, blocks read as floating.
const BUILDING_POINT_BUDGET = 24_000;

/** `?bld=off` disables the VWorld building layer (dev-proxy dependent). */
function buildingsWanted(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('bld') !== 'off';
}

/** The world streamer runs only in the textured CONTROL view. */
function satModeOn(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('tex') === 'sat';
}

/** Append a point patch (e.g., buildings) onto a base scene cloud. */
function mergePatch(base: SceneData, patch: PointPatch): SceneData {
  if (patch.count === 0) return base;
  const positions = new Float32Array(base.positions.length + patch.positions.length);
  positions.set(base.positions, 0);
  positions.set(patch.positions, base.positions.length);
  const colors = new Float32Array(base.colors.length + patch.colors.length);
  colors.set(base.colors, 0);
  colors.set(patch.colors, base.colors.length);
  const bounds = base.bounds.clone().union(patch.bounds);
  return { positions, colors, count: base.count + patch.count, bounds, groundY: base.groundY };
}

export async function loadScene(opts: LoadSceneOptions): Promise<LoadedScene> {
  // Real-world terrain mode (?map=…): DEM tiles instead of a splat. Same
  // deterministic pipeline on both computers → identical clouds (§1). No splat
  // exists for these scenes, so STATUS falls back to its reveal cloud.
  const mapSpec = resolveMapSpec();
  if (mapSpec) {
    try {
      const wantBld = buildingsWanted();
      const terrain = await loadTerrainScene(
        mapSpec,
        opts.onProgress,
        wantBld ? TERRAIN_POINT_SHARE : undefined,
      );
      let data = terrain.data;
      const terrainPointCount = terrain.data.count;
      let buildingVisual: BuildingVisual | undefined;
      if (wantBld) {
        // Building layer is best-effort: VWorld is proxy-only (no CORS), so a
        // missing proxy/key or API hiccup degrades to terrain-only — never a
        // broken scene during a demo.
        try {
          const bld = await loadBuildings(mapSpec, terrain.ctx, BUILDING_POINT_BUDGET);
          data = mergePatch(data, bld.points);
          buildingVisual = bld.visual ?? undefined;
        } catch (err) {
          console.warn('[buildings] layer unavailable — terrain-only scene:', err);
        }
      }
      return {
        data,
        splat: null,
        terrainVisual: terrain.visual,
        terrainPointCount,
        buildingVisual,
        surroundVisual: terrain.surround,
        // World streamer (CONTROL only): surrounding cells load as the drone moves.
        streamSeed:
          wantBld && buildingVisual && satModeOn()
            ? { coreBbox: mapSpec, ctx: terrain.ctx }
            : undefined,
      };
    } catch {
      return fallback();
    }
  }
  if (!opts.url) return fallback();
  try {
    const buffer = await SplatLoader.loadFromURL(
      opts.url,
      (pct: number) => opts.onProgress?.(pct),
      false, // full buffer, not progressive
      undefined,
      1,
      0,
      true,
    );
    return deriveFromSplat(buffer, opts.url);
  } catch {
    // Network/parse failure — degrade to the shared procedural cloud.
    return fallback();
  }
}

function fallback(): LoadedScene {
  return { data: buildSceneData(), splat: null };
}

/** Percentile of a sorted ascending array. */
function pct(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

function deriveFromSplat(
  buffer: import('@mkkellogg/gaussian-splats-3d').SplatBuffer,
  url: string,
): LoadedScene {
  const total = buffer.getSplatCount();
  if (total === 0) return fallback();

  const c = new THREE.Vector3();

  // Gather sampled raw centers (for auto-leveling + robust bounds).
  const sampleStride = Math.max(1, Math.floor(total / 60_000));
  const samples: THREE.Vector3[] = [];
  for (let i = 0; i < total; i += sampleStride) {
    buffer.getSplatCenter(i, c);
    samples.push(c.clone());
  }

  // Orientation. Clean dataset splats are already gravity-aligned, so default to
  // NO rotation. ?up=<preset|euler> overrides manually; ?level=on opts into the
  // automatic PCA leveling (only useful for arbitrarily-tilted SfM/photo splats).
  const wantLevel =
    typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('level') === 'on';
  const UPRIGHT =
    manualUpQuat() ??
    (wantLevel
      ? autoLevelQuat(samples)
      : new THREE.Quaternion().setFromEuler(UP_PRESETS.x180));

  // --- ROBUST bounds (leveled frame) ---
  // Photo/SfM splats carry lots of outlier "floater" gaussians far from the real
  // structure. Naive min/max is dominated by them, so we fit to robust
  // percentile bounds instead — the building, not the noise, sets the scale.
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  for (const sv of samples) {
    c.copy(sv).applyQuaternion(UPRIGHT);
    xs.push(c.x);
    ys.push(c.y);
    zs.push(c.z);
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);

  const LO = 0.05;
  const HI = 0.95;
  const rMin = new THREE.Vector3(pct(xs, LO), pct(ys, LO), pct(zs, LO));
  const rMax = new THREE.Vector3(pct(xs, HI), pct(ys, HI), pct(zs, HI));
  const size = new THREE.Vector3().subVectors(rMax, rMin);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const s = TARGET_EXTENT / maxDim;

  const rCenter = new THREE.Vector3().addVectors(rMin, rMax).multiplyScalar(0.5);
  // world = s*(R*raw) + P ; center XZ at origin, floor the robust bottom at y=0.
  const P = new THREE.Vector3(-s * rCenter.x, -s * rMin.y, -s * rCenter.z);

  // Keep points a bit beyond the robust box so the structure isn't clipped, but
  // drop far floaters. Margin in rotated (pre-scale) units.
  const margin = maxDim * 0.15;
  const clipMin = rMin.clone().subScalar(margin);
  const clipMax = rMax.clone().addScalar(margin);

  // --- Build the downsampled, clipped world-space cloud ---
  const stride = Math.max(1, Math.floor(total / TARGET_POINTS));
  const cap = Math.ceil(total / stride);
  const positions = new Float32Array(cap * 3);
  const colors = new Float32Array(cap * 3);
  const col = new THREE.Vector4();
  const rot = new THREE.Vector3();
  const bounds = new THREE.Box3(
    new THREE.Vector3(Infinity, Infinity, Infinity),
    new THREE.Vector3(-Infinity, -Infinity, -Infinity),
  );

  let j = 0;
  for (let i = 0; i < total; i += stride) {
    buffer.getSplatCenter(i, rot);
    rot.applyQuaternion(UPRIGHT);
    // Reject floaters outside the robust box (+margin).
    if (
      rot.x < clipMin.x || rot.x > clipMax.x ||
      rot.y < clipMin.y || rot.y > clipMax.y ||
      rot.z < clipMin.z || rot.z > clipMax.z
    ) {
      continue;
    }
    c.copy(rot).multiplyScalar(s).add(P);
    positions[j * 3] = c.x;
    positions[j * 3 + 1] = c.y;
    positions[j * 3 + 2] = c.z;
    buffer.getSplatColor(i, col);
    colors[j * 3] = col.x / 255;
    colors[j * 3 + 1] = col.y / 255;
    colors[j * 3 + 2] = col.z / 255;
    bounds.expandByPoint(c);
    j++;
  }

  const data: SceneData = {
    positions: positions.subarray(0, j * 3),
    colors: colors.subarray(0, j * 3),
    count: j,
    bounds,
    groundY: 0,
  };

  const splat: SplatTransform = {
    url,
    position: [P.x, P.y, P.z],
    rotation: [UPRIGHT.x, UPRIGHT.y, UPRIGHT.z, UPRIGHT.w],
    scale: [s, s, s],
  };

  return { data, splat };
}
