// VWorld building layer for terrain scenes (?map=…) — real national building
// footprints (lt_c_bldginfo) extruded onto the DEM and sampled into the same
// point-cloud contract, using the terrain's own transform so both layers align.
//
// Measured facts this code relies on (2026-08-08 curl checks):
// - WFS 1.1.0 + EPSG:4326 BBOX axis order is LAT,LON (lon,lat order returns 0).
// - Response GeoJSON coordinates are lon,lat (normal order).
// - maxFeatures caps at 1000/request → adaptive bbox quad-split + id dedupe.
// - `height`/`grnd_flr` are often null in rural areas → floors×3 m → 6 m.
// - api.vworld.kr sends no CORS headers → all requests go through the Vite
//   `/vworld` dev proxy, which injects key+domain server-side (vite.config.ts).
//   No proxy (prod build) → fetch fails → caller degrades to terrain-only.

import * as THREE from 'three';
import type { Bbox, TerrainContext } from './terrainSource.ts';

const WFS_PATH = '/vworld/wfs';
const MAX_FEATURES = 1000;
/** Don't quad-split below ~50 m cells. */
const MIN_CELL_DEG = 0.0005;
// Depth 7 reaches ~55 m cells from ring-sized bboxes — depth 5 truncated
// dense city blocks (>1000 rows in a 200 m cell) and left building holes.
const MAX_DEPTH = 7;

/** Same deterministic PRNG as sceneData.ts — identical clouds on both computers. */
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

type LonLat = [number, number];

interface Building {
  id: string;
  /** Outer rings only (holes ignored — point sampling doesn't need them). */
  rings: LonLat[][];
  heightM: number;
}

export interface PointPatch {
  positions: Float32Array;
  colors: Float32Array;
  count: number;
  bounds: THREE.Box3;
}

/** Non-indexed triangle soup (flat-shaded prisms) — display-only. */
export interface BuildingVisual {
  positions: Float32Array;
  colors: Float32Array;
}

export interface BuildingLayer {
  points: PointPatch;
  /** Extruded walls+roof per footprint; SIM `?tex=sat` renders these INSTEAD
   *  of the building points, so buildings can't read as floating clumps. */
  visual: BuildingVisual | null;
}

export interface BuildingLoadOptions {
  /** Skip point sampling — prisms only. Used by the display-only context ring;
   *  consumes NO PRNG draws, so the shared point stream stays untouched. */
  prismsOnly?: boolean;
  /** Drop buildings whose centroid falls inside this bbox (already rendered
   *  by another layer). */
  excludeBbox?: Bbox;
  /** Keep ONLY buildings whose centroid falls inside this bbox — streamed
   *  cells use it so border-crossing footprints aren't drawn by two cells. */
  onlyBbox?: Bbox;
  /** World-Y offset for building bases (the ring terrain sits slightly lower). */
  yOffset?: number;
}

function parseFeatures(json: unknown, out: Map<string, Building>): number {
  const feats: any[] = (json as any)?.features ?? [];
  for (const f of feats) {
    const id = String(f?.id ?? '');
    if (!id || out.has(id)) continue;
    const geom = f?.geometry;
    if (!geom) continue;
    const rings: LonLat[][] = [];
    if (geom.type === 'MultiPolygon') {
      for (const poly of geom.coordinates ?? []) {
        if (poly?.[0]?.length >= 3) rings.push(poly[0] as LonLat[]);
      }
    } else if (geom.type === 'Polygon') {
      if (geom.coordinates?.[0]?.length >= 3) rings.push(geom.coordinates[0] as LonLat[]);
    }
    if (rings.length === 0) continue;

    const props = f?.properties ?? {};
    const h = Number(props.height);
    const floors = Number(props.grnd_flr);
    // Rural rows often carry null height AND null floors — fall back to 6 m.
    const heightM = h > 2 ? h : floors > 0 ? floors * 3 : 6;
    out.set(id, { id, rings, heightM });
  }
  return feats.length;
}

/** Fetch one bbox cell; quad-split when the 1000-row cap suggests truncation. */
async function fetchCell(
  b: Bbox,
  out: Map<string, Building>,
  depth: number,
): Promise<void> {
  const [w, s, e, n] = b;
  // BBOX is LAT,LON order for WFS 1.1.0 + EPSG:4326 (measured — see header).
  const url =
    `${WFS_PATH}?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature` +
    `&TYPENAME=lt_c_bldginfo&SRSNAME=EPSG:4326` +
    `&BBOX=${s},${w},${n},${e},EPSG:4326` +
    `&maxFeatures=${MAX_FEATURES}&OUTPUT=application/json`;
  let json: unknown;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`vworld wfs: HTTP ${res.status}`);
    json = await res.json();
  } catch (err) {
    // One flaky cell must not nuke the whole layer — keep what the rest found.
    console.warn('[buildings] cell fetch failed (partial layer):', err);
    return;
  }

  // Probe row count BEFORE committing: a full page means truncation is likely,
  // so re-fetch the area as 4 sub-cells instead (children see everything).
  const probe = new Map<string, Building>();
  const rows = parseFeatures(json, probe);
  const canSplit = depth < MAX_DEPTH && (e - w) / 2 > MIN_CELL_DEG && (n - s) / 2 > MIN_CELL_DEG;
  if (rows >= MAX_FEATURES && canSplit) {
    const mx = (w + e) / 2;
    const my = (s + n) / 2;
    await Promise.all([
      fetchCell([w, s, mx, my], out, depth + 1),
      fetchCell([mx, s, e, my], out, depth + 1),
      fetchCell([w, my, mx, n], out, depth + 1),
      fetchCell([mx, my, e, n], out, depth + 1),
    ]);
    return;
  }
  for (const [id, bld] of probe) if (!out.has(id)) out.set(id, bld);
}

function ringAreaAndPerimeter(ring: LonLat[], ctx: TerrainContext): { area: number; perim: number } {
  // Shoelace in local meters (equirectangular — fine at building scale).
  let area = 0;
  let perim = 0;
  for (let i = 0; i < ring.length; i++) {
    const [lon1, lat1] = ring[i];
    const [lon2, lat2] = ring[(i + 1) % ring.length];
    const x1 = (lon1 - ctx.lon0) * ctx.mPerDegLon;
    const y1 = (lat1 - ctx.lat0) * ctx.mPerDegLat;
    const x2 = (lon2 - ctx.lon0) * ctx.mPerDegLon;
    const y2 = (lat2 - ctx.lat0) * ctx.mPerDegLat;
    area += x1 * y2 - x2 * y1;
    perim += Math.hypot(x2 - x1, y2 - y1);
  }
  return { area: Math.abs(area) / 2, perim };
}

function pointInRing(lon: number, lat: number, ring: LonLat[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Fetch buildings in the bbox and sample them into points in the terrain's
 * world frame: base snapped to the DEM surface, height extruded with the same
 * vertical exaggeration as the terrain so the layers stay proportionate.
 */
export async function loadBuildings(
  bbox: Bbox,
  ctx: TerrainContext,
  budget: number,
  opts?: BuildingLoadOptions,
): Promise<BuildingLayer> {
  const found = new Map<string, Building>();
  await fetchCell(bbox, found, 0);

  const empty: PointPatch = {
    positions: new Float32Array(0),
    colors: new Float32Array(0),
    count: 0,
    bounds: new THREE.Box3(),
  };
  if (found.size === 0) return { points: empty, visual: null };

  // Weight per building: wall area + roof area (meters²-ish) for fair budget split.
  const items = [...found.values()].map((b) => {
    const outer = b.rings[0];
    const { area, perim } = ringAreaAndPerimeter(outer, ctx);
    return { b, area, perim, weight: perim * b.heightM + area };
  });
  const totalWeight = items.reduce((acc, it) => acc + it.weight, 0) || 1;

  const rand = mulberry32(0xb17d5eed);
  const positions: number[] = [];
  const colors: number[] = [];
  const bounds = new THREE.Box3(
    new THREE.Vector3(Infinity, Infinity, Infinity),
    new THREE.Vector3(-Infinity, -Infinity, -Infinity),
  );
  const wall = new THREE.Color(0x8e959e);
  const roof = new THREE.Color(0xd3d8de);
  const p = new THREE.Vector3();

  // Display prisms (walls + roof cap). No PRNG draws in this path — the shared
  // point stream must stay identical on both computers even though only SIM
  // renders the prisms.
  const vPos: number[] = [];
  const vCol: number[] = [];
  const wallC = new THREE.Color(0x9aa2ab).convertSRGBToLinear();
  const roofC = new THREE.Color(0xdde2e7).convertSRGBToLinear();
  const pushTri = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
    c: THREE.Color,
  ): void => {
    vPos.push(ax, ay, az, bx, by, bz, cx, cy, cz);
    vCol.push(c.r, c.g, c.b, c.r, c.g, c.b, c.r, c.g, c.b);
  };

  const toWorld = (lon: number, lat: number, elevY: number): THREE.Vector3 =>
    p.set(
      (lon - ctx.lon0) * ctx.mPerDegLon * ctx.s,
      elevY,
      -(lat - ctx.lat0) * ctx.mPerDegLat * ctx.s,
    );

  const push = (v: THREE.Vector3, c: THREE.Color, dim: number): void => {
    positions.push(v.x, v.y, v.z);
    colors.push(
      Math.min(c.r * dim, 1),
      Math.min(c.g * dim, 1),
      Math.min(c.b * dim, 1),
    );
    bounds.expandByPoint(v);
  };

  // Cartographic minimum size: on wide scenes (uljin preset = ~14 km across) a
  // 6 m house is sub-pixel at true scale, so buildings get scaled UP to stay
  // readable as landmarks. Narrow scenes (uljinup) pass the threshold untouched.
  const MIN_HEIGHT_WORLD = 0.25;
  const MAX_INFLATE = 8;

  for (const { b, weight } of items) {
    const outerRaw = b.rings[0];

    // Ground snap at the LOWEST ground under the footprint (vertices +
    // centroid) — a centroid-only snap leaves downhill edges hovering on slopes.
    let cLon = 0;
    let cLat = 0;
    for (const [lon, lat] of outerRaw) {
      cLon += lon;
      cLat += lat;
    }
    cLon /= outerRaw.length;
    cLat /= outerRaw.length;
    if (opts?.excludeBbox) {
      const [xw, xs, xe, xn] = opts.excludeBbox;
      if (cLon >= xw && cLon <= xe && cLat >= xs && cLat <= xn) continue;
    }
    if (opts?.onlyBbox) {
      const [ow, os, oe, on] = opts.onlyBbox;
      if (cLon < ow || cLon > oe || cLat < os || cLat > on) continue;
    }
    let baseE = ctx.elevationAt(cLon, cLat);
    for (const [lon, lat] of outerRaw) {
      const e = ctx.elevationAt(lon, lat);
      if (e < baseE) baseE = e;
    }
    const baseY = (baseE - ctx.minE) * ctx.s * ctx.exaggeration + (opts?.yOffset ?? 0);

    let hWorld = b.heightM * ctx.s * ctx.exaggeration;
    let outer = outerRaw;
    if (hWorld < MIN_HEIGHT_WORLD && hWorld > 0) {
      const inflate = Math.min(MAX_INFLATE, MIN_HEIGHT_WORLD / hWorld);
      hWorld *= inflate;
      // Footprint grows gently (sqrt) so dense villages don't fuse into blobs.
      const fp = Math.min(3, Math.sqrt(inflate));
      outer = outerRaw.map(
        ([lon, lat]) =>
          [cLon + (lon - cLon) * fp, cLat + (lat - cLat) * fp] as LonLat,
      );
    }

    if (!opts?.prismsOnly) {
      const n = Math.max(6, Math.round((budget * weight) / totalWeight));

      // Footprint bbox for rejection sampling.
      let minLon = Infinity;
      let maxLon = -Infinity;
      let minLat = Infinity;
      let maxLat = -Infinity;
      for (const [lon, lat] of outer) {
        if (lon < minLon) minLon = lon;
        if (lon > maxLon) maxLon = lon;
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
      }

      const nRoof = Math.max(2, Math.floor(n * 0.35));
      const nBase = Math.max(2, Math.floor(n * 0.25));
      const nWall = Math.max(2, n - nRoof - nBase);

      // Roof: rejection-sample inside the outer ring at base+height.
      let placed = 0;
      for (let tries = 0; placed < nRoof && tries < nRoof * 8; tries++) {
        const lon = minLon + rand() * (maxLon - minLon);
        const lat = minLat + rand() * (maxLat - minLat);
        if (!pointInRing(lon, lat, outer)) continue;
        push(toWorld(lon, lat, baseY + hWorld), roof, 0.9 + rand() * 0.2);
        placed++;
      }

      // Walls: random position along the ring edges, random height.
      const edgeLens: number[] = [];
      let perim = 0;
      for (let i = 0; i < outer.length; i++) {
        const [lon1, lat1] = outer[i];
        const [lon2, lat2] = outer[(i + 1) % outer.length];
        const len = Math.hypot(
          (lon2 - lon1) * ctx.mPerDegLon,
          (lat2 - lat1) * ctx.mPerDegLat,
        );
        edgeLens.push(len);
        perim += len;
      }
      const onRing = (): [number, number] => {
        let target = rand() * perim;
        let edge = 0;
        while (edge < edgeLens.length - 1 && target > edgeLens[edge]) {
          target -= edgeLens[edge];
          edge++;
        }
        const t = edgeLens[edge] > 0 ? target / edgeLens[edge] : 0;
        const [lon1, lat1] = outer[edge];
        const [lon2, lat2] = outer[(edge + 1) % outer.length];
        return [lon1 + (lon2 - lon1) * t, lat1 + (lat2 - lat1) * t];
      };
      for (let k = 0; k < nWall; k++) {
        const [lon, lat] = onRing();
        push(toWorld(lon, lat, baseY + rand() * hWorld), wall, 0.8 + rand() * 0.3);
      }
      // Ground ring: anchor points that sit ON the local terrain height along
      // the footprint. Tight budgets (~6 pts/bldg) put everything at
      // roof/mid-wall height otherwise, and whole blocks read as floating.
      for (let k = 0; k < nBase; k++) {
        const [lon, lat] = onRing();
        const gy = (ctx.elevationAt(lon, lat) - ctx.minE) * ctx.s * ctx.exaggeration;
        push(toWorld(lon, lat, gy), wall, 0.7 + rand() * 0.2);
      }
    }

    // Prism: wall quad per ring edge + triangulated roof cap, continuous from
    // just under the terrain to the roofline.
    const yb = baseY - 0.03;
    const yt = baseY + hWorld;
    const ringXZ = outer.map(
      ([lon, lat]) =>
        [
          (lon - ctx.lon0) * ctx.mPerDegLon * ctx.s,
          -(lat - ctx.lat0) * ctx.mPerDegLat * ctx.s,
        ] as [number, number],
    );
    for (let i = 0; i < ringXZ.length; i++) {
      const [x1, z1] = ringXZ[i];
      const [x2, z2] = ringXZ[(i + 1) % ringXZ.length];
      pushTri(x1, yb, z1, x2, yb, z2, x2, yt, z2, wallC);
      pushTri(x1, yb, z1, x2, yt, z2, x1, yt, z1, wallC);
    }
    try {
      const shape = ringXZ.map(([x, z]) => new THREE.Vector2(x, z));
      for (const tri of THREE.ShapeUtils.triangulateShape(shape, [])) {
        pushTri(
          ringXZ[tri[0]][0], yt, ringXZ[tri[0]][1],
          ringXZ[tri[1]][0], yt, ringXZ[tri[1]][1],
          ringXZ[tri[2]][0], yt, ringXZ[tri[2]][1],
          roofC,
        );
      }
    } catch {
      // Degenerate/self-intersecting ring — walls alone still read as a building.
    }
  }

  const count = positions.length / 3;
  console.info(
    `[buildings] ${found.size} footprints → ${count} points, ${vPos.length / 9} prism tris`,
  );
  return {
    points: {
      positions: new Float32Array(positions),
      colors: new Float32Array(colors),
      count,
      bounds,
    },
    visual: vPos.length
      ? { positions: new Float32Array(vPos), colors: new Float32Array(vCol) }
      : null,
  };
}
