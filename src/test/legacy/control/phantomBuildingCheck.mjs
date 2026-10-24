// Is anything drawn that VWorld never sent?
//
//   npm run demo
//   node src/test/control/phantomBuildingCheck.mjs
//
// buildingRegistrationCheck.mjs asks whether every real footprint is drawn in
// the right place. This asks the opposite question, which is the one an
// operator actually hits: is everything drawn a real building? A stand-in block
// laid over real ground looks exactly like a building that is not there, and
// the route then appears to cross a block that does not exist.
//
// Method: take the corners of what is rendered around a fix, convert them back
// to GPS through the scene's own frame, and measure each against the nearest
// VWorld footprint vertex in the same area.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const AT = { lat: Number(process.argv[2] ?? 36.3665), lon: Number(process.argv[3] ?? 127.3448) };
const BOX_M = 200;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.waitForTimeout(22_000);

const out = await page.evaluate(
  async ([at, boxM]) => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const frame = window.skylens.frame;
    const scene = window.skylens.viewer.debugScene();

    const dLat = boxM / 111_320;
    const dLon = boxM / (111_320 * Math.cos((at.lat * Math.PI) / 180));
    const box = {
      south: at.lat - dLat / 2,
      north: at.lat + dLat / 2,
      west: at.lon - dLon / 2,
      east: at.lon + dLon / 2,
    };

    // Every real footprint in the box, kept whole so coverage can be counted
    // building by building — a layer can be perfectly placed and still be
    // missing half the blocks, which reads to an operator as flying over the
    // wrong ground just as strongly as an offset does.
    const url =
      `/vworld/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature&TYPENAME=lt_c_bldginfo` +
      `&SRSNAME=EPSG:4326&BBOX=${box.south - 0.002},${box.west - 0.002},${box.north + 0.002},${box.east + 0.002},EPSG:4326` +
      `&maxFeatures=500&OUTPUT=application/json`;
    const res = await fetch(url);
    const feats = res.ok ? ((await res.json()).features ?? []) : [];
    const real = [];
    const buildings = [];
    for (const f of feats) {
      const polys =
        f.geometry?.type === 'MultiPolygon' ? f.geometry.coordinates : [f.geometry?.coordinates];
      const corners = [];
      for (const poly of polys ?? []) for (const c of poly?.[0] ?? []) corners.push(c);
      if (corners.length === 0) continue;
      real.push(...corners);
      const cLon = corners.reduce((a, c) => a + c[0], 0) / corners.length;
      const cLat = corners.reduce((a, c) => a + c[1], 0) / corners.length;
      // Only buildings whose middle is inside the box: an edge case straddling
      // the boundary is neither drawn nor missing in any meaningful sense.
      if (cLat < box.south || cLat > box.north || cLon < box.west || cLon > box.east) continue;
      buildings.push({
        id: f.id,
        name: f.properties?.dong_nm ?? null,
        floors: f.properties?.grnd_flr ?? null,
        height: f.properties?.height ?? null,
        area: f.properties?.archarea ?? null,
        corners,
      });
    }

    // Every rendered vertex in the box, above the ground (roof/wall corners).
    const v = new THREE.Vector3();
    const rendered = [];
    scene.traverse((o) => {
      const pos = o.geometry?.attributes?.position;
      // Buildings only. Drone rigs, the route line and the danger arcs are all
      // meshes sitting above the ground, and their corners are nowhere near a
      // footprint by design.
      if (!pos || !o.isMesh || !/^buildings/.test(o.name)) return;
      o.updateWorldMatrix(true, false);
      for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld);
        const gps = frame.toGps(v);
        if (gps.lat < box.south || gps.lat > box.north) continue;
        if (gps.lon < box.west || gps.lon > box.east) continue;
        // The terrain drape is a mesh too; its vertices sit ON the ground.
        if (gps.alt - frame.groundAltAt(gps) < 2) continue;
        rendered.push([gps.lon, gps.lat]);
      }
    });

    // Distance from each rendered corner to the nearest real footprint vertex.
    const mLat = 111_320;
    const mLon = 111_320 * Math.cos((at.lat * Math.PI) / 180);
    const dists = [];
    for (const [lon, lat] of rendered) {
      let best = Infinity;
      for (const [rlon, rlat] of real) {
        const d = Math.hypot((lon - rlon) * mLon, (lat - rlat) * mLat);
        if (d < best) best = d;
      }
      dists.push(best);
    }
    // Which real buildings made it onto the screen at all.
    const missing = [];
    for (const b of buildings) {
      let drawn = false;
      for (const [lon, lat] of b.corners) {
        for (const [rlon, rlat] of rendered) {
          if (Math.hypot((lon - rlon) * mLon, (lat - rlat) * mLat) < 2) {
            drawn = true;
            break;
          }
        }
        if (drawn) break;
      }
      if (!drawn) missing.push({ id: b.id, name: b.name, floors: b.floors, area: b.area });
    }

    dists.sort((a, b) => a - b);
    const q = (p) => (dists.length ? Math.round(dists[Math.floor((dists.length - 1) * p)] * 10) / 10 : null);
    return {
      realVertices: real.length,
      renderedVertices: rendered.length,
      median: q(0.5),
      p90: q(0.9),
      max: q(1),
      farCount: dists.filter((d) => d > 20).length,
      buildings: buildings.length,
      missing,
    };
  },
  [AT, BOX_M],
);

console.log(`around ${AT.lat}, ${AT.lon} (${BOX_M} m box):`);
console.log('  VWorld footprint vertices :', out.realVertices);
console.log('  rendered building vertices:', out.renderedVertices);
console.log(`  distance to the nearest real vertex — median ${out.median} m · p90 ${out.p90} m · max ${out.max} m`);
console.log(`  vertices further than 20 m from any real footprint: ${out.farCount}`);
console.log(`  buildings in the box: ${out.buildings} · not drawn: ${out.missing.length}`);
for (const m of out.missing) {
  console.log(`    missing: ${m.name ?? '(no name)'} · ${m.floors ?? '?'}층 · ${m.area ?? '?'} m2 · ${m.id}`);
}

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['VWorld has footprints here', out.realVertices > 0],
  ['the tower drew buildings here', out.renderedVertices > 0],
  ['nothing is drawn that VWorld does not have', out.farCount === 0],
  ['every real building in the box is drawn', out.missing.length === 0],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
