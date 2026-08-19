// Do the buildings stand as tall as the buildings they represent?
//
//   npm run demo
//   node src/test/control/buildingHeightCheck.mjs
//
// Position is not the only way a 3D view can stop matching the map. If a
// one-storey warehouse and a four-storey hall are both drawn as the same slab,
// the operator has no skyline to recognise the place by — the aircraft looks
// like it is somewhere else even though its coordinates are exact.
//
// So: for the buildings around the demo route, compare what the tower drew with
// what VWorld says (height when it has one, floors otherwise).

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const AT = { lat: 36.3665, lon: 127.3448 };
const BOX_M = 400;
/** Storey height used to read a floor count as metres. */
const FLOOR_M = 3.3;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.waitForTimeout(20_000);

const rows = await page.evaluate(
  async ([at, boxM, floorM]) => {
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

    const url =
      `/vworld/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature&TYPENAME=lt_c_bldginfo` +
      `&SRSNAME=EPSG:4326&BBOX=${box.south},${box.west},${box.north},${box.east},EPSG:4326` +
      `&maxFeatures=300&OUTPUT=application/json`;
    const res = await fetch(url);
    const feats = res.ok ? ((await res.json()).features ?? []) : [];

    // Rendered building corners, as GPS + height above their own ground.
    const v = new THREE.Vector3();
    const drawn = [];
    scene.traverse((o) => {
      const pos = o.geometry?.attributes?.position;
      if (!pos || !o.isMesh || !/^buildings/.test(o.name)) return;
      o.updateWorldMatrix(true, false);
      for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld);
        const gps = frame.toGps(v);
        if (gps.lat < box.south || gps.lat > box.north) continue;
        if (gps.lon < box.west || gps.lon > box.east) continue;
        drawn.push({ lat: gps.lat, lon: gps.lon, above: gps.alt - frame.groundAltAt(gps) });
      }
    });

    const mLat = 111_320;
    const mLon = 111_320 * Math.cos((at.lat * Math.PI) / 180);

    // Rings first, so a drawn corner can be attributed to exactly one building.
    // Campus blocks share walls: a corner within reach of two footprints says
    // nothing about either one's height, and counting it made a one-storey
    // store look like it had been drawn five times too tall.
    const rings = [];
    for (const f of feats) {
      const polys =
        f.geometry?.type === 'MultiPolygon' ? f.geometry.coordinates : [f.geometry?.coordinates];
      const ring = polys?.[0]?.[0];
      if (ring && ring.length >= 3) rings.push(ring);
    }
    const claims = (ring, d) => {
      for (const [lon, lat] of ring) {
        if (Math.hypot((d.lon - lon) * mLon, (d.lat - lat) * mLat) < 1.5) return true;
      }
      return false;
    };

    const out = [];
    for (const f of feats) {
      const polys =
        f.geometry?.type === 'MultiPolygon' ? f.geometry.coordinates : [f.geometry?.coordinates];
      const ring = polys?.[0]?.[0];
      if (!ring || ring.length < 3) continue;
      const cLon = ring.reduce((a, c) => a + c[0], 0) / ring.length;
      const cLat = ring.reduce((a, c) => a + c[1], 0) / ring.length;
      if (cLat < box.south || cLat > box.north || cLon < box.west || cLon > box.east) continue;

      // Tallest drawn corner that belongs to THIS footprint and no other.
      let tallest = 0;
      let claimed = 0;
      for (const d of drawn) {
        if (!claims(ring, d)) continue;
        if (rings.filter((r) => claims(r, d)).length > 1) continue;
        claimed++;
        if (d.above > tallest) tallest = d.above;
      }
      if (claimed === 0) continue;

      const h = Number(f.properties?.height ?? 0);
      const floors = Number(f.properties?.grnd_flr ?? 0);
      const stated = h > 2 ? h : floors > 0 ? floors * floorM : null;
      out.push({
        name: f.properties?.dong_nm ?? null,
        floors: floors || null,
        heightProp: h || null,
        statedM: stated == null ? null : Math.round(stated * 10) / 10,
        drawnM: Math.round(tallest * 10) / 10,
      });
    }
    out.sort((a, b) => b.drawnM - a.drawnM);
    return out;
  },
  [AT, BOX_M, FLOOR_M],
);

console.log('building heights around the demo route:');
console.log('  (drawn = height above its own ground in the scene)');
for (const r of rows) {
  const ratio = r.statedM ? (r.drawnM / r.statedM).toFixed(2) : '—';
  console.log(
    `  ${(r.name ?? '(이름 없음)').padEnd(24)} ` +
      `${String(r.floors ?? '?').padStart(2)}층 · height=${String(r.heightProp ?? '없음').padStart(5)} · ` +
      `실제 ${String(r.statedM ?? '?').padStart(5)} m → 화면 ${String(r.drawnM).padStart(5)} m  (x${ratio})`,
  );
}

const known = rows.filter((r) => r.statedM != null);
const ratios = known.map((r) => r.drawnM / r.statedM).sort((a, b) => a - b);
const median = ratios.length ? ratios[Math.floor(ratios.length / 2)] : null;
const flattened = known.filter((r) => r.drawnM > r.statedM * 1.8).length;

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['buildings were measured', rows.length > 0],
  [
    `heights are near their stated size (median x${median ? median.toFixed(2) : 'n/a'})`,
    median != null && median <= 1.8,
  ],
  [`no building is drawn far taller than it is (${flattened} are)`, flattened === 0],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
