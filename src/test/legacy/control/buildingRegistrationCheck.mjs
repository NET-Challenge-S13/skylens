// Are the building prisms where VWorld says the buildings are?
//
//   npm run demo
//   node src/test/control/buildingRegistrationCheck.mjs
//
// registrationCheck.mjs compares the satellite drape against the planner map.
// It cannot see a building-layer error: in the two building display modes the
// drape is not on screen, so prisms could sit tens of metres off their real
// footprints and every other check would still pass — while an operator
// watching the flight sees the aircraft cross the wrong block.
//
// So this asks VWorld for footprints directly, converts each one into the
// scene's own frame, and looks for a rendered prism vertex there. The answer is
// a distance in metres per building, and a median offset over the sample.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
// Spread across the loaded area: a uniform shift and a scale error look the
// same at one point and different across the map.
const SAMPLES = [
  { name: '충남대 공대 (route)', lat: 36.36649, lon: 127.34478 },
  { name: '충남대 정문 방면', lat: 36.3625, lon: 127.3455 },
  { name: '유성 시가지 북', lat: 36.3755, lon: 127.3405 },
  { name: '동편 카이스트 방면', lat: 36.3705, lon: 127.3605 },
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.waitForTimeout(22_000);

const results = await page.evaluate(async (samples) => {
  const THREE = await import('/node_modules/three/build/three.module.js');
  const frame = window.skylens.frame;
  const scene = window.skylens.viewer.debugScene();

  // Every rendered building vertex, in world units. Buildings are the meshes
  // with many vertices that are not the terrain drape (which is a regular grid
  // spanning the whole bbox), so collect candidates and let the distance decide.
  const clouds = [];
  scene.traverse((o) => {
    const pos = o.geometry?.attributes?.position;
    if (!pos || pos.count < 100) return;
    o.updateWorldMatrix(true, false);
    clouds.push({ pos, matrix: o.matrixWorld.clone(), count: pos.count });
  });

  const out = [];
  for (const s of samples) {
    const d = 0.0012;
    const url =
      `/vworld/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature&TYPENAME=lt_c_bldginfo` +
      `&SRSNAME=EPSG:4326&BBOX=${s.lat - d},${s.lon - d},${s.lat + d},${s.lon + d},EPSG:4326` +
      `&maxFeatures=200&OUTPUT=application/json`;
    let feats = [];
    try {
      const res = await fetch(url);
      if (res.ok) feats = (await res.json()).features ?? [];
    } catch {
      /* reported as no data below */
    }
    if (feats.length === 0) {
      out.push({ ...s, footprints: 0, nearestM: null });
      continue;
    }

    // A corner of the footprint nearest the sample: corners are what a prism
    // reproduces exactly, unlike an interpolated edge point.
    const f = feats[0];
    const polys = f.geometry?.type === 'MultiPolygon' ? f.geometry.coordinates : [f.geometry?.coordinates];
    const ring = polys?.[0]?.[0] ?? [];
    if (ring.length < 3) {
      out.push({ ...s, footprints: feats.length, nearestM: null });
      continue;
    }

    // Where that corner should be in the scene, and what is actually there.
    let worst = 0;
    const perCorner = [];
    for (const [lon, lat] of ring.slice(0, 6)) {
      const want = frame.toScene({ lat, lon, alt: 0 });
      let best = Infinity;
      const v = new THREE.Vector3();
      for (const c of clouds) {
        for (let i = 0; i < c.count; i++) {
          v.fromBufferAttribute(c.pos, i).applyMatrix4(c.matrix);
          // Horizontal only: height is exaggerated on purpose.
          const dist = Math.hypot(v.x - want.x, v.z - want.z);
          if (dist < best) best = dist;
        }
      }
      const metres = best / frame.unitsPerMeter;
      perCorner.push(Math.round(metres * 10) / 10);
      worst = Math.max(worst, metres);
    }
    perCorner.sort((a, b) => a - b);
    out.push({
      ...s,
      footprints: feats.length,
      id: f.id,
      heightProp: f.properties?.height ?? null,
      floors: f.properties?.grnd_flr ?? null,
      cornersM: perCorner,
      medianM: perCorner[Math.floor(perCorner.length / 2)],
    });
  }
  return out;
}, SAMPLES);

for (const r of results) {
  console.log(
    `${r.name.padEnd(20)} footprints ${String(r.footprints).padStart(3)}` +
      (r.medianM == null
        ? '  — no geometry to compare'
        : `  nearest prism vertex per corner (m): ${JSON.stringify(r.cornersM)}` +
          `  median ${r.medianM} m   [height=${r.heightProp} floors=${r.floors}]`),
  );
}

console.log('');
console.log('===== RESULT =====');
const medians = results.map((r) => r.medianM).filter((m) => m != null);
const overall = medians.length ? medians.sort((a, b) => a - b)[Math.floor(medians.length / 2)] : null;
const checks = [
  ['VWorld answered for the sampled areas', results.every((r) => r.footprints > 0)],
  [
    `prisms sit on their footprints (median ${overall ?? 'n/a'} m)`,
    overall != null && overall <= 8,
  ],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
