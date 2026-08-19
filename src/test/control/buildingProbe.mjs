// Are the buildings under the route the real ones?
//
//   npm run demo
//   node src/test/control/buildingProbe.mjs
//
// The tower falls back to a deterministic block grid when VWorld returns no
// footprints, and that stand-in city has nothing to do with the ground the
// operator planned over: a route drawn along 충남대 제5공학관 then runs past
// invented blocks. This says which source is on screen and, for a fix given on
// the command line, how far the nearest footprint is and how big it is.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
// 충남대 제5공학관 — the building the route in the report was drawn along.
const AT = { lat: Number(process.argv[2] ?? 36.36649), lon: Number(process.argv[3] ?? 127.34478) };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const warnings = [];
page.on('console', (m) => {
  if (/buildings|vworld/i.test(m.text())) warnings.push(m.text().slice(0, 200));
});
await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.waitForTimeout(20_000);

// What the loader reported, straight from the settings panel's own text.
const reported = await page.evaluate(() => {
  const el = document.querySelector('.settings-panel');
  return el ? el.textContent.replace(/\s+/g, ' ').trim().slice(0, 400) : null;
});

// Ask VWorld the same question the viewer asks, for a small box around the fix.
const wfs = await page.evaluate(async (at) => {
  const d = 0.0012; // ~130 m
  const url =
    `/vworld/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature&TYPENAME=lt_c_bldginfo` +
    `&SRSNAME=EPSG:4326&BBOX=${at.lat - d},${at.lon - d},${at.lat + d},${at.lon + d},EPSG:4326` +
    `&maxFeatures=1000&OUTPUT=application/json`;
  const res = await fetch(url);
  if (!res.ok) return { error: `HTTP ${res.status}` };
  const json = await res.json();
  const feats = json.features ?? [];
  // Nearest footprint to the fix, in metres.
  let best = null;
  for (const f of feats) {
    const polys = f.geometry?.type === 'MultiPolygon' ? f.geometry.coordinates : [f.geometry?.coordinates];
    for (const poly of polys ?? []) {
      for (const [lon, lat] of poly?.[0] ?? []) {
        const dx = (lon - at.lon) * 111_320 * Math.cos((at.lat * Math.PI) / 180);
        const dy = (lat - at.lat) * 111_320;
        const dist = Math.hypot(dx, dy);
        if (!best || dist < best.dist) {
          best = { dist: Math.round(dist), id: f.id, height: f.properties?.height ?? null, floors: f.properties?.grnd_flr ?? null };
        }
      }
    }
  }
  return { count: feats.length, nearest: best };
}, AT);

console.log('settings panel says :', reported);
console.log('warnings            :', warnings.length ? warnings.join(' | ') : '(none)');
console.log('');
console.log(`VWorld around ${AT.lat}, ${AT.lon}:`, JSON.stringify(wfs));

await browser.close();
