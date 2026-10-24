// Are the drones drawn above the ground they are flying over?
//
//   npm run demo
//   node src/test/control/clearanceCheck.mjs
//
// The planner asks for an altitude in metres and the drones report one. The
// scene turns that number into a height with the terrain's own vertical scale.
// If the number means "above sea level" in one place and "above the ground" in
// another, the aircraft is drawn inside the hill it is supposed to be flying
// over — which looks exactly like "the drone is not on the route" even when
// every coordinate is correct.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.bringToFront();
await page.waitForTimeout(18_000);

const probe = await page.evaluate(() => {
  const frame = window.skylens.frame;
  const drones = window.skylens.fleet.drones();
  const rows = drones.map((d) => {
    const groundAlt = frame.groundAltAt(d.gps);
    const y = frame.toScene(d.gps).y;
    const groundY = frame.groundYAt(d.gps);
    return {
      station: d.station,
      reportedAltM: Math.round(d.gps.alt),
      groundAltM: Math.round(groundAlt),
      clearanceM: Math.round(d.gps.alt - groundAlt),
      sceneY: Number(y.toFixed(2)),
      groundSceneY: Number(groundY.toFixed(2)),
      aboveSurface: y > groundY,
    };
  });
  const route = window.skylens.viewer.debugRoute?.() ?? null;
  const routeRows = (route ?? []).map((v) => {
    const gps = frame.toGps(v);
    return {
      altM: Math.round(gps.alt),
      groundAltM: Math.round(frame.groundAltAt(gps)),
      clearanceM: Math.round(gps.alt - frame.groundAltAt(gps)),
    };
  });
  return { anchor: frame.anchor, rows, routeRows };
});

console.log('frame anchor (bbox lowest elevation):', JSON.stringify(probe.anchor));
console.log('');
console.log('drones:');
for (const r of probe.rows) {
  console.log(
    `  ${r.station.padEnd(7)} alt ${String(r.reportedAltM).padStart(4)} m · ground ` +
      `${String(r.groundAltM).padStart(4)} m · clearance ${String(r.clearanceM).padStart(5)} m` +
      `   y ${String(r.sceneY).padStart(7)} vs surface ${String(r.groundSceneY).padStart(7)}` +
      `   ${r.aboveSurface ? 'above ground' : 'UNDER THE TERRAIN'}`,
  );
}
console.log('');
console.log('route vertices:');
for (const r of probe.routeRows) {
  console.log(
    `  alt ${String(r.altM).padStart(4)} m · ground ${String(r.groundAltM).padStart(4)} m ` +
      `· clearance ${String(r.clearanceM).padStart(5)} m`,
  );
}

console.log('');
console.log('===== RESULT =====');
// The planner sets ONE clearance for the route, so every vertex must hold it
// over its own ground — that is the whole difference between an altitude and
// an elevation, and getting it wrong drew the flight inside the hills.
const clearances = probe.routeRows.map((r) => r.clearanceM);
const spread = clearances.length
  ? Math.max(...clearances) - Math.min(...clearances)
  : Infinity;
const checks = [
  ['every drone is drawn above the ground', probe.rows.every((r) => r.aboveSurface)],
  ['the drones have real clearance (>15 m)', probe.rows.every((r) => r.clearanceM > 15)],
  ['a route was drawn', probe.routeRows.length > 0],
  [
    `every route vertex holds the same clearance (spread ${spread} m)`,
    probe.routeRows.length > 0 && spread <= 5,
  ],
  ['the route clears the terrain', clearances.every((c) => c > 15)],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
