// The route from the report, seen on both maps.
//
//   npm run demo
//   node src/test/control/routeOverBuildingCheck.mjs
//
// Plans the exact waypoints the operator used (along 충남대 제5공학관) and then
// photographs the same ground twice at the same scale: the planner map with the
// markers on it, and the 3D view from straight above with the route line and
// the aircraft on it. If the flight is over the building it was drawn on, the
// two pictures put the line on the same roof.

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';
import { matchImages } from './matchImages.mjs';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const SHOTS = 'C:/tmp/skylens-shots';
const SPAN_M = 300;

// Read off the operator's planner list, in order.
const WAYPOINTS = [
  { lat: 36.36644, lon: 127.34523 },
  { lat: 36.36649, lon: 127.34478 },
  { lat: 36.36652, lon: 127.34446 },
  { lat: 36.36656, lon: 127.34405 },
  { lat: 36.36662, lon: 127.34369 },
];
const MID = { lat: 36.36652, lon: 127.34446 };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.bringToFront();
await page.waitForTimeout(22_000);

// --- plan it exactly as the operator did ----------------------------------
await page.evaluate(
  async ([wps, mid, span]) => {
    const rm = window.skylens.routeModal;
    rm.debugView(mid, span);
    rm.open();
    const frame = window.skylens.frame;
    // Same conversion the planner does: the operator's clearance over the
    // ground under each point.
    const waypoints = wps.map((w) => ({
      ...w,
      alt: Math.round(frame.groundAltAt({ ...w, alt: 0 }) + 60),
    }));
    const leader = window.skylens.fleet.drones().find((d) => d.station === 'center');
    window.skylens.core.send({
      kind: 'assign-route',
      droneId: leader?.id ?? 1,
      waypoints,
      loop: true,
    });
    return waypoints;
  },
  [WAYPOINTS, MID, SPAN_M],
);
await page.waitForTimeout(5000);
const plannerPng = await page.locator('.route-modal__canvas').screenshot();
writeFileSync(`${SHOTS}/route-planner.png`, plannerPng);
await page.evaluate(() => window.skylens.routeModal.close());

// --- let the formation reach it, then look straight down --------------------
await page.waitForTimeout(35_000);
const cam = await page.evaluate(
  async ([mid, span]) => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const frame = window.skylens.frame;
    window.skylens.viewer.setDisplay('aerial');
    const at = frame.toScene({ ...mid, alt: 0 });
    const groundY = frame.groundYAt({ ...mid, alt: 0 });
    const above = ((span / 2) * frame.unitsPerMeter) / Math.tan((55 / 2) * (Math.PI / 180));
    window.skylens.viewer.debugTopDown(new THREE.Vector3(at.x, groundY, at.z), above);
    const drones = window.skylens.fleet.drones().map((d) => ({
      station: d.station,
      lat: Number(d.gps.lat.toFixed(5)),
      lon: Number(d.gps.lon.toFixed(5)),
    }));
    return { drones };
  },
  [MID, SPAN_M],
);
console.log('drones at capture:', JSON.stringify(cam.drones));
await page.waitForTimeout(2500);
const aerialPng = await page.locator('canvas').first().screenshot();
writeFileSync(`${SHOTS}/route-3d-aerial.png`, aerialPng);

// The same view in the default building mode — what the operator was looking at.
await page.evaluate(() => window.skylens.viewer.setDisplay('black'));
await page.waitForTimeout(2500);
writeFileSync(`${SHOTS}/route-3d-buildings.png`, await page.locator('canvas').first().screenshot());

// Same ground, same span: whatever separates the two pictures is the error an
// operator would see between planning and watching.
const match = await matchImages(page, plannerPng, aerialPng, SPAN_M);
console.log('planner vs 3D:', JSON.stringify(match));

console.log('shots written:');
console.log(`  ${SHOTS}/route-planner.png`);
console.log(`  ${SHOTS}/route-3d-aerial.png`);
console.log(`  ${SHOTS}/route-3d-buildings.png`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
