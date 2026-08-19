// Do VWorld's building outlines sit on VWorld's own satellite imagery?
//
//   npm run demo
//   node src/test/control/footprintOverlayCheck.mjs
//
// This is the comparison the other registration checks could not make. They
// matched the planner map against the 3D drape — but both are the same WMTS
// tiles, so they agree by construction. The operator plans on the imagery and
// then watches the flight against the BUILDING layer, and those come from two
// different VWorld products. If the vectors are offset from the pictures, the
// aircraft crosses the right coordinates over the wrong building.
//
// So: draw the WFS footprints straight onto the planner map, in the planner's
// own projection, and look at whether the outlines land on the roofs.

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const SHOTS = 'C:/tmp/skylens-shots';
// 충남대 공대 일대 — the ground the report's route was planned over.
const AT = { lat: 36.3665, lon: 127.3448, alt: 0 };
const SPAN_M = 400;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.bringToFront();
await page.waitForTimeout(15_000);

const info = await page.evaluate(
  async ([at, span]) => {
    const rm = window.skylens.routeModal;
    rm.debugView(at, span);
    rm.open();
    await new Promise((r) => setTimeout(r, 5000));

    const canvas = document.querySelector('.route-modal__canvas');
    const { bounds } = rm.debugBounds();
    const W = canvas.width;

    // The planner's own mapping, so the overlay cannot introduce its own error.
    const toPx = (lat, lon) => [
      ((lon - bounds.west) / (bounds.east - bounds.west)) * W,
      ((bounds.north - lat) / (bounds.north - bounds.south)) * W,
    ];

    const url =
      `/vworld/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature&TYPENAME=lt_c_bldginfo` +
      `&SRSNAME=EPSG:4326&BBOX=${bounds.south},${bounds.west},${bounds.north},${bounds.east},EPSG:4326` +
      `&maxFeatures=500&OUTPUT=application/json`;
    const res = await fetch(url);
    const json = await res.json();
    const feats = json.features ?? [];

    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255,60,60,0.95)';
    ctx.fillStyle = 'rgba(255,60,60,0.18)';
    let drawn = 0;
    for (const f of feats) {
      const polys =
        f.geometry?.type === 'MultiPolygon' ? f.geometry.coordinates : [f.geometry?.coordinates];
      for (const poly of polys ?? []) {
        const ring = poly?.[0];
        if (!ring || ring.length < 3) continue;
        ctx.beginPath();
        ring.forEach(([lon, lat], i) => {
          const [x, y] = toPx(lat, lon);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        drawn++;
      }
    }
    return { footprints: feats.length, drawn, bounds };
  },
  [AT, SPAN_M],
);

console.log('footprints drawn over the map:', JSON.stringify(info));
writeFileSync(
  `${SHOTS}/footprint-overlay.png`,
  await page.locator('.route-modal__canvas').screenshot(),
);
console.log(`shot: ${SHOTS}/footprint-overlay.png`);

// The same ground as the tower draws it, in the mode the operator flies in.
// A narrow lens from high up keeps the roofs where their footprints are: at the
// scene's own 55 deg field of view a tall roof swells over the street beside
// it, which is what makes the city hard to recognise from the chase camera.
await page.evaluate(() => window.skylens.routeModal.close());
const cam = await page.evaluate(
  async ([at, span]) => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const frame = window.skylens.frame;
    const v = window.skylens.viewer;
    v.setDisplay('black');
    const p = frame.toScene({ ...at, alt: 0 });
    const groundY = frame.groundYAt({ ...at, alt: 0 });
    const fov = 8;
    const above = ((span / 2) * frame.unitsPerMeter) / Math.tan((fov / 2) * (Math.PI / 180));
    v.debugTopDown(new THREE.Vector3(p.x, groundY, p.z), above, fov);
    return { fov, altitudeM: Math.round(above / frame.unitsPerMeter) };
  },
  [AT, SPAN_M],
);
console.log('near-orthographic camera:', JSON.stringify(cam));
await page.waitForTimeout(3000);
writeFileSync(`${SHOTS}/footprint-3d-black.png`, await page.locator('canvas').first().screenshot());
console.log(`shot: ${SHOTS}/footprint-3d-black.png`);

await browser.close();
