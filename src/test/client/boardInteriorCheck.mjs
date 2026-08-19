// Interior-mode verification: is the camera inside the corridor, and does the
// person marker sit on the reconstructed person?
//
//   npm run demo  (+ route assigned, chunks arrived)
//   node src/test/client/boardInteriorCheck.mjs

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const SHOTS = 'C:/tmp/skylens-shots';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message.slice(0, 200)));
await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page.bringToFront();
await page
  .waitForFunction(() => window.skylens.splat?.status === 'ready', undefined, { timeout: 180_000 })
  .catch(() => console.log('splat never ready'));
await page.waitForTimeout(12_000); // first sort + interior mode kick-in

writeFileSync(`${SHOTS}/interior-follow.png`, await page.screenshot({ timeout: 120_000 }));
console.log('saved interior-follow.png');

const info = await page.evaluate(() => ({
  markers: window.skylens.markers,
  detections: window.skylens.state.detections.map((d) => ({
    id: d.id,
    kind: d.kind,
    pos: d.pos.map((n) => Math.round(n * 10) / 10),
  })),
  dbg: { camPos: window.skylens.dbg.camPos, target: window.skylens.dbg.target },
}));
console.log(JSON.stringify(info, null, 1));

// Park 8 m from the person marker at eye height, looking at it.
const parked = await page.evaluate(async () => {
  const THREE = await import('/node_modules/three/build/three.module.js');
  const det = window.skylens.state.detections.find((d) => d.kind === 'person');
  if (!det) return null;
  const [x, y, z] = det.pos;
  window.skylens.viewer.debugCamera(
    new THREE.Vector3(x + 6, y + 2.6, z + 5),
    new THREE.Vector3(x, y + 1.0, z),
  );
  return det.pos;
});
await page.waitForTimeout(6000);
writeFileSync(`${SHOTS}/interior-person.png`, await page.screenshot({ timeout: 120_000 }));
console.log('saved interior-person.png, marker at', JSON.stringify(parked));
await browser.close();
