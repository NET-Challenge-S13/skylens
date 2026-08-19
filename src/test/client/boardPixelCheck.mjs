// Is anything actually drawn?
//
//   npm run demo   (+ a route assigned)
//   node src/test/client/boardPixelCheck.mjs
//
// Every other check says the geometry has arrived, survives the clip, and sits
// in front of the camera — and the screen can still be empty. Count the pixels
// that are not the background, from a camera parked right on top of a chunk, so
// "nothing renders" and "nothing is in view" stop looking the same.

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const SHOTS = 'C:/tmp/skylens-shots';
const WAIT_MS = Number(process.env.SKYLENS_BOARD_WAIT ?? 180_000);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
// A patched shader that fails to link renders nothing and says so only here.
const console_ = [];
page.on('console', (m) => {
  if (m.type() === 'error' || /shader|glsl|program|compile/i.test(m.text())) {
    console_.push(`${m.type()}: ${m.text().slice(0, 300)}`);
  }
});
page.on('pageerror', (e) => console_.push(`pageerror: ${e.message.slice(0, 200)}`));
await page.goto(`${BOARD}?reveal=off`, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page.bringToFront();
await page
  .waitForFunction(() => (window.skylens.splat?.chunks ?? 0) >= 2, undefined, { timeout: WAIT_MS })
  .catch(() => console.log('[pixels] fewer than 2 chunks'));
await page.waitForTimeout(3000);

/** Fraction of the canvas that is not the background colour. */
async function ink(label) {
  // The whole page: an element screenshot of the canvas scrolls it into view,
  // which on this layout waits forever.
  const png = await page.screenshot();
  writeFileSync(`${SHOTS}/board-${label}.png`, png);
  return page.evaluate(async (b64) => {
    const res = await fetch(`data:image/png;base64,${b64}`);
    const bmp = await createImageBitmap(await res.blob());
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const g = c.getContext('2d');
    g.drawImage(bmp, 0, 0);
    const d = g.getImageData(0, 0, bmp.width, bmp.height).data;
    // The board's background is a near-black blue; anything brighter is drawn.
    let lit = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] + d[i + 1] + d[i + 2] > 90) lit += 1;
    }
    return Math.round((lit / (d.length / 4)) * 1000) / 10;
  }, png.toString('base64'));
}

const asIs = await ink('asis');

// Park the camera just above the newest chunk, looking down at it.
const parked = await page.evaluate(async () => {
  const THREE = await import('/node_modules/three/build/three.module.js');
  const v = window.skylens.viewer;
  if (!v?.debugLookAt) return null;
  const chunks = window.skylens.splat.loadedChunks();
  if (chunks.length === 0) return null;
  const c = chunks[chunks.length - 1].center;
  v.debugLookAt(new THREE.Vector3(c[0], c[1], c[2]), 60);
  return c.map((n) => Math.round(n));
});
await page.waitForTimeout(2500);
const close = parked ? await ink('close') : null;

console.log(`ink as-is        : ${asIs}% of the canvas`);
console.log(`ink from 60 m up : ${close === null ? 'no debugLookAt hook' : `${close}%`} (over ${JSON.stringify(parked)})`);
console.log(`shots: ${SHOTS}/board-asis.png · ${SHOTS}/board-close.png`);
if (console_.length) {
  console.log('');
  console.log('console:');
  for (const l of console_.slice(0, 12)) console.log('  ', l);
}

await browser.close();
