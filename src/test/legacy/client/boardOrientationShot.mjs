// What ANGLE is the reconstruction standing at?
//
//   npm run demo  (+ route assigned, chunks arrived)
//   node src/test/client/boardOrientationShot.mjs
//
// Three parked views — top-down over the strip, oblique along it, side-on at
// ground level — so a rolled/pitched/yawed scene can be told apart from a
// correctly placed one by eye. Screenshots land in C:/tmp/skylens-shots.

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const SHOTS = 'C:/tmp/skylens-shots';

const browser = await chromium.launch({ args: ['--use-angle=default'] });
const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
const logs = [];
page.on('console', (m) => {
  if (m.type() === 'error') logs.push(`${m.type()}: ${m.text().slice(0, 300)}`);
});
await page.goto(`${BOARD}?reveal=off`, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page.bringToFront();
await page
  .waitForFunction(() => window.skylens.splat?.status === 'ready', undefined, { timeout: 180_000 })
  .catch(() => logs.push('splat never became ready'));
// Let the sort worker finish its first pass before judging pixels.
await page.waitForTimeout(8000);

async function park(label, calc) {
  await page.evaluate(async (fn) => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const chunks = window.skylens.splat.loadedChunks();
    if (chunks.length === 0) return;
    const mid = chunks[Math.floor(chunks.length / 2)].center;
    const first = chunks[0].center;
    const last = chunks[chunks.length - 1].center;
    // eslint-disable-next-line no-new-func
    const make = new Function('THREE', 'mid', 'first', 'last', `return (${fn})(THREE, mid, first, last);`);
    const { pos, target } = make(THREE, mid, first, last);
    window.skylens.viewer.debugCamera(pos, target);
  }, calc.toString());
  await page.waitForTimeout(2500);
  writeFileSync(`${SHOTS}/orient-${label}.png`, await page.screenshot());
  console.log(`saved ${SHOTS}/orient-${label}.png`);
}

// Top-down over the middle of the strip: judges yaw/mirror vs the chunk line.
await park('top', (THREE, mid) => ({
  pos: new THREE.Vector3(mid[0], mid[1] + 220, mid[2] + 1),
  target: new THREE.Vector3(mid[0], mid[1], mid[2]),
}));

// Side-on at low height, looking across the strip: judges roll (horizon).
await park('side', (THREE, mid, first, last) => {
  const dir = new THREE.Vector3(last[0] - first[0], 0, last[2] - first[2]).normalize();
  const cross = new THREE.Vector3(-dir.z, 0, dir.x);
  return {
    pos: new THREE.Vector3(mid[0] + cross.x * 120, mid[1] + 25, mid[2] + cross.z * 120),
    target: new THREE.Vector3(mid[0], mid[1] + 5, mid[2]),
  };
});

// Oblique along the strip: the "demo" view.
await park('along', (THREE, mid, first, last) => {
  const dir = new THREE.Vector3(last[0] - first[0], 0, last[2] - first[2]).normalize();
  return {
    pos: new THREE.Vector3(
      first[0] - dir.x * 60,
      first[1] + 55,
      first[2] - dir.z * 60,
    ),
    target: new THREE.Vector3(mid[0], mid[1], mid[2]),
  };
});

if (logs.length) {
  console.log('console:');
  for (const l of logs.slice(0, 10)) console.log('  ', l);
}
await browser.close();
