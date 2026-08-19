// Why is the board's 3D view empty?
//
//   npm run demo   (+ a route assigned)
//   node src/test/client/boardViewCheck.mjs
//
// Chunks can arrive, be placed correctly, and still draw nothing: the camera
// can be pointed elsewhere, the fog can swallow them, the floater clip can
// discard them, or the far plane can cut them off. All four look identical on a
// screenshot — an empty screen — and identical to geometry that never came. So
// print each of them against where the geometry actually is.

import { chromium } from '@playwright/test';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const WAIT_MS = Number(process.env.SKYLENS_BOARD_WAIT ?? 180_000);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page.bringToFront();
await page
  .waitForFunction(() => (window.skylens.splat?.chunks ?? 0) >= 2, undefined, { timeout: WAIT_MS })
  .catch(() => console.log('[view] fewer than 2 chunks — reporting what there is'));
await page.waitForTimeout(3000);

const probe = await page.evaluate(() => {
  const dbg = window.skylens.dbg;
  const samples = window.skylens.splat.samples(3000);
  const cam = dbg.camPos;
  const target = dbg.target;

  // How the geometry sits relative to the camera: distance, and whether it is
  // inside the clip box the shader tests against.
  const clip = dbg.clip;
  let inClip = 0;
  let nearest = Infinity;
  let furthest = 0;
  for (const [x, y, z] of samples) {
    const d = Math.hypot(x - cam[0], y - cam[1], z - cam[2]);
    nearest = Math.min(nearest, d);
    furthest = Math.max(furthest, d);
    if (
      clip &&
      x >= clip[0][0] && x <= clip[1][0] &&
      y >= clip[0][1] && y <= clip[1][1] &&
      z >= clip[0][2] && z <= clip[1][2]
    ) {
      inClip += 1;
    }
  }

  // Is any of it in front of the camera, within the view cone?
  const fwd = [target[0] - cam[0], target[1] - cam[1], target[2] - cam[2]];
  const flen = Math.hypot(...fwd) || 1;
  let inFront = 0;
  for (const [x, y, z] of samples) {
    const v = [x - cam[0], y - cam[1], z - cam[2]];
    const vlen = Math.hypot(...v) || 1;
    const cos = (v[0] * fwd[0] + v[1] * fwd[1] + v[2] * fwd[2]) / (vlen * flen);
    if (cos > 0.5) inFront += 1; // within ~60 deg of where the camera looks
  }

  return {
    cam,
    target,
    fog: dbg.fog,
    camFar: dbg.camFar,
    clip,
    chunkCenters: dbg.chunkCenters,
    revealEnabled: dbg.revealEnabled,
    samples: samples.length,
    inClip,
    inFront,
    nearest: Math.round(nearest),
    furthest: Math.round(furthest),
  };
});

console.log('camera at', JSON.stringify(probe.cam), 'looking at', JSON.stringify(probe.target));
console.log('fog near/far', JSON.stringify(probe.fog), '· camera far', probe.camFar);
console.log('floater clip', JSON.stringify(probe.clip));
console.log('chunks at', JSON.stringify(probe.chunkCenters));
console.log(
  `${probe.samples} splat samples: ${probe.inClip} inside the clip, ${probe.inFront} in front of ` +
    `the camera, ${probe.nearest}–${probe.furthest} m away`,
);

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['geometry has arrived', probe.samples > 0],
  [`it survives the floater clip (${probe.inClip}/${probe.samples})`, probe.inClip > probe.samples * 0.5],
  [`the camera is pointed at it (${probe.inFront}/${probe.samples})`, probe.inFront > probe.samples * 0.2],
  [
    `it is inside the fog (nearest ${probe.nearest} m, fog ends ${probe.fog?.[1]} m)`,
    probe.fog != null && probe.nearest < probe.fog[1],
  ],
  [`it is inside the far plane (${probe.furthest} m < ${probe.camFar} m)`, probe.furthest < probe.camFar],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join(String.fromCharCode(10)));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
