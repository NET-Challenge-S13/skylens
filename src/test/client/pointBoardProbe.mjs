// Point layer against the REAL core feed, on the machine's real GPU.
//
//   npm run demo:points                                   (stage the capture once)
//   npx vite & npx tsx src/skylens_client/server/index.ts &
//   node src/test/client/pointBoardProbe.mjs &            (board connects first)
//   SKYLENS_DEMO=1 SKYLENS_DEMO_POINTS_AUTOSTART=1 SKYLENS_CORE_WEB_MODE=off \
//     npx tsx src/skylens_core/server/index.ts
//
// What it records, into test-results/points/:
//   - a screenshot whenever the delivered levels change (first pass on screen,
//     SKYLENS_FIXED_CAMERA=1 keeps one viewpoint so the shots compare directly;
//     first pass replaced by the refined pass, everything refined)
//   - per-segment timings the board measured (worker fetch, decode, build,
//     mount → first frame, total)
//   - frame rate at the largest accumulated point count, and again with a
//     SYNTHETIC stress load (the refined files re-offered at shifted positions
//     to reach ~20 segments). The stress numbers are a load test, not data.
//
// Playwright's test config forces SwiftShader (software GL); frame rates from
// that mean nothing, so this launches Chrome with the GPU enabled instead and
// prints which renderer it actually got.

import fs from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const BOARD =
  process.env.SKYLENS_BOARD ?? 'http://localhost:5173/res/static/status.html';
const OUT = path.resolve(process.env.SKYLENS_OUT ?? 'test-results/points');
const WAIT_MS = Number(process.env.SKYLENS_WAIT_MS ?? 180_000);
const STRESS_SEGMENTS = Number(process.env.SKYLENS_STRESS_SEGMENTS ?? 20);
/** Hold the camera after the first shot, so before/after shots compare. */
const FIXED = !!process.env.SKYLENS_FIXED_CAMERA;
fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({
  channel: process.env.SKYLENS_CHANNEL ?? 'chrome',
  headless: process.env.SKYLENS_HEADED ? false : true,
  args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=metal'],
});
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const logs = [];
page.on('console', (m) => {
  const t = m.text();
  if (/point|error/i.test(t)) logs.push(`${m.type()}: ${t.slice(0, 240)}`);
});
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message.slice(0, 240)}`));

await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, {
  timeout: 120_000,
});

const gpu = await page.evaluate(() => {
  const gl = document.createElement('canvas').getContext('webgl2');
  const ext = gl?.getExtension('WEBGL_debug_renderer_info');
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'unknown';
});
console.log(`[probe] renderer: ${gpu}`);

const stats = () => page.evaluate(() => window.skylens.points.stats);

/** Frames per second over `ms`, plus the slowest frame. */
async function fps(ms = 5000) {
  return page.evaluate(
    (dur) =>
      new Promise((resolve) => {
        const times = [];
        let last = performance.now();
        const t0 = last;
        const tick = (now) => {
          times.push(now - last);
          last = now;
          if (now - t0 < dur) requestAnimationFrame(tick);
          else {
            times.shift();
            const sorted = [...times].sort((a, b) => a - b);
            resolve({
              fps: Math.round((times.length / ((now - t0) / 1000)) * 10) / 10,
              p50ms: Math.round(sorted[Math.floor(sorted.length / 2)] * 10) / 10,
              p95ms: Math.round(sorted[Math.floor(sorted.length * 0.95)] * 10) / 10,
              worstMs: Math.round(sorted[sorted.length - 1] * 10) / 10,
            });
          }
        };
        requestAnimationFrame(tick);
      }),
    ms,
  );
}

// --- 1. watch the real feed -------------------------------------------------
let lastKey = '';
let shot = 0;
const started = Date.now();
let total = 0;
while (Date.now() - started < WAIT_MS) {
  const s = await stats();
  const segs = s?.segments ?? [];
  const key = segs.filter((x) => x.shown > 0).map((x) => `${x.segment}:${x.shown}`).join(' ');
  if (key && key !== lastKey) {
    await page.waitForTimeout(FIXED ? 600 : 1800); // let the change draw (and the framing settle)
    // Name the shot after what is on screen NOW, not what triggered it.
    const now = (await stats()).segments.filter((x) => x.shown > 0);
    lastKey = now.map((x) => `${x.segment}:${x.shown}`).join(' ');
    const name = `${String(++shot).padStart(2, '0')}_${now.map((x) => `r${x.segment}L${x.shown}`).join('_')}.png`;
    await page.screenshot({ path: path.join(OUT, name) });
    const cam = await page.evaluate(() => window.skylens.points.inset?.camPos ?? []);
    console.log(`[probe] ${lastKey}  (${(await stats()).points} pts)  cam ${cam.join(',')}  → ${name}`);
    if (FIXED && shot === 1) {
      // A press in the panel is the operator taking its camera: the framing
      // stops following, so every later shot shares this one's viewpoint.
      await page.locator('#point-inset .point-inset__view').click();
    }
  }
  total = segs.length;
  if (total > 0 && segs.every((x) => x.shown === 2 && x.loading === 0 && x.wanted === 0) && total >= 7) {
    break;
  }
  await page.waitForTimeout(150);
}

const done = await stats();
console.log(`[probe] feed settled: ${done.segments.length} segments, ${done.points} points`);
await page.evaluate(() => window.skylens.points.frame());
await page.waitForTimeout(800);
await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}_all_refined_framed.png`) });
const fpsReal = await fps();
console.log(`[probe] frame rate @ ${done.points} pts:`, fpsReal);

// --- 2. synthetic stress ---------------------------------------------------
// Re-offer the refined files as extra segments shifted east, past the capture,
// so nothing overlaps and the camera sees all of it.
const manifest = JSON.parse(
  fs.readFileSync('res/static/demo/points/points.json', 'utf8'),
);
const refined = manifest.segments.map((s) => s.levels.find((l) => l.level === 2)).filter(Boolean);
const extra = [];
for (let k = manifest.segments.length; k < STRESS_SEGMENTS; k++) {
  const src = refined[k % refined.length];
  extra.push({
    kind: 'point-chunk',
    id: `stress-${k}`,
    segment: 1000 + k,
    level: 2,
    final: true,
    url: `/res/static/demo/points/${src.url}`,
    bytes: src.bytes,
    points: src.points,
    format: 'xyz-rgb-nxyz-27',
    voxel: 0.1,
    bbox: src.bbox,
    align: {
      anchor: manifest.anchor,
      position: [400 * (1 + Math.floor((k - manifest.segments.length) / refined.length)), 0, 0],
      rotation: [0, 0, 0, 1],
      scale: [1, 1, 1],
    },
  });
}
for (const c of extra) await page.evaluate((chunk) => window.skylens.points.offer(chunk), c);
await page.waitForFunction(
  (n) => {
    const s = window.skylens.points.stats;
    return s.segments.length >= n && s.segments.every((x) => x.shown === 2 && x.loading === 0);
  },
  STRESS_SEGMENTS,
  { timeout: 180_000 },
);
await page.evaluate(() => window.skylens.points.frame());
await page.waitForTimeout(1000);
const stress = await stats();
await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}_stress_synthetic.png`) });
const fpsStress = await fps();
console.log(`[probe] SYNTHETIC frame rate @ ${stress.points} pts:`, fpsStress);

const report = {
  renderer: gpu,
  board: BOARD,
  real: {
    segments: done.segments.length,
    points: done.points,
    heightRange: done.heightRange,
    fps: fpsReal,
    timings: done.timings,
    stale: done.stale,
    cancelled: done.cancelled,
    errors: done.errors,
  },
  syntheticStress: {
    note: 'refined files re-offered at shifted positions; a load test, not data',
    segments: stress.segments.length,
    points: stress.points,
    fps: fpsStress,
    timings: stress.timings.slice(done.timings.length),
  },
  logs,
};
fs.writeFileSync(path.join(OUT, 'report.json'), JSON.stringify(report, null, 2));
console.log(`[probe] report → ${path.join(OUT, 'report.json')}`);
await browser.close();
