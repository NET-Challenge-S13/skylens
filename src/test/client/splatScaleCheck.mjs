// Does the reconstruction cover the ground it was flown over?
//
//   npm run demo   (+ a route assigned)
//   node src/test/client/splatScaleCheck.mjs
//
// A reconstruction built from images has shape but no size — structure from
// motion cannot recover metric scale — so someone has to resolve it against
// something measured. When nobody did, each chunk was drawn 4.7 m wide for a 40
// m stretch of flight and the board showed specks over empty ground.
//
// The property that matters is not an exact ratio: it is whether the geometry
// forms a continuous strip along the track. So project the splats onto the
// assigned route and look for holes.

import { chromium } from '@playwright/test';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const WAIT_MS = Number(process.env.SKYLENS_BOARD_WAIT ?? 180_000);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page
  .waitForFunction(() => (window.skylens.splat?.chunks ?? 0) >= 3, undefined, { timeout: WAIT_MS })
  .catch(() => console.log('[splat] fewer than 3 chunks — reporting what there is'));
await page.waitForTimeout(3000);

const probe = await page.evaluate(async () => {
  const { sceneToEnu } = await import('/src/shared/geo.ts');
  const { state } = await import('/src/shared/viewer/store.ts');

  const loaded = window.skylens.splat.loadedChunks();
  const route = (state.route ?? []).map((p) => {
    const enu = sceneToEnu(p);
    return { e: enu.e, n: enu.n };
  });
  if (route.length < 2) return { route: 0 };

  // Arc position along the route, in metres, of a point in scene space.
  const legs = [];
  let total = 0;
  for (let i = 0; i + 1 < route.length; i++) {
    const dx = route[i + 1].e - route[i].e;
    const dy = route[i + 1].n - route[i].n;
    const len = Math.hypot(dx, dy);
    legs.push({ ...route[i], dx, dy, len, cum: total });
    total += len;
  }
  const arcOf = (e, n) => {
    let best = null;
    for (const leg of legs) {
      const t = leg.len > 0 ? Math.max(0, Math.min(1, ((e - leg.e) * leg.dx + (n - leg.n) * leg.dy) / (leg.len * leg.len))) : 0;
      const px = leg.e + leg.dx * t;
      const py = leg.n + leg.dy * t;
      const off = Math.hypot(e - px, n - py);
      if (best === null || off < best.off) best = { off, arc: leg.cum + leg.len * t };
    }
    return best;
  };

  const anchors = loaded
    .map((c) => {
      const enu = sceneToEnu(c.center);
      const a = arcOf(enu.e, enu.n);
      return { segment: c.segment, level: c.level, arc: a ? Math.round(a.arc) : null };
    })
    .sort((a, b) => a.segment - b.segment);

  // Where the geometry actually is, along the route.
  const BIN_M = 10;
  const bins = new Map();
  let offTrack = 0;
  const samples = window.skylens.splat.samples(8000);
  for (const [x, y, z] of samples) {
    const enu = sceneToEnu([x, y, z]);
    const a = arcOf(enu.e, enu.n);
    if (!a) continue;
    // Splats far to the side are the scene's flanks, not route coverage.
    if (a.off > 60) {
      offTrack += 1;
      continue;
    }
    const bin = Math.floor(a.arc / BIN_M);
    bins.set(bin, (bins.get(bin) ?? 0) + 1);
  }

  const covered = [...bins.keys()].sort((a, b) => a - b);
  const first = covered[0] ?? null;
  const last = covered[covered.length - 1] ?? null;
  const span = first === null ? 0 : last - first + 1;
  const holes = [];
  for (let b = first; b !== null && b <= last; b++) if (!bins.has(b)) holes.push(b * BIN_M);

  return {
    route: route.length,
    routeLengthM: Math.round(total),
    anchors,
    binM: BIN_M,
    coveredM: covered.length * BIN_M,
    spanM: span * BIN_M,
    holesM: holes,
    offTrack,
    samples: samples.length,
  };
});

if (!probe.route) {
  console.log('the board has no route yet — assign one first (src/test/control/assignRoute.mjs)');
  await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
  process.exit(1);
}

console.log(`route: ${probe.route} waypoints, ${probe.routeLengthM} m`);
console.log('chunk anchors, as arc length along that route (m):');
for (const a of probe.anchors) console.log(`  segment ${a.segment} level ${a.level}: ${a.arc}`);
console.log(
  `geometry covers ${probe.coveredM} m of the ${probe.spanM} m it spans ` +
    `(${probe.binM} m bins, ${probe.samples} samples, ${probe.offTrack} off to the side)`,
);
if (probe.holesM.length) console.log('holes at (m along route):', JSON.stringify(probe.holesM));

console.log('');
console.log('===== RESULT =====');
const continuity = probe.spanM > 0 ? probe.coveredM / probe.spanM : 0;
const checks = [
  ['the reconstruction is on the route', probe.coveredM > 0],
  [
    `it covers a real stretch of it (${probe.spanM} m)`,
    probe.spanM >= 40 * Math.max(1, probe.anchors.length - 1),
  ],
  [`it is continuous (${Math.round(continuity * 100)}% of the bins it spans)`, continuity >= 0.8],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join(String.fromCharCode(10)));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
