// Where did the final scene actually end up?
//
//   npm run demo  (+ route assigned, chunks arrived)
//   node src/test/client/boardFinalProbe.mjs
//
// Prints the splat mesh's real sample bounds, the reveal state, and every
// console/network event around the final asset, so "loaded but invisible"
// stops being one undifferentiated symptom.

import { chromium } from '@playwright/test';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text().slice(0, 400)}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message.slice(0, 300)}`));
page.on('response', (r) => {
  if (/step30000|segments\.json|\.ply/.test(r.url())) logs.push(`net ${r.status()} ${r.url()}`);
});
await page.goto(`${BOARD}?reveal=off`, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page.bringToFront();
await page
  .waitForFunction(() => window.skylens.splat?.status === 'ready', undefined, { timeout: 120_000 })
  .catch(() => logs.push('splat never became ready'));
await page.waitForTimeout(4000);

const probe = await page.evaluate(() => {
  const s = window.skylens;
  return {
    status: s.splat.status,
    progress: s.splat.progress,
    chunks: s.splat.chunks,
    hasGeometry: s.splat.hasGeometry,
    bounds: s.splat.bounds(),
    dbg: s.dbg,
  };
});
console.log(JSON.stringify(probe, null, 2));
console.log('--- events ---');
for (const l of logs.slice(0, 40)) console.log(l);
await browser.close();
