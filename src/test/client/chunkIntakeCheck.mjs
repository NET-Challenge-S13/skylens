// What happens to a chunk between the core and the board's scene?
//
//   npm run demo   (+ a route assigned)
//   node src/test/client/chunkIntakeCheck.mjs
//
// The core logs every level it publishes. If the board ends up holding one
// segment while the flight is hundreds of metres further on, the loss is
// somewhere in between: the relay, the queue, or the loader. This follows the
// scene's own counters while the flight continues and prints whatever the
// loader said about the difference.

import { chromium } from '@playwright/test';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const WATCH_MS = Number(process.env.SKYLENS_WATCH_MS ?? 90_000);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const logs = [];
page.on('console', (m) => {
  const t = m.text();
  if (/splat|chunk|scene|load/i.test(t)) logs.push(`${m.type()}: ${t.slice(0, 200)}`);
});
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message.slice(0, 200)}`));

await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });

const started = Date.now();
let last = -1;
while (Date.now() - started < WATCH_MS) {
  const snap = await page.evaluate(() => ({
    chunks: window.skylens.splat.chunks,
    replaced: window.skylens.splat.replaced,
    levels: window.skylens.splat.segmentLevels,
    status: window.skylens.splat.status,
    progress: window.skylens.splat.progress,
    loaded: window.skylens.splat.loadedChunks().map((c) => `s${c.segment}l${c.level}`),
  }));
  if (snap.chunks !== last) {
    console.log(
      `[+${String(Math.round((Date.now() - started) / 1000)).padStart(3)}s] ` +
        `chunks=${snap.chunks} replaced=${snap.replaced} status=${snap.status} ` +
        `progress=${Math.round(snap.progress)}%  in scene: ${JSON.stringify(snap.loaded)}  ` +
        `levels=${JSON.stringify(snap.levels)}`,
    );
    last = snap.chunks;
  }
  await new Promise((r) => setTimeout(r, 2000));
}

console.log('');
console.log('page log lines about the splat pipeline:');
for (const l of logs.slice(-25)) console.log('  ', l);

await browser.close();
