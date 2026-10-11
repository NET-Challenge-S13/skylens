// Which space do the splat centres come back in?
//
//   npm run demo   (+ a route assigned)
//   node src/test/client/splatSpaceProbe.mjs
//
// The library can hand back a splat centre in its own scene's coordinates or in
// the world, depending on a flag and on whether the mesh is dynamic. Getting it
// wrong makes every measurement of "where is the reconstruction" quietly wrong,
// which is how a clip box ended up fitted to the wrong space.

import { chromium } from '@playwright/test';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const WAIT_MS = Number(process.env.SKYLENS_BOARD_WAIT ?? 180_000);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page
  .waitForFunction(() => (window.skylens.splat?.chunks ?? 0) >= 2, undefined, { timeout: WAIT_MS })
  .catch(() => console.log('[space] fewer than 2 chunks'));
await page.waitForTimeout(3000);

console.log(
  JSON.stringify(
    await page.evaluate(() => {
      const chunks = window.skylens.splat.loadedChunks();
      const s = window.skylens.splat.samples(2000);
      const range = (i) => {
        const v = s.map((p) => p[i]).sort((a, b) => a - b);
        return [Math.round(v[0]), Math.round(v[v.length - 1])];
      };
      return {
        chunkPositions: chunks.map((c) => c.center.map((n) => Math.round(n))),
        sampleRangeX: range(0),
        sampleRangeY: range(1),
        sampleRangeZ: range(2),
        samples: s.length,
      };
    }),
    null,
    2,
  ),
);

await browser.close();
