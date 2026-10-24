// Photograph the whole situation board.
//
//   npm run demo   (+ a route assigned)
//   node src/test/client/boardShot.mjs
//
// The board is the deliverable an operator looks at, and its failures are
// visual: geometry too small to see, a camera pointed at empty ground. Numbers
// answer "is it in the right place"; only the picture answers "can it be read".

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const SHOTS = 'C:/tmp/skylens-shots';
const WAIT_MS = Number(process.env.SKYLENS_BOARD_WAIT ?? 180_000);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page.bringToFront();
await page
  .waitForFunction(() => (window.skylens.splat?.chunks ?? 0) >= 3, undefined, { timeout: WAIT_MS })
  .catch(() => console.log('[board] fewer than 3 chunks — shooting anyway'));
await page.waitForTimeout(4000);

writeFileSync(`${SHOTS}/board.png`, await page.screenshot());
console.log(`shot: ${SHOTS}/board.png`);
console.log(
  'state:',
  JSON.stringify(
    await page.evaluate(async () => {
      const { state } = await import('/src/shared/viewer/store.ts');
      return {
        chunks: window.skylens.splat.chunks,
        splats: window.skylens.splat.bounds()?.splats ?? 0,
        drones: state.drones.length,
        detections: state.detections.length,
      };
    }),
  ),
);

await browser.close();
