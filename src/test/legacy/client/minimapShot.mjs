// Photograph the board's minimap.
//
//   npm run demo   (+ a route assigned)
//   node src/test/client/minimapShot.mjs
//
// The minimap is the board's only plan view, so this is the quickest way to see
// whether the track, the aircraft, the reconstruction and the findings describe
// one piece of ground.

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const SHOTS = 'C:/tmp/skylens-shots';
const WAIT_MS = Number(process.env.SKYLENS_BOARD_WAIT ?? 120_000);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page
  .waitForFunction(() => (window.skylens.splat?.chunks ?? 0) >= 2, undefined, { timeout: WAIT_MS })
  .catch(() => console.log('[minimap] fewer than 2 chunks — shooting anyway'));
await page.waitForTimeout(2000);

writeFileSync(`${SHOTS}/board-minimap.png`, await page.locator('#minimap').screenshot());
console.log(`shot: ${SHOTS}/board-minimap.png`);
console.log(
  'contents:',
  JSON.stringify(
    await page.evaluate(async () => {
      const { state } = await import('/src/shared/viewer/store.ts');
      return {
        route: state.route?.length ?? 0,
        drones: state.drones.length,
        detections: state.detections.length,
        chunks: window.skylens.splat.chunks,
      };
    }),
  ),
);

await browser.close();
