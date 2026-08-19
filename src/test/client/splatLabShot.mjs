// Screenshot one ply in the splat lab bench.
//
//   node src/test/client/splatLabShot.mjs <ply-url> <label> [view] [h]
//
// Needs the Vite dev server (npm run demo or npm run dev).

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const [ply, label, view = 'side', h = ''] = process.argv.slice(2);
if (!ply || !label) {
  console.error('usage: splatLabShot.mjs <ply-url> <label> [side|top|along] [h]');
  process.exit(2);
}
const VITE = process.env.SKYLENS_VITE ?? 'http://localhost:5173';
const SHOTS = 'C:/tmp/skylens-shots';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message.slice(0, 200)));
page.on('console', (m) => {
  if (m.type() === 'error') console.log('console:', m.text().slice(0, 200));
});
const u = `${VITE}/res/static/splatlab.html?ply=${encodeURIComponent(ply)}&view=${view}${h ? `&h=${h}` : ''}`;
await page.goto(u, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.bringToFront();
await page.waitForFunction(() => window.lab?.ready === true, undefined, { timeout: 240_000 });
writeFileSync(`${SHOTS}/lab-${label}.png`, await page.screenshot({ timeout: 120_000 }));
const bounds = await page.evaluate(() => window.lab.bounds);
console.log(`saved ${SHOTS}/lab-${label}.png  bounds lo/hi:`, JSON.stringify(bounds));
await browser.close();
