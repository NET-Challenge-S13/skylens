// Assign the demo route the way an operator would, then leave.
//
//   node src/test/control/assignRoute.mjs
//
// The scenario starts when someone plans a route (COMPONENTS.md §5.2), so any
// check that needs a flight in progress — anything about segments, chunks or
// the situation board — needs this first.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.waitForTimeout(15_000);

const sent = await page.evaluate(async () => {
  const { DEMO_ROUTE } = await import('/src/skylens_drone/core/config.ts');
  const leader = window.skylens.fleet.drones().find((d) => d.station === 'center');
  const ok = window.skylens.core.send({
    kind: 'assign-route',
    droneId: leader?.id ?? 1,
    waypoints: DEMO_ROUTE,
    loop: true,
  });
  return { ok, waypoints: DEMO_ROUTE.length };
});

console.log('[assign]', JSON.stringify(sent));
await page.waitForTimeout(1500);
await browser.close();
process.exit(sent.ok ? 0 : 1);
