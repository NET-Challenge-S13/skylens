// Does the assigned track reach the board — live, and on a late join?
//
//   npm run demo
//   node src/test/client/routeArrivalCheck.mjs
//
// The track crosses three components before the board draws it: the core
// broadcasts it, the client relay forwards and caches it, the board applies it.
// Each hop can drop it silently, and the failure looks identical from the
// screen: a board with drones, findings and geometry, and nothing to read them
// against. So test the hops separately — a board already open when the route is
// assigned, and one that opens afterwards.

import { chromium } from '@playwright/test';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';

const browser = await chromium.launch();

/** Open a board and read what it knows about the track. */
async function openBoard(label) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('pageerror', (e) => console.log(`  [${label}] pageerror: ${e.message.slice(0, 120)}`));
  await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
  return page;
}
const routeOf = (page) =>
  page.evaluate(async () => {
    const { state } = await import('/src/shared/viewer/store.ts');
    return state.route ? state.route.length : 0;
  });

// 1. a board that is already watching when the operator assigns
const early = await openBoard('early');
await early.waitForTimeout(3000);
console.log('early board, before assignment :', await routeOf(early), 'waypoints');

const tower = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await tower.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await tower.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await tower.waitForTimeout(12_000);
await tower.evaluate(async () => {
  const { DEMO_ROUTE } = await import('/src/skylens_drone/core/config.ts');
  const leader = window.skylens.fleet.drones().find((d) => d.station === 'center');
  window.skylens.core.send({
    kind: 'assign-route',
    droneId: leader?.id ?? 1,
    waypoints: DEMO_ROUTE,
    loop: true,
  });
});
await early.waitForTimeout(4000);
const live = await routeOf(early);
console.log('early board, after assignment  :', live, 'waypoints');

// 2. a board that opens afterwards and has to be told
const late = await openBoard('late');
await late.waitForTimeout(5000);
const replayed = await routeOf(late);
console.log('late board, on join            :', replayed, 'waypoints');

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['a watching board is given the track', live >= 2],
  ['a board that joins later is given it too', replayed >= 2],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
