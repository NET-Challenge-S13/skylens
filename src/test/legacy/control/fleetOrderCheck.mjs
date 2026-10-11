// Does the fleet list read like the formation looks?
//
//   npm run demo
//   node src/test/control/fleetOrderCheck.mjs
//
// The operator picks an aircraft by where it flies, so the list has to run
// left, centre, right — with the centre selected until they choose otherwise.
// The order comes from the station each aircraft reports, so a drone whose
// station never arrives falls to the end of the list and the formation reads
// wrong on screen.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.waitForFunction(() => window.skylens.fleet.drones().length >= 3, undefined, {
  timeout: 120_000,
});
await page.waitForTimeout(4000);

const probe = await page.evaluate(() => ({
  // What the fleet believes, in the order it hands to the panel.
  fleet: window.skylens.fleet.drones().map((d) => ({ id: d.id, station: d.station })),
  // What is actually on screen, top to bottom.
  rows: [...document.querySelectorAll('.telemetry-panel__list > .telemetry-row')].map((el) => ({
    text: (el.querySelector('.telemetry-row__name') ?? el).textContent.trim().slice(0, 12),
    selected: el.classList.contains('is-active'),
  })),
  selectedId: window.skylens.state?.activeDroneId ?? null,
}));

console.log('fleet order :', JSON.stringify(probe.fleet));
console.log('rows on screen:');
for (const r of probe.rows) console.log(`   ${r.selected ? '>' : ' '} ${r.text}`);
console.log('selected id :', probe.selectedId);

const stations = probe.fleet.map((d) => d.station);
const selected = probe.fleet.find((d) => d.id === probe.selectedId);
const rowOrder = probe.rows.map((r) => r.text);

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['every aircraft reports a station', stations.every((s) => s === 'left' || s === 'center' || s === 'right')],
  ['the fleet is ordered left, centre, right', JSON.stringify(stations) === JSON.stringify(['left', 'center', 'right'])],
  [
    'the rows are in that order too',
    rowOrder.length === 3 && /좌측/.test(rowOrder[0]) && /중앙/.test(rowOrder[1]) && /우측/.test(rowOrder[2]),
  ],
  ['the centre aircraft is selected', selected?.station === 'center'],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
