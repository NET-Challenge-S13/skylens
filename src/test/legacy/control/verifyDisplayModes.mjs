// Manual verification harness for the control tower's three display options
// (COMPONENTS.md §4). Not part of `npm test` — it needs a live Vite dev server
// and a working VWorld key, so it is run by hand:
//
//   npm run dev
//   node src/test/control/verifyDisplayModes.mjs [--headed]
//
// It drives the REAL settings panel rather than poking the viewer directly, so
// a pass means the operator-facing path works, not just the internals.

import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const BASE = process.env.SKYLENS_BASE ?? 'http://localhost:5173';
const URL = `${BASE}/res/static/control.html`;
const OUT = process.env.SKYLENS_SHOTS ?? 'C:/tmp/skylens-shots';
const headed = process.argv.includes('--headed');

mkdirSync(OUT, { recursive: true });

const log = (...a) => console.log('[verify]', ...a);

const browser = await chromium.launch({ headless: !headed });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
page.on('console', (m) => {
  const t = m.text();
  // The core is not running in this harness, so a refused /viewer socket is the
  // EXPECTED result — it is what the disconnected-state assertions below check.
  const expectedCoreRefusal = /ws:\/\/[^ ]*:8080\/viewer/.test(t);
  if (m.type() === 'error' && !expectedCoreRefusal) errors.push(`console.error: ${t}`);
  if (/\[buildings\]|\[terrain\]|\[control\]|\[stream\]/.test(t)) log(t);
});

await page.goto(URL, { waitUntil: 'domcontentloaded' });

// The scene is a live network load (DEM + satellite + WFS); wait for the app
// handle rather than a fixed sleep.
await page.waitForFunction(() => window.skylens?.role === 'control', null, {
  timeout: 180_000,
});
log('app booted');

const scene = await page.evaluate(() => ({
  buildingSource: window.skylens.scene.buildingSource,
  footprints: window.skylens.scene.footprints,
  imageryAvailable: window.skylens.scene.imageryAvailable,
  points: window.skylens.scene.data.count,
  hasTerrain: window.skylens.scene.terrainVisual != null,
  hasBuildingVisual: window.skylens.scene.buildingVisual != null,
  hasUvs: window.skylens.scene.buildingVisual?.uvs != null,
  hasEdges: window.skylens.scene.buildingVisual?.edges != null,
  bbox: window.skylens.scene.bbox,
  anchor: window.skylens.frame.anchor,
  coreUrl: window.skylens.core.url,
  coreState: window.skylens.core.state,
}));
log('scene:', JSON.stringify(scene, null, 2));

// Give the renderer a few frames to settle before the first capture.
await page.waitForTimeout(2500);

for (const mode of ['points', 'black', 'aerial']) {
  // Drive it the way an operator does: open 설정, click the option.
  await page.click('.control-toolbar__btn:has-text("설정")');
  await page.waitForSelector('.settings-overlay:not(.is-hidden)');
  const btn = page.locator(`.settings-option[data-mode="${mode}"]`);
  const disabled = await btn.isDisabled();
  if (disabled) {
    errors.push(`option "${mode}" is DISABLED in the panel`);
    log(`option ${mode}: DISABLED`);
  }
  if (mode === 'points') {
    await page.screenshot({ path: `${OUT}/00-settings-panel.png` });
  }
  await btn.click();
  await page.waitForTimeout(1200);
  await page.click('.settings-panel__close');
  await page.waitForTimeout(1800);

  const applied = await page.evaluate(() => ({
    viewer: window.skylens.viewer.display,
    stored: JSON.parse(localStorage.getItem('skylens.control.settings.v1') ?? 'null'),
  }));
  const ok = applied.viewer === mode && applied.stored?.display === mode;
  if (!ok) errors.push(`mode ${mode}: viewer=${applied.viewer} stored=${applied.stored?.display}`);
  log(`mode ${mode}: viewer=${applied.viewer} persisted=${applied.stored?.display} ${ok ? 'OK' : 'FAIL'}`);

  await page.screenshot({ path: `${OUT}/mode-${mode}.png` });
}

// Persistence across a full reload: no ?display= in the URL, so the value can
// only come back from localStorage.
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => window.skylens?.role === 'control', null, {
  timeout: 180_000,
});
await page.waitForTimeout(2000);
const restored = await page.evaluate(() => window.skylens.viewer.display);
log(`after reload: ${restored} ${restored === 'aerial' ? 'OK (persisted)' : 'FAIL'}`);
if (restored !== 'aerial') errors.push(`persistence: expected aerial, got ${restored}`);
await page.screenshot({ path: `${OUT}/reload-persisted.png` });

// Disconnected core state — nothing is running on 8080 in this harness.
const link = await page.evaluate(() => ({
  state: window.skylens.core.state,
  text: document.querySelector('.mission-panel__link-text')?.textContent,
  message: document.querySelector('.mission-panel__message')?.textContent,
  drones: window.skylens.state.drones.length,
  empty: document.querySelector('.telemetry-panel__empty')?.textContent,
}));
log('core link:', JSON.stringify(link, null, 2));
// The core may or may not be running. Both are valid — assert the RULE, which
// is that drones exist if and only if the core is feeding telemetry. The tower
// must never conjure a fleet on its own.
if (link.state === 'connected') {
  log(`core is UP — ${link.drones} drone(s) from telemetry`);
} else if (link.drones !== 0) {
  errors.push(`fleet must be empty while core is ${link.state}, got ${link.drones}`);
} else {
  log('core is DOWN — fleet correctly empty');
}

// Route modal renders and refuses a 1-waypoint route.
await page.click('.control-toolbar__btn:has-text("경로")');
await page.waitForSelector('.route-modal-overlay:not(.is-hidden)');
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/route-modal.png` });
const canvasBox = await page.locator('.route-modal__canvas').boundingBox();
await page.mouse.click(canvasBox.x + canvasBox.width * 0.4, canvasBox.y + canvasBox.height * 0.4);
await page.click('.route-modal__btn--primary');
const warn = await page.locator('.route-modal__hint').textContent();
log('1-waypoint assign →', warn);
if (!warn.includes('2개')) errors.push('modal did not refuse a 1-waypoint route');
await page.mouse.click(canvasBox.x + canvasBox.width * 0.6, canvasBox.y + canvasBox.height * 0.6);
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/route-2wp.png` });
const wps = await page.locator('.route-modal__item').count();
log(`waypoints placed: ${wps}`);

console.log('\n===== RESULT =====');
if (errors.length === 0) {
  console.log('PASS — no page errors, all three modes applied and persisted');
} else {
  console.log(`${errors.length} problem(s):`);
  for (const e of errors) console.log('  - ' + e);
}
console.log(`screenshots: ${OUT}`);

await browser.close();
process.exit(errors.length === 0 ? 0 : 1);
