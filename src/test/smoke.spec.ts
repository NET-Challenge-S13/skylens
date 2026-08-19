// Browser smoke tests for the two operator screens.
//
// SCOPE. These load each page from the Vite dev server and check that it boots,
// renders, and survives its dependencies being absent. What they deliberately do
// NOT test is the pipeline: the two screens no longer produce anything on their
// own, so a "does the delay pattern work" assertion here would be a test of the
// fake data it was given. That behaviour is exercised where it actually lives:
//
//   src/test/demo/driveScenario.ts   whole pipeline, real components
//   src/test/proxy/pipelineDrill.ts  gateway/proxy failover and hole punching
//   src/test/client/boardCheck.ts    board against a scripted core
//   src/test/geography.spec.ts       the anchor/map/waypoint coupling
//
// The previous version of this file drove a control-tower→board WebRTC pairing
// and a client-side reconstruction mock. Both were removed by design: the
// screens are siblings fed by the pipeline, never by each other.

import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';

interface SkylensHandle {
  role: 'control' | 'status';
  state: { drones: Array<{ id: number }> };
  /** Board only. */
  splat?: { chunks: number; segmentLevels: Record<number, number> };
}

declare global {
  interface Window {
    skylens: SkylensHandle;
  }
}

/** Network chatter is expected while the core and the relay are not running. */
const EXPECTED_NOISE =
  /websocket|ws:\/\/|failed to fetch|net::err|connection|econnrefused|vworld|wmts|wfs|502|503/i;

/** A closed port: reachable host, nothing accepting. Makes "dependency absent"
 *  a property of the test rather than of the machine it runs on. */
const DEAD_HOST = '127.0.0.1:9';

function trackPageErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error' && !EXPECTED_NOISE.test(msg.text())) {
      errors.push(`console: ${msg.text()}`);
    }
  });
  return errors;
}

async function hasWebGL(page: Page, canvasId: string): Promise<boolean> {
  return page.evaluate((id) => {
    const el = document.getElementById(id);
    if (!(el instanceof HTMLCanvasElement)) return false;
    return el.width > 0 && el.height > 0;
  }, canvasId);
}

test.describe('control tower', () => {
  test('boots and renders without the core', async ({ page }) => {
    test.setTimeout(90_000);
    const errors = trackPageErrors(page);

    // Point at a port nothing listens on, so "no core" is the state under test
    // rather than whatever happens to be running on this machine.
    await page.goto(`/res/static/control.html?core=${DEAD_HOST}`);
    await page.waitForFunction(() => window.skylens?.role === 'control', undefined, {
      timeout: 60_000,
    });

    expect(await hasWebGL(page, 'control-view')).toBeTruthy();

    // With no core reachable the tower must SAY so rather than invent a fleet:
    // it holds no simulation of its own any more.
    expect(await page.evaluate(() => window.skylens.state.drones.length)).toBe(0);

    await page.waitForTimeout(1000);
    expect(errors, errors.join('\n')).toEqual([]);
  });

  test('the display-mode setting survives a reload', async ({ page }) => {
    test.setTimeout(90_000);
    await page.goto(`/res/static/control.html?display=points&core=${DEAD_HOST}`);
    await page.waitForFunction(() => window.skylens?.role === 'control', undefined, {
      timeout: 60_000,
    });

    // `?display=` is a one-load override and must NOT overwrite the stored
    // choice — otherwise a shared debug link silently reconfigures the operator.
    const stored = await page.evaluate(() => localStorage.getItem('skylens.control.settings.v1'));
    expect(stored === null || !/points/.test(stored)).toBeTruthy();
  });
});

test.describe('situation board', () => {
  test('boots and shows a waiting state without the relay', async ({ page }) => {
    test.setTimeout(90_000);
    const errors = trackPageErrors(page);

    await page.goto(`/res/static/status.html?relay=${DEAD_HOST}`);
    await page.waitForFunction(() => window.skylens?.role === 'status', undefined, {
      timeout: 60_000,
    });

    expect(await hasWebGL(page, 'status-view')).toBeTruthy();
    await expect(page.locator('#server-status')).toBeAttached();
    await expect(page.locator('#minimap')).toBeAttached();

    // Nothing has arrived, so nothing may be claimed as reconstructed.
    // Run this with the demo STOPPED (npm run demo:clean): a live client server
    // on 8090 feeds the board real chunks, and the failure then says the demo is
    // running rather than anything about this page.
    expect(await page.evaluate(() => window.skylens.splat?.chunks ?? 0)).toBe(0);

    await page.waitForTimeout(1000);
    expect(errors, errors.join('\n')).toEqual([]);
  });
});
