import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';

// End-to-end smoke tests for the two-page WebRTC build.
//
//  - Per-page tests verify each role boots the real browser runtime (WebGL
//    shader compilation, first-frame code paths) without uncaught exceptions.
//  - The integration test opens BOTH pages against the public PeerJS broker in
//    a unique room and verifies live state actually flows SIM -> RECON and the
//    detection choreography runs on RECON.
//
// Each page exposes `window.skylens = { role, state, scene, transport, CONFIG? }`.

type Vec = { x: number; y: number; z: number };
interface DetectionRuntime {
  id: string;
  kind: 'person' | 'danger';
  revealed: boolean;
  confirmed: boolean;
}
interface SkylensHandle {
  role: 'sim' | 'recon';
  state: {
    time: number;
    running: boolean;
    drones: { id: number; mode: string; pos: Vec }[];
    visited: unknown[];
    detections: DetectionRuntime[];
    activeDroneId: number;
    cameraSync: string;
    focusedDetectionId: string | null;
  };
  transport: { status: string };
  scene: { count: number; positions: Float32Array };
  splat?: { status: string; progress: number; chunks: number };
  server?: { connected: boolean; receiving: boolean; chunks: number; detections: number };
  CONFIG?: { sim: { speed: number } };
}

declare global {
  interface Window {
    skylens: SkylensHandle;
  }
}

// PeerJS/STUN/websocket chatter is expected noise, especially while a peer is
// waiting for its partner. We only care about genuine app-level failures.
const NET_NOISE = /peerjs|peer-unavailable|ice|stun|turn|websocket|could not connect/i;

function trackPageErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error' && !NET_NOISE.test(msg.text())) {
      errors.push(`console: ${msg.text()}`);
    }
  });
  return errors;
}

function uniqueRoom(): string {
  return `t${Date.now()}${Math.floor(Math.random() * 1000)}`;
}

async function hasWebGL(page: Page, canvasId: string): Promise<boolean> {
  return page.$eval(`#${canvasId}`, (el) => {
    const c = el as HTMLCanvasElement;
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  });
}

test.describe('per-page boot', () => {
  test('SIM page boots, drives drones, no uncaught errors', async ({ page }) => {
    const errors = trackPageErrors(page);
    // splat=off → shared procedural fallback: fast + CDN-independent.
    // ?demo → drones auto-fly; real (default) mode idles until a route is assigned.
    await page.goto(`/res/static/sim.html?room=${uniqueRoom()}&splat=off&demo`);
    await page.waitForFunction(
      () => window.skylens?.role === 'sim' && window.skylens.state.drones.length === 3,
      undefined,
      { timeout: 15_000 },
    );

    expect(await hasWebGL(page, 'view1')).toBeTruthy();

    // Clock advances and the drone moves along its path (poll to avoid
    // flakiness from rAF throttling / slow warmup).
    const p0 = await page.evaluate(() => window.skylens.state.drones[0].pos);
    await expect
      .poll(
        () =>
          page.evaluate((start) => {
            const p = window.skylens.state.drones[0].pos;
            return (
              Math.abs(p.x - start.x) +
              Math.abs(p.y - start.y) +
              Math.abs(p.z - start.z)
            );
          }, p0),
        { timeout: 8_000 },
      )
      .toBeGreaterThan(0.05);

    // visited buffer (reveal source) accumulates locally on SIM.
    await expect
      .poll(() => page.evaluate(() => window.skylens.state.visited.length), {
        timeout: 8_000,
      })
      .toBeGreaterThan(0);

    // Active-drone switch via number key.
    await page.locator('#view1').click({ position: { x: 50, y: 50 } });
    await page.keyboard.press('3');
    await expect
      .poll(() => page.evaluate(() => window.skylens.state.activeDroneId))
      .toBe(3);

    expect(errors, errors.join('\n')).toEqual([]);
  });

  test('RECON page boots cleanly, no uncaught errors', async ({
    page,
  }) => {
    const errors = trackPageErrors(page);
    // splat=off keeps this test fast + independent of the CDN asset.
    // Real (default, no ?demo) mode: detections arrive only from the server.
    await page.goto(`/res/static/recon.html?room=${uniqueRoom()}&splat=off`);
    await page.waitForFunction(
      () => window.skylens?.role === 'recon',
      undefined,
      { timeout: 15_000 },
    );

    expect(await hasWebGL(page, 'view2')).toBeTruthy();

    // Server-status + minimap panels are present (real-mode UI).
    await expect(page.locator('.server-status')).toBeAttached();
    await expect(page.locator('#minimap')).toBeAttached();

    // Render a few frames with no drones yet (unconnected) — must not throw.
    await page.waitForTimeout(800);
    expect(errors, errors.join('\n')).toEqual([]);
  });
});

test.describe('WebRTC integration (SIM <-> RECON)', () => {
  // Needs outbound access to the public PeerJS broker + STUN.
  test('connects, streams state, runs detection choreography', async ({
    browser,
  }) => {
    test.setTimeout(90_000);
    const room = uniqueRoom();
    const ctx = await browser.newContext();
    const sim = await ctx.newPage();
    const recon = await ctx.newPage();
    const simErrors = trackPageErrors(sim);
    const reconErrors = trackPageErrors(recon);

    await sim.goto(`/res/static/sim.html?room=${room}&splat=off&demo`);
    await recon.goto(`/res/static/recon.html?room=${room}&splat=off&demo`);

    // Both peers report a live DataChannel.
    await expect
      .poll(() => sim.evaluate(() => window.skylens.transport.status), {
        timeout: 45_000,
      })
      .toBe('connected');
    await expect
      .poll(() => recon.evaluate(() => window.skylens.transport.status), {
        timeout: 45_000,
      })
      .toBe('connected');

    // State flows SIM -> RECON: drones + clock + visited appear on RECON.
    await expect
      .poll(() => recon.evaluate(() => window.skylens.state.drones.length), {
        timeout: 15_000,
      })
      .toBe(3);
    await expect
      .poll(() => recon.evaluate(() => window.skylens.state.time), {
        timeout: 15_000,
      })
      .toBeGreaterThan(0);
    await expect
      .poll(() => recon.evaluate(() => window.skylens.state.visited.length), {
        timeout: 15_000,
      })
      .toBeGreaterThan(0);

    // Fast-forward the SIM so detections get revealed on RECON quickly.
    await sim.evaluate(() => {
      window.skylens.CONFIG!.sim.speed = 25;
    });

    // Detections now trickle in from RECON's own mock server on a real-time
    // timer (independent of sim speed), landing by ~8s. Wait for all of them
    // to arrive before counting "total" below, or late arrivals would keep
    // kicking the camera back into FOCUSING after we think we're done.
    await expect
      .poll(() => recon.evaluate(() => window.skylens.state.detections.length), {
        timeout: 20_000,
      })
      .toBe(3);

    // RECON's camera focuses a detection once its area is revealed.
    await recon.waitForFunction(
      () =>
        window.skylens.state.cameraSync === 'FOCUSING' ||
        window.skylens.state.cameraSync === 'LOCKED',
      undefined,
      { timeout: 30_000 },
    );
    await expect(recon.locator('.detect-card.is-open')).toBeVisible({
      timeout: 10_000,
    });

    const total = await recon.evaluate(
      () => window.skylens.state.detections.length,
    );

    // Confirm every revealed detection; the board cycles through the queue.
    const deadline = Date.now() + 45_000;
    while (Date.now() < deadline) {
      const confirmed = await recon.evaluate(
        () => window.skylens.state.detections.filter((d) => d.confirmed).length,
      );
      if (confirmed >= total) break;
      const card = recon.locator('.detect-card.is-open');
      if ((await card.count()) === 0) {
        await recon.waitForTimeout(150);
        continue;
      }
      const fid = await recon.evaluate(
        () => window.skylens.state.focusedDetectionId,
      );
      await card.locator('.detect-card__btn').click();
      await expect
        .poll(
          () =>
            recon.evaluate(
              (id) =>
                window.skylens.state.detections.find((d) => d.id === id)
                  ?.confirmed,
              fid,
            ),
          { timeout: 6_000 },
        )
        .toBe(true);
    }

    expect(
      await recon.evaluate(
        () => window.skylens.state.detections.filter((d) => d.confirmed).length,
      ),
    ).toBe(total);

    // Slow down; with nothing left to focus the camera rests at SYNCED.
    await sim.evaluate(() => {
      window.skylens.CONFIG!.sim.speed = 1;
    });
    // RECON must be foreground so its render loop (rAF) runs — a backgrounded
    // page throttles/pauses rAF and the camera tween would freeze mid-return.
    await recon.bringToFront();
    await expect
      .poll(() => recon.evaluate(() => window.skylens.state.cameraSync), {
        timeout: 15_000,
      })
      .toBe('SYNCED');

    expect(simErrors, simErrors.join('\n')).toEqual([]);
    expect(reconErrors, reconErrors.join('\n')).toEqual([]);
    await ctx.close();
  });
});

test.describe('real Gaussian splat (public sample)', () => {
  // Downloads the lighter public sample and builds it in the browser. Needs
  // outbound access to the Hugging Face CDN.
  test('loads the sample splat to ready and the reveal shader compiles', async ({
    page,
  }) => {
    test.setTimeout(90_000);
    // Shader-compile failures (e.g. a bad onBeforeCompile injection) surface as
    // console errors — track them so a broken splat-reveal patch fails here.
    const errors = trackPageErrors(page);
    // ?demo → the mock server streams the splat chunk; real mode idles.
    await page.goto(`/res/static/recon.html?room=${uniqueRoom()}&splat=light&demo`);
    await page.waitForFunction(() => window.skylens?.role === 'recon', undefined, {
      timeout: 15_000,
    });

    // Reaches 'ready' — download + parse + GL build all succeeded. (The scene
    // itself already loaded before window.skylens was set, so we assert the
    // final render-ready state rather than racing the intermediate 'loading'.)
    await expect
      .poll(() => page.evaluate(() => window.skylens.splat?.status), {
        timeout: 75_000,
      })
      .toBe('ready');

    // Let a few frames render so the patched splat material actually compiles.
    await page.waitForTimeout(1500);
    expect(errors, errors.join('\n')).toEqual([]);
  });

  // The core invariant (PROJECT.md §1): SIM and RECON must derive the SAME point
  // cloud from the SAME splat — SIM shows it low-fi, RECON renders the full splat.
  test('SIM and RECON derive an identical point cloud from the same splat', async ({
    browser,
  }) => {
    test.setTimeout(90_000);
    const room = uniqueRoom();
    const ctx = await browser.newContext();
    const sim = await ctx.newPage();
    const recon = await ctx.newPage();

    await sim.goto(`/res/static/sim.html?room=${room}&splat=light`);
    await recon.goto(`/res/static/recon.html?room=${room}&splat=light`);

    const ready = (p: Page) =>
      p.waitForFunction(() => window.skylens?.scene?.count > 0, undefined, {
        timeout: 60_000,
      });
    await Promise.all([ready(sim), ready(recon)]);

    // Read point count + a deterministic fingerprint of sampled positions.
    const fingerprint = (p: Page) =>
      p.evaluate(() => {
        const s = window.skylens.scene;
        const pts = s.positions;
        const samples: number[] = [];
        const step = Math.max(1, Math.floor(pts.length / 60));
        for (let i = 0; i < pts.length; i += step) samples.push(Math.round(pts[i] * 1000));
        return { count: s.count, samples };
      });
    const [a, b] = await Promise.all([fingerprint(sim), fingerprint(recon)]);

    expect(a.count).toBeGreaterThan(1000);
    expect(b.count).toBe(a.count);
    expect(b.samples).toEqual(a.samples);
    await ctx.close();
  });

  // With the splat on, RECON reveals the SPLAT itself (coverage texture + patched
  // shader) — no point overlay. Verify drones scanning drive detections revealed
  // through the splat-reveal path (isAreaRevealed reads the same coverage the
  // shader samples), with no shader/runtime errors.
  test('splat-mode: drones scanning reveal the splat and detections', async ({
    browser,
  }) => {
    test.setTimeout(120_000);
    const room = uniqueRoom();
    const ctx = await browser.newContext();
    const sim = await ctx.newPage();
    const recon = await ctx.newPage();
    const reconErrors = trackPageErrors(recon);

    await sim.goto(`/res/static/sim.html?room=${room}&splat=light&demo`);
    await recon.goto(`/res/static/recon.html?room=${room}&splat=light&demo`);

    await Promise.all([
      sim.waitForFunction(() => window.skylens?.role === 'sim', undefined, { timeout: 60_000 }),
      recon.waitForFunction(() => window.skylens?.role === 'recon', undefined, { timeout: 60_000 }),
    ]);

    await expect
      .poll(() => recon.evaluate(() => window.skylens.transport.status), { timeout: 45_000 })
      .toBe('connected');

    // Fast-forward so drones sweep and the splat coverage reveals detections.
    await sim.evaluate(() => {
      window.skylens.CONFIG!.sim.speed = 25;
    });

    await expect
      .poll(
        () =>
          recon.evaluate(
            () => window.skylens.state.detections.filter((d) => d.revealed).length,
          ),
        { timeout: 45_000 },
      )
      .toBeGreaterThan(0);

    expect(reconErrors, reconErrors.join('\n')).toEqual([]);
    await ctx.close();
  });
});

test.describe('real vs demo server gating', () => {
  test('real mode (no ?demo): RECON idles waiting for the server', async ({
    page,
  }) => {
    const errors = trackPageErrors(page);
    await page.goto(`/res/static/recon.html?room=${uniqueRoom()}&splat=off`);
    await page.waitForFunction(() => window.skylens?.role === 'recon', undefined, {
      timeout: 15_000,
    });

    await expect(page.locator('.server-status')).toBeAttached();

    // No server wired yet in real mode → nothing streams in over several seconds.
    await page.waitForTimeout(4_000);
    expect(await page.evaluate(() => window.skylens.splat?.status)).toBe('idle');
    expect(await page.evaluate(() => window.skylens.state.detections.length)).toBe(0);

    expect(errors, errors.join('\n')).toEqual([]);
  });

  test('demo mode: RECON receives a splat chunk and detections from the mock server', async ({
    page,
  }) => {
    test.setTimeout(60_000);
    const errors = trackPageErrors(page);
    await page.goto(`/res/static/recon.html?room=${uniqueRoom()}&splat=light&demo`);
    await page.waitForFunction(() => window.skylens?.role === 'recon', undefined, {
      timeout: 15_000,
    });

    await expect(page.locator('#minimap')).toBeAttached();

    await expect
      .poll(() => page.evaluate(() => window.skylens.splat?.status), { timeout: 30_000 })
      .toBe('ready');
    await expect
      .poll(() => page.evaluate(() => window.skylens.state.detections.length), {
        timeout: 30_000,
      })
      .toBeGreaterThan(0);

    expect(errors, errors.join('\n')).toEqual([]);
  });

  test('SIM: demo drones auto-fly; real drones idle until a route is assigned', async ({
    browser,
  }) => {
    test.setTimeout(60_000);
    const ctx = await browser.newContext();
    const demoPage = await ctx.newPage();
    const realPage = await ctx.newPage();
    const demoErrors = trackPageErrors(demoPage);
    const realErrors = trackPageErrors(realPage);

    await demoPage.goto(`/res/static/sim.html?room=${uniqueRoom()}&splat=off&demo`);
    await realPage.goto(`/res/static/sim.html?room=${uniqueRoom()}&splat=off`);
    await Promise.all([
      demoPage.waitForFunction(
        () => window.skylens?.role === 'sim' && window.skylens.state.drones.length === 3,
        undefined,
        { timeout: 15_000 },
      ),
      realPage.waitForFunction(
        () => window.skylens?.role === 'sim' && window.skylens.state.drones.length === 3,
        undefined,
        { timeout: 15_000 },
      ),
    ]);

    const demoStart = await demoPage.evaluate(() => window.skylens.state.drones[0].pos);
    const realStart = await realPage.evaluate(() => window.skylens.state.drones[0].pos);

    // Demo drone moves within a few seconds.
    await expect
      .poll(
        () =>
          demoPage.evaluate((start) => {
            const p = window.skylens.state.drones[0].pos;
            return Math.abs(p.x - start.x) + Math.abs(p.y - start.y) + Math.abs(p.z - start.z);
          }, demoStart),
        { timeout: 8_000 },
      )
      .toBeGreaterThan(0.05);

    // Real drone stays idle over the same window (no route assigned).
    await realPage.waitForTimeout(4_000);
    const realDelta = await realPage.evaluate((start) => {
      const p = window.skylens.state.drones[0].pos;
      return Math.abs(p.x - start.x) + Math.abs(p.y - start.y) + Math.abs(p.z - start.z);
    }, realStart);
    expect(realDelta).toBeLessThan(0.05);

    expect(demoErrors, demoErrors.join('\n')).toEqual([]);
    expect(realErrors, realErrors.join('\n')).toEqual([]);
    await ctx.close();
  });
});
