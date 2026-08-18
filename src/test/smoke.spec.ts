import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';

// End-to-end smoke tests for the two-page WebRTC build.
//
//  - Per-page tests verify each role boots the real browser runtime (WebGL
//    shader compilation, first-frame code paths) without uncaught exceptions.
//  - The integration test opens BOTH pages against the public PeerJS broker in
//    a unique room and verifies live state actually flows CONTROL -> STATUS and the
//    detection choreography runs on STATUS.
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
  role: 'control' | 'status';
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
  splat?: {
    status: string;
    progress: number;
    chunks: number;
    replaced: number;
    segmentLevels: Record<number, number>;
  };
  server?: {
    connected: boolean;
    receiving: boolean;
    chunks: number;
    detections: number;
    segments: { index: number; level: number; levels: number; steps: number }[];
  };
  CONFIG?: { clock: { speed: number } };
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
  test('CONTROL page boots, drives drones, no uncaught errors', async ({ page }) => {
    const errors = trackPageErrors(page);
    // splat=off → shared procedural fallback: fast + CDN-independent.
    // ?demo → drones auto-fly; real (default) mode idles until a route is assigned.
    await page.goto(`/res/static/control.html?room=${uniqueRoom()}&splat=off&demo`);
    await page.waitForFunction(
      () => window.skylens?.role === 'control' && window.skylens.state.drones.length === 3,
      undefined,
      { timeout: 15_000 },
    );

    expect(await hasWebGL(page, 'control-view')).toBeTruthy();

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

    // visited buffer (reveal source) accumulates locally on CONTROL.
    await expect
      .poll(() => page.evaluate(() => window.skylens.state.visited.length), {
        timeout: 8_000,
      })
      .toBeGreaterThan(0);

    // Active-drone switch via number key.
    await page.locator('#control-view').click({ position: { x: 50, y: 50 } });
    await page.keyboard.press('3');
    await expect
      .poll(() => page.evaluate(() => window.skylens.state.activeDroneId))
      .toBe(3);

    expect(errors, errors.join('\n')).toEqual([]);
  });

  test('STATUS page boots cleanly, no uncaught errors', async ({
    page,
  }) => {
    const errors = trackPageErrors(page);
    // splat=off keeps this test fast + independent of the CDN asset.
    // Real (default, no ?demo) mode: detections arrive only from the server.
    await page.goto(`/res/static/status.html?room=${uniqueRoom()}&splat=off`);
    await page.waitForFunction(
      () => window.skylens?.role === 'status',
      undefined,
      { timeout: 15_000 },
    );

    expect(await hasWebGL(page, 'status-view')).toBeTruthy();

    // Server-status + minimap panels are present (real-mode UI).
    await expect(page.locator('.server-status')).toBeAttached();
    await expect(page.locator('#minimap')).toBeAttached();

    // Render a few frames with no drones yet (unconnected) — must not throw.
    await page.waitForTimeout(800);
    expect(errors, errors.join('\n')).toEqual([]);
  });
});

test.describe('WebRTC integration (CONTROL <-> STATUS)', () => {
  // Needs outbound access to the public PeerJS broker + STUN.
  test('connects, streams state, runs detection choreography', async ({
    browser,
  }) => {
    test.setTimeout(90_000);
    const room = uniqueRoom();
    const ctx = await browser.newContext();
    const control = await ctx.newPage();
    const status = await ctx.newPage();
    const controlErrors = trackPageErrors(control);
    const statusErrors = trackPageErrors(status);

    await control.goto(`/res/static/control.html?room=${room}&splat=off&demo`);
    await status.goto(`/res/static/status.html?room=${room}&splat=off&demo`);

    // Both peers report a live DataChannel.
    await expect
      .poll(() => control.evaluate(() => window.skylens.transport.status), {
        timeout: 45_000,
      })
      .toBe('connected');
    await expect
      .poll(() => status.evaluate(() => window.skylens.transport.status), {
        timeout: 45_000,
      })
      .toBe('connected');

    // State flows CONTROL -> STATUS: drones + clock + visited appear on STATUS.
    await expect
      .poll(() => status.evaluate(() => window.skylens.state.drones.length), {
        timeout: 15_000,
      })
      .toBe(3);
    await expect
      .poll(() => status.evaluate(() => window.skylens.state.time), {
        timeout: 15_000,
      })
      .toBeGreaterThan(0);
    await expect
      .poll(() => status.evaluate(() => window.skylens.state.visited.length), {
        timeout: 15_000,
      })
      .toBeGreaterThan(0);

    // Fast-forward the CONTROL so detections get revealed on STATUS quickly.
    await control.evaluate(() => {
      window.skylens.CONFIG!.clock.speed = 25;
    });

    // Detections now trickle in from STATUS's own mock server on a real-time
    // timer (independent of control speed), landing by ~8s. Wait for all of them
    // to arrive before counting "total" below, or late arrivals would keep
    // kicking the camera back into FOCUSING after we think we're done.
    await expect
      .poll(() => status.evaluate(() => window.skylens.state.detections.length), {
        timeout: 20_000,
      })
      .toBe(3);

    // STATUS's camera focuses a detection once its area is revealed.
    await status.waitForFunction(
      () =>
        window.skylens.state.cameraSync === 'FOCUSING' ||
        window.skylens.state.cameraSync === 'LOCKED',
      undefined,
      { timeout: 30_000 },
    );
    await expect(status.locator('.detect-card.is-open')).toBeVisible({
      timeout: 10_000,
    });

    const total = await status.evaluate(
      () => window.skylens.state.detections.length,
    );

    // Confirm every revealed detection; the board cycles through the queue.
    const deadline = Date.now() + 45_000;
    while (Date.now() < deadline) {
      const confirmed = await status.evaluate(
        () => window.skylens.state.detections.filter((d) => d.confirmed).length,
      );
      if (confirmed >= total) break;
      const card = status.locator('.detect-card.is-open');
      if ((await card.count()) === 0) {
        await status.waitForTimeout(150);
        continue;
      }
      const fid = await status.evaluate(
        () => window.skylens.state.focusedDetectionId,
      );
      await card.locator('.detect-card__btn').click();
      await expect
        .poll(
          () =>
            status.evaluate(
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
      await status.evaluate(
        () => window.skylens.state.detections.filter((d) => d.confirmed).length,
      ),
    ).toBe(total);

    // Slow down; with nothing left to focus the camera rests at SYNCED.
    await control.evaluate(() => {
      window.skylens.CONFIG!.clock.speed = 1;
    });
    // STATUS must be foreground so its render loop (rAF) runs — a backgrounded
    // page throttles/pauses rAF and the camera tween would freeze mid-return.
    await status.bringToFront();
    await expect
      .poll(() => status.evaluate(() => window.skylens.state.cameraSync), {
        timeout: 15_000,
      })
      .toBe('SYNCED');

    expect(controlErrors, controlErrors.join('\n')).toEqual([]);
    expect(statusErrors, statusErrors.join('\n')).toEqual([]);
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
    await page.goto(`/res/static/status.html?room=${uniqueRoom()}&splat=light&demo`);
    await page.waitForFunction(() => window.skylens?.role === 'status', undefined, {
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

  // The core invariant (PROJECT.md §1): CONTROL and STATUS must derive the SAME point
  // cloud from the SAME splat — CONTROL shows it low-fi, STATUS renders the full splat.
  test('CONTROL and STATUS derive an identical point cloud from the same splat', async ({
    browser,
  }) => {
    test.setTimeout(90_000);
    const room = uniqueRoom();
    const ctx = await browser.newContext();
    const control = await ctx.newPage();
    const status = await ctx.newPage();

    await control.goto(`/res/static/control.html?room=${room}&splat=light`);
    await status.goto(`/res/static/status.html?room=${room}&splat=light`);

    const ready = (p: Page) =>
      p.waitForFunction(() => window.skylens?.scene?.count > 0, undefined, {
        timeout: 60_000,
      });
    await Promise.all([ready(control), ready(status)]);

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
    const [a, b] = await Promise.all([fingerprint(control), fingerprint(status)]);

    expect(a.count).toBeGreaterThan(1000);
    expect(b.count).toBe(a.count);
    expect(b.samples).toEqual(a.samples);
    await ctx.close();
  });

  // With the splat on, STATUS reveals the SPLAT itself (coverage texture + patched
  // shader) — no point overlay. Verify drones scanning drive detections revealed
  // through the splat-reveal path (isAreaRevealed reads the same coverage the
  // shader samples), with no shader/runtime errors.
  test('splat-mode: drones scanning reveal the splat and detections', async ({
    browser,
  }) => {
    test.setTimeout(120_000);
    const room = uniqueRoom();
    const ctx = await browser.newContext();
    const control = await ctx.newPage();
    const status = await ctx.newPage();
    const statusErrors = trackPageErrors(status);

    await control.goto(`/res/static/control.html?room=${room}&splat=light&demo`);
    await status.goto(`/res/static/status.html?room=${room}&splat=light&demo`);

    await Promise.all([
      control.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 60_000 }),
      status.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 60_000 }),
    ]);

    await expect
      .poll(() => status.evaluate(() => window.skylens.transport.status), { timeout: 45_000 })
      .toBe('connected');

    // Fast-forward so drones sweep and the splat coverage reveals detections.
    await control.evaluate(() => {
      window.skylens.CONFIG!.clock.speed = 25;
    });

    await expect
      .poll(
        () =>
          status.evaluate(
            () => window.skylens.state.detections.filter((d) => d.revealed).length,
          ),
        { timeout: 45_000 },
      )
      .toBeGreaterThan(0);

    expect(statusErrors, statusErrors.join('\n')).toEqual([]);
    await ctx.close();
  });
});


test.describe('delay-pattern reconstruction stream', () => {
  // Interim report Ⅱ-3-다: the flight is cut into segments; each segment is
  // delivered at a low training-step level first and refined afterwards, and one
  // segment's refinement OVERLAPS the next segment's first delivery. Needs the
  // local capture in res/static/demo (split_segments.py); skipped without it.
  test('segments refine in place while the next segment starts', async ({ page }) => {
    test.setTimeout(90_000);
    const manifest = await page.request.get('/res/static/demo/segments.json');
    test.skip(!manifest.ok(), 'res/static/demo segment assets not generated');

    const errors = trackPageErrors(page);
    // No ?splat → the local capture, which is what the segment assets are cut from.
    await page.goto(`/res/static/status.html?room=${uniqueRoom()}&demo`);
    await page.waitForFunction(() => window.skylens?.role === 'status', undefined, {
      timeout: 15_000,
    });

    // Segment 2's first level lands while segment 1 is still being refined —
    // that overlap IS the delay pattern.
    await expect
      .poll(() => page.evaluate(() => window.skylens.splat?.segmentLevels[1] ?? 0), {
        timeout: 45_000,
      })
      .toBeGreaterThan(0);

    const levels = await page.evaluate(() => window.skylens.splat!.segmentLevels);
    const segCount = await page.evaluate(() => window.skylens.server!.segments.length);
    const levelCount = await page.evaluate(() => window.skylens.server!.segments[0].levels);
    expect(segCount).toBeGreaterThan(1);
    // Segment 1 got past its first level before segment 2 arrived...
    expect(levels[0]).toBeGreaterThan(1);
    // ...and is still short of its final level, so the two overlap.
    expect(levels[0]).toBeLessThan(levelCount);

    // Refinements REPLACE the level they supersede instead of stacking on it.
    expect(await page.evaluate(() => window.skylens.splat!.replaced)).toBeGreaterThan(0);
    expect(await page.evaluate(() => window.skylens.splat?.status)).toBe('ready');

    // Let the rebuilt splat material render: the reveal/clip shader patch must be
    // re-applied after a removal, and a broken injection shows up as a console error.
    await page.waitForTimeout(1500);
    expect(errors, errors.join('\n')).toEqual([]);

    // Tear the scene down before the next test: this run holds a dozen splat
    // scenes, each with its own sort worker + GL buffers, and leaving them for
    // the fixture to reap starves the next browser context under SwiftShader.
    await page.goto('about:blank');
    await page.waitForTimeout(500);
  });
});

test.describe('real vs demo server gating', () => {
  test('real mode (no ?demo): STATUS idles waiting for the server', async ({
    page,
  }) => {
    const errors = trackPageErrors(page);
    await page.goto(`/res/static/status.html?room=${uniqueRoom()}&splat=off`);
    await page.waitForFunction(() => window.skylens?.role === 'status', undefined, {
      timeout: 15_000,
    });

    await expect(page.locator('.server-status')).toBeAttached();

    // No server wired yet in real mode → nothing streams in over several seconds.
    await page.waitForTimeout(4_000);
    expect(await page.evaluate(() => window.skylens.splat?.status)).toBe('idle');
    expect(await page.evaluate(() => window.skylens.state.detections.length)).toBe(0);

    expect(errors, errors.join('\n')).toEqual([]);
  });

  test('demo mode: STATUS receives a splat chunk and detections from the mock server', async ({
    page,
  }) => {
    test.setTimeout(60_000);
    const errors = trackPageErrors(page);
    await page.goto(`/res/static/status.html?room=${uniqueRoom()}&splat=light&demo`);
    await page.waitForFunction(() => window.skylens?.role === 'status', undefined, {
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

  test('CONTROL: demo drones auto-fly; real drones idle until a route is assigned', async ({
    browser,
  }) => {
    test.setTimeout(60_000);
    const ctx = await browser.newContext();
    const demoPage = await ctx.newPage();
    const realPage = await ctx.newPage();
    const demoErrors = trackPageErrors(demoPage);
    const realErrors = trackPageErrors(realPage);

    await demoPage.goto(`/res/static/control.html?room=${uniqueRoom()}&splat=off&demo`);
    await realPage.goto(`/res/static/control.html?room=${uniqueRoom()}&splat=off`);
    await Promise.all([
      demoPage.waitForFunction(
        () => window.skylens?.role === 'control' && window.skylens.state.drones.length === 3,
        undefined,
        { timeout: 15_000 },
      ),
      realPage.waitForFunction(
        () => window.skylens?.role === 'control' && window.skylens.state.drones.length === 3,
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
