// The board's point layer end to end: relay frame → worker fetch/decode →
// THREE.Points → first frame. The test plays the relay itself (a WebSocket on
// a free port, `?relay=<port>`) and serves small generated PLYs from the same
// port, so every request the board makes is visible and can be delayed or
// failed on purpose.

import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { WebSocketServer } from 'ws';
import type { WebSocket } from 'ws';
import { encodePointPly } from '../shared/pointPly.ts';
import type { PointChunk } from '../shared/protocol.ts';
import { CONFIG } from '../shared/viewer/config.ts';

interface Served {
  body: Buffer;
  delayMs: number;
  status: number;
}

interface Relay {
  port: number;
  files: Map<string, Served>;
  requests: string[];
  send(msg: unknown): void;
  connected(): number;
  close(): Promise<void>;
}

async function startRelay(): Promise<Relay> {
  const files = new Map<string, Served>();
  const requests: string[] = [];
  const server = http.createServer((req, res) => {
    const p = (req.url ?? '').split('?')[0];
    requests.push(p);
    const f = files.get(p);
    res.setHeader('access-control-allow-origin', '*');
    if (!f) {
      res.statusCode = 404;
      res.end();
      return;
    }
    setTimeout(() => {
      res.statusCode = f.status;
      res.setHeader('content-type', 'application/octet-stream');
      res.end(f.status === 200 ? f.body : undefined);
    }, f.delayMs);
  });
  const wss = new WebSocketServer({ server, path: '/stream' });
  const boards = new Set<WebSocket>();
  wss.on('connection', (ws) => {
    boards.add(ws);
    ws.on('close', () => boards.delete(ws));
    ws.send(JSON.stringify({ kind: 'relay-hello', boardId: 'b1', upstream: 'online' }));
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const port = (server.address() as AddressInfo).port;
  return {
    port,
    files,
    requests,
    send: (msg) => {
      for (const ws of boards) ws.send(JSON.stringify(msg));
    },
    connected: () => boards.size,
    close: () =>
      new Promise<void>((r) => {
        for (const ws of boards) ws.terminate();
        wss.close();
        // Keep-alive sockets (and a deliberately slow response the board has
        // already abandoned) would otherwise hold close() open indefinitely.
        server.closeAllConnections();
        server.close(() => r());
      }),
  };
}

/** A small slab of ground: n points on a grid, z spread over [z0, z1]. */
function slab(n: number, x0: number, z0: number, z1: number): Buffer {
  const pos: number[] = [];
  const nrm: number[] = [];
  const side = Math.ceil(Math.sqrt(n));
  for (let i = 0; i < n; i++) {
    pos.push(x0 + (i % side) * 0.5, Math.floor(i / side) * 0.5, z0 + ((z1 - z0) * i) / (n - 1));
    nrm.push(0, 0, 1);
  }
  return Buffer.from(encodePointPly(pos, null, nrm));
}

function chunk(relay: Relay, segment: number, level: number, points: number, final = level === 2): PointChunk {
  return {
    kind: 'point-chunk',
    id: `pts-seg${segment}-l${level}`,
    segment,
    level,
    final,
    url: `http://127.0.0.1:${relay.port}/pts/s${segment}l${level}.ply`,
    bytes: 0,
    points,
    format: 'xyz-rgb-nxyz-27',
    voxel: 0.1,
    bbox: { min: [0, 0, 0], max: [1, 1, 1] },
    // The board's own anchor: scene y equals ENU z, so heights can be checked
    // straight against the file.
    align: { anchor: CONFIG.geo.anchor, position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  };
}

function serve(relay: Relay, segment: number, level: number, body: Buffer, delayMs = 0, status = 200): void {
  relay.files.set(`/pts/s${segment}l${level}.ply`, { body, delayMs, status });
}

interface PointStats {
  enabled: boolean;
  segments: Array<{ segment: number; shown: number; loading: number; wanted: number }>;
  points: number;
  heightRange: [number, number] | null;
  timings: Array<{ segment: number; level: number; totalMs: number }>;
  errors: Array<{ segment: number; level: number; message: string }>;
  stale: number;
  drawObjects: number;
}

async function stats(page: Page): Promise<PointStats | null> {
  return page.evaluate(
    () => (window as unknown as { skylens: { points: { stats: PointStats | null } } }).skylens.points.stats,
  );
}

async function openBoard(page: Page, relay: Relay, extra = ''): Promise<void> {
  await page.goto(`/res/static/status.html?relay=${relay.port}${extra}`);
  await page.waitForFunction(
    () => (window as unknown as { skylens?: { role: string } }).skylens?.role === 'status',
    undefined,
    { timeout: 60_000 },
  );
  await expect.poll(() => relay.connected(), { timeout: 15_000 }).toBe(1);
}

const seg = (s: PointStats | null, k: number) => s?.segments.find((x) => x.segment === k);

test.describe('board point layer', () => {
  let relay: Relay;
  test.beforeEach(async () => {
    relay = await startRelay();
  });
  test.afterEach(async () => {
    await relay.close();
  });

  test('a refined pass replaces the first pass; a replayed first pass is ignored', async ({ page }) => {
    test.setTimeout(90_000);
    serve(relay, 0, 1, slab(1000, 0, -40, -20));
    serve(relay, 0, 2, slab(3000, 0, -35, -15));
    await openBoard(page, relay);

    relay.send(chunk(relay, 0, 1, 1000));
    await expect.poll(async () => seg(await stats(page), 0)?.shown, { timeout: 15_000 }).toBe(1);
    expect((await stats(page))!.points).toBe(1000);
    // Let a frame actually draw it: a first pass replaced before any frame is
    // never drawn, and rightly gets no timing.
    await expect.poll(async () => (await stats(page))?.timings.length, { timeout: 5_000 }).toBe(1);

    relay.send(chunk(relay, 0, 2, 3000));
    await expect.poll(async () => seg(await stats(page), 0)?.shown, { timeout: 15_000 }).toBe(2);
    let s = (await stats(page))!;
    expect(s.points).toBe(3000);
    expect(s.drawObjects).toBe(1);

    // The relay replays its cache on reconnect; an old level must not win.
    relay.send(chunk(relay, 0, 1, 1000));
    await page.waitForTimeout(500);
    s = (await stats(page))!;
    expect(seg(s, 0)?.shown).toBe(2);
    expect(s.points).toBe(3000);
    expect(s.stale).toBe(1);
    expect(relay.requests.filter((r) => r === '/pts/s0l1.ply').length).toBe(1);

    // Height colouring follows the points on screen (1st..99th percentile).
    const [lo, hi] = s.heightRange!;
    expect(lo).toBeGreaterThan(-35.5);
    expect(lo).toBeLessThan(-34.5);
    expect(hi).toBeGreaterThan(-15.5);
    expect(hi).toBeLessThan(-14.5);

    // Each arrival was timed through to the frame that drew it.
    expect(s.timings.map((t) => `${t.segment}/${t.level}`)).toEqual(['0/1', '0/2']);
    for (const t of s.timings) expect(t.totalMs).toBeGreaterThan(0);
  });

  test('a refined pass arriving mid-load cancels the first pass, which never shows', async ({ page }) => {
    test.setTimeout(90_000);
    serve(relay, 1, 1, slab(800, 10, -40, -20), 1500);
    serve(relay, 1, 2, slab(1600, 10, -35, -15));
    await openBoard(page, relay);

    relay.send(chunk(relay, 1, 1, 800));
    await expect.poll(async () => seg(await stats(page), 1)?.loading, { timeout: 15_000 }).toBe(1);
    relay.send(chunk(relay, 1, 2, 1600));
    await expect.poll(async () => seg(await stats(page), 1)?.shown, { timeout: 15_000 }).toBe(2);
    await page.waitForTimeout(3000); // well past the slow first pass (1.5 s)
    const s = (await stats(page))!;
    expect(seg(s, 1)?.shown).toBe(2);
    expect(s.points).toBe(1600);
    expect(s.timings.map((t) => `${t.segment}/${t.level}`)).toEqual(['1/2']);
  });

  test('while off nothing is fetched; turning it on loads only the newest level', async ({ page }) => {
    test.setTimeout(90_000);
    serve(relay, 0, 1, slab(500, 0, -40, -20));
    serve(relay, 0, 2, slab(900, 0, -35, -15));
    serve(relay, 1, 1, slab(400, 20, -38, -18));
    await openBoard(page, relay, '&points=off');

    relay.send(chunk(relay, 0, 1, 500));
    relay.send(chunk(relay, 0, 2, 900));
    relay.send(chunk(relay, 1, 1, 400, false));
    await expect.poll(async () => (await stats(page))?.segments.length ?? 0).toBe(2);
    await page.waitForTimeout(300);
    expect(relay.requests.filter((r) => r.startsWith('/pts/'))).toEqual([]);
    await expect(page.locator('#point-toggle')).toContainText('2구간 수신');

    await page.locator('#point-toggle').click();
    await expect.poll(async () => (await stats(page))?.points ?? 0, { timeout: 15_000 }).toBe(1300);
    expect(relay.requests.filter((r) => r.startsWith('/pts/')).sort()).toEqual([
      '/pts/s0l2.ply',
      '/pts/s1l1.ply',
    ]);
    await expect(page.locator('#point-toggle')).toHaveClass(/is-on/);

    // And off again: hidden, follow camera handed back, data kept.
    await page.locator('#point-toggle').click();
    const s = (await stats(page))!;
    expect(s.enabled).toBe(false);
    expect(s.points).toBe(1300);
  });

  test('a failed refined pass keeps the first pass on screen', async ({ page }) => {
    test.setTimeout(90_000);
    serve(relay, 2, 1, slab(700, 0, -40, -20));
    serve(relay, 2, 2, Buffer.alloc(0), 0, 404);
    await openBoard(page, relay);

    relay.send(chunk(relay, 2, 1, 700));
    await expect.poll(async () => seg(await stats(page), 2)?.shown, { timeout: 15_000 }).toBe(1);
    relay.send(chunk(relay, 2, 2, 1400));
    await expect.poll(async () => (await stats(page))?.errors.length, { timeout: 15_000 }).toBe(1);
    const s = (await stats(page))!;
    expect(seg(s, 2)?.shown).toBe(1);
    expect(s.points).toBe(700);
    expect(s.errors[0]).toMatchObject({ segment: 2, level: 2 });
  });

  test('the panel framing follows new segments until the operator takes it', async ({ page }) => {
    test.setTimeout(90_000);
    serve(relay, 0, 1, slab(4000, 0, -40, -20));
    serve(relay, 1, 1, slab(4000, 300, -40, -20));
    serve(relay, 2, 1, slab(4000, 600, -40, -20));
    await openBoard(page, relay);
    const cam = () =>
      page.evaluate(
        () =>
          (window as unknown as { skylens: { points: { inset: { camPos: number[] } } } }).skylens.points
            .inset.camPos,
      );

    relay.send(chunk(relay, 0, 1, 4000));
    await expect.poll(async () => seg(await stats(page), 0)?.shown, { timeout: 15_000 }).toBe(1);
    await page.waitForTimeout(300);
    const first = await cam();

    // A new segment 300 m east: the view eases out to hold both.
    relay.send(chunk(relay, 1, 1, 4000));
    await expect.poll(async () => seg(await stats(page), 1)?.shown, { timeout: 15_000 }).toBe(1);
    await expect.poll(async () => (await cam())[0] - first[0], { timeout: 10_000 }).toBeGreaterThan(50);

    // The operator presses on the panel: from then on arrivals leave it alone.
    await page.locator('#point-inset .point-inset__view').click();
    await page.waitForTimeout(300);
    const held = await cam();
    relay.send(chunk(relay, 2, 1, 4000));
    await expect.poll(async () => seg(await stats(page), 2)?.shown, { timeout: 15_000 }).toBe(1);
    await page.waitForTimeout(1500);
    expect(await cam()).toEqual(held);

    // framePoints() hands the framing back.
    await page.evaluate(() => (window as unknown as { skylens: { points: { frame(): boolean } } }).skylens.points.frame());
    await expect.poll(async () => (await cam())[0] - held[0], { timeout: 10_000 }).toBeGreaterThan(50);
  });

  test('over the point budget every segment is thinned by the same fraction', async ({ page }) => {
    test.setTimeout(90_000);
    serve(relay, 0, 2, slab(3000, 0, -35, -15));
    serve(relay, 1, 2, slab(1000, 100, -35, -15));
    await openBoard(page, relay);
    await page.evaluate(() =>
      (window as unknown as { skylens: { points: { setBudget(n: number): void } } }).skylens.points.setBudget(2000),
    );
    relay.send(chunk(relay, 0, 2, 3000));
    relay.send(chunk(relay, 1, 2, 1000));
    await expect.poll(async () => (await stats(page))?.points ?? 0, { timeout: 15_000 }).toBe(4000);
    let s = (await stats(page)) as PointStats & { drawn: number; drawFraction: number };
    expect(s.drawFraction).toBe(0.5);
    expect(s.drawn).toBe(2000);
    // Back under budget: everything is drawn again.
    await page.evaluate(() =>
      (window as unknown as { skylens: { points: { setBudget(n: number): void } } }).skylens.points.setBudget(10_000),
    );
    s = (await stats(page)) as PointStats & { drawn: number; drawFraction: number };
    expect(s.drawFraction).toBe(1);
    expect(s.drawn).toBe(4000);
  });

  test('the main view is untouched: the points get their own small panel', async ({ page }) => {
    test.setTimeout(90_000);
    serve(relay, 0, 1, slab(20_000, 0, -40, -20));
    await openBoard(page, relay);
    const main = () =>
      page.evaluate(() => {
        const d = (window as unknown as { skylens: { dbg: { camPos: number[]; target: number[] } } }).skylens.dbg;
        return { camPos: d.camPos, target: d.target };
      });
    // No panel until the feed offers something.
    await expect(page.locator('#point-inset')).toBeHidden();
    await page.waitForTimeout(500);
    const before = await main();

    relay.send(chunk(relay, 0, 1, 20_000));
    await expect.poll(async () => seg(await stats(page), 0)?.shown, { timeout: 15_000 }).toBe(1);
    await expect(page.locator('#point-inset')).toBeVisible();
    await page.waitForTimeout(800);

    // The main board camera did not move for the points.
    expect(await main()).toEqual(before);

    // The panel's camera looks at the slab: x 0..~70, ENU y 0..~70 → scene
    // z 0..-70, height -40..-20.
    const t = await page.evaluate(
      () =>
        (window as unknown as { skylens: { points: { inset: { target: number[] } } } }).skylens.points.inset
          .target,
    );
    expect(t[0]).toBeGreaterThan(0);
    expect(t[0]).toBeLessThan(80);
    expect(t[1]).toBeGreaterThan(-45);
    expect(t[1]).toBeLessThan(-15);
    expect(t[2]).toBeLessThan(0);

    // Height legend reads the delivered range, relative to its low end.
    await expect(page.locator('.point-inset__tick--low')).toHaveText('0 m');
    await expect(page.locator('.point-inset__tick--high')).toHaveText(/^\+1\d\.\d m$/);

    // The toggle closes the panel without touching the main view either.
    await page.locator('#point-toggle').click();
    await expect(page.locator('#point-inset')).toBeHidden();
    expect(await main()).toEqual(before);
  });
});
