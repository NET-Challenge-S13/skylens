// Point layer, the parts that need no browser: the PLY decoder every hop
// shares, the per-segment replacement rules, and the core's demo feed order.
//
// The board itself (fetch → worker → THREE.Points → first frame) is exercised
// in pointLayer.spec.ts against a scripted relay.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test, expect } from '@playwright/test';
import { encodePointPly, parsePointPly, readPlyLayout, shufflePoints } from '../shared/pointPly.ts';
import { SegmentSlots } from '../skylens_client/statusview/segmentSlots.ts';
import { PointFeed } from '../skylens_core/server/pointFeed.ts';
import type { PointChunk } from '../shared/protocol.ts';

function header(lines: string[]): string {
  return ['ply', 'format binary_little_endian 1.0', ...lines, 'end_header', ''].join('\n');
}

function withHeader(h: string, body: Uint8Array): ArrayBuffer {
  const buf = new Uint8Array(h.length + body.length);
  for (let i = 0; i < h.length; i++) buf[i] = h.charCodeAt(i);
  buf.set(body, h.length);
  return buf.buffer;
}

test.describe('point PLY decoder', () => {
  test('round-trips the 27-byte record', () => {
    const pos = [1, 2, 3, -4.5, 5.25, -35, 10, 20, 30];
    const col = [255, 0, 10, 1, 2, 3, 100, 150, 200];
    const nrm = [0, 0, 1, 1, 0, 0, 0, -1, 0];
    const pc = parsePointPly(encodePointPly(pos, col, nrm));
    expect(pc.count).toBe(3);
    expect(Array.from(pc.positions)).toEqual(pos);
    expect(Array.from(pc.colors!)).toEqual(col);
    expect(Array.from(pc.normals!)).toEqual([0, 0, 127, 127, 0, 0, 0, -127, 0]);
    expect(pc.bbox).toEqual({ min: [-4.5, 2, -35], max: [10, 20, 30] });
    expect(pc.badNormals).toBe(0);
  });

  test('accepts both header spellings (float/uchar and float32/uint8)', () => {
    const body = new Uint8Array(encodePointPly([1, 2, 3]).byteLength);
    const src = new Uint8Array(encodePointPly([1, 2, 3]));
    const h0 = new TextDecoder().decode(src).indexOf('end_header') + 'end_header\n'.length;
    body.set(src.subarray(h0));
    const rec = body.subarray(0, 27);
    const h = header([
      'element vertex 1',
      'property float32 x',
      'property float32 y',
      'property float32 z',
      'property uint8 red',
      'property uint8 green',
      'property uint8 blue',
      'property float32 nx',
      'property float32 ny',
      'property float32 nz',
    ]);
    const pc = parsePointPly(withHeader(h, rec));
    expect(Array.from(pc.positions)).toEqual([1, 2, 3]);
    expect(pc.colors).not.toBeNull();
    expect(pc.normals).not.toBeNull();
  });

  test('drops non-finite positions and zeroes normals that are not unit length', () => {
    const pos = [0, 0, 0, NaN, 1, 1, 2, 2, 2];
    const nrm = [0, 0, 1, 0, 0, 1, 1e15, NaN, 3];
    const pc = parsePointPly(encodePointPly(pos, null, nrm));
    expect(pc.declared).toBe(3);
    expect(pc.count).toBe(2);
    expect(Array.from(pc.positions)).toEqual([0, 0, 0, 2, 2, 2]);
    expect(Array.from(pc.normals!)).toEqual([0, 0, 127, 0, 0, 0]);
    expect(pc.badNormals).toBe(1);
  });

  test('near-unit normals are renormalised, not wrapped past Int8', () => {
    const pc = parsePointPly(encodePointPly([0, 0, 0, 1, 1, 1], null, [0, 0, 1.02, 0.6, 0, 0.82]));
    expect(Array.from(pc.normals!.subarray(0, 3))).toEqual([0, 0, 127]);
    for (const v of pc.normals!) expect(Math.abs(v)).toBeLessThanOrEqual(127);
    expect(pc.normals![3]).toBeGreaterThan(0);
  });

  test('the framing core ignores a stray point far off the capture', () => {
    const pos: number[] = [];
    for (let i = 0; i < 1000; i++) pos.push(i % 40, Math.floor(i / 40), -30);
    pos.push(1000, 1000, -30);
    const pc = parsePointPly(encodePointPly(pos));
    expect(pc.bbox.max[0]).toBe(1000);
    expect(pc.core.max[0]).toBeLessThan(41);
    expect(pc.core.max[1]).toBeLessThan(26);
  });

  test('height range is the 1st..99th percentile, not the extremes', () => {
    const pos: number[] = [];
    for (let i = 0; i < 1000; i++) pos.push(i % 37, i % 11, -35 + (i / 1000) * 20);
    // Two stray points far below and above the surface.
    pos.push(0, 0, -400, 0, 0, 300);
    const pc = parsePointPly(encodePointPly(pos));
    expect(pc.bbox.min[2]).toBe(-400);
    expect(pc.bbox.max[2]).toBe(300);
    expect(pc.zLow).toBeGreaterThan(-36);
    expect(pc.zLow).toBeLessThan(-34);
    expect(pc.zHigh).toBeGreaterThan(-16);
    expect(pc.zHigh).toBeLessThan(-14);
  });

  test('shuffling keeps every point whole (position, colour and normal move together)', () => {
    const pos: number[] = [];
    const col: number[] = [];
    const nrm: number[] = [];
    for (let i = 0; i < 500; i++) {
      pos.push(i, i * 2, -i);
      col.push(i % 256, (i * 7) % 256, (i * 13) % 256);
      nrm.push(0, 0, 1);
    }
    const pc = parsePointPly(encodePointPly(pos, col, nrm));
    shufflePoints(pc, 42);
    const seen = new Set<number>();
    let moved = 0;
    for (let k = 0; k < pc.count; k++) {
      const i = pc.positions[k * 3];
      expect(pc.positions[k * 3 + 1]).toBe(i * 2);
      expect(pc.positions[k * 3 + 2]).toBe(-i);
      expect(pc.colors![k * 3]).toBe(i % 256);
      expect(pc.colors![k * 3 + 2]).toBe((i * 13) % 256);
      seen.add(i);
      if (i !== k) moved++;
    }
    expect(seen.size).toBe(500);
    expect(moved).toBeGreaterThan(400);
    // Seeded: the same seed gives the same order.
    const again = parsePointPly(encodePointPly(pos, col, nrm));
    shufflePoints(again, 42);
    expect(Array.from(again.positions.subarray(0, 30))).toEqual(Array.from(pc.positions.subarray(0, 30)));
  });

  test('refuses what it cannot honour', () => {
    expect(() => parsePointPly(new TextEncoder().encode('hello').buffer)).toThrow(/not a PLY/);
    const ascii = header(['element vertex 0', 'property float x']).replace(
      'binary_little_endian',
      'ascii',
    );
    expect(() => readPlyLayout(withHeader(ascii, new Uint8Array()))).toThrow(/unsupported/);
    const full = new Uint8Array(encodePointPly([1, 2, 3, 4, 5, 6]));
    expect(() => parsePointPly(full.slice(0, full.length - 5).buffer)).toThrow(/truncated/);
  });
});

// ---------------------------------------------------------------------------

interface Msg {
  segment: number;
  level: number;
}

/** A controllable loader: each load waits until the test resolves or fails it. */
function harness() {
  const loads: Array<{
    msg: Msg;
    signal: AbortSignal;
    resolve: (v: string) => void;
    reject: (e: Error) => void;
  }> = [];
  const mounted: string[] = [];
  const released: string[] = [];
  const errors: string[] = [];
  const slots = new SegmentSlots<Msg, string>({
    load: (msg, signal) =>
      new Promise<string>((resolve, reject) => {
        loads.push({ msg, signal, resolve, reject });
        signal.addEventListener('abort', () => reject(new Error('aborted')));
      }),
    mount: (msg, obj) => mounted.push(`${msg.segment}:${obj}`),
    release: (obj) => released.push(obj),
    onError: (msg) => errors.push(`${msg.segment}/${msg.level}`),
  });
  const tick = () => new Promise((r) => setTimeout(r, 0));
  return { slots, loads, mounted, released, errors, tick };
}

test.describe('per-segment replacement', () => {
  test('a higher level replaces the lower one; the old object is released after the swap', async () => {
    const h = harness();
    expect(h.slots.offer({ segment: 0, level: 1 })).toBe('accepted');
    h.loads[0].resolve('s0l1');
    await h.tick();
    expect(h.mounted).toEqual(['0:s0l1']);
    h.slots.offer({ segment: 0, level: 2 });
    h.loads[1].resolve('s0l2');
    await h.tick();
    expect(h.mounted).toEqual(['0:s0l1', '0:s0l2']);
    expect(h.released).toEqual(['s0l1']);
    expect(h.slots.states()).toEqual([{ segment: 0, shown: 2, loading: 0, wanted: 0 }]);
  });

  test('stale and repeated levels change nothing', async () => {
    const h = harness();
    h.slots.offer({ segment: 3, level: 2 });
    h.loads[0].resolve('s3l2');
    await h.tick();
    expect(h.slots.offer({ segment: 3, level: 1 })).toBe('stale');
    expect(h.slots.offer({ segment: 3, level: 2 })).toBe('stale');
    expect(h.loads.length).toBe(1);
    expect(h.slots.counters.stale).toBe(2);
  });

  test('a higher level arriving mid-load cancels the lower one, which is never shown', async () => {
    const h = harness();
    h.slots.offer({ segment: 1, level: 1 });
    h.slots.offer({ segment: 1, level: 2 });
    expect(h.loads[0].signal.aborted).toBe(true);
    h.loads[1].resolve('s1l2');
    await h.tick();
    expect(h.mounted).toEqual(['1:s1l2']);
    // Even if the cancelled load had finished anyway, it is released unseen.
    h.loads[0].resolve('s1l1-late');
    await h.tick();
    expect(h.mounted).toEqual(['1:s1l2']);
    expect(h.slots.counters.cancelled).toBe(1);
  });

  test('segments are independent', async () => {
    const h = harness();
    h.slots.offer({ segment: 0, level: 1 });
    h.slots.offer({ segment: 1, level: 1 });
    h.slots.offer({ segment: 0, level: 2 });
    expect(h.loads.map((l) => `${l.msg.segment}/${l.msg.level}`)).toEqual(['0/1', '1/1', '0/2']);
    expect(h.loads[1].signal.aborted).toBe(false);
  });

  test('a failed load keeps what the segment showed and can be retried', async () => {
    const h = harness();
    h.slots.offer({ segment: 0, level: 1 });
    h.loads[0].resolve('s0l1');
    await h.tick();
    h.slots.offer({ segment: 0, level: 2 });
    h.loads[1].reject(new Error('HTTP 404'));
    await h.tick();
    expect(h.errors).toEqual(['0/2']);
    expect(h.slots.states()[0].shown).toBe(1);
    expect(h.slots.counters.failed).toEqual([0]);
    // The relay replays its cache on reconnect: the same level is accepted again.
    expect(h.slots.offer({ segment: 0, level: 2 })).toBe('accepted');
    h.loads[2].resolve('s0l2');
    await h.tick();
    expect(h.slots.states()[0].shown).toBe(2);
  });

  test('turning slots off cancels loads in flight and resumes them when turned on', async () => {
    const h = harness();
    h.slots.offer({ segment: 0, level: 2 });
    h.slots.setActive(false);
    expect(h.loads[0].signal.aborted).toBe(true);
    await h.tick();
    expect(h.slots.states()).toEqual([{ segment: 0, shown: 0, loading: 0, wanted: 2 }]);
    h.slots.setActive(true);
    expect(h.loads.length).toBe(2);
    h.loads[1].resolve('s0l2');
    await h.tick();
    expect(h.mounted).toEqual(['0:s0l2']);
  });

  test('inactive slots remember offers and load only the newest level when turned on', async () => {
    const h = harness();
    h.slots.setActive(false);
    h.slots.offer({ segment: 0, level: 1 });
    h.slots.offer({ segment: 0, level: 2 });
    h.slots.offer({ segment: 1, level: 1 });
    expect(h.loads.length).toBe(0);
    h.slots.setActive(true);
    expect(h.loads.map((l) => `${l.msg.segment}/${l.msg.level}`)).toEqual(['0/2', '1/1']);
  });
});

// ---------------------------------------------------------------------------

test.describe('core point feed (demo)', () => {
  function stage(segments: number[][]): string {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'skylens-points-'));
    const manifest = {
      anchor: { lat: 36.36577312, lon: 127.34226545, alt: 29.9 },
      frame: 'enu',
      voxel: 0.1,
      format: 'xyz-rgb-nxyz-27',
      segments: segments.map((levels, index) => ({
        index,
        levels: levels.map((level) => {
          const url = `r${index}_l${level}.ply`;
          fs.writeFileSync(path.join(dir, url), 'x');
          return {
            level,
            url,
            bytes: 1,
            points: 10 * level,
            bbox: { min: [0, 0, 0], max: [1, 1, 1] },
          };
        }),
      })),
    };
    fs.writeFileSync(path.join(dir, 'points.json'), JSON.stringify(manifest));
    return dir;
  }

  test('plays the delay pattern: segment k first pass, then segment k-1 refined', async () => {
    const dir = stage([[1, 2], [1, 2], [1, 2]]);
    const sent: PointChunk[] = [];
    const feed = new PointFeed({
      manifestPath: path.join(dir, 'points.json'),
      urlBase: '/pts',
      intervalMs: 5,
      emit: (c) => {
        sent.push(c);
        return true;
      },
    });
    expect(feed.available).toBe(true);
    feed.start('test');
    feed.start('again'); // idempotent
    await expect.poll(() => feed.counters().state).toBe('done');
    expect(sent.map((c) => `${c.segment}/${c.level}${c.final ? 'F' : ''}`)).toEqual([
      '0/1',
      '1/1',
      '0/2F',
      '2/1',
      '1/2F',
      '2/2F',
    ]);
    expect(sent[0].url).toBe('/pts/r0_l1.ply');
    expect(sent[0].align.anchor).toEqual({ lat: 36.36577312, lon: 127.34226545, alt: 29.9 });
    expect(sent[0].format).toBe('xyz-rgb-nxyz-27');
  });

  test('refuses a manifest whose files are missing, and says which', () => {
    const dir = stage([[1, 2]]);
    fs.rmSync(path.join(dir, 'r0_l2.ply'));
    const feed = new PointFeed({
      manifestPath: path.join(dir, 'points.json'),
      urlBase: '/pts',
      intervalMs: 5,
      emit: () => true,
    });
    expect(feed.available).toBe(false);
    expect(feed.counters().detail).toContain('r0_l2.ply');
  });

  test('is off, not broken, without a manifest', () => {
    const feed = new PointFeed({
      manifestPath: path.join(os.tmpdir(), 'no-such-dir', 'points.json'),
      urlBase: '/pts',
      intervalMs: 5,
      emit: () => true,
    });
    expect(feed.available).toBe(false);
    feed.start('test');
    expect(feed.counters().state).toBe('idle');
  });
});
