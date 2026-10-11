// Point layer DEMO feed: plays a staged capture as PointChunks (protocol.ts).
//
// Why this lives in the core: scheduling belongs to the core alone
// (COMPONENTS.md §3.4). The transport from the reconstruction node to the core
// is not decided yet, so in demo mode the core replays the staged files
// (res/static/demo/points, `npm run demo:points`) on the same delay pattern the
// live node will follow:
//
//   tick 0      seg 0 L1
//   tick k      seg k L1, then seg k-1 L2     (a new segment's first pass never waits)
//   tick N      seg N-1 L2 [final]
//
// The clock here only stands in for "the pipeline finished the next pass"; it
// does not decide anything a live feed would not. Messages go out through the
// same Store + Distributor as everything else, so a late-joining board gets
// the highest level per segment and nothing superseded.

import fs from 'node:fs';
import path from 'node:path';
import type { Gps } from '../../shared/geo.ts';
import type { PointChunk } from '../../shared/protocol.ts';

/** Shape of points.json (written by src/demo/preparePoints.ts). */
interface Manifest {
  anchor: Gps;
  frame: 'enu';
  voxel: number | null;
  format: 'xyz-rgb-nxyz-27';
  segments: Array<{
    index: number;
    levels: Array<{
      level: number;
      url: string;
      bytes: number;
      points: number;
      bbox: { min: [number, number, number]; max: [number, number, number] };
    }>;
  }>;
}

export interface PointFeedOptions {
  manifestPath: string;
  urlBase: string;
  intervalMs: number;
  /** Store + broadcast. Returns false when the chunk was superseded. */
  emit: (chunk: PointChunk) => boolean;
}

export interface PointFeedCounters {
  available: boolean;
  detail: string;
  state: 'idle' | 'playing' | 'done';
  segments: number;
  sent: number;
  ticks: number;
}

export class PointFeed {
  private readonly opts: PointFeedOptions;
  private readonly schedule: PointChunk[][] = [];
  private readonly manifest: Manifest | null = null;
  private readonly detail: string;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private state: PointFeedCounters['state'] = 'idle';
  private tick = 0;
  private sent = 0;

  constructor(opts: PointFeedOptions) {
    this.opts = opts;
    const file = path.resolve(opts.manifestPath);
    if (!fs.existsSync(file)) {
      this.detail = `no manifest at ${file} (run \`npm run demo:points\`)`;
      return;
    }
    try {
      this.manifest = JSON.parse(fs.readFileSync(file, 'utf8')) as Manifest;
    } catch (err) {
      this.detail = `unreadable manifest ${file}: ${(err as Error).message}`;
      return;
    }
    // A manifest entry whose file is gone would hand the boards a 404 they
    // render as a missing segment; refuse it here instead, by name.
    const dir = path.dirname(file);
    const missing = this.manifest.segments
      .flatMap((s) => s.levels)
      .filter((l) => !fs.existsSync(path.join(dir, l.url)))
      .map((l) => l.url);
    if (missing.length) {
      this.manifest = null;
      this.detail = `manifest lists files that are not on disk: ${missing.join(', ')}`;
      return;
    }
    this.buildSchedule(this.manifest);
    this.detail = `${this.manifest.segments.length} segment(s) from ${file}`;
  }

  get available(): boolean {
    return this.manifest !== null;
  }

  private chunkFor(m: Manifest, seg: Manifest['segments'][number], level: number): PointChunk {
    const l = seg.levels.find((x) => x.level === level)!;
    const top = Math.max(...seg.levels.map((x) => x.level));
    return {
      kind: 'point-chunk',
      id: `pts-seg${seg.index}-l${level}`,
      segment: seg.index,
      level,
      final: level === top,
      url: `${this.opts.urlBase}/${l.url}`,
      bytes: l.bytes,
      points: l.points,
      format: m.format,
      voxel: m.voxel,
      bbox: l.bbox,
      align: {
        anchor: m.anchor,
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
      },
    };
  }

  /** Tick k: segment k's first level, then segment k-1's remaining levels. */
  private buildSchedule(m: Manifest): void {
    const segs = [...m.segments].sort((a, b) => a.index - b.index);
    for (let k = 0; k <= segs.length; k++) {
      const tick: PointChunk[] = [];
      const cur = segs[k];
      const prev = segs[k - 1];
      if (cur) {
        const first = Math.min(...cur.levels.map((l) => l.level));
        tick.push(this.chunkFor(m, cur, first));
      }
      if (prev) {
        const first = Math.min(...prev.levels.map((l) => l.level));
        for (const l of [...prev.levels].sort((a, b) => a.level - b.level)) {
          if (l.level !== first) tick.push(this.chunkFor(m, prev, l.level));
        }
      }
      if (tick.length) this.schedule.push(tick);
    }
  }

  /** Begin playback. Idempotent: one capture plays once per core process. */
  start(reason: string): void {
    if (!this.manifest || this.state !== 'idle') return;
    this.state = 'playing';
    console.log(
      `[core] point feed: playing ${this.schedule.length} tick(s) every ` +
        `${this.opts.intervalMs} ms (${reason})`,
    );
    this.step();
  }

  private step(): void {
    const tick = this.schedule[this.tick];
    if (!tick) {
      this.state = 'done';
      this.timer = null;
      console.log(`[core] point feed: done, ${this.sent} chunk(s) sent`);
      return;
    }
    for (const c of tick) {
      if (this.opts.emit(c)) {
        this.sent += 1;
        console.log(
          `[core] point segment ${c.segment} level ${c.level} · ${c.points} pts → viewers` +
            `${c.final ? ' [final]' : ''}`,
        );
      }
    }
    this.tick += 1;
    this.timer = setTimeout(() => this.step(), this.opts.intervalMs);
  }

  stop(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  counters(): PointFeedCounters {
    return {
      available: this.available,
      detail: this.detail,
      state: this.state,
      segments: this.manifest?.segments.length ?? 0,
      sent: this.sent,
      ticks: this.tick,
    };
  }
}
