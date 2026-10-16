// Stage the point-layer demo assets (protocol.ts PointChunk).
//
//   npm run demo:points -- [--src <dir>] [--out <dir>]
//
// Copies the per-segment point files of one capture into res/static/demo/points
// and writes points.json, the manifest the core's demo feed (skylens_core/
// server/pointFeed.ts) streams from. The source is READ ONLY: files are copied,
// never rewritten, so what the board shows is byte-for-byte what the pipeline
// produced.
//
// Expected source layout (pipeline 04d after the 0.1 m voxel pass):
//   r{k}_preview_new.ply   level 1 · first pass, only the ground segment k adds
//   r{k}_refined.ply       level 2 · refined, the whole of segment k
//
// Every file is decoded with the same decoder the board uses, so a file the
// board cannot read fails HERE, and its counts/bounds in the manifest are
// measured rather than copied from somewhere else.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { parsePointPly } from '../shared/pointPly.ts';
import type { Gps } from '../shared/geo.ts';

/** ENU origin of the 04d capture (agreed with the pipeline owner). */
const CAPTURE_ANCHOR: Gps = { lat: 36.36577312, lon: 127.34226545, alt: 29.9 };
// Local copy of the T7 capture, relative to the repo root (npm runs from there).
const DEFAULT_SRC = '../texture_lab/data/1001/snapshots_split/vox010';
const DEFAULT_OUT = 'res/static/demo/points';
const VOXEL_M = 0.1;

export interface PointLevelEntry {
  level: number;
  /** 'preview' = first pass (new ground only), 'refined' = whole segment. */
  pass: 'preview' | 'refined';
  /** Relative to the manifest's directory. */
  url: string;
  bytes: number;
  points: number;
  bbox: { min: [number, number, number]; max: [number, number, number] };
  zLow: number;
  zHigh: number;
  badNormals: number;
}

export interface PointManifest {
  source: string;
  anchor: Gps;
  frame: 'enu';
  voxel: number;
  format: 'xyz-rgb-nxyz-27';
  segments: Array<{ index: number; levels: PointLevelEntry[] }>;
}

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function round(v: number): number {
  return Math.round(v * 1000) / 1000;
}

function main(): void {
  const src = arg('--src', DEFAULT_SRC);
  const out = path.resolve(arg('--out', DEFAULT_OUT));
  if (!fs.existsSync(src)) {
    console.error(`[points] source not found: ${src} (is the drive mounted?)`);
    process.exit(1);
  }
  fs.mkdirSync(out, { recursive: true });

  const PASSES: Array<{ level: number; pass: 'preview' | 'refined'; suffix: string }> = [
    { level: 1, pass: 'preview', suffix: '_preview_new.ply' },
    { level: 2, pass: 'refined', suffix: '_refined.ply' },
  ];

  const indices = new Set<number>();
  for (const f of fs.readdirSync(src)) {
    const m = /^r(\d+)_(preview_new|refined)\.ply$/.exec(f);
    if (m) indices.add(Number(m[1]));
  }
  if (indices.size === 0) {
    console.error(`[points] no r{k}_preview_new.ply / r{k}_refined.ply in ${src}`);
    process.exit(1);
  }

  const manifest: PointManifest = {
    source: src,
    anchor: CAPTURE_ANCHOR,
    frame: 'enu',
    voxel: VOXEL_M,
    format: 'xyz-rgb-nxyz-27',
    segments: [],
  };

  for (const k of [...indices].sort((a, b) => a - b)) {
    const levels: PointLevelEntry[] = [];
    for (const p of PASSES) {
      const name = `r${k}${p.suffix}`;
      const from = path.join(src, name);
      if (!fs.existsSync(from)) {
        console.warn(`[points] segment ${k}: ${name} missing, level ${p.level} skipped`);
        continue;
      }
      const raw = fs.readFileSync(from);
      const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
      const t0 = performance.now();
      const pc = parsePointPly(buf);
      const ms = performance.now() - t0;
      fs.copyFileSync(from, path.join(out, name));
      levels.push({
        level: p.level,
        pass: p.pass,
        url: name,
        bytes: raw.byteLength,
        points: pc.count,
        bbox: {
          min: pc.bbox.min.map(round) as [number, number, number],
          max: pc.bbox.max.map(round) as [number, number, number],
        },
        zLow: round(pc.zLow),
        zHigh: round(pc.zHigh),
        badNormals: pc.badNormals,
      });
      console.log(
        `[points] r${k} L${p.level} ${p.pass.padEnd(7)} ${String(pc.count).padStart(8)} pts ` +
          `${(raw.byteLength / 1e6).toFixed(1).padStart(5)} MB  z ${pc.zLow.toFixed(1)}..${pc.zHigh.toFixed(1)}` +
          `  decode ${ms.toFixed(0)} ms${pc.badNormals ? `  badNormals ${pc.badNormals}` : ''}` +
          `${pc.count !== pc.declared ? `  dropped ${pc.declared - pc.count}` : ''}`,
      );
    }
    if (levels.length) manifest.segments.push({ index: k, levels });
  }

  fs.writeFileSync(path.join(out, 'points.json'), JSON.stringify(manifest, null, 1) + '\n');
  const total = manifest.segments.reduce(
    (a, s) => a + (s.levels.find((l) => l.level === 2)?.points ?? 0),
    0,
  );
  console.log(
    `[points] ${manifest.segments.length} segment(s) → ${out}/points.json · refined total ${total} pts`,
  );
}

main();
