// Turn the recorded demo footage into the codec the drone actually claims.
//
//   npx tsx src/skylens_drone/tools/transcodeDemoFootage.ts [--force]
//
// WHY THIS EXISTS
// ---------------
// res/static/video/*.mp4 are the clips that were flown for the demo, and they
// are H.264 High 10. The wire contract says a VideoSegment is `codec: 'h265'`,
// because a real 5G drone uplink is H.265. Shipping an H.264 file under an
// h265 label would make every latency/size number downstream a measurement of
// something that never happened, so instead we really encode them once:
//
//   res/static/video/*.mp4        H.264 High 10   (source, committed)
//   res/static/video/h265/*.mp4   HEVC Main 10    (generated, what we ship)
//
// The encode targets ~12 Mbps VBR with a 2 s keyframe interval, which is both a
// plausible 5G uplink budget for 4K and the same GOP structure LiveCapture asks
// WebCodecs for — so a demo segment is the same SHAPE of thing a flying drone
// would produce, only prepared ahead of time.
//
// Output is NOT committed by this script; run it once after cloning.

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, statSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const SOURCE_DIR = path.resolve('res/static/video');
const OUT_DIR = path.join(SOURCE_DIR, 'h265');

/** Order matters only for the printed table. */
const FILES = [
  'demo_left.mp4',
  'demo_center.mp4',
  'demo_right.mp4',
  'demo_right_backward.mp4',
  'demo_left_backward.mp4',
];

/** ~12 Mbps VBR HEVC, 10-bit, hvc1-tagged, 2 s GOP at 60 fps. */
const ENCODE_ARGS = [
  '-c:v', 'hevc_nvenc',
  '-preset', 'p5',
  '-profile:v', 'main10',
  '-pix_fmt', 'p010le',
  '-rc', 'vbr',
  '-b:v', '12M',
  '-maxrate', '16M',
  '-bufsize', '24M',
  '-g', '120',
  '-tag:v', 'hvc1',
  '-movflags', '+faststart',
  '-an',
];

/** Software fallback for machines with no NVENC. Much slower, same output. */
const ENCODE_ARGS_CPU = [
  '-c:v', 'libx265',
  '-preset', 'medium',
  '-profile:v', 'main10',
  '-pix_fmt', 'yuv420p10le',
  '-b:v', '12M',
  '-x265-params', 'keyint=120:min-keyint=120',
  '-tag:v', 'hvc1',
  '-movflags', '+faststart',
  '-an',
];

function hasNvenc(): boolean {
  try {
    const out = execFileSync('ffmpeg', ['-hide_banner', '-encoders'], { encoding: 'utf8' });
    return out.includes('hevc_nvenc');
  } catch {
    return false;
  }
}

function probe(file: string): Record<string, string> {
  const out = execFileSync(
    'ffprobe',
    [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'stream=codec_name,profile,width,height,r_frame_rate',
      '-show_entries', 'format=duration',
      '-of', 'default=nw=1',
      file,
    ],
    { encoding: 'utf8' },
  );
  const fields: Record<string, string> = {};
  for (const line of out.trim().split(/\r?\n/)) {
    const eq = line.indexOf('=');
    if (eq > 0) fields[line.slice(0, eq)] = line.slice(eq + 1);
  }
  return fields;
}

function main(): void {
  const force = process.argv.includes('--force');
  if (!existsSync(SOURCE_DIR)) {
    console.error(`[transcode] no source footage at ${SOURCE_DIR}`);
    process.exit(1);
  }
  mkdirSync(OUT_DIR, { recursive: true });
  const args = hasNvenc() ? ENCODE_ARGS : ENCODE_ARGS_CPU;
  console.log(`[transcode] encoder: ${args[1]}`);

  const rows: string[] = [];
  for (const file of FILES) {
    const src = path.join(SOURCE_DIR, file);
    const dst = path.join(OUT_DIR, file);
    if (!existsSync(src)) {
      console.warn(`[transcode] missing source ${file} — skipped`);
      continue;
    }
    if (existsSync(dst) && !force) {
      console.log(`[transcode] ${file} already encoded, skipping (--force to redo)`);
    } else {
      const started = Date.now();
      execFileSync('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-i', src, ...args, dst], {
        stdio: 'inherit',
      });
      console.log(`[transcode] ${file} -> h265/${file} in ${((Date.now() - started) / 1000).toFixed(1)}s`);
    }
    const info = probe(dst);
    const bytes = statSync(dst).size;
    const [n, d] = (info.r_frame_rate ?? '0/1').split('/').map(Number);
    rows.push(
      `  ${file.padEnd(26)} ${info.codec_name}/${info.profile} ${info.width}x${info.height} ` +
        `${(n / d).toFixed(2)}fps ${Math.round(Number(info.duration) * 1000)}ms ${bytes} bytes`,
    );
  }

  console.log('\n[transcode] measured — these are the numbers core/demoAssets.ts must carry:');
  for (const row of rows) console.log(row);
}

main();
