// The return leg of a 왕복 route, seen through the camera panel.
//
//   npm run demo                                (one shell)
//   node src/test/demo/reverseCheck.mjs [--headed]
//
// Flying home, an aircraft sees the outbound view backwards. Left and right
// were filmed in both directions, so their return leg is its own footage played
// forward. The centre pass exists only outbound, so its return leg is that clip
// played in REVERSE — restarting it would show the drone flying away while it
// comes home.
//
// Both are checked the same way: sample the video element's currentTime twice
// and see which way it moves.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const headed = process.argv.includes('--headed');

const log = (...a) => console.log('[reverse]', ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ headless: !headed });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.bringToFront();
await sleep(4000);

await page.evaluate(async () => {
  const { DEMO_ROUTE } = await import('/src/skylens_drone/core/config.ts');
  window.skylens.core.send({ kind: 'assign-route', droneId: 1, waypoints: DEMO_ROUTE, loop: true });
});
log('route assigned — waiting for the outbound leg to finish');

/** Watch the video element and report which way its clock runs. */
async function sample(station) {
  await page.evaluate(async (want) => {
    const { emit, state } = await import('/src/shared/viewer/store.ts');
    const d = window.skylens.fleet.drones().find((x) => x.station === want);
    if (!d) return;
    state.activeDroneId = d.id;
    emit({ type: 'active-drone', id: d.id });
  }, station);
  // A clip is tens of megabytes; sampling before it has decoded anything reads
  // a stationary zero and says nothing about direction.
  await page
    .waitForFunction(
      () => (document.querySelector('.video-panel__video')?.readyState ?? 0) >= 2,
      undefined,
      { timeout: 20_000 },
    )
    .catch(() => log('warning: the clip never became playable'));
  await sleep(600);
  const read = () =>
    page.evaluate(() => {
      const v = document.querySelector('.video-panel__video');
      const feed = window.skylens.videoPanel?.debugFeed?.() ?? null;
      return v
        ? {
            t: v.currentTime,
            src: (v.getAttribute('src') ?? '').split('/').pop(),
            paused: v.paused,
            readyState: v.readyState,
            reverse: feed?.reverse ?? null,
          }
        : null;
    });
  const a = await read();
  await sleep(1200);
  const b = await read();
  if (!a || !b) return null;
  return {
    src: b.src,
    reverse: b.reverse,
    paused: b.paused,
    readyState: b.readyState,
    from: Number(a.t.toFixed(2)),
    to: Number(b.t.toFixed(2)),
    // Same clip both samples, or the drone changed leg mid-read and the
    // comparison means nothing.
    stable: a.src === b.src && a.reverse === b.reverse,
  };
}

/** Whichever leg it is on, the clock must run the way the feed says. */
function runsCorrectly(s) {
  if (!s || !s.stable) return null;
  return s.reverse ? s.to < s.from : s.to > s.from;
}

// The drone flies ~300 m at 12 m/s after a 10 s transit, so the first
// turnaround lands around t+35 s.
await sleep(40_000);

const legs = await page.evaluate(() => window.skylens.fleet.drones().map((d) => d.station));
log('fleet:', JSON.stringify(legs));

// Which leg the drone is on when a sample lands is not ours to choose, so keep
// sampling the centre until the return leg comes round. Without this the check
// passes on an outbound sample and proves nothing about reverse playback.
let centre = null;
let reversedSeen = null;
for (let i = 0; i < 12; i++) {
  centre = await sample('center');
  log(`centre #${i}:`, JSON.stringify(centre));
  if (centre?.reverse === true && centre.stable) {
    reversedSeen = centre;
    break;
  }
  await sleep(4000);
}

const left = await sample('left');
log('left:', JSON.stringify(left));

console.log('');
console.log('===== RESULT =====');
// The leg a drone is on when the sample lands is not ours to choose — the route
// loops. So the assertion is the invariant, not the leg: a feed flagged reverse
// must run its clock backwards, and any other feed must run it forwards.
const checks = [
  ['a return leg was observed on the centre camera', reversedSeen !== null],
  ['it ran the footage backwards', runsCorrectly(reversedSeen) === true],
  ['it is still the centre pass, not another clip', centre !== null && /center/.test(centre.src)],
  ['the left camera obeys its own direction flag', runsCorrectly(left) === true],
  [
    'the centre pass is the one that has to be reversed',
    // Left and right were flown both ways; only the centre needs substituting.
    left !== null && left.reverse === false,
  ],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
