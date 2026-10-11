// Does the 3D world show the same ground as the planner map?
//
//   npm run demo
//   node src/test/control/registrationCheck.mjs [--headed]
//
// The GPS chain is already checked end to end (routeFidelityCheck.mjs): the
// centre drone flies the planned line to within a metre. That leaves the other
// half of "the drone is not where I pointed" — whether the two PICTURES agree.
// The operator plans on a satellite map and then watches the flight over a
// draped 3D terrain; if those two images are out of register, every number can
// be right and the aircraft will still appear to fly over the wrong building.
//
// So both views are rendered over the same metric footprint centred on the same
// fix, and matched against each other by correlation rather than by eye: the
// answer is an offset in metres and a scale ratio. Reading it off screenshots
// is what produced two wrong diagnoses already — a chase camera that quietly
// overrode the top-down made a 60 m view look like a 600 m one.

import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const SITE = { lat: 36.3685, lon: 127.3475 };
const SPAN_M = 600; // ground width of both captures
const SHOTS = 'C:/tmp/skylens-shots';
const headed = process.argv.includes('--headed');

const log = (...a) => console.log('[register]', ...a);

const browser = await chromium.launch({ headless: !headed });
const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(`${TOWER}?display=aerial`, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.bringToFront();
// The drape streams in cell by cell; correlating against half-loaded imagery
// measures the loader, not the registration.
await page.waitForTimeout(20_000);

// --- 1. the planner, centred on the site ----------------------------------
await page.evaluate(
  ([site, span]) => window.skylens.routeModal.debugView(site, span) ?? window.skylens.routeModal.open(),
  [SITE, SPAN_M],
);
await page.evaluate(() => window.skylens.routeModal.open());
await page.waitForTimeout(6000);
const plannerPng = await page.locator('.route-modal__canvas').screenshot();
writeFileSync(`${SHOTS}/reg-planner.png`, plannerPng);

const view = await page.evaluate(() => window.skylens.routeModal.debugBounds());
log('planner:', JSON.stringify(view));

// --- 2. the 3D view, straight down over the same fix ------------------------
await page.evaluate(() => window.skylens.routeModal.close());
const cam = await page.evaluate(
  async ([site, span]) => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const frame = window.skylens.frame;
    const at = frame.toScene({ lat: site.lat, lon: site.lon, alt: 0 });

    // What the camera covers is set by its height above the GROUND, not above
    // the frame's zero — the zero is the bbox's lowest elevation, tens of
    // metres below the site, and ignoring that magnified the view by 1.2x and
    // read as a registration error. Ask the terrain directly: raycasting down
    // hits whatever is in the way, and once a route is drawn that is the route.
    const groundY = frame.groundYAt({ lat: site.lat, lon: site.lon, alt: 0 });

    // Ground half-width a perspective camera sees looking straight down.
    const halfWorld = (span / 2) * frame.unitsPerMeter;
    const above = halfWorld / Math.tan((55 / 2) * (Math.PI / 180));
    window.skylens.viewer.debugTopDown(new THREE.Vector3(at.x, groundY, at.z), above);
    return {
      aboveGroundM: Math.round(above / frame.unitsPerMeter),
      groundAltM: Math.round(frame.groundAltAt({ lat: site.lat, lon: site.lon, alt: 0 })),
    };
  },
  [SITE, SPAN_M],
);
log('camera:', JSON.stringify(cam));
await page.waitForTimeout(2500);
const threePng = await page.locator('canvas').first().screenshot();
writeFileSync(`${SHOTS}/reg-3d.png`, threePng);

// --- 3. match them ----------------------------------------------------------
// Both images cover SPAN_M across their full width, so a feature at the same
// place on the ground must land at the same fraction of each image.
const match = await page.evaluate(
  async ([aB64, bB64, spanM]) => {
    const load = async (b64) => {
      const res = await fetch(`data:image/png;base64,${b64}`);
      return createImageBitmap(await res.blob());
    };
    /** Grayscale square, `n` px across, resampled from a bitmap. */
    const gray = (bmp, n) => {
      const c = new OffscreenCanvas(n, n);
      const g = c.getContext('2d');
      g.drawImage(bmp, 0, 0, n, n);
      const d = g.getImageData(0, 0, n, n).data;
      const out = new Float32Array(n * n);
      for (let i = 0; i < n * n; i++) {
        out[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
      }
      return out;
    };
    /** Normalised cross-correlation of a template against a window. */
    const ncc = (t, tN, s, sN, ox, oy, scale) => {
      let sumT = 0;
      let sumS = 0;
      let n = 0;
      const vals = [];
      for (let y = 0; y < tN; y += 2) {
        for (let x = 0; x < tN; x += 2) {
          // Where this template pixel lands in the search image.
          const sx = Math.round((x - tN / 2) * scale + sN / 2 + ox);
          const sy = Math.round((y - tN / 2) * scale + sN / 2 + oy);
          if (sx < 0 || sy < 0 || sx >= sN || sy >= sN) continue;
          const a = t[y * tN + x];
          const b = s[sy * sN + sx];
          vals.push(a, b);
          sumT += a;
          sumS += b;
          n++;
        }
      }
      if (n < 200) return -1;
      const mT = sumT / n;
      const mS = sumS / n;
      let num = 0;
      let dT = 0;
      let dS = 0;
      for (let i = 0; i < vals.length; i += 2) {
        const a = vals[i] - mT;
        const b = vals[i + 1] - mS;
        num += a * b;
        dT += a * a;
        dS += b * b;
      }
      return dT > 0 && dS > 0 ? num / Math.sqrt(dT * dS) : -1;
    };

    const N = 150; // 4 m per pixel at a 600 m span
    const planner = gray(await load(aB64), N);
    const three = gray(await load(bB64), N);

    // Template: the middle of the planner. The edges of the 3D shot carry HUD
    // panels, and the middle is the part an operator plans in anyway.
    const tN = 80;
    const tpl = new Float32Array(tN * tN);
    const off = Math.floor((N - tN) / 2);
    for (let y = 0; y < tN; y++) {
      for (let x = 0; x < tN; x++) tpl[y * tN + x] = planner[(y + off) * N + (x + off)];
    }

    let best = { score: -2, dx: 0, dy: 0, scale: 1 };
    for (let scale = 0.6; scale <= 1.65; scale += 0.05) {
      for (let dy = -30; dy <= 30; dy++) {
        for (let dx = -30; dx <= 30; dx++) {
          const score = ncc(tpl, tN, three, N, dx, dy, scale);
          if (score > best.score) best = { score, dx, dy, scale };
        }
      }
    }
    const mPerPx = spanM / N;
    return {
      score: Number(best.score.toFixed(3)),
      offsetM: {
        east: Number((best.dx * mPerPx).toFixed(1)),
        north: Number((-best.dy * mPerPx).toFixed(1)),
      },
      scale: Number(best.scale.toFixed(2)),
    };
  },
  [plannerPng.toString('base64'), threePng.toString('base64'), SPAN_M],
);

log('best match:', JSON.stringify(match));

console.log('');
console.log('===== RESULT =====');
const offset = Math.hypot(match.offsetM.east, match.offsetM.north);
const checks = [
  ['the two views show recognisably the same ground', match.score > 0.35],
  [`the 3D view is at the planner's scale (x${match.scale})`, Math.abs(match.scale - 1) <= 0.1],
  [`the 3D view is not shifted (${offset.toFixed(0)} m)`, offset <= 20],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));
console.log(`shots: ${SHOTS}/reg-planner.png · ${SHOTS}/reg-3d.png`);

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
