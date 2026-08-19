// What is one world unit worth, and does every layer agree?
//
//   npm run demo
//   node src/test/control/frameProbe.mjs
//
// The tower's scene is not in metres: the terrain normalises an arbitrary bbox
// to a fixed span and everything else has to follow that same factor. This
// prints the factor each layer is actually using — terrain mesh, geo frame,
// drones, buildings — so a layer drawn at the wrong scale is a number, not a
// judgement call about a screenshot.

import { chromium } from '@playwright/test';

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1000, height: 1000 } });
await p.goto('http://localhost:8080/res/static/control.html?display=aerial', {
  waitUntil: 'domcontentloaded',
  timeout: 120_000,
});
await p.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await p.bringToFront();
await p.waitForTimeout(15_000);

const probe = await p.evaluate(async () => {
  // Vite serves bare specifiers only to modules it transformed, so pull three
  // through the app's own graph rather than importing it here.
  const THREE = await import('/node_modules/three/build/three.module.js');
  const frame = window.skylens.frame;
  const scene = window.skylens.viewer.debugScene?.() ?? null;

  // 1. The frame's own claim.
  const claimed = frame.unitsPerMeter;

  // 2. Measure it: two fixes 1000 m apart on the ground.
  const a = frame.toScene({ lat: 36.36, lon: 127.34, alt: 0 });
  const c = frame.toScene({ lat: 36.36, lon: 127.34 + 1000 / (111_320 * Math.cos((36.36 * Math.PI) / 180)), alt: 0 });
  const measured = Math.hypot(c.x - a.x, c.z - a.z) / 1000;

  // 3. Every mesh in the scene, with its world footprint.
  const layers = [];
  if (scene) {
    scene.traverse((o) => {
      if (!o.isMesh && !o.isPoints) return;
      const g = o.geometry;
      if (!g) return;
      g.computeBoundingBox();
      const bb = g.boundingBox;
      if (!bb) return;
      const size = new THREE.Vector3();
      bb.getSize(size);
      const scale = o.getWorldScale(new THREE.Vector3());
      layers.push({
        name: o.name || o.type,
        // Footprint in world units, then in metres via the frame's factor.
        unitsX: Number((size.x * scale.x).toFixed(2)),
        unitsZ: Number((size.z * scale.z).toFixed(2)),
        metresX: Math.round((size.x * scale.x) / claimed),
        metresZ: Math.round((size.z * scale.z) / claimed),
        verts: g.attributes?.position?.count ?? 0,
      });
    });
  }

  return { claimed, measured, layers: layers.slice(0, 25) };
});

console.log('unitsPerMeter claimed :', probe.claimed);
console.log('unitsPerMeter measured:', probe.measured);
console.log('1000 m of ground =', (probe.measured * 1000).toFixed(2), 'world units');
console.log('');
console.log('layer footprints (world units → metres at that factor):');
for (const l of probe.layers) {
  console.log(
    `  ${l.name.padEnd(22)} ${String(l.unitsX).padStart(8)} x ${String(l.unitsZ).padStart(8)} u` +
      `  = ${String(l.metresX).padStart(7)} x ${String(l.metresZ).padStart(7)} m   (${l.verts} verts)`,
  );
}

await b.close();
