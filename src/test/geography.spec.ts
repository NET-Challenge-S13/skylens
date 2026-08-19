// The demo only makes sense if three independently-owned constants describe the
// SAME piece of ground:
//
//   shared/viewer/config.ts        geo.anchor          (ENU origin)
//   shared/viewer/config.ts        control.defaultMap  (what the tower renders)
//   skylens_drone/core/config.ts   DEMO_ROUTE          (where the drone flies)
//
// They drifted once already — the fleet flew over Seoul while the tower drew
// Daejeon, 150 km away, which on screen looks exactly like a rendering bug. No
// browser needed to catch that: it is arithmetic on four numbers.

import { test, expect } from '@playwright/test';
import { CONFIG } from '../shared/viewer/config.ts';
import { PRESETS } from '../shared/viewer/sources/terrainSource.ts';
import { DEMO_ROUTE } from '../skylens_drone/core/config.ts';

test.describe('demo geography stays on one patch of ground', () => {
  const bbox = PRESETS[CONFIG.control.defaultMap];

  test('the control tower default map is a real preset', () => {
    expect(bbox, `no preset named "${CONFIG.control.defaultMap}"`).toBeTruthy();
  });

  test('the ENU anchor sits inside the map the tower draws', () => {
    const [west, south, east, north] = bbox;
    const { lat, lon } = CONFIG.geo.anchor;
    expect(lon, `anchor lon ${lon} outside [${west}, ${east}]`).toBeGreaterThanOrEqual(west);
    expect(lon).toBeLessThanOrEqual(east);
    expect(lat, `anchor lat ${lat} outside [${south}, ${north}]`).toBeGreaterThanOrEqual(south);
    expect(lat).toBeLessThanOrEqual(north);
  });

  test('every demo waypoint sits inside the same map', () => {
    const [west, south, east, north] = bbox;
    for (const [i, wp] of DEMO_ROUTE.entries()) {
      expect(wp.lon, `waypoint ${i} lon ${wp.lon} outside [${west}, ${east}]`)
        .toBeGreaterThanOrEqual(west);
      expect(wp.lon).toBeLessThanOrEqual(east);
      expect(wp.lat, `waypoint ${i} lat ${wp.lat} outside [${south}, ${north}]`)
        .toBeGreaterThanOrEqual(south);
      expect(wp.lat).toBeLessThanOrEqual(north);
    }
  });

  // A route far shorter than the segment length never closes a segment, so the
  // delay pattern would never start; far longer and the demo drags.
  test('the route is long enough to close several segments', () => {
    let metres = 0;
    for (let i = 1; i < DEMO_ROUTE.length; i++) {
      const a = DEMO_ROUTE[i - 1];
      const b = DEMO_ROUTE[i];
      const dLat = (b.lat - a.lat) * 111_320;
      const dLon = (b.lon - a.lon) * 111_320 * Math.cos((a.lat * Math.PI) / 180);
      metres += Math.hypot(dLat, dLon);
    }
    expect(metres).toBeGreaterThan(120);
    expect(metres).toBeLessThan(3000);
  });
});
