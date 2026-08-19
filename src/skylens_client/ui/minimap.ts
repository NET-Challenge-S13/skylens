// Minimap — bottom-right top-down view of the reconstructed scene, live drone
// positions, the assigned route (if any), and detection pins. Positions are
// converted scene -> ENU (skylens_core/geo.ts) so the minimap reads in the
// same real-world frame as the server's GPS detections/telemetry.

import * as THREE from 'three';
import { state } from '../../shared/viewer/store.ts';
import { sceneToEnu } from '../../shared/geo.ts';
import type { Vec3 } from '../../shared/viewer/types.ts';

export interface Minimap {
  update(): void;
  dispose(): void;
}

/** Shape of an optional assigned-route field, if the store ever carries one. */
interface RouteLike {
  droneId?: number;
  waypoints?: Vec3[];
}

function noop(): Minimap {
  return { update() {}, dispose() {} };
}

/** Mount into `#minimap` (see status.html). Bounds come from the scene's own box3. */
export function mountMinimap(bounds: THREE.Box3): Minimap {
  const hostEl = document.getElementById('minimap');
  if (!hostEl) return noop();
  const host: HTMLElement = hostEl;

  const canvas = document.createElement('canvas');
  canvas.className = 'minimap__canvas';
  host.appendChild(canvas);
  const rawCtx = canvas.getContext('2d');
  if (!rawCtx) return noop();
  const ctx: CanvasRenderingContext2D = rawCtx;

  // Compute the ENU footprint from the scene bounds' ground-plane corners.
  const min = bounds.min, max = bounds.max;
  const cornersScene: Vec3[] = [
    [min.x, 0, min.z],
    [max.x, 0, min.z],
    [max.x, 0, max.z],
    [min.x, 0, max.z],
  ];
  let minE = Infinity, maxE = -Infinity, minN = Infinity, maxN = -Infinity;
  for (const v of cornersScene) {
    const enu = sceneToEnu(v);
    minE = Math.min(minE, enu.e);
    maxE = Math.max(maxE, enu.e);
    minN = Math.min(minN, enu.n);
    maxN = Math.max(maxN, enu.n);
  }
  if (!isFinite(minE) || !isFinite(maxE) || maxE <= minE) { minE = -20; maxE = 20; }
  if (!isFinite(minN) || !isFinite(maxN) || maxN <= minN) { minN = -20; maxN = 20; }
  // Margin so drones/detections slightly outside the fitted core still show.
  const padE = (maxE - minE) * 0.3 || 10;
  const padN = (maxN - minN) * 0.3 || 10;
  minE -= padE; maxE += padE; minN -= padN; maxN += padN;
  const spanE = maxE - minE || 1;
  const spanN = maxN - minN || 1;

  function project(e: number, n: number, w: number, h: number): [number, number] {
    const x = ((e - minE) / spanE) * w;
    // North is "up" on the minimap -> invert y.
    const y = h - ((n - minN) / spanN) * h;
    return [x, y];
  }

  let cssW = 160, cssH = 160;
  function resize(): void {
    cssW = host.clientWidth || 160;
    cssH = host.clientHeight || 160;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(cssH * dpr));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  function draw(): void {
    const w = cssW, h = cssH;
    ctx.clearRect(0, 0, w, h);

    // Scene extent frame.
    ctx.strokeStyle = 'rgba(140, 200, 255, 0.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    // Assigned route polyline, if the store carries one (optional field).
    const route = (state as unknown as { route?: RouteLike | Vec3[] }).route;
    const pts = route ? (Array.isArray(route) ? route : route.waypoints) : undefined;
    if (pts && pts.length > 1) {
      ctx.strokeStyle = 'rgba(255, 210, 127, 0.85)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      pts.forEach((p, i) => {
        const enu = sceneToEnu(p);
        const [x, y] = project(enu.e, enu.n, w, h);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Detection pins.
    for (const det of state.detections) {
      const enu = sceneToEnu(det.pos);
      const [x, y] = project(enu.e, enu.n, w, h);
      ctx.fillStyle = det.kind === 'person' ? '#39d98a' : '#ff4d4d';
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Live drones.
    for (const d of state.drones) {
      const enu = sceneToEnu([d.pos.x, d.pos.y, d.pos.z]);
      const [x, y] = project(enu.e, enu.n, w, h);
      const active = d.id === state.activeDroneId;
      ctx.fillStyle = active ? '#9fe8ff' : '#4a90d9';
      ctx.beginPath();
      ctx.arc(x, y, active ? 4 : 3, 0, Math.PI * 2);
      ctx.fill();
      if (active) {
        ctx.strokeStyle = 'rgba(159, 232, 255, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, 6.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  return {
    update: draw,
    dispose(): void {
      window.removeEventListener('resize', resize);
      host.removeChild(canvas);
    },
  };
}
