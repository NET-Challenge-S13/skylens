// Minimap — top-right top-down view: live drone positions and detection pins,
// in the same real-world ENU frame as the GPS the pipeline sends.
//
// The extent FOLLOWS THE CONTENT. It used to be fixed to the locally loaded
// splat's bounding box, which is the scaffold scene — a few tens of metres of
// corridor — while the drones fly a route hundreds of metres long. Everything
// live then sat outside the frame and the map looked empty. Now the box is
// fitted to what is actually on it and eased toward changes, so a drone that
// flies off the edge pulls the view with it.

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

  interface Extent {
    minE: number;
    maxE: number;
    minN: number;
    maxN: number;
  }

  /** The scaffold scene, used only until something live arrives. */
  function sceneExtent(): Extent {
    const min = bounds.min, max = bounds.max;
    const corners: Vec3[] = [
      [min.x, 0, min.z],
      [max.x, 0, min.z],
      [max.x, 0, max.z],
      [min.x, 0, max.z],
    ];
    let e0 = Infinity, e1 = -Infinity, n0 = Infinity, n1 = -Infinity;
    for (const v of corners) {
      const enu = sceneToEnu(v);
      e0 = Math.min(e0, enu.e);
      e1 = Math.max(e1, enu.e);
      n0 = Math.min(n0, enu.n);
      n1 = Math.max(n1, enu.n);
    }
    if (!isFinite(e0) || e1 <= e0) { e0 = -20; e1 = 20; }
    if (!isFinite(n0) || n1 <= n0) { n0 = -20; n1 = 20; }
    return { minE: e0, maxE: e1, minN: n0, maxN: n1 };
  }

  /** Square, padded box around everything currently on the map. Square because
   *  a non-uniform scale would make a straight flight path look bent. */
  function targetExtent(points: Array<{ e: number; n: number }>): Extent {
    if (points.length === 0) return pad(sceneExtent());
    let e0 = Infinity, e1 = -Infinity, n0 = Infinity, n1 = -Infinity;
    for (const p of points) {
      e0 = Math.min(e0, p.e);
      e1 = Math.max(e1, p.e);
      n0 = Math.min(n0, p.n);
      n1 = Math.max(n1, p.n);
    }
    const cx = (e0 + e1) / 2;
    const cy = (n0 + n1) / 2;
    // A minimum span keeps a single stationary drone from being magnified to
    // the point where noise reads as movement.
    const half = Math.max((e1 - e0) / 2, (n1 - n0) / 2, 60);
    return pad({ minE: cx - half, maxE: cx + half, minN: cy - half, maxN: cy + half });
  }

  function pad(x: Extent): Extent {
    const padE = (x.maxE - x.minE) * 0.15 || 10;
    const padN = (x.maxN - x.minN) * 0.15 || 10;
    return {
      minE: x.minE - padE,
      maxE: x.maxE + padE,
      minN: x.minN - padN,
      maxN: x.maxN + padN,
    };
  }

  let view: Extent = pad(sceneExtent());

  function ease(to: Extent): void {
    const k = 0.12;
    view = {
      minE: view.minE + (to.minE - view.minE) * k,
      maxE: view.maxE + (to.maxE - view.maxE) * k,
      minN: view.minN + (to.minN - view.minN) * k,
      maxN: view.maxN + (to.maxN - view.maxN) * k,
    };
  }

  function project(e: number, n: number, w: number, h: number): [number, number] {
    const spanE = view.maxE - view.minE || 1;
    const spanN = view.maxN - view.minN || 1;
    const x = ((e - view.minE) / spanE) * w;
    // North is "up" on the minimap -> invert y.
    const y = h - ((n - view.minN) / spanN) * h;
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

    // Fit to what is actually on the map: drones first, detections too.
    const live: Array<{ e: number; n: number }> = [];
    for (const d of state.drones) live.push(sceneToEnu([d.pos.x, d.pos.y, d.pos.z]));
    for (const det of state.detections) live.push(sceneToEnu(det.pos));
    ease(targetExtent(live));

    // Frame + scale bar, so the operator can read distance off the map.
    ctx.strokeStyle = 'rgba(140, 200, 255, 0.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    drawScale(w, h);

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

  /** A bar whose label is a round number of metres, sized to the current fit. */
  function drawScale(w: number, h: number): void {
    const metresPerPx = (view.maxE - view.minE) / Math.max(1, w);
    const rough = metresPerPx * (w * 0.28);
    const pow = 10 ** Math.floor(Math.log10(Math.max(1, rough)));
    const step = [1, 2, 5, 10].map((m) => m * pow).find((v) => v >= rough) ?? pow * 10;
    const px = step / metresPerPx;
    const x0 = 10;
    const y0 = h - 12;
    ctx.strokeStyle = 'rgba(200, 230, 255, 0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, y0 - 3);
    ctx.lineTo(x0, y0);
    ctx.lineTo(x0 + px, y0);
    ctx.lineTo(x0 + px, y0 - 3);
    ctx.stroke();
    ctx.fillStyle = 'rgba(200, 230, 255, 0.75)';
    ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(step >= 1000 ? `${step / 1000} km` : `${step} m`, x0 + px + 5, y0 + 1);
  }

  return {
    update: draw,
    dispose(): void {
      window.removeEventListener('resize', resize);
      host.removeChild(canvas);
    },
  };
}
