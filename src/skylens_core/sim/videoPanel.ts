// Bottom-right "main drone camera" placeholder. Renders a synthetic feed
// (animated gradient + scanlines + crosshair + a small telemetry overlay) on
// a canvas until a real feed is wired up via setSource().

import { state } from '../store.ts';
import { CONFIG } from '../config.ts';
import { sceneToGps } from '../geo.ts';

export interface VideoPanel {
  /** Seam for a real feed later: a MediaStream, a video URL, or null to fall
   *  back to the synthetic placeholder. */
  setSource(src: MediaStream | string | null): void;
  dispose(): void;
}

export function createVideoPanel(container: HTMLElement): VideoPanel {
  const root = document.createElement('div');
  root.className = 'video-panel';

  const canvas = document.createElement('canvas');
  canvas.className = 'video-panel__canvas';
  canvas.width = 320;
  canvas.height = 200;

  const video = document.createElement('video');
  video.className = 'video-panel__video is-hidden';
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;

  const label = document.createElement('div');
  label.className = 'video-panel__label';
  label.textContent = 'MAIN CAM';

  const overlay = document.createElement('div');
  overlay.className = 'video-panel__overlay';

  root.append(canvas, video, label, overlay);
  container.appendChild(root);

  const ctx = canvas.getContext('2d');
  let usingRealSource = false;
  let raf = 0;
  let t = 0;

  function draw(): void {
    raf = requestAnimationFrame(draw);
    if (usingRealSource || !ctx) {
      updateOverlay();
      return;
    }
    t += 1 / 60;
    const w = canvas.width;
    const h = canvas.height;

    const grad = ctx.createLinearGradient(0, 0, w, h);
    const hue = (t * 8) % 360;
    grad.addColorStop(0, `hsl(${hue}, 40%, 12%)`);
    grad.addColorStop(1, `hsl(${(hue + 60) % 360}, 35%, 6%)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Scanlines
    ctx.strokeStyle = 'rgba(160, 220, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 3) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Crosshair
    const cx = w / 2;
    const cy = h / 2;
    ctx.strokeStyle = 'rgba(159, 232, 255, 0.55)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cx - 14, cy);
    ctx.lineTo(cx - 4, cy);
    ctx.moveTo(cx + 4, cy);
    ctx.lineTo(cx + 14, cy);
    ctx.moveTo(cx, cy - 14);
    ctx.lineTo(cx, cy - 4);
    ctx.moveTo(cx, cy + 4);
    ctx.lineTo(cx, cy + 14);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 22, 0, Math.PI * 2);
    ctx.stroke();

    updateOverlay();
  }

  function updateOverlay(): void {
    const leader = state.drones.find((d) => d.id === state.activeDroneId) ?? state.drones[0];
    if (!leader) {
      overlay.textContent = 'NO SIGNAL';
      return;
    }
    const p = leader.pos;
    // Show real GPS (lat/lon/alt) instead of raw scene X/Y/Z — convert the
    // drone's scene position through the shared ENU frame to WGS84.
    const g = sceneToGps([p.x, p.y, p.z], CONFIG.geo.anchor);
    const ns = g.lat >= 0 ? 'N' : 'S';
    const ew = g.lon >= 0 ? 'E' : 'W';
    overlay.textContent =
      `DRONE #${leader.id} · ${leader.mode}  ` +
      `${Math.abs(g.lat).toFixed(5)}°${ns} ${Math.abs(g.lon).toFixed(5)}°${ew} · ${g.alt.toFixed(0)}m`;
  }

  raf = requestAnimationFrame(draw);

  return {
    setSource(src: MediaStream | string | null): void {
      if (src instanceof MediaStream) {
        video.srcObject = src;
        video.removeAttribute('src');
        usingRealSource = true;
        video.classList.remove('is-hidden');
        canvas.classList.add('is-hidden');
        void video.play().catch(() => {});
      } else if (typeof src === 'string') {
        video.srcObject = null;
        video.src = src;
        usingRealSource = true;
        video.classList.remove('is-hidden');
        canvas.classList.add('is-hidden');
        void video.play().catch(() => {});
      } else {
        video.pause();
        video.srcObject = null;
        video.removeAttribute('src');
        usingRealSource = false;
        video.classList.add('is-hidden');
        canvas.classList.remove('is-hidden');
      }
    },
    dispose(): void {
      cancelAnimationFrame(raf);
      root.remove();
    },
  };
}
