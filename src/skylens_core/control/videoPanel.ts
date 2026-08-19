// Bottom-right MAIN CAM. Plays what the main drone is actually uplinking: the
// core turns each arriving video slice into a `camera-feed` message and this
// panel plays that footage. Until one arrives — no drone, or a drone still in
// transit — it draws a synthetic standby pattern and says so.
//
// Why the feed carries two addresses: the uplink artifact is HEVC because that
// is what the radio carries, and a browser video element generally cannot
// decode HEVC in MP4. `previewUri` is the same footage in a rendition it can.

import type { CameraFeed, DroneTelemetry } from '../../shared/protocol.ts';

export interface VideoPanel {
  /** Show the footage the drone is transmitting; null returns to standby. */
  setFeed(feed: CameraFeed | null): void;
  /** Telemetry for the readout strip. */
  setTelemetry(t: DroneTelemetry | null): void;
  /** Escape hatch for a live MediaStream or an arbitrary URL. */
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
  let feed: CameraFeed | null = null;
  let telemetry: DroneTelemetry | null = null;
  /** Set when the browser refuses the footage, so the panel can say WHY. */
  let playbackError = '';

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
    if (playbackError) {
      overlay.textContent = playbackError;
      return;
    }
    if (!telemetry) {
      // Genuinely nothing on the link — say so rather than implying a feed.
      overlay.textContent = feed ? '수신 대기' : 'NO SIGNAL';
      return;
    }
    const g = telemetry.gps;
    const ns = g.lat >= 0 ? 'N' : 'S';
    const ew = g.lon >= 0 ? 'E' : 'W';
    const where =
      `${Math.abs(g.lat).toFixed(5)}°${ns} ${Math.abs(g.lon).toFixed(5)}°${ew} · ${g.alt.toFixed(0)}m`;
    overlay.textContent = feed
      ? `DRONE #${telemetry.droneId} · ${where}`
      : `DRONE #${telemetry.droneId} · ${where} · 영상 대기`;
  }

  raf = requestAnimationFrame(draw);

  function showVideo(url: string): void {
    video.srcObject = null;
    video.src = url;
    video.loop = true;
    usingRealSource = true;
    playbackError = '';
    video.classList.remove('is-hidden');
    canvas.classList.add('is-hidden');
    void video.play().catch((err: unknown) => {
      // Autoplay refusal or an undecodable codec: fall back to the standby
      // pattern and name the reason instead of showing a black rectangle.
      playbackError = '영상 재생 불가 — 코덱 미지원';
      usingRealSource = false;
      video.classList.add('is-hidden');
      canvas.classList.remove('is-hidden');
      console.warn('[video-panel] playback failed', err);
    });
  }

  video.addEventListener('error', () => {
    playbackError = '영상 재생 불가 — 소스 오류';
    usingRealSource = false;
    video.classList.add('is-hidden');
    canvas.classList.remove('is-hidden');
  });

  return {
    setFeed(next: CameraFeed | null): void {
      feed = next;
      if (!next) {
        video.pause();
        video.removeAttribute('src');
        usingRealSource = false;
        video.classList.add('is-hidden');
        canvas.classList.remove('is-hidden');
        return;
      }
      const url = next.previewUri ?? next.uri;
      // Re-issuing the same URL would restart the clip on every slice; the
      // footage is already looping.
      if (video.getAttribute('src') !== url) showVideo(url);
    },

    setTelemetry(t: DroneTelemetry | null): void {
      telemetry = t;
    },

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
