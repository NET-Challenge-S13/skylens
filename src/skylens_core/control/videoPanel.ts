// Bottom-right camera panel. It shows ONE aircraft's feed — whichever the
// operator has selected in the fleet list — and is titled after that aircraft's
// station: LEFT CAM / CENTER CAM / RIGHT CAM. The core turns each arriving video
// slice into a `camera-feed` message; the panel keeps the latest per drone and
// plays the selected one. Until a feed arrives — no drone, or one still in
// transit — it draws a synthetic standby pattern and says so.
//
// Why the feed carries two addresses: the uplink artifact is HEVC because that
// is what the radio carries, and a browser video element generally cannot
// decode HEVC in MP4. `previewUri` is the same footage in a rendition it can.

import type { CameraFeed, DroneStation, DroneTelemetry } from '../../shared/protocol.ts';
import { stationCamLabel, stationLabel } from '../ui/stationLabel.ts';

export interface VideoPanel {
  /** File a feed under its drone. The panel shows it only if that drone is the
   *  one currently selected. */
  setFeed(feed: CameraFeed | null): void;
  /** Switch the panel to a drone. Its title and footage follow. */
  select(droneId: number, station: DroneStation): void;
  /** Telemetry for the readout strip; ignored for drones not on screen. */
  setTelemetry(t: DroneTelemetry | null): void;
  /** Escape hatch for a live MediaStream or an arbitrary URL. */
  setSource(src: MediaStream | string | null): void;
  dispose(): void;
}

export function createVideoPanel(container: HTMLElement): VideoPanel {
  const root = document.createElement('div');
  root.className = 'sl-surface sl-surface--frame video-panel';

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
  label.textContent = stationCamLabel('center');

  const overlay = document.createElement('div');
  overlay.className = 'video-panel__overlay';

  root.append(canvas, video, label, overlay);
  container.appendChild(root);

  const ctx = canvas.getContext('2d');
  let usingRealSource = false;
  let raf = 0;
  let t = 0;
  /** Latest feed per drone: switching cameras must not wait for the next slice. */
  const feeds = new Map<number, CameraFeed>();
  let selected: number | null = null;
  let selectedStation: DroneStation = 'center';
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
    // The aircraft is named by its station, and the position shown must be the
    // position OF THAT AIRCRAFT — a readout carrying another drone's fix under
    // this one's name is worse than no readout.
    const who = stationLabel(selectedStation);
    if (!telemetry || (selected !== null && telemetry.droneId !== selected)) {
      overlay.textContent = feed ? `${who} · 수신 대기` : `${who} · NO SIGNAL`;
      return;
    }
    const g = telemetry.gps;
    const ns = g.lat >= 0 ? 'N' : 'S';
    const ew = g.lon >= 0 ? 'E' : 'W';
    const where =
      `${Math.abs(g.lat).toFixed(5)}°${ns} ${Math.abs(g.lon).toFixed(5)}°${ew} · ${g.alt.toFixed(0)}m`;
    overlay.textContent = feed ? `${who} · ${where}` : `${who} · ${where} · 영상 대기`;
  }

  raf = requestAnimationFrame(draw);

  /** Point the panel at a drone: retitle, and play its feed if one has landed. */
  function select(droneId: number, station: DroneStation): void {
    selected = droneId;
    selectedStation = station;
    label.textContent = stationCamLabel(station);
    telemetry = null;
    const known = feeds.get(droneId);
    if (known) show(known);
    else standby();
  }

  function show(next: CameraFeed): void {
    feed = next;
    const url = next.previewUri ?? next.uri;
    // Re-issuing the same URL would restart the clip on every slice; the
    // footage is already looping.
    if (video.getAttribute('src') !== url) showVideo(url);
  }

  function standby(): void {
    feed = null;
    video.pause();
    video.removeAttribute('src');
    usingRealSource = false;
    playbackError = '';
    video.classList.add('is-hidden');
    canvas.classList.remove('is-hidden');
  }

  function showVideo(url: string): void {
    video.srcObject = null;
    video.src = url;
    video.loop = true;
    usingRealSource = true;
    playbackError = '';
    video.classList.remove('is-hidden');
    canvas.classList.add('is-hidden');
    void video.play().catch((err: unknown) => {
      // Switching cameras replaces the source mid-load, which rejects the
      // in-flight play() with AbortError. That is the switch working, not a
      // failure — the load that replaced it will start on its own. Only a real
      // refusal (autoplay policy, undecodable media) drops back to standby.
      const name = err instanceof Error ? err.name : '';
      if (name === 'AbortError') return;
      fail(name === 'NotAllowedError' ? '영상 재생 불가 — 자동재생 차단' : '영상 재생 불가');
      console.warn('[video-panel] playback failed', err);
    });
  }

  function fail(reason: string): void {
    playbackError = reason;
    usingRealSource = false;
    video.classList.add('is-hidden');
    canvas.classList.remove('is-hidden');
  }

  video.addEventListener('error', () => {
    // MEDIA_ERR_SRC_NOT_SUPPORTED is the codec case; the others are transport.
    const code = video.error?.code ?? 0;
    fail(code === 4 ? '영상 재생 불가 — 코덱 미지원' : '영상 재생 불가 — 소스 오류');
  });

  // A source that recovers should clear the banner rather than leave the panel
  // claiming a failure that is no longer true.
  video.addEventListener('playing', () => {
    playbackError = '';
    usingRealSource = true;
    video.classList.remove('is-hidden');
    canvas.classList.add('is-hidden');
  });

  return {
    setFeed(next: CameraFeed | null): void {
      if (!next) {
        // The whole link dropped: forget every feed, not just the visible one.
        feeds.clear();
        standby();
        return;
      }
      feeds.set(next.droneId, next);
      // A feed for a drone the operator is not watching is filed, not shown.
      if (selected === null) select(next.droneId, next.station);
      else if (next.droneId === selected) show(next);
    },

    select(droneId: number, station: DroneStation): void {
      select(droneId, station);
    },

    setTelemetry(t: DroneTelemetry | null): void {
      if (t !== null && selected !== null && t.droneId !== selected) return;
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
