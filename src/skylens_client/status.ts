// STATUS page bootstrap — the "3D reconstruction situation board" computer (컴퓨터 B).
//
// Access URL:  http://<서버IP>:5173/res/static/status.html?room=<방이름>
// Pair with CONTROL at the SAME room: /res/static/control.html?room=<방이름>
//
// Consumes state snapshots from the CONTROL computer over WebRTC and computes reveal,
// detection markers, and the camera state machine locally. The clock and drone
// poses come from the wire; confirmations happen here and stay local.
//
// The scene is the SAME splat CONTROL shows as low-fi points. The point cloud +
// auto-fit transform (loadScene) is still computed locally for framing/reveal/
// detection-gating, but the actual splat RENDER and the detections now arrive
// progressively from the server (server/serverSource.ts): splat chunks via
// onSplatChunk (ingested into the splat scene one at a time), and detections
// via onDetection (placed with gpsToScene against the shared geo anchor).

import '../skylens_core/style.css';
import './ui/status-panels.css';
import { state } from '../skylens_core/store.ts';
import { CONFIG } from '../skylens_core/config.ts';
import { isDemo } from '../skylens_core/mode.ts';
import { gpsToScene } from '../skylens_core/geo.ts';
import { IDENTITY_ALIGN } from '../skylens_core/protocol.ts';
import type { DetectionRuntime } from '../skylens_core/types.ts';
import {
  loadScene,
  resolveSegmentManifest,
  resolveSplatUrl,
} from '../skylens_core/sources/sceneSource.ts';
import { StatusViewer } from './statusview/statusViewer.ts';
import { initUI } from './ui/overlay.ts';
import { createTransport } from '../skylens_core/net/peer.ts';
import { applyState } from '../skylens_core/protocol.ts';
import type { StateSnapshot } from '../skylens_core/protocol.ts';
import { roomFromQuery, mountNetBadge } from '../skylens_core/net/statusUi.ts';
import { createLoadingScreen } from '../skylens_core/ui/loadingScreen.ts';
import { createServerSource } from '../skylens_core/server/serverSource.ts';
import { mountMinimap } from './ui/minimap.ts';
import { mountServerStatus } from './ui/serverStatus.ts';

function getCanvas(id: string): HTMLCanvasElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLCanvasElement)) throw new Error(`Canvas #${id} missing`);
  return el;
}

async function main(): Promise<void> {
  const loading = createLoadingScreen('현황판 · 3D 복원 데이터 로딩');
  const loaded = await loadScene({
    url: resolveSplatUrl(),
    onProgress: (p) => loading.progress(p),
  });
  loading.done();

  // Debug: ?render=points renders the extracted point cloud instead of the splat
  // (to check the fit/camera independently of the splat renderer).
  const renderPoints = new URLSearchParams(window.location.search).get('render') === 'points';

  const status = new StatusViewer(getCanvas('status-view'), loaded.data, !!loaded.splat && !renderPoints);
  const ui = initUI();

  // ?reveal=on|off overrides the splat reveal MASK (default from CONFIG.reveal.splatMask).
  const revealQ = new URLSearchParams(window.location.search).get('reveal');
  if (revealQ === 'on') status.setSplatMask(true);
  else if (revealQ === 'off') status.setSplatMask(false);

  if (renderPoints) {
    status.revealAll();
  } else if (loaded.splat) {
    mountSplatLoading(status);
  }

  // --- Server data source: progressive splat chunks + GPS detections. ---
  const serverSource = createServerSource({
    demo: isDemo(),
    splatUrl: resolveSplatUrl(),
    manifestUrl: resolveSegmentManifest(),
  });

  if (!renderPoints && loaded.splat) {
    const localSplat = loaded.splat;
    serverSource.onSplatChunk((chunk) => {
      // An identity align means the server hasn't computed its own placement
      // yet — use the locally fit transform. Every segment is cut from the SAME
      // capture, so one transform places all of them.
      const isIdentity =
        chunk.align.position.every((v, i) => v === IDENTITY_ALIGN.position[i]) &&
        chunk.align.rotation.every((v, i) => v === IDENTITY_ALIGN.rotation[i]) &&
        chunk.align.scale.every((v, i) => v === IDENTITY_ALIGN.scale[i]) &&
        !chunk.align.anchor;
      const align = isIdentity
        ? {
            position: localSplat.position,
            rotation: localSplat.rotation,
            scale: localSplat.scale,
          }
        : chunk.align;
      // The viewer owns the ingest queue: it runs loads one at a time and drops
      // levels that a later refinement has already overtaken.
      void status.ingestSplatChunk({
        url: chunk.url,
        align,
        segment: chunk.segment,
        level: chunk.level,
      });
    });
  }

  serverSource.onDetection((d) => {
    const pos = gpsToScene(d.gps, CONFIG.geo.anchor);
    const det: DetectionRuntime = {
      id: d.id,
      kind: d.category,
      pos,
      label: d.label,
      confidence: d.confidence,
      revealed: false,
      confirmed: false,
      revealedAt: null,
    };
    status.addDetection(det);
  });

  mountServerStatus(serverSource);
  serverSource.start();

  mountMinimap(loaded.data.bounds);

  const transport = createTransport('status', roomFromQuery());
  mountNetBadge(transport, 'status');

  transport.onData((d) => {
    const snap = d as StateSnapshot;
    if (snap && snap.kind === 'state') applyState(snap, state);
  });

  const resize = (): void => status.resize();
  window.addEventListener('resize', resize);
  resize();

  // --- Render loop ---
  // state.time advances from incoming snapshots; dt here only drives frame-rate-
  // independent camera damping/tweening and marker pulsing.
  const MAX_DT = 1 / 20;
  let last = -1; // seed from first rAF timestamp (different epoch than performance.now())

  const frame = (now: number): void => {
    if (last < 0) last = now;
    const dt = Math.max(0, Math.min((now - last) / 1000, MAX_DT));
    last = now;
    status.update(dt);
    ui.update();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  // Debug/e2e handle.
  (window as unknown as { skylens?: unknown }).skylens = {
    role: 'status',
    state,
    scene: loaded.data,
    transport,
    splat: {
      get status() {
        return status.splatStatus;
      },
      get progress() {
        return status.splatProgress;
      },
      get chunks() {
        return status.splatChunks;
      },
      get replaced() {
        return status.splatReplaced;
      },
      get segmentLevels() {
        return status.splatSegmentLevels;
      },
    },
    get server() {
      return serverSource.status;
    },
    get dbg() {
      return status.debugInfo;
    },
  };
}

// A small badge showing splat render-build progress until the scene is ready.
function mountSplatLoading(status: StatusViewer): void {
  const host = document.getElementById('overlay-status');
  if (!host) return;
  const el = document.createElement('div');
  el.className = 'splat-loading';
  host.appendChild(el);

  let lastText = '';
  const tick = (): void => {
    const s = status.splatStatus;
    let text = '';
    if (s === 'idle') text = '서버 · 실사 3D 스트리밍 대기 중';
    else if (s === 'loading') text = `실사 3D 렌더 준비… ${Math.round(status.splatProgress)}%`;
    else if (s === 'error') text = '실사 3D 로드 실패 — 포인트클라우드로 진행';
    if (text !== lastText) {
      lastText = text;
      el.textContent = text;
      el.classList.toggle('is-visible', text !== '');
    }
    if (s === 'loading' || s === 'idle') requestAnimationFrame(tick);
    else if (s === 'error') setTimeout(() => el.classList.remove('is-visible'), 4000);
    else if (s === 'ready') setTimeout(() => el.classList.remove('is-visible'), 1500);
  };
  requestAnimationFrame(tick);
}

void main();
