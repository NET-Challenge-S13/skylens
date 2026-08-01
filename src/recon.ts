// RECON page bootstrap — the "3D reconstruction situation board" computer (컴퓨터 B).
//
// Access URL:  http://<서버IP>:5173/recon.html?room=<방이름>
// Pair with SIM at the SAME room: /sim.html?room=<방이름>
//
// Consumes state snapshots from the SIM computer over WebRTC and computes reveal,
// detection markers, and the camera state machine locally. The clock and drone
// poses come from the wire; confirmations happen here and stay local.
//
// The scene is the SAME splat SIM shows as low-fi points — here rendered as the
// full Gaussian splat, fit with the transform derived from its own bounds.

import './style.css';
import { state } from './core/store.ts';
import { loadScene, resolveSplatUrl } from './data/sceneSource.ts';
import { buildDetections } from './data/detections.ts';
import { ReconViewer } from './viewer2/reconViewer.ts';
import { initUI } from './ui/overlay.ts';
import { createTransport } from './net/peer.ts';
import { applyState } from './net/protocol.ts';
import type { StateSnapshot } from './net/protocol.ts';
import { roomFromQuery, mountNetBadge } from './net/statusUi.ts';
import { createLoadingScreen } from './ui/loadingScreen.ts';

function getCanvas(id: string): HTMLCanvasElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLCanvasElement)) throw new Error(`Canvas #${id} missing`);
  return el;
}

async function main(): Promise<void> {
  const loading = createLoadingScreen('RECON · 3D 복원 데이터 로딩');
  const loaded = await loadScene({
    url: resolveSplatUrl(),
    onProgress: (p) => loading.progress(p),
  });
  loading.done();

  // Debug: ?render=points renders the extracted point cloud instead of the splat
  // (to check the fit/camera independently of the splat renderer).
  const renderPoints = new URLSearchParams(window.location.search).get('render') === 'points';

  const detections = buildDetections(loaded.data);
  const recon = new ReconViewer(
    getCanvas('view2'),
    loaded.data,
    detections,
    !!loaded.splat && !renderPoints,
  );
  const ui = initUI();

  // ?reveal=on|off overrides the splat reveal MASK (default from CONFIG.reveal.splatMask).
  const revealQ = new URLSearchParams(window.location.search).get('reveal');
  if (revealQ === 'on') recon.setSplatMask(true);
  else if (revealQ === 'off') recon.setSplatMask(false);

  if (renderPoints) {
    recon.revealAll();
  } else if (loaded.splat) {
    // Render the full splat with the same transform the point cloud was fit with.
    // (Re-downloads the same URL, served from browser cache — see sceneSource.ts.)
    recon.loadSplat(loaded.splat);
    mountSplatLoading(recon);
  }

  const transport = createTransport('recon', roomFromQuery());
  mountNetBadge(transport, 'recon');

  transport.onData((d) => {
    const snap = d as StateSnapshot;
    if (snap && snap.kind === 'state') applyState(snap, state);
  });

  const resize = (): void => recon.resize();
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
    recon.update(dt);
    ui.update();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  // Debug/e2e handle.
  (window as unknown as { skylens?: unknown }).skylens = {
    role: 'recon',
    state,
    scene: loaded.data,
    transport,
    splat: {
      get status() {
        return recon.splatStatus;
      },
      get progress() {
        return recon.splatProgress;
      },
    },
    get dbg() {
      return recon.debugInfo;
    },
  };
}

// A small badge showing splat render-build progress until the scene is ready.
function mountSplatLoading(recon: ReconViewer): void {
  const host = document.getElementById('overlay-recon');
  if (!host) return;
  const el = document.createElement('div');
  el.className = 'splat-loading';
  host.appendChild(el);

  let lastText = '';
  const tick = (): void => {
    const s = recon.splatStatus;
    let text = '';
    if (s === 'loading') text = `실사 3D 렌더 준비… ${Math.round(recon.splatProgress)}%`;
    else if (s === 'error') text = '실사 3D 로드 실패 — 포인트클라우드로 진행';
    if (text !== lastText) {
      lastText = text;
      el.textContent = text;
      el.classList.toggle('is-visible', text !== '');
    }
    if (s === 'loading' || s === 'idle') requestAnimationFrame(tick);
    else if (s === 'error') setTimeout(() => el.classList.remove('is-visible'), 4000);
  };
  requestAnimationFrame(tick);
}

void main();
