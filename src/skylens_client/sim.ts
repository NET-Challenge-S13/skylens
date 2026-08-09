// SIM page bootstrap — the "drone simulation" computer (컴퓨터 A).
//
// Access URL:  http://<서버IP>:5173/sim.html?room=<방이름>
// Pair with RECON at the SAME room: /recon.html?room=<방이름>
//
// Owns the simulation: runs the drone controller + the low-fi viewer, advances
// the shared clock, and streams state snapshots to the RECON computer over the
// WebRTC DataChannel. Keyboard control (fly / switch drone / pause) lives here.
//
// The scene is the SAME splat RECON renders — here it's shown as a low-fi point
// cloud derived from the splat's own points (PROJECT.md §1: one source, two views).

import './style.css';
import './sim/sim.css';
import { state } from '../skylens_core/store.ts';
import { CONFIG } from '../skylens_core/config.ts';
import { isDemo } from '../skylens_core/mode.ts';
import { loadScene, resolveSplatUrl } from './data/sceneSource.ts';
import { buildDronePaths } from './data/paths.ts';
import { buildIdlePaths, buildRouteFromGps } from './data/routes.ts';
import { createDroneController } from './drones/pathFollower.ts';
import { LowfiViewer } from './viewer1/lowfiViewer.ts';
import { createTransport } from './net/peer.ts';
import { encodeState } from '../skylens_core/protocol.ts';
import { roomFromQuery, mountNetBadge } from './net/statusUi.ts';
import { createLoadingScreen } from './ui/loadingScreen.ts';
import { createServerSource } from './server/serverSource.ts';
import { createRouteModal } from './sim/routeModal.ts';
import { createVideoPanel } from './sim/videoPanel.ts';

function getCanvas(id: string): HTMLCanvasElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLCanvasElement)) throw new Error(`Canvas #${id} missing`);
  return el;
}

async function main(): Promise<void> {
  const loading = createLoadingScreen('SIM · 장면 로딩');
  const loaded = await loadScene({
    url: resolveSplatUrl(),
    onProgress: (p) => loading.progress(p),
  });
  loading.done();

  const demo = isDemo();
  // Demo: auto full-scene sweep, leader flies automatically. Real (default):
  // drones idle-hover until the operator assigns a route via the modal.
  const paths = demo ? buildDronePaths(loaded.data.bounds) : buildIdlePaths();
  const drones = createDroneController(paths);
  const lowfi = new LowfiViewer(getCanvas('view1'), loaded.data);

  const transport = createTransport('sim', roomFromQuery());
  mountNetBadge(transport, 'sim');

  // --- Server connection (route commands out; status shown in the toolbar) ---
  const server = createServerSource({ demo, splatUrl: resolveSplatUrl() });
  server.start();

  // --- Route planning modal (control-tower route assignment) ---
  const modal = createRouteModal({
    anchor: CONFIG.geo.anchor,
    getLeaderId: () => state.activeDroneId,
    onAssign: ({ droneId, waypoints }) => {
      const route = buildRouteFromGps(waypoints, undefined, droneId);
      drones.setLeaderRoute(route);
      server.assignRoute({ kind: 'assign-route', droneId, waypoints });
    },
  });

  const toolbar = document.getElementById('sim-toolbar');
  if (toolbar) {
    const routeBtn = document.createElement('button');
    routeBtn.type = 'button';
    routeBtn.className = 'sim-toolbar__btn';
    routeBtn.textContent = '경로 계획 · Route';
    routeBtn.addEventListener('click', () => modal.open());
    toolbar.appendChild(routeBtn);
  }

  // --- Main drone camera placeholder (bottom-right) ---
  const videoMount = document.getElementById('video-panel-mount');
  const videoPanel = videoMount ? createVideoPanel(videoMount) : null;

  const resize = (): void => lowfi.resize();
  window.addEventListener('resize', resize);
  resize();

  // --- Snapshot streaming (throttled) ---
  const SEND_INTERVAL = 1 / 30; // 30 Hz
  let sendAcc = 0;
  let sentVisited = 0;

  const stream = (): void => {
    if (transport.status !== 'connected') return;
    const snap = encodeState(state, sentVisited);
    transport.send(snap);
    sentVisited = snap.visitedTotal;
  };

  // --- Main loop ---
  const MAX_DT = 1 / 20;
  // Seed from the first rAF timestamp (its epoch differs from performance.now()).
  let last = -1;

  const frame = (now: number): void => {
    if (last < 0) last = now;
    const real = (now - last) / 1000;
    last = now;
    // Clamp to [0, MAX_DT]: never negative (guards damping from exploding).
    const dt = Math.max(0, Math.min(real, MAX_DT)) * CONFIG.sim.speed;

    if (state.running) {
      state.time += dt;
      drones.update(dt);
      sendAcc += real;
      if (sendAcc >= SEND_INTERVAL) {
        sendAcc = 0;
        stream();
      }
    }

    lowfi.update(dt);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  // --- Pause toggle (space) ---
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      state.running = !state.running;
    }
  });

  // Debug/e2e handle.
  (window as unknown as { skylens?: unknown }).skylens = {
    role: 'sim',
    state,
    scene: loaded.data,
    CONFIG,
    transport,
    server,
    videoPanel,
    routeModal: modal,
  };
}

void main();
