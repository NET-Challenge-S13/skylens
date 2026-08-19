// STATUS page bootstrap — the 3D reconstruction situation board (COMPONENTS.md §3.6).
//
// Access:  http://<현황판 서버>:8090/res/static/status.html
//          (dev without the relay: http://<IP>:5173/res/static/status.html — the
//           board then reaches the relay on :8090 of the same host, `?relay=` overrides)
//
// WHERE THE DATA COMES FROM. One socket, to this component's own server, which
// relays what the core pushed:
//
//   드론 → 게이트웨이 → 프록시 → 코어 → skylens_client(8090) → 현황판
//
// This page used to take StateSnapshots peer-to-peer from the CONTROL TOWER and
// drive drone poses and reveal from that simulation. COMPONENTS.md §2 has no
// such edge and §8 retires it outright: the two screens are siblings, both fed
// by the pipeline, never by each other. So there is no WebRTC transport here,
// no `applyState`, and no local clock authority — every ViewerMessage kind maps
// to exactly one thing on screen:
//
//   telemetry       → drone poses (minimap + camera follow)
//   splat-chunk     → geometry, and therefore visibility
//   detection       → markers, gated on their segment's arrival
//   mission-status  → the operator line
//   server-status   → the delay-pattern ladder
//   link-status     → hop health
//
// The locally loaded point cloud is NOT the reconstruction: it is a scaffold for
// framing and minimap bounds while the board waits, and it is dropped the moment
// the first chunk lands.

import '../shared/viewer/style.css';
import './ui/status-panels.css';
import * as THREE from 'three';
import { state } from '../shared/viewer/store.ts';
import { CONFIG } from '../shared/viewer/config.ts';
import { gpsToScene } from '../shared/geo.ts';
import { IDENTITY_ALIGN } from '../shared/protocol.ts';
import type { DroneTelemetry } from '../shared/protocol.ts';
import type { DetectionRuntime } from '../shared/viewer/types.ts';
import { loadScene, resolveSplatUrl } from '../shared/viewer/sources/sceneSource.ts';
import { StatusViewer } from './statusview/statusViewer.ts';
import { initUI } from './ui/overlay.ts';
import { createLoadingScreen } from '../shared/viewer/ui/loadingScreen.ts';
import { mountPaneLabel } from '../shared/viewer/ui/surface.ts';
import { createRelayClient } from './sources/relayClient.ts';
import type { RelayClient } from './sources/relayClient.ts';
import { mountMinimap } from './ui/minimap.ts';
import { mountServerStatus } from './ui/serverStatus.ts';
import { mountRelayBadge } from './ui/relayBadge.ts';

function getCanvas(id: string): HTMLCanvasElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLCanvasElement)) throw new Error(`Canvas #${id} missing`);
  return el;
}

async function main(): Promise<void> {
  const loading = createLoadingScreen('현황판 · 3D 복원 데이터 로딩');
  const loaded = await loadScene({
    url: resolveSplatUrl(),
    onProgress: (p: number) => loading.progress(p),
  });
  loading.done();

  // Debug: ?render=points renders the extracted point cloud instead of the splat
  // (to check the fit/camera independently of the splat renderer).
  const renderPoints = new URLSearchParams(window.location.search).get('render') === 'points';

  const status = new StatusViewer(
    getCanvas('status-view'),
    loaded.data,
    !!loaded.splat && !renderPoints,
  );
  const ui = initUI();

  // ?reveal=on|off overrides the per-segment FADE (default CONFIG.reveal.splatMask).
  // It never changes WHAT is visible — arrival decides that — only whether newly
  // arrived segments ease in or pop.
  const revealQ = new URLSearchParams(window.location.search).get('reveal');
  if (revealQ === 'on') status.setSplatMask(true);
  else if (revealQ === 'off') status.setSplatMask(false);

  if (renderPoints) status.revealAll();

  // --- The one data source ------------------------------------------------
  const relay = createRelayClient();

  if (!renderPoints && loaded.splat) {
    const localSplat = loaded.splat;
    relay.onSplatChunk((chunk) => {
      // An identity align means the core has not computed its own placement yet
      // — fall back to the locally fit transform. Every segment is cut from the
      // SAME capture, so one transform places all of them.
      const isIdentity =
        !chunk.align.anchor &&
        chunk.align.position.every((v, i) => v === IDENTITY_ALIGN.position[i]) &&
        chunk.align.rotation.every((v, i) => v === IDENTITY_ALIGN.rotation[i]) &&
        chunk.align.scale.every((v, i) => v === IDENTITY_ALIGN.scale[i]);
      let align: {
        position: [number, number, number];
        rotation: [number, number, number, number];
        scale: [number, number, number];
      };
      if (isIdentity) {
        align = {
          position: localSplat.position,
          rotation: localSplat.rotation,
          scale: localSplat.scale,
        };
      } else {
        // A GPS anchor places the chunk in the shared ENU frame; the explicit
        // transform is applied on top of it.
        const base = chunk.align.anchor
          ? gpsToScene(chunk.align.anchor, CONFIG.geo.anchor)
          : ([0, 0, 0] as [number, number, number]);
        align = {
          position: [
            base[0] + chunk.align.position[0],
            base[1] + chunk.align.position[1],
            base[2] + chunk.align.position[2],
          ],
          rotation: chunk.align.rotation,
          scale: chunk.align.scale,
        };
      }
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

  // The plan, so the board has a fixed reference to read everything else
  // against. The core replays it to any viewer that joins (index.ts onJoin), so
  // a board opened mid-mission gets it too.
  relay.onRoute((r) => {
    state.route =
      r.waypoints.length >= 2 ? r.waypoints.map((wp) => gpsToScene(wp, CONFIG.geo.anchor)) : null;
  });

  relay.onDetection((d) => {
    const det: DetectionRuntime = {
      id: d.id,
      kind: d.category,
      pos: gpsToScene(d.gps, CONFIG.geo.anchor),
      label: d.label,
      confidence: d.confidence,
      revealed: false,
      confirmed: false,
      revealedAt: null,
    };
    // The marker stays hidden until d.segment's geometry has arrived.
    status.addDetection(det, d.segment);
  });

  // Telemetry drives the drone rendering, minimap, and camera follow. The board
  // holds no drone simulation: a drone exists here because the pipeline said so.
  relay.onTelemetry((t: DroneTelemetry) => {
    const pos = gpsToScene(t.gps, CONFIG.geo.anchor);
    let drone = state.drones.find((d) => d.id === t.droneId);
    if (!drone) {
      drone = {
        id: t.droneId,
        zone: `드론 ${t.droneId}`,
        pos: new THREE.Vector3(),
        quat: new THREE.Quaternion(),
        forward: new THREE.Vector3(0, 0, 1),
        mode: 'AUTO',
        pathTime: 0,
      };
      state.drones.push(drone);
      if (state.drones.length === 1) state.activeDroneId = t.droneId;
    }
    drone.pos.set(pos[0], pos[1], pos[2]);
    // headingDeg is a compass bearing (0 = North); scene north is -Z. The slight
    // downward tilt is what makes the chase camera look at the ground the drone
    // is capturing rather than the horizon.
    const rad = (t.headingDeg * Math.PI) / 180;
    drone.forward.set(Math.sin(rad), -0.35, -Math.cos(rad)).normalize();
    drone.quat.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -rad);
  });

  const pane = document.querySelector('.pane');
  if (pane instanceof HTMLElement) mountPaneLabel(pane, '현황판 · 실시간 3D 복원', 'status');

  mountServerStatus(relay);
  mountRelayBadge(relay);
  // The minimap shows the reconstruction too, so the operator can see how far
  // it has got along the track rather than guessing from the 3D view.
  const minimap = mountMinimap(loaded.data.bounds, () =>
    status.loadedChunks().map((c) => ({ segment: c.segment, position: c.position })),
  );
  mountWaitingBanner(status, relay);
  relay.start();

  const resize = (): void => status.resize();
  window.addEventListener('resize', resize);
  resize();

  // --- Render loop ---
  // state.time is now just the board's own wall clock (seconds since load): it
  // drives marker pulsing and the camera tweens. Nothing on screen depends on it
  // agreeing with another machine — mission time comes from mission-status.
  const MAX_DT = 1 / 20;
  let last = -1;

  const frame = (now: number): void => {
    if (last < 0) last = now;
    const dt = Math.max(0, Math.min((now - last) / 1000, MAX_DT));
    last = now;
    state.time += dt;
    status.update(dt);
    ui.update();
    minimap.update();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  // Debug/e2e handle.
  (window as unknown as { skylens?: unknown }).skylens = {
    role: 'status',
    state,
    scene: loaded.data,
    relayUrl: relay.url,
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
      get hasGeometry() {
        return status.hasGeometry;
      },
      loadedChunks() {
        return status.loadedChunks();
      },
    },
    get server() {
      return relay.status;
    },
    get markers() {
      return status.markerStates;
    },
    get dbg() {
      return status.debugInfo;
    },
  };
}

/**
 * The waiting state. While no geometry has arrived the board says why, in the
 * operator's terms, and keeps saying it — a board that goes quiet is
 * indistinguishable from a board that has crashed. Once geometry is flowing it
 * only reports render progress, then gets out of the way.
 */
function mountWaitingBanner(status: StatusViewer, relay: RelayClient): void {
  const host = document.getElementById('overlay-status');
  if (!host) return;
  const el = document.createElement('div');
  el.className = 'splat-loading';
  el.id = 'board-waiting';
  host.appendChild(el);

  let lastText = '';
  const tick = (): void => {
    const feed = relay.status;
    const splat = status.splatStatus;
    const feedUp = feed.relay === 'online' && feed.upstream === 'online';
    let text = '';
    if (!feedUp) {
      // The feed is broken. Say so whether or not geometry is on screen —
      // otherwise a board holding the last delivered segments looks identical to
      // a live one, which is the exact failure this banner exists to prevent.
      text = status.hasGeometry
        ? `${feed.detail} — 마지막 수신 상태 표시 중`
        : `${feed.detail} — 복원 데이터 없음`;
    } else if (!status.hasGeometry) {
      text = '코어 연결됨 · 첫 구간 복원 대기 중';
    } else if (splat === 'loading') {
      text = `구간 수신 · 렌더 준비 ${Math.round(status.splatProgress)}%`;
    } else if (splat === 'error') {
      text = '구간 로드 실패 — 다음 수준 대기';
    }
    if (text !== lastText) {
      lastText = text;
      el.textContent = text;
      el.classList.toggle('is-visible', text !== '');
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

void main();
