// 관제탑 (control tower) bootstrap — COMPONENTS.md §3.4-1, §4.
//
// Access:  http://<서버IP>:5173/res/static/control.html
//
// What this screen is, after the VWorld unification:
//
//   ONE SCENE       VWorld real terrain + real building footprints. There is no
//                   longer a splat-derived point cloud world here; "점" is one
//                   of three ways to draw the buildings of this same scene.
//   GPS COORDS      Routes, telemetry and drone placement are lat/lon/alt end to
//                   end. geoFrame.ts converts, and only at the render boundary.
//   ONE UPSTREAM    The tower talks to the CORE and to nothing else. It does not
//                   stream state to the situation board (the old peer.ts path is
//                   gone — see COMPONENTS.md §8) and it does not simulate the
//                   fleet: drone positions arrive as DroneTelemetry.
//
// When the core is unreachable the tower shows the operating area and says it is
// disconnected. It does NOT invent drones.

import '../shared/viewer/style.css';
import './control/control.css';
import './ui/panels.css';
import type { Gps } from '../shared/geo.ts';
import { state, subscribe } from '../shared/viewer/store.ts';
import { CONFIG } from '../shared/viewer/config.ts';
import { loadControlScene } from '../shared/viewer/sources/sceneSource.ts';
import { startWorldStream } from '../shared/viewer/sources/streamSource.ts';
import { createLoadingScreen } from '../shared/viewer/ui/loadingScreen.ts';
import { mountPaneLabel } from '../shared/viewer/ui/surface.ts';
import { showToast } from '../shared/viewer/ui/toast.ts';
import { createGeoFrame } from './geoFrame.ts';
import { createSettings } from './settings.ts';
import { createCoreLink } from './coreLink.ts';
import { TowerViewer } from './controlview/towerViewer.ts';
import { createTelemetryFleet } from './drones/telemetryFleet.ts';
import { createManualLink } from './drones/manualLink.ts';
import { createRouteModal } from './control/routeModal.ts';
import { createVideoPanel } from './control/videoPanel.ts';
import { createSettingsPanel } from './ui/settingsPanel.ts';
import { createMissionPanel } from './ui/missionPanel.ts';
import { createTelemetryPanel } from './ui/telemetryPanel.ts';

function getCanvas(id: string): HTMLCanvasElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLCanvasElement)) throw new Error(`Canvas #${id} missing`);
  return el;
}

function mount(id: string): HTMLElement | null {
  return document.getElementById(id);
}

function toolbarButton(label: string, onClick: () => void): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'control-toolbar__btn';
  btn.textContent = label;
  btn.addEventListener('click', onClick);
  return btn;
}

async function main(): Promise<void> {
  const settings = createSettings();

  const loading = createLoadingScreen('관제탑 · 실지형 장면 로딩');
  const scene = await loadControlScene((p) => loading.progress(p));
  loading.done();

  // The tower's coordinate boundary. Everything above this line is GPS.
  const frame = createGeoFrame(scene.ctx);

  const viewer = new TowerViewer(getCanvas('control-view'), {
    sceneData: scene.data,
    terrainVisual: scene.terrainVisual,
    surroundVisual: scene.surroundVisual,
    buildingVisual: scene.buildingVisual,
    display: settings.value.display,
  });

  // --- Fleet: telemetry in, rigs out. No local motion model. ---
  const fleet = createTelemetryFleet(frame);

  // --- Panels ---
  const missionPanel = mount('mission-panel-mount')
    ? createMissionPanel(mount('mission-panel-mount') as HTMLElement)
    : null;
  const telemetryPanel = mount('telemetry-panel-mount')
    ? createTelemetryPanel(mount('telemetry-panel-mount') as HTMLElement)
    : null;
  const videoMount = mount('video-panel-mount');
  const pane = document.querySelector('.pane');
  if (pane instanceof HTMLElement) mountPaneLabel(pane, '관제탑 · 실지형 상황도', 'control');

  const videoPanel = videoMount ? createVideoPanel(videoMount) : null;
  /** Aircraft the camera panel is currently pointed at. */
  let camDroneId: number | null = null;

  /** Point the camera panel at whichever aircraft is selected. Driven by the
   *  selection EVENT rather than the render loop: a click has to be answered
   *  immediately, and the loop runs on requestAnimationFrame, which stalls
   *  whenever the window is not the foreground one. */
  const syncCamera = (): void => {
    const active = fleet.drones().find((d) => d.id === state.activeDroneId);
    if (!active || active.id === camDroneId) return;
    camDroneId = active.id;
    videoPanel?.select(active.id, active.station);
  };
  subscribe((e) => {
    if (e.type === 'active-drone') syncCamera();
  });
  /** Last logged state per hop, so only changes reach the console. */
  const linkSeen = new Map<string, string>();

  // --- Core link: the ONE upstream. ---
  const core = createCoreLink({
    onState: (linkState, detail) => {
      missionPanel?.setLink(linkState, detail);
      if (linkState === 'disconnected') {
        // Telemetry stops meaning anything once the link drops; drop the rigs
        // rather than leave ghosts frozen over the map.
        fleet.clear();
        telemetryPanel?.render([]);
        // The camera panel follows the link: no core, no feed.
        videoPanel?.setFeed(null);
        videoPanel?.setTelemetry(null);
      }
    },
    onTelemetry: (t) => {
      fleet.ingest(t);
      // The panel itself decides whether this drone is the one on screen.
      videoPanel?.setTelemetry(t);
    },
    onCameraFeed: (f) => videoPanel?.setFeed(f),
    onMission: (m) => missionPanel?.setMission(m),
    onLinkStatus: (l) => {
      // The core repeats every hop's status on a timer; only transitions are
      // worth a line, or the console fills with the same two rows per second.
      const now = `${l.connected ? 'up' : 'down'} (${l.mode})`;
      if (linkSeen.get(l.hop) === now) return;
      linkSeen.set(l.hop, now);
      console.info(`[control] link ${l.hop}: ${now}`);
    },
  });
  core.start();

  // --- Route planning (GPS in, GPS out) ---
  const routeModal = createRouteModal({
    // Centered on the loaded operating area, NOT on a hardcoded anchor — the
    // planner and the 3D scene must describe the same piece of ground.
    anchor: frame.anchor,
    // A route is assigned to the FORMATION, and the centre aircraft is the one
    // that carries it: the wingmen take the same track and offset themselves,
    // and segmentation follows the centre. Selecting a wingman to look at its
    // camera must not change who the route belongs to.
    getLeaderId: () =>
      fleet.drones().find((d) => d.station === 'center')?.id ?? state.activeDroneId,
    onAssign: ({ droneId, waypoints, loop }) => {
      const sent = core.send({ kind: 'assign-route', droneId, waypoints, loop });
      showToast(
        sent
          ? `경로 전송됨 · 웨이포인트 ${waypoints.length}개${loop ? ' · 왕복' : ''}`
          : '코어에 연결되어 있지 않아 경로를 전송하지 못했습니다',
        sent ? 'info' : 'danger',
      );
    },
  });

  // Where the planner should open. The core knows where the operations centre
  // is; this browser's memory of where the operator left the map wins over it,
  // and routeModal is what decides between them.
  void fetch(new URL('/site', core.url.replace(/^ws/, 'http')).toString(), {
    signal: AbortSignal.timeout(4000),
  })
    .then((res) => (res.ok ? (res.json() as Promise<{ gps: Gps; source: string }>) : null))
    .then((site) => {
      if (!site?.gps) return;
      routeModal.suggestCenter(site.gps);
      console.info(`[control] planner centred on the core site (${site.source})`);
    })
    .catch(() => {
      // Core unreachable, or an older core with no /site: the planner keeps the
      // operating-area anchor it was built with. Not worth telling the operator.
    });

  // --- Manual control: keyboard → wire, never → local position ---
  const manual = createManualLink({
    send: (msg) => core.send(msg),
    onActiveChange: (active) => {
      document.body.classList.toggle('is-manual', active);
    },
  });

  // --- Settings panel (the three display options) ---
  const settingsPanel = createSettingsPanel({
    settings,
    onDisplayChange: (mode) => viewer.setDisplay(mode),
    aerialAvailable: scene.imageryAvailable,
    buildingSource: scene.buildingSource,
    footprints: scene.footprints,
    areaLabel: `${CONFIG.control.defaultMap} (${scene.bbox[1].toFixed(3)}, ${scene.bbox[0].toFixed(3)})`,
  });

  const toolbar = mount('control-toolbar');
  if (toolbar) {
    toolbar.append(
      toolbarButton('경로 계획 · Route', () => routeModal.open()),
      toolbarButton('설정 · Display', () => settingsPanel.toggle()),
    );
  }

  // The world streamer loads surrounding cells as the ACTIVE drone travels.
  // Display-only, and only meaningful once a drone actually exists.
  startWorldStream(scene.streamSeed.coreBbox, scene.streamSeed.ctx, viewer, () => {
    const d = state.drones.find((x) => x.id === state.activeDroneId) ?? state.drones[0];
    return d ? { x: d.pos.x, z: d.pos.z } : null;
  });

  const resize = (): void => viewer.resize();
  window.addEventListener('resize', resize);
  resize();

  // --- Main loop ---
  const MAX_DT = 1 / 20;
  /** Telemetry rows re-render at 5 Hz; 60 Hz would be churn for no gain. */
  const PANEL_INTERVAL = 0.2;
  let panelAcc = PANEL_INTERVAL;
  // Seeded from the first rAF timestamp (its epoch differs from performance.now()).
  let last = -1;

  const step = (now: number): void => {
    if (last < 0) last = now;
    const real = (now - last) / 1000;
    last = now;
    // Clamp to [0, MAX_DT] — never negative, so smoothing can't explode.
    const dt = Math.max(0, Math.min(real, MAX_DT));

    state.time += dt;
    fleet.update(dt);

    panelAcc += dt;
    if (panelAcc >= PANEL_INTERVAL) {
      panelAcc = 0;
      telemetryPanel?.render(fleet.drones());
      // Covers the first aircraft to report, which becomes active without a
      // click and therefore without an event.
      syncCamera();
    }

    viewer.update(dt);
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);

  // Debug/e2e handle.
  (window as unknown as { skylens?: unknown }).skylens = {
    role: 'control',
    state,
    scene,
    frame,
    settings,
    viewer,
    core,
    fleet,
    manual,
    routeModal,
    settingsPanel,
    videoPanel,
    CONFIG,
  };
}

void main();
