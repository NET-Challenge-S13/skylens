// Browser entry point for the drone app.
//
// This is the whole application. The Tauri shell (src-tauri/) loads this exact
// page in a webview and adds nothing to it, which is what keeps `cargo build`
// off the demo launcher's critical path (COMPONENTS.md §5).
//
//   http://localhost:5173/src/skylens_drone/index.html?demo&drone=1
//   http://localhost:5173/src/skylens_drone/index.html?mode=webrtc&gateway=ws://10.0.0.4:8081
//
// Query flags map onto the same SKYLENS_* keys the Node runner reads from the
// environment (core/config.ts), so a browser tab and the headless runner are
// configured identically.

import { DEMO_ROUTE, envFromQuery, resolveConfig } from './core/config.ts';
import { DemoCapture, type CaptureSource } from './core/capture.ts';
import { DEMO_CLIPS, DEMO_FOOTAGE_DIR } from './core/demoAssets.ts';
import { DroneApp, type DroneSnapshot } from './core/drone.ts';
import { openLiveCamera } from './liveCamera.ts';
import { OperatorPanel } from './ui/panel.ts';
import { LogView } from './ui/log.ts';
import { StickPad } from './ui/sticks.ts';
import { CameraPreview } from './ui/preview.ts';

const LIVE_DEFAULTS = { width: 1920, height: 1080, framerate: 30, bitrate: 12_000_000 };

function mount(root: HTMLElement): {
  panel: HTMLElement;
  side: HTMLElement;
} {
  const panel = document.createElement('div');
  panel.className = 'drone__panel';
  const side = document.createElement('div');
  side.className = 'drone__side';
  root.append(panel, side);
  return { panel, side };
}

async function boot(): Promise<void> {
  const root = document.getElementById('drone-root');
  if (!root) throw new Error('#drone-root missing');

  const env = envFromQuery(window.location.search);
  const config = resolveConfig(env);
  const { panel: panelHost, side } = mount(root);

  const logHost = document.createElement('div');
  const previewHost = document.createElement('div');
  const stickHost = document.createElement('div');
  side.append(previewHost, stickHost, logHost);

  const log = new LogView(logHost);
  const preview = new CameraPreview(previewHost);
  const panel = new OperatorPanel(panelHost);

  log.push(
    `id=${config.droneId} mode=${config.mode} gateway=${config.gatewayUrl} ` +
      `demo=${config.demo} slices/leg=${config.slicesPerLeg} telemetry=${config.telemetryHz}Hz`,
  );

  // Capture source. Demo mode never touches the camera; live mode falls back to
  // demo footage (and says so) when the browser has no HEVC encoder, because
  // shipping H.264 under `codec: 'h265'` outside demo mode would be a lie.
  let capture: CaptureSource = new DemoCapture();
  if (!config.demo) {
    try {
      capture = await openLiveCamera({
        ...LIVE_DEFAULTS,
        uploadUrl: env.SKYLENS_DRONE_UPLOAD_URL ?? null,
        droneId: config.droneId,
        onLog: (line) => log.push(line),
      });
      log.push('camera open — slices are really H.265 encoded by WebCodecs');
    } catch (err) {
      log.push(`live capture unavailable (${String(err)}) — falling back to demo footage`);
    }
  }

  // The H.265 footage is generated, not committed. Say so before the first
  // segment points at a 404 rather than after.
  if (capture.kind === 'demo') {
    void fetch(DEMO_CLIPS[0].uri, { method: 'HEAD' })
      .then((res) => {
        if (!res.ok) throw new Error(String(res.status));
      })
      .catch(() => {
        log.push(
          `demo footage missing at ${DEMO_FOOTAGE_DIR} — run ` +
            'npx tsx src/skylens_drone/tools/transcodeDemoFootage.ts',
        );
      });
  }

  let lastSegmentSeq = -1;
  const app = new DroneApp({
    config,
    capture,
    onLog: (line) => log.push(line),
    onUpdate: (snap: DroneSnapshot) => {
      panel.render(snap);
      const head = snap.segments[0];
      if (head && head.seq !== lastSegmentSeq) {
        lastSegmentSeq = head.seq;
        preview.show(
          head.uri,
          `slice #${head.seq} · ${head.uri} · ${(head.bytes / 1e6).toFixed(1)} MB · ${head.poses.length} poses`,
        );
      }
    },
  });

  const sticks = new StickPad(stickHost, config.droneId, (msg) => app.handleControl(msg));
  window.addEventListener('beforeunload', () => {
    sticks.dispose();
    app.stop();
  });

  await app.start();

  if (config.autoRoute) {
    app.assignRoute({ kind: 'assign-route', droneId: config.droneId, waypoints: DEMO_ROUTE, loop: true });
    log.push('autoroute: built-in demo route assigned locally (no core involved)');
  }

  // Handle for the Tauri shell and for E2E: the page exposes the live app.
  (window as unknown as { skylensDrone: DroneApp }).skylensDrone = app;
}

void boot().catch((err) => {
  const root = document.getElementById('drone-root');
  if (root) root.textContent = `드론 앱 시작 실패: ${String(err)}`;
  console.error('[drone] boot failed', err);
});
