// Viewer 2 — the "real" 3D reconstruction situation board (PROJECT.md §5, §8).
//
// Everything it shows arrives from the pipeline (skylens_client/sources/
// relayClient.ts). It places the detection markers the core reports, follows
// the drone by its telemetry, and drives the camera through the
// SYNCED/FOCUSING/LOCKED/RETURNING state machine (§8.3).
//
// GEOMETRY. The board renders ONE static splat scene — the final-quality
// export of the capture — loaded when the stream's first chunk arrives and
// placed by that chunk's alignment. The delay-pattern stream then only drives
// VISIBILITY: each arriving segment fades its slab of the scene in
// (splatReveal.ts), refinement levels firm it up. The stream's own PLYs are
// cuts of the same file, so nothing shown is more than what has arrived.
//
// Visibility has exactly one rule: a segment is visible once its chunk has
// landed (COMPONENTS.md §8). Until the first chunk lands, the locally-loaded
// point cloud is drawn as a dim scaffold so the board reads as WAITING rather
// than broken; it is removed the moment real geometry arrives.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SceneData } from '../../shared/viewer/sources/sceneData';
import type { DetectionRuntime } from '../../shared/viewer/types';
import { state, emit } from '../../shared/viewer/store';
import { CONFIG } from '../../shared/viewer/config';
import type { ChunkPlacement, SplatChunkInput } from './splatScene.ts';
import { SplatReveal } from './splatReveal.ts';
import { CameraSync } from './cameraSync.ts';
import { SplatScene } from './splatScene.ts';
import type { SplatStatus } from './splatScene.ts';

const POINT_VERT = /* glsl */ `
  attribute float aReveal;
  varying vec3 vColor;
  varying float vReveal;
  void main() {
    vColor = color;
    vReveal = aReveal;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    // Freshly revealed points pop slightly larger, then settle.
    float pop = 1.0 + 0.6 * (1.0 - vReveal) * step(0.001, vReveal);
    gl_PointSize = (2.2 * pop) * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const POINT_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vReveal;
  void main() {
    if (vReveal <= 0.001) discard;
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    // Brief emissive pop while fading in, settling to base color at full reveal.
    vec3 pop = mix(vColor * 1.8 + 0.2, vColor, smoothstep(0.0, 1.0, vReveal));
    gl_FragColor = vec4(pop, vReveal);
  }
`;

const STATUS_UP = new THREE.Vector3(0, 1, 0);

/** See StatusViewer.corridorPath. Median x/z per 6 m along-axis bin. */
function buildCorridorPath(samples: Array<[number, number, number]>): {
  dir: THREE.Vector2;
  mean: THREE.Vector2;
  u0: number;
  du: number;
  pts: Array<[number, number] | null>;
} | null {
  if (samples.length < 500) return null;
  let mx = 0;
  let mz = 0;
  for (const p of samples) {
    mx += p[0];
    mz += p[2];
  }
  mx /= samples.length;
  mz /= samples.length;
  let sxx = 0;
  let sxz = 0;
  let szz = 0;
  for (const p of samples) {
    const dx = p[0] - mx;
    const dz = p[2] - mz;
    sxx += dx * dx;
    sxz += dx * dz;
    szz += dz * dz;
  }
  const theta = 0.5 * Math.atan2(2 * sxz, sxx - szz);
  const dir = new THREE.Vector2(Math.cos(theta), Math.sin(theta));

  const us = samples.map((p) => (p[0] - mx) * dir.x + (p[2] - mz) * dir.y).sort((a, b) => a - b);
  const u0 = us[Math.floor(us.length * 0.01)];
  const u1 = us[Math.floor(us.length * 0.99)];
  const du = 6;
  const n = Math.max(2, Math.ceil((u1 - u0) / du) + 1);
  const binsX: number[][] = Array.from({ length: n }, () => []);
  const binsZ: number[][] = Array.from({ length: n }, () => []);
  for (const p of samples) {
    const u = (p[0] - mx) * dir.x + (p[2] - mz) * dir.y;
    const i = Math.round((u - u0) / du);
    if (i >= 0 && i < n) {
      binsX[i].push(p[0]);
      binsZ[i].push(p[2]);
    }
  }
  const median = (a: number[]): number => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
  const raw = binsX.map((xs, i): [number, number] | null =>
    xs.length >= 40 ? [median(xs), median(binsZ[i])] : null,
  );
  if (raw.filter(Boolean).length < 2) return null;
  // Moving average over ±2 bins: per-bin medians zigzag with whatever stands
  // in each slice (side rooms, plants), and a camera riding the raw polyline
  // sways wall to wall.
  const pts = raw.map((p, i): [number, number] | null => {
    if (!p) return null;
    let ax = 0;
    let az = 0;
    let n0 = 0;
    for (let k = Math.max(0, i - 2); k <= Math.min(raw.length - 1, i + 2); k++) {
      const q = raw[k];
      if (!q) continue;
      ax += q[0];
      az += q[1];
      n0++;
    }
    return [ax / n0, az / n0];
  });
  return { dir, mean: new THREE.Vector2(mx, mz), u0, du, pts };
}

/** Centerline point at along-position u, lerped between the nearest bins. */
function corridorPointAt(
  path: NonNullable<ReturnType<typeof buildCorridorPath>>,
  u: number,
): THREE.Vector2 {
  const f = (u - path.u0) / path.du;
  let lo = Math.min(path.pts.length - 1, Math.max(0, Math.floor(f)));
  let hi = Math.min(path.pts.length - 1, lo + 1);
  while (lo > 0 && path.pts[lo] === null) lo--;
  while (hi < path.pts.length - 1 && path.pts[hi] === null) hi++;
  const a = path.pts[lo] ?? path.pts[hi];
  const b = path.pts[hi] ?? path.pts[lo];
  if (!a || !b) return path.mean.clone();
  if (lo === hi) return new THREE.Vector2(a[0], a[1]);
  const t = Math.min(1, Math.max(0, (f - lo) / (hi - lo)));
  return new THREE.Vector2(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
}

/** Opacity of the "not reconstructed yet" scaffold point cloud. */
const SCAFFOLD_ALPHA = 0.2;

/** One marker's live visuals (pin + pulsing ring), kept in sync with a
 *  DetectionRuntime. `segment` is the capture segment the core found it in, so
 *  the marker can be gated on the SAME arrival the geometry is gated on: a pin
 *  floating over a piece of scene that has not been reconstructed yet would be
 *  claiming more than the board knows. */
interface MarkerVisual {
  det: DetectionRuntime;
  segment: number;
  group: THREE.Group;
  ring: THREE.Mesh;
  visible: boolean;
}

/** The final reconstruction asset + the manifest geometry that cut it into
 *  segments. axis/origin are in the ASSET's own frame; boundaries are cut
 *  positions along the axis relative to the origin (split_segments.py). */
export interface ReconSource {
  url: string;
  axis: [number, number, number];
  origin: [number, number, number];
  boundaries: number[];
}

export class StatusViewer {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly canvas: HTMLCanvasElement;

  private readonly points: THREE.Points;
  private readonly pointGeom: THREE.BufferGeometry;
  private readonly revealAttr: THREE.BufferAttribute;
  /** Built on the first arrived chunk; fades each segment's slab in. */
  private splatReveal: SplatReveal | null;
  private readonly splatCapable: boolean;
  /** Where the final scene comes from — null until the manifest resolves
   *  (a board on a non-demo scene has no final asset and keeps the scaffold). */
  private recon: ReconSource | null = null;
  /** Material the reveal shader is patched into, re-checked every frame in
   *  case the library rebuilds the mesh. */
  private attachedMaterial: THREE.ShaderMaterial | null = null;
  private splatMaskEnabled: boolean = CONFIG.reveal.splatMask;
  private readonly camSync: CameraSync;
  private readonly markers: MarkerVisual[] = [];
  private splat: SplatScene | null = null;
  /** True once real geometry has arrived (the scaffold has been dropped). */
  private geometryArrived = false;

  // Free-orbit navigation of the reconstructed space.
  private readonly controls: OrbitControls;
  /** True disables the follow entirely — the operator owns the camera. */
  private cameraHeld = false;
  private readonly sceneCenter = new THREE.Vector3();
  private followDist = 40;
  private followHeight = 30;
  private floorY = 0;
  /** Follow distance the placeholder cloud implies; the floor for the framing
   *  that the reconstruction's own size drives. */
  private baseFollowDist = 1;
  /** Far plane the placeholder cloud implies. */
  private baseFar = 100;
  /** Camera eye height once the reconstruction turns out to be an INTERIOR
   *  (roof over a walkway). Null until the scene has loaded. The demo capture
   *  is the inside of a building, and a chase camera hovering over the roof
   *  shows the operator nothing but roof — the board has to go inside. */
  private interiorEyeY: number | null = null;
  /** The corridor's CENTERLINE, measured from the splats: median (x,z) per
   *  along-axis bin. The flight line hugs one wall, so a camera following the
   *  route stares into plaster; the corridor's own middle is where a view
   *  down it exists. `dir` is the along axis, `mean` its origin, pts[i] the
   *  centre at u0 + i*du (NaN pairs where the strip has a gap). */
  private corridorPath: {
    dir: THREE.Vector2;
    mean: THREE.Vector2;
    u0: number;
    du: number;
    pts: Array<[number, number] | null>;
  } | null = null;
  /** Smoothed along-corridor camera position and latched travel direction:
   *  the raw scan point jitters with the drone's gaze and the travel dot
   *  product wobbles through zero at turns — chased raw, the camera slaloms. */
  private followU: number | null = null;
  private travelSign = 1;
  private lastDt = 1 / 60;
  /** Placement the final scene was loaded with, for ?pick=on measuring. */
  private finalPlacement: ChunkPlacement | null = null;
  // Recent poses of the active drone, from telemetry, for a damped chase view.
  private readonly history: Array<{ t: number; pos: THREE.Vector3; forward: THREE.Vector3 }> = [];
  private userDragging = false;
  private dragGraceUntil = 0;

  constructor(canvas: HTMLCanvasElement, sceneData: SceneData, useSplat: boolean) {
    this.canvas = canvas;
    this.camSync = new CameraSync(sceneData.bounds);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

    this.scene.background = new THREE.Color(CONFIG.color.statusBg);
    // Subtle warm-neutral fill so the "real" reconstruction reads photographic.
    this.scene.add(new THREE.AmbientLight(0xffe8cf, 0.35));
    const key = new THREE.DirectionalLight(0xfff1d8, 0.5);
    key.position.set(10, 20, 8);
    this.scene.add(key);

    // Frame on the DENSE CORE, not the full bounds: outdoor scenes have a wide
    // spread-out ground that inflates the bounding sphere and leaves the actual
    // subject tiny. Use the median point-distance from the center as the scale.
    sceneData.bounds.getCenter(this.sceneCenter);
    const cx = this.sceneCenter.x, cy = this.sceneCenter.y, cz = this.sceneCenter.z;
    const pos = sceneData.positions;
    const dists: number[] = [];
    for (let i = 0; i < pos.length; i += 3) {
      const dx = pos[i] - cx, dy = pos[i + 1] - cy, dz = pos[i + 2] - cz;
      dists.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
    dists.sort((a, b) => a - b);
    const core = dists.length ? dists[Math.floor(dists.length * 0.6)] : 10;
    const radius = Math.max(1, core);
    this.scene.fog = new THREE.Fog(CONFIG.color.statusBg, radius * 3, radius * 9);

    this.floorY = sceneData.groundY;
    this.followDist = radius * 1.6;
    this.followHeight = radius * 1.3;
    // What the placeholder cloud is worth as a scale. Everything the mission
    // adds is measured in metres and can run far past it, so the camera and the
    // fog are re-derived from the reconstruction as it grows (see followPose).
    this.baseFollowDist = this.followDist;
    this.baseFar = radius * 40;

    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    this.camera = new THREE.PerspectiveCamera(50, aspect, 0.05, radius * 40);
    const dist = radius * 2.6;
    this.camera.position.set(
      this.sceneCenter.x + dist * 0.75,
      this.sceneCenter.y + dist * 0.4,
      this.sceneCenter.z + dist * 0.75,
    );

    // STATUS trails the active drone's view (PROJECT.md §8.3). OrbitControls stay
    // available so the operator can drag to look around; the follow pauses while
    // dragging and for a few seconds after.
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.copy(this.sceneCenter);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = radius * 0.2;
    this.controls.maxDistance = radius * 12;
    this.controls.addEventListener('start', () => {
      this.userDragging = true;
    });
    this.controls.addEventListener('end', () => {
      this.userDragging = false;
      this.dragGraceUntil = performance.now() + 3500;
    });
    this.controls.update();

    // Full point cloud with a per-point aReveal attribute we drive over time.
    this.pointGeom = new THREE.BufferGeometry();
    this.pointGeom.setAttribute('position', new THREE.BufferAttribute(sceneData.positions, 3));
    this.pointGeom.setAttribute('color', new THREE.BufferAttribute(sceneData.colors, 3));
    // The scaffold is drawn dim from the start (the point shader uses aReveal as
    // its alpha). Nothing "reveals" it: it is a placeholder for geometry that has
    // not arrived, and it disappears wholesale when the first chunk lands.
    const revealArray = new Float32Array(sceneData.count).fill(SCAFFOLD_ALPHA);
    this.revealAttr = new THREE.BufferAttribute(revealArray, 1);
    this.revealAttr.setUsage(THREE.DynamicDrawUsage);
    this.pointGeom.setAttribute('aReveal', this.revealAttr);

    const material = new THREE.ShaderMaterial({
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
    });
    this.points = new THREE.Points(this.pointGeom, material);

    // The point cloud stands in until the stream delivers real geometry. If
    // splat rendering is disabled (useSplat=false — e.g. ?render=points), it
    // stays for good and is shown at full opacity instead.
    this.scene.add(this.points);
    this.splatReveal = null;
    this.splatCapable = useSplat;

    this.resize();
  }

  /** Tell the board where the final reconstruction lives. Without this the
   *  stream is bookkeeping only and the scaffold stays up. */
  configureRecon(source: ReconSource): void {
    this.recon = source;
  }

  /**
   * Add a detection the core reported. `segment` is the capture segment it was
   * found in; the marker stays hidden until that segment's geometry has landed.
   * Idempotent per id (the relay replays its cache on reconnect).
   */
  addDetection(det: DetectionRuntime, segment: number): void {
    if (state.detections.some((d) => d.id === det.id)) return;
    state.detections.push(det);
    this.markers.push(this.buildMarker(det, segment));
  }

  private buildMarker(det: DetectionRuntime, segment: number): MarkerVisual {
    const color = det.kind === 'person' ? CONFIG.color.markerPerson : CONFIG.color.markerDanger;
    const group = new THREE.Group();

    const pinGeom = new THREE.ConeGeometry(0.5, 1.6, 12);
    const pinMat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.6 });
    const pin = new THREE.Mesh(pinGeom, pinMat);
    // Marker position is at the FEET; the pin floats well above a standing
    // person's head (a cone at body height reads as stabbing them, not
    // marking them — and the reconstruction smears people taller than life).
    pin.position.y = 8.5;
    pin.rotation.x = Math.PI;
    group.add(pin);

    const ringGeom = new THREE.RingGeometry(0.8, 1.0, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.05;
    group.add(ring);

    group.position.set(det.pos[0], det.pos[1], det.pos[2]);
    group.visible = false;
    this.scene.add(group);

    return { det, segment, group, ring, visible: false };
  }

  /** Gate for DETECTIONS, asked per segment: not first arrival but the LAST
   *  refinement level — a '구조 대상자' card popping up seconds after the drone
   *  passes, while the segment is still a rough shell, reads as fake. The
   *  finding surfaces when the segment finishes processing. (?render=points
   *  has no stream, so everything counts as done.) */
  private isSegmentArrived(segment: number): boolean {
    if (!this.splatCapable) return true;
    const scenes = this.splat?.scenes;
    if (!scenes) return false;
    for (const s of scenes) if (s.segment === segment && s.final) return true;
    return false;
  }

  update(dt: number): void {
    const now = state.time;
    this.lastDt = dt;

    // Patch the splat shader whenever its material appears OR is replaced (the
    // library can rebuild the mesh), so the reveal mask is never lost.
    if (this.splatReveal && this.splat) {
      const mat = this.splat.material;
      if (mat && mat !== this.attachedMaterial) {
        this.splatReveal.attachTo(mat);
        this.splatReveal.setRevealEnabled(this.splatMaskEnabled);
        this.attachedMaterial = mat;
      }
    }

    // Per-segment fade-in. Timed on the render clock: it is an animation, not
    // a piece of mission state.
    this.splatReveal?.update(dt);

    // Once the final scene is up, read its floor/roof off the splats. A roofed
    // scene flips the follow camera into interior mode (see followPose).
    if (this.interiorEyeY === null && this.splat?.status === 'ready') {
      const samples = this.splat.sampleCenters(9000);
      const ys = samples.map((p) => p[1]).sort((a, b) => a - b);
      if (ys.length > 200) {
        // Floor and roof are the two DENSEST horizontal sheets, not the low
        // and high percentiles: the glossy floor reconstructs its own mirror
        // image metres BELOW itself, and a percentile floor dives into that
        // reflection world — the camera then walks on the real floor's level.
        const lo = ys[Math.floor(ys.length * 0.02)];
        const hi = ys[Math.floor(ys.length * 0.98)];
        const bin = 0.4;
        const n = Math.max(4, Math.ceil((hi - lo) / bin));
        const counts = new Array<number>(n).fill(0);
        for (const y of ys) {
          const i = Math.floor((y - lo) / bin);
          if (i >= 0 && i < n) counts[i]++;
        }
        const mid = ys[Math.floor(ys.length / 2)];
        let floorBin = 0;
        let roofBin = n - 1;
        for (let i = 0; i < n; i++) {
          const c = lo + (i + 0.5) * bin;
          if (c < mid && counts[i] > counts[floorBin]) floorBin = i;
          if (c >= mid && counts[i] > (counts[roofBin] ?? -1)) roofBin = i;
        }
        const floor = lo + (floorBin + 0.5) * bin;
        const roof = lo + (roofBin + 0.5) * bin;
        if (roof - floor < 40 && roof - floor > 2.5) {
          this.interiorEyeY = Math.min(floor + 2.2, roof - 1.2);
          this.floorY = floor;
          this.corridorPath = buildCorridorPath(samples);
        }
      }
    }

    // Marker visibility — gated on the arrival of the segment the detection was
    // found in, the same rule the geometry follows.
    for (const m of this.markers) {
      if (!m.det.revealed) {
        if (this.isSegmentArrived(m.segment)) {
          m.det.revealed = true;
          m.det.revealedAt = now;
          emit({ type: 'detection-revealed', id: m.det.id });
        }
      }
      const shouldShow = m.det.revealed;
      if (shouldShow !== m.visible) {
        m.group.visible = shouldShow;
        m.visible = shouldShow;
      }
      if (shouldShow) {
        const pulse = 0.8 + 0.4 * Math.sin(now * 3 + m.group.position.x);
        m.ring.scale.setScalar(pulse);
      }
    }

    // Camera state machine (§8.3).
    // Detection state machine (focus card / confirm / reveal triggers).
    this.camSync.step(dt, state.detections);

    // Record the active drone's pose (fed from DroneTelemetry) for the follow.
    const active = state.drones.find((d) => d.id === state.activeDroneId) ?? state.drones[0];
    if (active) {
      this.history.push({ t: now, pos: active.pos.clone(), forward: active.forward.clone() });
      if (this.history.length > 400) this.history.shift();
    }

    // Desired camera pose: focus a detection, else trail the drone's view.
    const focused =
      state.cameraSync === 'FOCUSING' || state.cameraSync === 'LOCKED'
        ? this.markers.find((m) => m.det.id === state.focusedDetectionId)
        : undefined;
    const desired = focused ? this.focusPose(focused) : this.followPose();

    // Follow (or ease to focus) unless the operator is dragging to look around.
    const dragging =
      this.cameraHeld || this.userDragging || performance.now() < this.dragGraceUntil;
    if (!dragging && desired) {
      const k = 1 - Math.exp(-2.5 * dt);
      this.camera.position.lerp(desired.pos, k);
      this.controls.target.lerp(desired.target, k);
    }
    this.controls.update();

    this.renderer.render(this.scene, this.camera);
  }

  /**
   * One chunk of the delay-pattern stream arrived.
   *
   * The FIRST chunk does the heavy lifting: its alignment places the whole
   * final scene (every chunk of one flight shares a frame — the segment PLYs
   * are slabs of this exact file), the manifest geometry mapped through that
   * same placement anchors the reveal regions, and the scaffold comes down.
   * Every chunk, first included, is then bookkeeping: note the arrival, fade
   * the slab in.
   */
  ingestSplatChunk(chunk: SplatChunkInput): void {
    if (!this.splat) {
      this.splat = new SplatScene(this.scene);
      if (this.splatCapable && this.recon) {
        // Real geometry is on its way in — drop the scaffold and hand
        // visibility over to arrival.
        this.scene.remove(this.points);
        this.geometryArrived = true;
        this.splatReveal = new SplatReveal();
        this.splatReveal.setRevealEnabled(this.splatMaskEnabled);

        // The reveal regions in world space: the manifest's split geometry
        // pushed through the same placement that positions the scene
        // (world = R * (s * asset) + t). Boundaries are along-axis distances,
        // so only the scale touches them.
        const q = new THREE.Quaternion(...chunk.align.rotation);
        const s = chunk.align.scale[0];
        const origin = new THREE.Vector3(...this.recon.origin)
          .multiplyScalar(s)
          .applyQuaternion(q)
          .add(new THREE.Vector3(...chunk.align.position));
        const axis = new THREE.Vector3(...this.recon.axis).applyQuaternion(q);
        this.splatReveal.setFrame(origin, axis, this.recon.boundaries.map((b) => b * s));

        this.finalPlacement = chunk.align;
        void this.splat.loadFinal(this.recon.url, chunk.align);
      }
    }
    this.splat.noteChunk(chunk);
    this.splatReveal?.noteArrival(chunk.segment, chunk.level, chunk.final);
  }

  /** True once the first chunk has landed — the board's waiting state ends here. */
  get hasGeometry(): boolean {
    return this.geometryArrived;
  }

  /** Toggle the per-segment reveal. When off (?reveal=off) the whole loaded
   *  scene renders at full opacity — the debug view of the raw geometry. */
  setSplatMask(enabled: boolean): void {
    this.splatMaskEnabled = enabled;
    this.splatReveal?.setRevealEnabled(enabled);
  }

  /**
   * Park the camera above a point and hold it there. Only the checks use this:
   * an empty board looks the same whether nothing was drawn or the camera was
   * pointed elsewhere, and this separates the two.
   */
  debugLookAt(at: THREE.Vector3, height: number): void {
    this.camera.position.set(at.x, at.y + height, at.z + height * 0.15);
    this.controls.target.copy(at);
    this.camera.lookAt(at);
    this.controls.update();
    // Hold it: the follow camera would drag it back within a frame.
    this.userDragging = true;
  }

  /**
   * ?pick=on measuring: which splat was clicked, in world AND asset coords.
   * The asset coordinate is what people.json wants — this is how a person seen
   * in the reconstruction becomes a detection anchored to them.
   */
  pickAt(
    ndcX: number,
    ndcY: number,
  ): { world: [number, number, number]; asset: [number, number, number] | null } | null {
    if (!this.splat) return null;
    const dir = new THREE.Vector3(ndcX, ndcY, 0.5)
      .unproject(this.camera)
      .sub(this.camera.position)
      .normalize();
    const hit = this.splat.pick(this.camera.position, dir);
    if (!hit) return null;
    let asset: [number, number, number] | null = null;
    if (this.finalPlacement) {
      const q = new THREE.Quaternion(...this.finalPlacement.rotation);
      const p = new THREE.Vector3(...hit.point)
        .sub(new THREE.Vector3(...this.finalPlacement.position))
        .applyQuaternion(q.invert())
        .divideScalar(this.finalPlacement.scale[0] || 1);
      asset = [p.x, p.y, p.z];
    }
    return { world: hit.point, asset };
  }

  /** Hand the camera fully to the operator: no follow, near-unlimited zoom.
   *  ?pick=on uses this — you cannot click a person the camera keeps
   *  yanking you away from. */
  freeCamera(): void {
    this.cameraHeld = true;
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 5000;
  }

  /** Drop the camera INSIDE the reconstructed corridor at eye height, at the
   *  start of the built strip looking along it. False until the scene has
   *  loaded far enough to know where "inside" is. */
  parkInside(): boolean {
    if (this.interiorEyeY === null || !this.splat) return false;
    const path = this.corridorPath;
    if (path) {
      // Start of the corridor's own centerline, looking down it.
      const a = corridorPointAt(path, path.u0 + path.du);
      const b = corridorPointAt(path, path.u0 + path.du + 12);
      const pos = new THREE.Vector3(a.x, this.interiorEyeY, a.y);
      const target = new THREE.Vector3(b.x, this.interiorEyeY - 0.6, b.y);
      this.debugCamera(pos, target);
      return true;
    }
    const chunks = this.splat.loadedChunks();
    if (chunks.length === 0) return false;
    const a = chunks[0].center;
    const b = chunks[Math.min(1, chunks.length - 1)].center;
    const dir = new THREE.Vector3(b[0] - a[0], 0, b[2] - a[2]);
    if (dir.lengthSq() < 1e-6) dir.set(1, 0, 0);
    dir.normalize();
    const pos = new THREE.Vector3(a[0], this.interiorEyeY, a[2]);
    const target = pos.clone().addScaledVector(dir, 12);
    target.y = this.interiorEyeY - 0.6;
    this.debugCamera(pos, target);
    return true;
  }

  /** Park the camera anywhere, looking anywhere — the checks judge the scene's
   *  ORIENTATION from views the follow camera never takes (horizon, side-on). */
  debugCamera(pos: THREE.Vector3, target: THREE.Vector3): void {
    this.camera.position.copy(pos);
    this.controls.target.copy(target);
    this.camera.lookAt(target);
    this.controls.update();
    this.userDragging = true;
  }

  /** Debug: show the scaffold point cloud at full opacity (for ?render=points). */
  revealAll(): void {
    const arr = this.revealAttr.array as Float32Array;
    arr.fill(1);
    this.revealAttr.needsUpdate = true;
  }

  /** Marker gating state, for the e2e check: which detections are showing and
   *  which segment each is waiting on. */
  get markerStates(): Array<{ id: string; segment: number; visible: boolean }> {
    return this.markers.map((m) => ({ id: m.det.id, segment: m.segment, visible: m.visible }));
  }

  get debugInfo(): Record<string, unknown> {
    return {
      camPos: this.camera.position.toArray().map((n) => Math.round(n * 10) / 10),
      target: this.controls.target.toArray().map((n) => Math.round(n * 10) / 10),
      center: this.sceneCenter.toArray().map((n) => Math.round(n * 10) / 10),
      camDist: Math.round(this.camera.position.distanceTo(this.sceneCenter) * 10) / 10,
      splatAttached: this.attachedMaterial !== null,
      segmentLevels: this.splat?.segmentLevels ?? {},
      scenes: (this.splat?.scenes ?? []).map((s) => `seg${s.segment}/lv${s.level}`),
      geometryArrived: this.geometryArrived,
      // Why an empty screen is empty: the frustum, the fog and the reveal each
      // throw geometry away silently, and from a screenshot they look the same
      // as geometry that never arrived.
      camFar: Math.round(this.camera.far),
      fog:
        this.scene.fog instanceof THREE.Fog
          ? [Math.round(this.scene.fog.near), Math.round(this.scene.fog.far)]
          : null,
      reveal: this.splatReveal?.fadeState ?? null,
      chunkCenters: (this.splat?.loadedChunks() ?? []).map((c) =>
        c.center.map((n) => Math.round(n)),
      ),
      revealEnabled: this.splatMaskEnabled,
    };
  }

  get splatStatus(): SplatStatus {
    return this.splat?.status ?? 'idle';
  }

  get splatProgress(): number {
    return this.splat?.progress ?? 0;
  }

  get splatChunks(): number {
    return this.splat?.chunks ?? 0;
  }

  /** Superseded levels dropped after a refinement landed (delay pattern). */
  get splatReplaced(): number {
    return this.splat?.replaced ?? 0;
  }

  /** Highest refinement level rendering per segment. */
  get splatSegmentLevels(): Record<number, number> {
    return this.splat?.segmentLevels ?? {};
  }

  /** How much ground the reconstruction covers, in scene units. */
  splatBounds(): {
    min: [number, number, number];
    max: [number, number, number];
    splats: number;
  } | null {
    return this.splat?.worldBounds() ?? null;
  }

  /** A sample of splat centres in world space. */
  splatSamples(limit?: number): Array<[number, number, number]> {
    return this.splat?.sampleCenters(limit) ?? [];
  }

  /** Every loaded chunk with the place it was put — the minimap draws them,
   *  and the checks measure them against the flight that produced them. */
  loadedChunks(): Array<{ segment: number; level: number; center: [number, number, number] }> {
    return this.splat?.loadedChunks() ?? [];
  }

  /** Follow the active drone: look at the ground it is scanning, from behind and
   *  above its heading, so the board tracks where the capture is happening.
   *  Uses the LATEST telemetry — the old lag existed to trail a simulation the
   *  board no longer receives, and the camera lerp already damps the motion. */
  /** The loaded chunk furthest from a point, or null when nothing is loaded. */
  private furthestChunkFrom(p: THREE.Vector3): THREE.Vector3 | null {
    let best: THREE.Vector3 | null = null;
    let bestD = 0;
    for (const c of this.splat?.loadedChunks() ?? []) {
      const v = new THREE.Vector3(c.center[0], c.center[1], c.center[2]);
      const d = v.distanceTo(p);
      if (d > bestD) {
        bestD = d;
        best = v;
      }
    }
    return best;
  }

  private followPose(): { pos: THREE.Vector3; target: THREE.Vector3 } | null {
    if (this.history.length === 0) return null;
    const s = this.history[this.history.length - 1];
    const P = s.pos;
    const F = s.forward;
    const headingXZ = new THREE.Vector3(F.x, 0, F.z);
    if (headingXZ.lengthSq() < 1e-6) headingXZ.set(0, 0, 1);
    headingXZ.normalize();

    // Where the drone's look ray meets the ground plane (the scanned spot).
    let t = F.y < -1e-3 ? (P.y - this.floorY) / -F.y : this.followDist;
    t = Math.min(Math.max(t, 0), this.followDist * 3);
    const groundHit = P.clone().addScaledVector(F, t);

    // INTERIOR scene: walk the camera through the corridor at eye height,
    // trailing the spot being scanned, instead of hovering over the roof —
    // from above, an indoor reconstruction shows nothing but its roof. The
    // camera rides the corridor's own CENTERLINE, not the flight line: the
    // aircraft hugged a wall, and a camera on its track sees only plaster.
    if (this.interiorEyeY !== null) {
      const dist = 14;
      let pos: THREE.Vector3;
      let target: THREE.Vector3;
      const path = this.corridorPath;
      if (path) {
        // Along-position from the AIRCRAFT, not its gaze ray — the gaze tilts
        // with every telemetry frame and the jitter read as camera sway.
        const uHit = (P.x - path.mean.x) * path.dir.x + (P.z - path.mean.y) * path.dir.y;
        // Ease the along-position and latch the travel direction (flip only
        // on a decisive heading, not while the dot wobbles through zero).
        if (this.followU === null) this.followU = uHit;
        this.followU += (uHit - this.followU) * (1 - Math.exp(-1.5 * this.lastDt));
        const dot = headingXZ.x * path.dir.x + headingXZ.z * path.dir.y;
        if (Math.abs(dot) > 0.35) this.travelSign = Math.sign(dot);
        const t2 = corridorPointAt(path, this.followU);
        const p2 = corridorPointAt(path, this.followU - this.travelSign * dist);
        target = new THREE.Vector3(t2.x, this.interiorEyeY - 0.5, t2.y);
        pos = new THREE.Vector3(p2.x, this.interiorEyeY, p2.y);
      } else {
        target = groundHit.clone();
        target.y = this.interiorEyeY - 0.5;
        pos = target.clone().addScaledVector(headingXZ, -dist);
        pos.y = this.interiorEyeY;
      }
      if (this.scene.fog instanceof THREE.Fog) {
        this.scene.fog.near = 40;
        this.scene.fog.far = 260;
      }
      if (this.camera.far < 500) {
        this.camera.far = 500;
        this.camera.updateProjectionMatrix();
      }
      return { pos, target };
    }

    // The reconstruction TRAILS the aircraft — a segment is only rebuilt once
    // it has been flown — so a camera framed on the aircraft alone sits between
    // the operator and everything that has been built, showing them unscanned
    // ground. Pull back far enough to hold the built strip in frame, and look
    // at the middle of it rather than at the aircraft's own footprint.
    const tail = this.furthestChunkFrom(groundHit);
    const lag = tail ? tail.distanceTo(groundHit) : 0;
    const dist = Math.min(Math.max(this.baseFollowDist, lag * 1.2), this.baseFollowDist * 10);
    const height = Math.max(this.followHeight, dist * 0.55);
    const target = tail ? groundHit.clone().lerp(tail, 0.45) : groundHit;

    // The fog and the far plane were sized from the placeholder cloud, which is
    // a fraction of what the mission covers; left alone they hide the far end of
    // the reconstruction the moment it grows past a segment or two.
    if (this.scene.fog instanceof THREE.Fog) {
      this.scene.fog.near = dist * 1.4;
      this.scene.fog.far = dist * 7;
    }
    const far = Math.max(this.baseFar, dist * 14);
    if (Math.abs(this.camera.far - far) > 1) {
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }

    const pos = target
      .clone()
      .addScaledVector(headingXZ, -dist)
      .addScaledVector(STATUS_UP, height);
    return { pos, target };
  }

  /** The corridor's direction where `p` is, in FLIGHT order: the segment
   *  chunk anchors are strung along the route the capture was flown, so the
   *  vector between the two nearest anchors IS the capture direction there. */
  private corridorAxisAt(p: THREE.Vector3): THREE.Vector3 {
    const chunks = this.splat?.loadedChunks() ?? [];
    const bySeg = [...chunks].sort((a, b) => a.segment - b.segment);
    if (bySeg.length >= 2) {
      let k = 0;
      let best = Infinity;
      for (let i = 0; i < bySeg.length; i++) {
        const c = bySeg[i].center;
        const d = (c[0] - p.x) ** 2 + (c[2] - p.z) ** 2;
        if (d < best) {
          best = d;
          k = i;
        }
      }
      const a = bySeg[Math.max(0, k - 1)].center;
      const b = bySeg[Math.min(bySeg.length - 1, k + 1)].center;
      const dir = new THREE.Vector3(b[0] - a[0], 0, b[2] - a[2]);
      if (dir.lengthSq() > 1e-6) return dir.normalize();
    }
    const h = this.history[this.history.length - 1];
    const dir = h ? new THREE.Vector3(h.forward.x, 0, h.forward.z) : new THREE.Vector3(1, 0, 0);
    return dir.lengthSq() > 1e-6 ? dir.normalize() : new THREE.Vector3(1, 0, 0);
  }

  /**
   * Approach a detection at EYE level, ALONG the corridor — never crosswise
   * and never from above. The capture was flown down the corridor, so that
   * axis is the only direction the scene is meant to be read from; a camera
   * stepping off it shows the unreconstructed gap between the walls.
   */
  private focusPose(m: MarkerVisual): { pos: THREE.Vector3; target: THREE.Vector3 } {
    const feet = new THREE.Vector3(m.det.pos[0], m.det.pos[1], m.det.pos[2]);
    const eyeY = this.interiorEyeY ?? feet.y + 1.7;
    // Look at the person's body, not the pin above them.
    const target = feet.clone();
    target.y = feet.y + 1.1;
    const axis = this.corridorAxisAt(feet);
    const back = CONFIG.camera.focusDistance * 0.7;
    let pos: THREE.Vector3;
    const path = this.corridorPath;
    if (path) {
      // Stand on the corridor CENTERLINE behind the person (in flight order):
      // backing straight off the person can put the camera inside a wall.
      const uFeet = (feet.x - path.mean.x) * path.dir.x + (feet.z - path.mean.y) * path.dir.y;
      const sign = Math.sign(axis.x * path.dir.x + axis.z * path.dir.y) || 1;
      const p2 = corridorPointAt(path, uFeet - sign * back);
      pos = new THREE.Vector3(p2.x, eyeY, p2.y);
    } else {
      pos = feet.clone().addScaledVector(axis, -back);
      pos.y = eyeY;
    }
    return { pos, target };
  }

  resize(): void {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.controls.dispose();
    this.splat?.dispose();
    this.pointGeom.dispose();
    (this.points.material as THREE.Material).dispose();
    for (const m of this.markers) {
      m.group.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          const mat = obj.material as THREE.Material | THREE.Material[];
          if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
          else mat.dispose();
        }
      });
    }
    this.renderer.dispose();
  }
}
