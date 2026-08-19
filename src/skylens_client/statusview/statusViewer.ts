// Viewer 2 — the "real" 3D reconstruction situation board (PROJECT.md §5, §8).
//
// Everything it shows arrives from the pipeline (skylens_client/sources/
// relayClient.ts). It renders the splat chunks the core delivers, places the
// detection markers the core reports, follows the drone by its telemetry, and
// drives the camera through the SYNCED/FOCUSING/LOCKED/RETURNING state machine
// (§8.3).
//
// Visibility has exactly one rule: a segment is visible once its chunk has
// landed (COMPONENTS.md §8). There is no trail-driven mask any more — the splat
// mesh only ever holds arrived geometry, and SplatReveal just fades each segment
// in. Until the first chunk lands, the locally-loaded point cloud is drawn as a
// dim scaffold so the board reads as WAITING rather than broken; it is removed
// the moment real geometry arrives.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SceneData } from '../../shared/viewer/sources/sceneData';
import type { DetectionRuntime } from '../../shared/viewer/types';
import { state, emit } from '../../shared/viewer/store';
import { CONFIG } from '../../shared/viewer/config';
import type { SplatChunkInput } from './splatScene.ts';
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

export class StatusViewer {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly canvas: HTMLCanvasElement;

  private readonly points: THREE.Points;
  private readonly pointGeom: THREE.BufferGeometry;
  private readonly revealAttr: THREE.BufferAttribute;
  /** Built on the first arrived chunk; fades each segment in from then on. */
  private splatReveal: SplatReveal | null;
  private readonly splatCapable: boolean;
  private readonly sceneBounds: THREE.Box3;
  /** Material the reveal shader is patched into. The library REBUILDS the splat
   *  mesh whenever a superseded level is removed, so this is re-checked every
   *  frame and re-patched when the material identity changes. */
  private attachedMaterial: THREE.ShaderMaterial | null = null;
  private splatMaskEnabled: boolean = CONFIG.reveal.splatMask;
  private readonly camSync: CameraSync;
  private readonly markers: MarkerVisual[] = [];
  private splat: SplatScene | null = null;
  /** True once real geometry has arrived (the scaffold has been dropped). */
  private geometryArrived = false;

  // Free-orbit navigation of the reconstructed space.
  private readonly controls: OrbitControls;
  private readonly sceneCenter = new THREE.Vector3();
  private followDist = 40;
  private followHeight = 30;
  private floorY = 0;
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
    this.sceneBounds = sceneData.bounds.clone();

    this.resize();
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
    pin.position.y = 0.8;
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

  /** The board's one visibility rule, asked per segment. Before any chunk has
   *  arrived nothing is revealed; with the splat renderer off (?render=points)
   *  the scaffold IS the scene, so everything is. */
  private isSegmentArrived(segment: number): boolean {
    if (!this.splatCapable) return true;
    const scenes = this.splat?.scenes;
    if (!scenes) return false;
    for (const s of scenes) if (s.segment === segment) return true;
    return false;
  }

  update(dt: number): void {
    const now = state.time;

    // Patch the splat shader whenever its material appears OR is replaced: the
    // library rebuilds the mesh (and material) each time a superseded level is
    // dropped, which would otherwise lose the floater clip + reveal mask.
    if (this.splatReveal && this.splat) {
      const mat = this.splat.material;
      if (mat && mat !== this.attachedMaterial) {
        this.splatReveal.attachTo(mat);
        this.splatReveal.setRevealEnabled(this.splatMaskEnabled);
        this.attachedMaterial = mat;
      }
    }

    // Per-segment fade-in. Timed on the real clock (performance.now()), not the
    // stream clock: it is an animation, not a piece of mission state.
    if (this.splatReveal && this.splat) {
      this.splatReveal.update(this.splat.scenes, performance.now());
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
    const dragging = this.userDragging || performance.now() < this.dragGraceUntil;
    if (!dragging && desired) {
      const k = 1 - Math.exp(-2.5 * dt);
      this.camera.position.lerp(desired.pos, k);
      this.controls.target.lerp(desired.target, k);
    }
    this.controls.update();

    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Progressive splat ingestion from the server: add one chunk placed by its
   * align transform, fit into the same frame as the procedural cloud so
   * reveal/markers/camera stay aligned. Creates the underlying SplatScene
   * lazily on the first chunk. Callers should serialize (await) calls.
   */
  ingestSplatChunk(chunk: SplatChunkInput): Promise<void> {
    if (!this.splat) {
      this.splat = new SplatScene(this.scene);
      if (this.splatCapable) {
        // Real geometry is on its way in — drop the scaffold and hand visibility
        // over to arrival.
        this.scene.remove(this.points);
        this.geometryArrived = true;
        this.splatReveal = new SplatReveal(this.sceneBounds);
      }
    }
    return this.splat.addChunk(chunk);
  }

  /** True once the first chunk has landed — the board's waiting state ends here. */
  get hasGeometry(): boolean {
    return this.geometryArrived;
  }

  /** Toggle the per-segment fade. When off, arrived chunks render at full
   *  opacity immediately (?reveal=off). Arrival still decides what exists. */
  setSplatMask(enabled: boolean): void {
    this.splatMaskEnabled = enabled;
    this.splatReveal?.setRevealEnabled(enabled);
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

  /** Every loaded chunk with the place it was put — the minimap draws them,
   *  and the checks measure them against the flight that produced them. */
  loadedChunks(): Array<{ segment: number; level: number; position: [number, number, number] }> {
    return this.splat?.loadedChunks() ?? [];
  }

  /** Follow the active drone: look at the ground it is scanning, from behind and
   *  above its heading, so the board tracks where the capture is happening.
   *  Uses the LATEST telemetry — the old lag existed to trail a simulation the
   *  board no longer receives, and the camera lerp already damps the motion. */
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

    const pos = groundHit
      .clone()
      .addScaledVector(headingXZ, -this.followDist)
      .addScaledVector(STATUS_UP, this.followHeight);
    return { pos, target: groundHit };
  }

  private focusPose(m: MarkerVisual): { pos: THREE.Vector3; target: THREE.Vector3 } {
    const det = new THREE.Vector3(m.det.pos[0], m.det.pos[1], m.det.pos[2]);
    const dir = det.clone().sub(this.camera.position);
    if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
    dir.normalize();
    const pos = det
      .clone()
      .addScaledVector(dir, -CONFIG.camera.focusDistance)
      .addScaledVector(STATUS_UP, CONFIG.camera.focusDistance * 0.35);
    return { pos, target: det };
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
