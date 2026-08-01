// Viewer 2 — the "real" 3D reconstruction situation board (PROJECT.md §5, §8).
// Renders the FULL point cloud with a progressive, lagged bloom-reveal driven
// by the drone module's `state.visited` trail, places detection markers that
// appear once their area is revealed, and drives the camera through the
// SYNCED/FOCUSING/LOCKED/RETURNING state machine (§8.3).
//
// A placeholder point cloud stands in for a future Gaussian splat; the
// reveal/camera choreography and coordinate frame are real and swappable.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SceneData } from '../data/sceneData';
import type { Detection, DetectionRuntime } from '../core/types';
import { state, emit } from '../core/store';
import { CONFIG } from '../core/config';
import { RevealField } from './reveal.ts';
import { SplatReveal } from './splatReveal.ts';
import { CameraSync } from './cameraSync.ts';
import { SplatScene } from './splatScene.ts';
import type { SplatOptions, SplatStatus } from './splatScene.ts';

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

const RECON_UP = new THREE.Vector3(0, 1, 0);

/** One marker's live visuals (pin + pulsing ring), kept in sync with a DetectionRuntime. */
interface MarkerVisual {
  det: DetectionRuntime;
  group: THREE.Group;
  ring: THREE.Mesh;
  visible: boolean;
}

export class ReconViewer {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly canvas: HTMLCanvasElement;

  private readonly points: THREE.Points;
  private readonly pointGeom: THREE.BufferGeometry;
  private readonly revealAttr: THREE.BufferAttribute;
  // Reveal is driven on the point cloud (fallback) OR on the splat itself
  // (photorealistic mode, via a coverage texture + patched splat shader).
  private readonly reveal: RevealField | null;
  private readonly splatReveal: SplatReveal | null;
  private splatAttached = false;
  private splatMaskEnabled: boolean = CONFIG.reveal.splatMask;
  private readonly camSync: CameraSync;
  private readonly markers: MarkerVisual[] = [];
  private splat: SplatScene | null = null;

  // Free-orbit navigation of the reconstructed space.
  private readonly controls: OrbitControls;
  private readonly sceneCenter = new THREE.Vector3();
  private followDist = 40;
  private followHeight = 30;
  private floorY = 0;
  // Lagged pose history of the active drone, so RECON trails the SIM view.
  private readonly history: Array<{ t: number; pos: THREE.Vector3; forward: THREE.Vector3 }> = [];
  private userDragging = false;
  private dragGraceUntil = 0;

  constructor(
    canvas: HTMLCanvasElement,
    sceneData: SceneData,
    detections: Detection[],
    useSplat: boolean,
  ) {
    this.canvas = canvas;
    this.camSync = new CameraSync(sceneData.bounds);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

    this.scene.background = new THREE.Color(CONFIG.color.reconBg);
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
    this.scene.fog = new THREE.Fog(CONFIG.color.reconBg, radius * 3, radius * 9);

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

    // RECON trails the active drone's view (PROJECT.md §8.3). OrbitControls stay
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
    const revealArray = new Float32Array(sceneData.count); // starts at 0 — fully hidden
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

    if (useSplat) {
      // RECON is photorealistic: the splat itself reveals. The point cloud is
      // NOT shown — it's the SIM's representation.
      this.reveal = null;
      this.splatReveal = new SplatReveal(sceneData.bounds);
    } else {
      // No splat (disabled/failed): show the point cloud and reveal on it.
      this.scene.add(this.points);
      this.reveal = new RevealField(sceneData.positions, sceneData.count);
      this.splatReveal = null;
    }

    // Initialize shared detection state on first construction (state owner).
    if (state.detections.length === 0) {
      state.detections = detections.map((d) => ({
        ...d,
        revealed: false,
        confirmed: false,
        revealedAt: null,
      }));
    }

    for (const det of state.detections) {
      this.markers.push(this.buildMarker(det));
    }

    this.resize();
  }

  private buildMarker(det: DetectionRuntime): MarkerVisual {
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

    return { det, group, ring, visible: false };
  }

  update(dt: number): void {
    const now = state.time;

    // Lag the visited trail before feeding the reveal (§5.2).
    const cutoff = now - CONFIG.sim.revealLagSeconds;
    const lagged = state.visited.filter((v) => v.t <= cutoff);

    // Patch the splat shader once its material exists (photorealistic reveal).
    // Always patch the splat shader (floater clip); the reveal mask is toggled.
    if (this.splatReveal && this.splat && !this.splatAttached) {
      const mat = this.splat.material;
      if (mat) {
        this.splatReveal.attachTo(mat);
        this.splatReveal.setRevealEnabled(this.splatMaskEnabled);
        this.splatAttached = true;
      }
    }

    if (this.splatReveal) {
      this.splatReveal.update(lagged, dt);
    } else if (this.reveal) {
      const dirty = this.reveal.update(lagged, now, dt);
      if (dirty) {
        (this.revealAttr.array as Float32Array).set(this.reveal.progress);
        this.revealAttr.needsUpdate = true;
      }
    }

    const isRevealed = (pos: [number, number, number]): boolean =>
      this.splatReveal ? this.splatReveal.isAreaRevealed(pos) : !!this.reveal?.isAreaRevealed(pos);

    // Detection reveal + marker visibility.
    for (const m of this.markers) {
      if (!m.det.revealed) {
        if (isRevealed(m.det.pos)) {
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

    // Record the active drone's pose for the lagged follow.
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
    const desired = focused ? this.focusPose(focused) : this.followPose(now);

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
   * Attach the real Gaussian splat layer (TEST asset). Fit into the same frame
   * as the procedural cloud so reveal/markers/camera stay aligned. Idempotent.
   */
  loadSplat(opts: SplatOptions): void {
    if (this.splat) return;
    this.splat = new SplatScene(this.scene);
    this.splat.load(opts);
  }

  /** Toggle the splat reveal mask. When off, the full splat renders regardless. */
  setSplatMask(enabled: boolean): void {
    this.splatMaskEnabled = enabled;
    this.splatReveal?.setRevealEnabled(enabled);
  }

  /** Debug: reveal the whole point cloud immediately (for ?render=points). */
  revealAll(): void {
    const arr = this.revealAttr.array as Float32Array;
    arr.fill(1);
    this.revealAttr.needsUpdate = true;
  }

  get debugInfo(): Record<string, unknown> {
    return {
      camPos: this.camera.position.toArray().map((n) => Math.round(n * 10) / 10),
      target: this.controls.target.toArray().map((n) => Math.round(n * 10) / 10),
      center: this.sceneCenter.toArray().map((n) => Math.round(n * 10) / 10),
      camDist: Math.round(this.camera.position.distanceTo(this.sceneCenter) * 10) / 10,
      splatAttached: this.splatAttached,
    };
  }

  get splatStatus(): SplatStatus {
    return this.splat?.status ?? 'idle';
  }

  get splatProgress(): number {
    return this.splat?.progress ?? 0;
  }

  /** Lagged follow of the active drone: look at the ground it's scanning, from
   *  behind + above its heading, so steering the drone rotates the RECON view. */
  private followPose(now: number): { pos: THREE.Vector3; target: THREE.Vector3 } | null {
    if (this.history.length === 0) return null;
    const targetT = now - CONFIG.sim.revealLagSeconds;
    let s = this.history[0];
    for (const h of this.history) {
      if (h.t <= targetT) s = h;
      else break;
    }
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
      .addScaledVector(RECON_UP, this.followHeight);
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
      .addScaledVector(RECON_UP, CONFIG.camera.focusDistance * 0.35);
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
