// Viewer 1 — the "drone simulation space" (PROJECT.md §4.3, §4.4).
// Deliberately LOW-FIDELITY: sparse cyan scanning dots, a dim grid, and
// glowing low-poly drone markers. This coldness is an intentional contrast
// device against viewer 2's warm realism — do not prettify this file.

import * as THREE from 'three';
import type { SceneData } from '../data/sceneData.ts';
import type { DroneRuntime } from '../types.ts';
import { state, subscribe } from '../store.ts';
import { CONFIG, DRONE_TINTS } from '../config.ts';

/** Keep 1 of every N points from the source cloud. */
const DOWNSAMPLE_STRIDE = 5;
/** Number of trail points kept per drone. */
const TRAIL_LENGTH = 40;
/** Camera chase damping — higher = snappier, lower = floatier. */
const CAMERA_LERP = 0.06;
/** Offset behind + above the active drone, in the drone's local frame. */
const CHASE_BACK = 9;
const CHASE_UP = 4;

interface DroneRig {
  id: number;
  core: THREE.Mesh;
  trailLine: THREE.Line;
  trailGeom: THREE.BufferGeometry;
  trailPositions: Float32Array;
  trailColors: Float32Array;
  trailCount: number;
}

export class LowfiViewer {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private canvas: HTMLCanvasElement;

  private points: THREE.Points;
  private pointsGeom: THREE.BufferGeometry;
  private pointsMat: THREE.PointsMaterial;
  private grid: THREE.GridHelper;

  private rigs = new Map<number, DroneRig>();
  private rigGroup: THREE.Group;

  private camPos = new THREE.Vector3();
  private camTarget = new THREE.Vector3();
  private camInitialized = false;

  private unsubscribe: () => void;

  constructor(canvas: HTMLCanvasElement, sceneData: SceneData) {
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(CONFIG.color.simBg);
    this.scene.fog = new THREE.Fog(CONFIG.color.simBg, 20, 90);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
    this.camera.position.set(0, 20, 40);

    // Sparse downsampled scanning-dot cloud.
    const { geom, mat } = buildPointCloud(sceneData);
    this.pointsGeom = geom;
    this.pointsMat = mat;
    this.points = new THREE.Points(geom, mat);
    this.scene.add(this.points);

    // Dim ground grid for spatial reference — deliberately minimal.
    this.grid = new THREE.GridHelper(80, 32, 0x2a4a8a, 0x16294f);
    this.grid.position.y = sceneData.groundY;
    (this.grid.material as THREE.Material).transparent = true;
    (this.grid.material as THREE.Material).opacity = 0.35;
    this.scene.add(this.grid);

    this.rigGroup = new THREE.Group();
    this.scene.add(this.rigGroup);

    this.unsubscribe = subscribe(() => {
      // Active-drone changes just ease the chase target; no snap logic needed
      // here since update() re-reads state.activeDroneId every frame.
    });

    this.resize();
  }

  private ensureRig(drone: DroneRuntime): DroneRig {
    let rig = this.rigs.get(drone.id);
    if (rig) return rig;

    const tint = DRONE_TINTS[(drone.id - 1) % DRONE_TINTS.length] ?? CONFIG.color.droneCore;

    const coreGeom = new THREE.OctahedronGeometry(0.6, 0);
    const coreMat = new THREE.MeshBasicMaterial({ color: tint, wireframe: true });
    const core = new THREE.Mesh(coreGeom, coreMat);
    this.rigGroup.add(core);

    const trailPositions = new Float32Array(TRAIL_LENGTH * 3);
    const trailColors = new Float32Array(TRAIL_LENGTH * 4);
    const trailGeom = new THREE.BufferGeometry();
    trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    trailGeom.setAttribute('color', new THREE.BufferAttribute(trailColors, 4));
    trailGeom.setDrawRange(0, 0);

    const trailMat = new THREE.LineBasicMaterial({
      color: CONFIG.color.droneTrail,
      transparent: true,
      opacity: 0.8,
      vertexColors: true,
    });
    const trailLine = new THREE.Line(trailGeom, trailMat);
    this.rigGroup.add(trailLine);

    rig = { id: drone.id, core, trailLine, trailGeom, trailPositions, trailColors, trailCount: 0 };
    this.rigs.set(drone.id, rig);
    return rig;
  }

  private updateRig(rig: DroneRig, drone: DroneRuntime, isActive: boolean): void {
    rig.core.position.copy(drone.pos);
    rig.core.quaternion.copy(drone.quat);
    const scale = isActive ? 1.6 : 1.0;
    rig.core.scale.setScalar(scale);
    const mat = rig.core.material as THREE.MeshBasicMaterial;
    mat.opacity = isActive ? 1 : 0.65;
    mat.transparent = !isActive;

    // Push a new trail sample every frame (rolling buffer, oldest dropped).
    const count = Math.min(rig.trailCount + 1, TRAIL_LENGTH);
    const pos = rig.trailPositions;
    const col = rig.trailColors;
    // Shift existing samples back by one slot.
    if (rig.trailCount > 0) {
      const shiftCount = Math.min(rig.trailCount, TRAIL_LENGTH - 1);
      pos.copyWithin(3, 0, shiftCount * 3);
      col.copyWithin(4, 0, shiftCount * 4);
    }
    pos[0] = drone.pos.x;
    pos[1] = drone.pos.y;
    pos[2] = drone.pos.z;
    rig.trailCount = count;

    // Recompute fade-toward-tail alpha for the whole buffer.
    const tint = new THREE.Color(DRONE_TINTS[(rig.id - 1) % DRONE_TINTS.length] ?? CONFIG.color.droneTrail);
    for (let i = 0; i < count; i++) {
      const alpha = (1 - i / count) * (isActive ? 0.9 : 0.5);
      col[i * 4] = tint.r;
      col[i * 4 + 1] = tint.g;
      col[i * 4 + 2] = tint.b;
      col[i * 4 + 3] = alpha;
    }

    rig.trailGeom.setDrawRange(0, count);
    (rig.trailGeom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (rig.trailGeom.attributes.color as THREE.BufferAttribute).needsUpdate = true;
  }

  update(_dt: number): void {
    const drones = state.drones;
    if (drones.length === 0) {
      this.renderer.render(this.scene, this.camera);
      return;
    }

    for (const drone of drones) {
      const rig = this.ensureRig(drone);
      this.updateRig(rig, drone, drone.id === state.activeDroneId);
    }

    const active = drones.find((d) => d.id === state.activeDroneId) ?? drones[0];
    this.updateChaseCamera(active);

    this.renderer.render(this.scene, this.camera);
  }

  private updateChaseCamera(drone: DroneRuntime): void {
    const forward = drone.forward.lengthSq() > 0.0001 ? drone.forward : new THREE.Vector3(0, 0, 1);
    const desiredPos = drone.pos
      .clone()
      .addScaledVector(forward, -CHASE_BACK)
      .add(new THREE.Vector3(0, CHASE_UP, 0));

    if (!this.camInitialized) {
      this.camPos.copy(desiredPos);
      this.camTarget.copy(drone.pos);
      this.camInitialized = true;
    } else {
      this.camPos.lerp(desiredPos, CAMERA_LERP);
      this.camTarget.lerp(drone.pos, CAMERA_LERP);
    }

    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camTarget);
  }

  resize(): void {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  dispose(): void {
    this.unsubscribe();
    this.pointsGeom.dispose();
    this.pointsMat.dispose();
    this.grid.geometry.dispose();
    (this.grid.material as THREE.Material).dispose();
    for (const rig of this.rigs.values()) {
      rig.core.geometry.dispose();
      (rig.core.material as THREE.Material).dispose();
      rig.trailGeom.dispose();
      (rig.trailLine.material as THREE.Material).dispose();
    }
    this.rigs.clear();
    this.renderer.dispose();
  }
}

/** Downsample the source cloud into a small cyan->navy scanning-dot set. */
function buildPointCloud(sceneData: SceneData): { geom: THREE.BufferGeometry; mat: THREE.PointsMaterial } {
  const srcCount = sceneData.count;
  const keep = Math.floor(srcCount / DOWNSAMPLE_STRIDE);
  const pos = new Float32Array(keep * 3);
  const col = new Float32Array(keep * 3);

  const near = new THREE.Color(CONFIG.color.simPoint);
  const far = new THREE.Color(CONFIG.color.simPointFar);
  const c = new THREE.Color();
  const heightSpan = Math.max(sceneData.bounds.max.y - sceneData.bounds.min.y, 0.001);

  for (let i = 0, srcI = 0; i < keep; i++, srcI += DOWNSAMPLE_STRIDE) {
    const j = srcI * 3;
    const k = i * 3;
    const y = sceneData.positions[j + 1];
    pos[k] = sceneData.positions[j];
    pos[k + 1] = y;
    pos[k + 2] = sceneData.positions[j + 2];

    const t = THREE.MathUtils.clamp((y - sceneData.bounds.min.y) / heightSpan, 0, 1);
    c.copy(far).lerp(near, t);
    col[k] = c.r;
    col[k + 1] = c.g;
    col[k + 2] = c.b;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.18,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  return { geom, mat };
}
