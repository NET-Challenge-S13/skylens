// Viewer 1 — the "drone simulation space" (PROJECT.md §4.3, §4.4).
// Deliberately LOW-FIDELITY: sparse cyan scanning dots, a dim grid, and
// glowing low-poly drone markers. This coldness is an intentional contrast
// device against viewer 2's warm realism — do not prettify this file.

import * as THREE from 'three';
import type { SceneData } from '../data/sceneData.ts';
import type { DroneRuntime } from '../types.ts';
import { state, subscribe } from '../store.ts';
import { CONFIG, DRONE_TINTS, droneViewScale } from '../config.ts';

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

  /** Rig size AND chase-camera distance shrink together (see droneViewScale):
   *  the drone keeps its screen size while the map appears magnified. */
  private droneScale = droneViewScale();

  private unsubscribe: () => void;

  constructor(
    canvas: HTMLCanvasElement,
    sceneData: SceneData,
    terrainVisual?: import('../data/terrainSource.ts').TerrainVisual,
    terrainPointCount?: number,
    buildingVisual?: import('../data/buildingSource.ts').BuildingVisual,
    surroundVisual?: import('../data/terrainSource.ts').TerrainVisual,
  ) {
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    // `?tex=sat` (real satellite colors): photo colors are unreadable on the
    // dark navy + tight fog, so that mode gets a neutral gray sky and far fog.
    const satMode =
      typeof window !== 'undefined' &&
      new URLSearchParams(window.location.search).get('tex') === 'sat';
    const bg = satMode ? new THREE.Color(0x4a515c) : new THREE.Color(CONFIG.color.simBg);
    this.scene.background = bg;
    this.scene.fog = satMode ? new THREE.Fog(bg.getHex(), 80, 320) : new THREE.Fog(bg.getHex(), 20, 90);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
    this.camera.position.set(0, 20, 40);

    // `?tex=sat` + a terrain visual: render the DEM as a CONTINUOUS textured
    // surface (satellite mosaic), and keep only the building points on top.
    // The full point cloud stays the data source for paths/detections either way.
    const useMesh = satMode && terrainVisual != null;
    if (useMesh && terrainVisual) {
      const meshGeom = new THREE.BufferGeometry();
      meshGeom.setAttribute('position', new THREE.BufferAttribute(terrainVisual.positions, 3));
      meshGeom.setAttribute('uv', new THREE.BufferAttribute(terrainVisual.uvs, 2));
      meshGeom.setIndex(new THREE.BufferAttribute(terrainVisual.indices, 1));
      const tex = new THREE.Texture(terrainVisual.texture);
      tex.flipY = false;
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      const meshMat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
      this.scene.add(new THREE.Mesh(meshGeom, meshMat));
    }

    // Context ring: low-res backdrop terrain around the sim area so the world
    // doesn't visibly end at the scene edge. Slightly dimmed (material.color
    // multiplies the texture) so the sharper sim area still reads as the focus.
    if (useMesh && surroundVisual) {
      const surGeom = new THREE.BufferGeometry();
      surGeom.setAttribute('position', new THREE.BufferAttribute(surroundVisual.positions, 3));
      surGeom.setAttribute('uv', new THREE.BufferAttribute(surroundVisual.uvs, 2));
      surGeom.setIndex(new THREE.BufferAttribute(surroundVisual.indices, 1));
      const surTex = new THREE.Texture(surroundVisual.texture);
      surTex.flipY = false;
      surTex.colorSpace = THREE.SRGBColorSpace;
      surTex.needsUpdate = true;
      const surMat = new THREE.MeshBasicMaterial({
        map: surTex,
        side: THREE.DoubleSide,
        color: 0xb8bdc4,
      });
      this.scene.add(new THREE.Mesh(surGeom, surMat));
    }

    // Buildings as solid extruded prisms (walls+roofs, flat-shaded) when the
    // textured mesh view is on: sparse point clumps read as "floating" over a
    // continuous terrain surface, so the display upgrades to real volumes.
    const hasBldMesh =
      useMesh && buildingVisual != null && buildingVisual.positions.length > 0;
    if (hasBldMesh && buildingVisual) {
      const bldGeom = new THREE.BufferGeometry();
      bldGeom.setAttribute('position', new THREE.BufferAttribute(buildingVisual.positions, 3));
      bldGeom.setAttribute('color', new THREE.BufferAttribute(buildingVisual.colors, 3));
      bldGeom.computeVertexNormals();
      const bldMat = new THREE.MeshLambertMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
      });
      this.scene.add(new THREE.Mesh(bldGeom, bldMat));
    }
    if (useMesh) {
      // Lambert needs light for volume; Basic materials (terrain/points) ignore
      // it. Intensities look high because three's physical lighting units make
      // legacy-looking values render dark.
      this.scene.add(new THREE.HemisphereLight(0xe8eef5, 0x5f676f, 2.6));
      const sun = new THREE.DirectionalLight(0xffffff, 1.8);
      sun.position.set(40, 70, -30);
      this.scene.add(sun);
    }

    // Sparse downsampled scanning-dot cloud (mesh mode: buildings only; prism
    // mode: none — the volumes replace the building points entirely).
    const { geom, mat } = buildPointCloud(
      sceneData,
      useMesh ? (hasBldMesh ? sceneData.count : (terrainPointCount ?? 0)) : 0,
    );
    this.pointsGeom = geom;
    this.pointsMat = mat;
    this.points = new THREE.Points(geom, mat);
    this.scene.add(this.points);

    // Dim ground grid for spatial reference — deliberately minimal.
    this.grid = new THREE.GridHelper(80, 32, 0x2a4a8a, 0x16294f);
    this.grid.position.y = sceneData.groundY;
    (this.grid.material as THREE.Material).transparent = true;
    (this.grid.material as THREE.Material).opacity = 0.35;
    // With a terrain backdrop the reference grid just reads as an artifact.
    this.grid.visible = !(useMesh && surroundVisual);
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

  /** Streamed building cells pop in as the drone travels (world streamer).
   *  Same prism look as the core so streamed territory reads first-class. */
  addSurroundBuildings(visual: import('../data/buildingSource.ts').BuildingVisual): void {
    if (visual.positions.length === 0) return;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(visual.positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(visual.colors, 3));
    geom.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
    });
    this.scene.add(new THREE.Mesh(geom, mat));
  }

  /** Streamed terrain cells (world streamer) — sharper than the coarse ring
   *  they cover, so streamed territory looks like the core scene. */
  addStreamedTerrain(visual: import('../data/terrainSource.ts').TerrainVisual): void {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(visual.positions, 3));
    geom.setAttribute('uv', new THREE.BufferAttribute(visual.uvs, 2));
    geom.setIndex(new THREE.BufferAttribute(visual.indices, 1));
    const tex = new THREE.Texture(visual.texture);
    tex.flipY = false;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
    this.scene.add(new THREE.Mesh(geom, mat));
  }

  private updateRig(rig: DroneRig, drone: DroneRuntime, isActive: boolean): void {
    rig.core.position.copy(drone.pos);
    rig.core.quaternion.copy(drone.quat);
    const scale = (isActive ? 1.6 : 1.0) * this.droneScale;
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
      .addScaledVector(forward, -CHASE_BACK * this.droneScale)
      .add(new THREE.Vector3(0, CHASE_UP * this.droneScale, 0));

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

/** Downsample the source cloud into a small cyan->navy scanning-dot set.
 *  `?tex=sat` keeps the source cloud's real colors (satellite drape) instead —
 *  the terrain then reads as the actual place in SIM too. */
function buildPointCloud(
  sceneData: SceneData,
  startIndex = 0,
): { geom: THREE.BufferGeometry; mat: THREE.PointsMaterial } {
  const realColors =
    typeof window !== 'undefined' &&
    new URLSearchParams(window.location.search).get('tex') === 'sat';

  const srcCount = sceneData.count;
  // Mesh mode passes startIndex = terrain count → building points only, denser.
  const stride = startIndex > 0 ? 2 : DOWNSAMPLE_STRIDE;
  const keep = Math.max(0, Math.floor((srcCount - startIndex) / stride));
  const pos = new Float32Array(keep * 3);
  const col = new Float32Array(keep * 3);

  const near = new THREE.Color(CONFIG.color.simPoint);
  const far = new THREE.Color(CONFIG.color.simPointFar);
  const c = new THREE.Color();
  const heightSpan = Math.max(sceneData.bounds.max.y - sceneData.bounds.min.y, 0.001);

  for (let i = 0, srcI = startIndex; i < keep; i++, srcI += stride) {
    const j = srcI * 3;
    const k = i * 3;
    const y = sceneData.positions[j + 1];
    pos[k] = sceneData.positions[j];
    pos[k + 1] = y;
    pos[k + 2] = sceneData.positions[j + 2];

    if (realColors) {
      col[k] = sceneData.colors[j];
      col[k + 1] = sceneData.colors[j + 1];
      col[k + 2] = sceneData.colors[j + 2];
    } else {
      const t = THREE.MathUtils.clamp((y - sceneData.bounds.min.y) / heightSpan, 0, 1);
      c.copy(far).lerp(near, t);
      col[k] = c.r;
      col[k + 1] = c.g;
      col[k + 2] = c.b;
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));

  // Real colors need normal blending — additive on a dark bg washes photos out.
  const mat = new THREE.PointsMaterial({
    size: realColors ? 0.42 : 0.18,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: realColors ? 1.0 : 0.85,
    blending: realColors ? THREE.NormalBlending : THREE.AdditiveBlending,
    depthWrite: realColors,
  });

  return { geom, mat };
}
