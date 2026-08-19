// The control tower's ONE 3D view (COMPONENTS.md §4).
//
// There used to be two worlds here: a point cloud derived from a Gaussian splat
// asset, and the VWorld real-terrain scene. They are now one. The scene IS
// VWorld terrain plus real building footprints, and "points" is no longer a
// separate world — it is one of three ways to DRAW those same buildings:
//
//   점              buildings as a sampled point cloud
//   검정 텍스처 건물   black prisms + cyan linework   (default)
//   실사 항공뷰        the satellite mosaic draped over terrain AND buildings
//
// All three are prepared from a SINGLE load: the point cloud, the prism
// triangles, and the aerial UVs all come from one pass of buildingSource. So
// switching mode swaps materials and visibility — it never refetches and never
// rebuilds geometry. That is what lets the settings panel switch live.

import * as THREE from 'three';
import type { SceneData } from '../../shared/viewer/sources/sceneData.ts';
import type { BuildingVisual } from '../../shared/viewer/sources/buildingSource.ts';
import type { TerrainVisual } from '../../shared/viewer/sources/terrainSource.ts';
import type { DroneRuntime } from '../../shared/viewer/types.ts';
import { state } from '../../shared/viewer/store.ts';
import { CONFIG, DRONE_TINTS, droneViewScale } from '../../shared/viewer/config.ts';
import type { DisplayMode } from '../settings.ts';

/** Trail points kept per drone. */
const TRAIL_LENGTH = 40;
/** Camera chase damping — higher = snappier, lower = floatier. */
const CAMERA_LERP = 0.06;
/** Offset behind + above the active drone, in the drone's local frame. */
const CHASE_BACK = 9;
const CHASE_UP = 4;
/** Degrees per second the overview camera orbits when no drone is connected. */
const OVERVIEW_SPIN = 4;

/** Per-mode look. The scene is identical in all three — only materials differ. */
interface ModeStyle {
  background: number;
  fogNear: number;
  fogFar: number;
  /** Multiplied onto the terrain drape. */
  terrainTint: number;
  ringTint: number;
  buildingPoints: boolean;
  buildingPrisms: boolean;
  buildingEdges: boolean;
}

const STYLES: Record<DisplayMode, ModeStyle> = {
  // Tactical: the drape drops back so the cyan building points own the frame.
  points: {
    background: CONFIG.color.controlBg,
    fogNear: 40,
    fogFar: 220,
    terrainTint: 0x3d4757,
    ringTint: 0x2a323d,
    buildingPoints: true,
    buildingPrisms: false,
    buildingEdges: false,
  },
  // Default. Black volumes over a legible-but-quiet map; the linework is what
  // keeps a block of black prisms readable as separate buildings.
  black: {
    background: 0x1b222c,
    fogNear: 70,
    fogFar: 320,
    terrainTint: 0xaab2bc,
    ringTint: 0x818992,
    buildingPoints: false,
    buildingPrisms: true,
    buildingEdges: true,
  },
  // Photoreal: one continuous aerial image across terrain AND buildings.
  aerial: {
    background: 0x4a515c,
    fogNear: 95,
    fogFar: 360,
    terrainTint: 0xffffff,
    ringTint: 0xdbe0e6,
    buildingPoints: false,
    buildingPrisms: true,
    buildingEdges: false,
  },
};

interface DroneRig {
  id: number;
  core: THREE.Mesh;
  halo: THREE.Mesh;
  trailLine: THREE.Line;
  trailGeom: THREE.BufferGeometry;
  trailPositions: Float32Array;
  trailColors: Float32Array;
  trailCount: number;
}

/** One streamed cell's building geometry, kept so mode changes reach it too. */
interface CellBuildings {
  mesh: THREE.Mesh;
  edges: THREE.LineSegments | null;
  black: THREE.Material;
  aerial: THREE.Material | null;
}

export interface TowerViewerInput {
  /** Building points (the 점 option). May be empty when VWorld gave nothing. */
  sceneData: SceneData;
  terrainVisual: TerrainVisual | null;
  surroundVisual: TerrainVisual | null;
  buildingVisual: BuildingVisual | null;
  display: DisplayMode;
}

export class TowerViewer {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private canvas: HTMLCanvasElement;
  private fog: THREE.Fog;

  private mode: DisplayMode;

  // --- Terrain ---
  private terrainMat: THREE.MeshBasicMaterial | null = null;
  private ringMat: THREE.MeshBasicMaterial | null = null;
  /** The aerial mosaic, shared by the terrain drape and the building drape. */
  private mosaic: THREE.Texture | null = null;
  private streamedTerrain = new Set<THREE.MeshBasicMaterial>();

  // --- Buildings (core scene) ---
  private bldPoints: THREE.Points | null = null;
  private bldMesh: THREE.Mesh | null = null;
  private bldEdges: THREE.LineSegments | null = null;
  private bldBlackMat: THREE.Material | null = null;
  private bldAerialMat: THREE.Material | null = null;

  // --- Buildings (streamed cells) ---
  private cells = new Set<CellBuildings>();

  private grid: THREE.GridHelper;
  private rigs = new Map<number, DroneRig>();
  private rigGroup: THREE.Group;

  private camPos = new THREE.Vector3();
  private camTarget = new THREE.Vector3();
  private camInitialized = false;
  private overviewAngle = 0.6;
  private overviewRadius = 40;
  private overviewCenter = new THREE.Vector3();

  private droneScale = droneViewScale();

  constructor(canvas: HTMLCanvasElement, input: TowerViewerInput) {
    this.canvas = canvas;
    this.mode = input.display;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    const style = STYLES[this.mode];
    this.scene.background = new THREE.Color(style.background);
    this.fog = new THREE.Fog(style.background, style.fogNear, style.fogFar);
    this.scene.fog = this.fog;

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 900);
    this.camera.position.set(0, 30, 55);

    if (input.terrainVisual) {
      this.mosaic = makeTexture(input.terrainVisual.texture);
      this.terrainMat = new THREE.MeshBasicMaterial({
        map: this.mosaic,
        side: THREE.DoubleSide,
      });
      this.scene.add(new THREE.Mesh(meshGeometry(input.terrainVisual), this.terrainMat));
    }

    if (input.surroundVisual) {
      this.ringMat = new THREE.MeshBasicMaterial({
        map: makeTexture(input.surroundVisual.texture),
        side: THREE.DoubleSide,
      });
      this.scene.add(new THREE.Mesh(meshGeometry(input.surroundVisual), this.ringMat));
    }

    this.buildBuildings(input);

    // Lambert (the black prisms) needs light; Basic materials ignore it. The
    // values look high because three's physical units render legacy ones dark.
    this.scene.add(new THREE.HemisphereLight(0xe8eef5, 0x5f676f, 2.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(40, 70, -30);
    this.scene.add(sun);

    // Reference grid — only meaningful when there is no terrain under it.
    this.grid = new THREE.GridHelper(80, 32, 0x2a4a8a, 0x16294f);
    this.grid.position.y = input.sceneData.groundY;
    (this.grid.material as THREE.Material).transparent = true;
    (this.grid.material as THREE.Material).opacity = 0.35;
    this.grid.visible = input.terrainVisual == null;
    this.scene.add(this.grid);

    this.rigGroup = new THREE.Group();
    this.scene.add(this.rigGroup);

    this.frameOverview(input);
    this.applyMode(this.mode);
    this.resize();
  }

  /** Frame the standby orbit on whatever geometry actually loaded. */
  private frameOverview(input: TowerViewerInput): void {
    const box = new THREE.Box3();
    if (input.terrainVisual) {
      box.setFromBufferAttribute(new THREE.BufferAttribute(input.terrainVisual.positions, 3));
    } else if (input.buildingVisual && input.buildingVisual.positions.length > 0) {
      box.setFromBufferAttribute(new THREE.BufferAttribute(input.buildingVisual.positions, 3));
    } else if (input.sceneData.count > 0) {
      box.copy(input.sceneData.bounds);
    }
    if (box.isEmpty()) {
      this.overviewRadius = 40;
      return;
    }
    box.getCenter(this.overviewCenter);
    const size = box.getSize(new THREE.Vector3());
    this.overviewRadius = Math.max(size.x, size.z) * 0.62 + 8;
  }

  // -------------------------------------------------------------------------
  // Display mode
  // -------------------------------------------------------------------------

  /** Swap materials/visibility in place. No refetch, no geometry rebuild. */
  setDisplay(mode: DisplayMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.applyMode(mode);
  }

  get display(): DisplayMode {
    return this.mode;
  }

  /** True when 실사 항공뷰 can actually be honored (imagery mosaic present). */
  get aerialAvailable(): boolean {
    return this.bldAerialMat != null || this.mosaic != null;
  }

  private applyMode(mode: DisplayMode): void {
    const style = STYLES[mode];
    (this.scene.background as THREE.Color).setHex(style.background);
    this.fog.color.setHex(style.background);
    this.fog.near = style.fogNear;
    this.fog.far = style.fogFar;

    this.terrainMat?.color.setHex(style.terrainTint);
    this.ringMat?.color.setHex(style.ringTint);
    for (const mat of this.streamedTerrain) mat.color.setHex(style.terrainTint);

    // Aerial falls back to the black look when no mosaic loaded, rather than
    // rendering untextured white slabs the operator cannot read.
    const wantAerial = mode === 'aerial' && this.bldAerialMat != null;

    if (this.bldPoints) this.bldPoints.visible = style.buildingPoints;
    if (this.bldMesh) {
      this.bldMesh.visible = style.buildingPrisms;
      const mat = wantAerial ? this.bldAerialMat : this.bldBlackMat;
      if (mat) this.bldMesh.material = mat;
    }
    if (this.bldEdges) this.bldEdges.visible = style.buildingEdges;

    for (const cell of this.cells) {
      cell.mesh.visible = style.buildingPrisms;
      cell.mesh.material = mode === 'aerial' && cell.aerial ? cell.aerial : cell.black;
      if (cell.edges) cell.edges.visible = style.buildingEdges;
    }
  }

  // -------------------------------------------------------------------------
  // Geometry
  // -------------------------------------------------------------------------

  private buildBuildings(input: TowerViewerInput): void {
    // 점 — sampled building points.
    if (input.sceneData.count > 0) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute(
        'position',
        new THREE.BufferAttribute(
          input.sceneData.positions.slice(0, input.sceneData.count * 3),
          3,
        ),
      );
      geom.setAttribute('color', new THREE.BufferAttribute(tacticalColors(input.sceneData), 3));
      this.bldPoints = new THREE.Points(
        geom,
        new THREE.PointsMaterial({
          size: 0.16,
          sizeAttenuation: true,
          vertexColors: true,
          transparent: true,
          opacity: 0.92,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        }),
      );
      this.scene.add(this.bldPoints);
    }

    const visual = input.buildingVisual;
    if (!visual || visual.positions.length === 0) return;

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(visual.positions, 3));
    const uvs = matchingUvs(visual);
    if (uvs) geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geom.computeVertexNormals();

    this.bldBlackMat = blackBuildingMaterial();
    if (this.mosaic && uvs) {
      this.bldAerialMat = aerialBuildingMaterial(this.mosaic);
    }
    this.bldMesh = new THREE.Mesh(geom, this.bldBlackMat);
    this.scene.add(this.bldMesh);

    if (visual.edges && visual.edges.length > 0) {
      this.bldEdges = buildEdges(visual.edges);
      this.scene.add(this.bldEdges);
    }
  }

  /** Streamed terrain cell (world streamer). Returns a disposer. */
  addStreamedTerrain(visual: TerrainVisual): () => void {
    const geom = meshGeometry(visual);
    const tex = makeTexture(visual.texture);
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      side: THREE.DoubleSide,
      color: STYLES[this.mode].terrainTint,
    });
    const mesh = new THREE.Mesh(geom, mat);
    this.scene.add(mesh);
    this.streamedTerrain.add(mat);
    return () => {
      this.streamedTerrain.delete(mat);
      this.scene.remove(mesh);
      geom.dispose();
      mat.dispose();
      tex.dispose();
    };
  }

  /**
   * Streamed building cell. `cellTexture` is that cell's OWN aerial mosaic — a
   * streamed cell lies outside the core drape, so it cannot share the core
   * mosaic; without its own, 실사 항공뷰 would stop at the scene edge.
   */
  addSurroundBuildings(visual: BuildingVisual, cellTexture?: ImageBitmap): () => void {
    if (visual.positions.length === 0) return () => {};
    const style = STYLES[this.mode];

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(visual.positions, 3));
    const uvs = matchingUvs(visual);
    if (uvs) geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geom.computeVertexNormals();

    const black = blackBuildingMaterial();
    let aerial: THREE.Material | null = null;
    let cellTex: THREE.Texture | null = null;
    if (cellTexture && uvs) {
      cellTex = makeTexture(cellTexture);
      aerial = aerialBuildingMaterial(cellTex);
    }
    const mesh = new THREE.Mesh(geom, this.mode === 'aerial' && aerial ? aerial : black);
    mesh.visible = style.buildingPrisms;
    this.scene.add(mesh);

    let edges: THREE.LineSegments | null = null;
    if (visual.edges && visual.edges.length > 0) {
      edges = buildEdges(visual.edges);
      edges.visible = style.buildingEdges;
      this.scene.add(edges);
    }

    const cell: CellBuildings = { mesh, edges, black, aerial };
    this.cells.add(cell);

    return () => {
      this.cells.delete(cell);
      this.scene.remove(mesh);
      geom.dispose();
      black.dispose();
      aerial?.dispose();
      cellTex?.dispose();
      if (edges) {
        this.scene.remove(edges);
        edges.geometry.dispose();
        (edges.material as THREE.Material).dispose();
      }
    };
  }

  // -------------------------------------------------------------------------
  // Drones
  // -------------------------------------------------------------------------

  private ensureRig(drone: DroneRuntime): DroneRig {
    const existing = this.rigs.get(drone.id);
    if (existing) return existing;

    const tint = DRONE_TINTS[(drone.id - 1) % DRONE_TINTS.length] ?? CONFIG.color.droneCore;
    const coreGeom = new THREE.OctahedronGeometry(0.6, 0);
    const core = new THREE.Mesh(
      coreGeom,
      new THREE.MeshBasicMaterial({ color: tint, wireframe: true }),
    );
    this.rigGroup.add(core);

    // The active drone gets a white cage: LUMINANCE, not another hue, is what
    // separates the mission focus from a map already full of cyan.
    const halo = new THREE.Mesh(
      coreGeom,
      new THREE.MeshBasicMaterial({
        color: 0xe8fbff,
        wireframe: true,
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
      }),
    );
    halo.visible = false;
    this.rigGroup.add(halo);

    const trailPositions = new Float32Array(TRAIL_LENGTH * 3);
    const trailColors = new Float32Array(TRAIL_LENGTH * 4);
    const trailGeom = new THREE.BufferGeometry();
    trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    trailGeom.setAttribute('color', new THREE.BufferAttribute(trailColors, 4));
    trailGeom.setDrawRange(0, 0);
    const trailLine = new THREE.Line(
      trailGeom,
      new THREE.LineBasicMaterial({
        color: CONFIG.color.droneTrail,
        transparent: true,
        opacity: 0.8,
        vertexColors: true,
      }),
    );
    this.rigGroup.add(trailLine);

    const rig: DroneRig = {
      id: drone.id,
      core,
      halo,
      trailLine,
      trailGeom,
      trailPositions,
      trailColors,
      trailCount: 0,
    };
    this.rigs.set(drone.id, rig);
    return rig;
  }

  /** Drop rigs for drones the core no longer reports. */
  private pruneRigs(live: Set<number>): void {
    for (const [id, rig] of this.rigs) {
      if (live.has(id)) continue;
      this.rigGroup.remove(rig.core, rig.halo, rig.trailLine);
      rig.core.geometry.dispose();
      (rig.core.material as THREE.Material).dispose();
      (rig.halo.material as THREE.Material).dispose();
      rig.trailGeom.dispose();
      (rig.trailLine.material as THREE.Material).dispose();
      this.rigs.delete(id);
    }
  }

  private updateRig(rig: DroneRig, drone: DroneRuntime, isActive: boolean): void {
    rig.core.position.copy(drone.pos);
    rig.core.quaternion.copy(drone.quat);
    const scale = (isActive ? 1.6 : 1.0) * this.droneScale;
    rig.core.scale.setScalar(scale);
    const mat = rig.core.material as THREE.MeshBasicMaterial;
    mat.opacity = isActive ? 1 : 0.65;
    mat.transparent = !isActive;

    rig.halo.visible = isActive;
    rig.halo.position.copy(drone.pos);
    rig.halo.quaternion.copy(drone.quat);
    rig.halo.scale.setScalar(scale * 1.32);

    const count = Math.min(rig.trailCount + 1, TRAIL_LENGTH);
    const pos = rig.trailPositions;
    const col = rig.trailColors;
    if (rig.trailCount > 0) {
      const shift = Math.min(rig.trailCount, TRAIL_LENGTH - 1);
      pos.copyWithin(3, 0, shift * 3);
      col.copyWithin(4, 0, shift * 4);
    }
    pos[0] = drone.pos.x;
    pos[1] = drone.pos.y;
    pos[2] = drone.pos.z;
    rig.trailCount = count;

    const tint = new THREE.Color(
      DRONE_TINTS[(rig.id - 1) % DRONE_TINTS.length] ?? CONFIG.color.droneTrail,
    );
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

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(dt: number): void {
    const drones = state.drones;
    this.pruneRigs(new Set(drones.map((d) => d.id)));

    if (drones.length === 0) {
      // No telemetry → no fleet. Show the operating area rather than a frozen
      // chase camera pointed at a drone that isn't there.
      this.updateOverviewCamera(dt);
      this.renderer.render(this.scene, this.camera);
      return;
    }

    for (const drone of drones) {
      this.updateRig(this.ensureRig(drone), drone, drone.id === state.activeDroneId);
    }
    const active = drones.find((d) => d.id === state.activeDroneId) ?? drones[0];
    this.updateChaseCamera(active);
    this.renderer.render(this.scene, this.camera);
  }

  private updateOverviewCamera(dt: number): void {
    this.overviewAngle += dt * OVERVIEW_SPIN * (Math.PI / 180);
    const r = this.overviewRadius;
    this.camera.position.set(
      this.overviewCenter.x + Math.cos(this.overviewAngle) * r,
      this.overviewCenter.y + r * 0.5,
      this.overviewCenter.z + Math.sin(this.overviewAngle) * r,
    );
    this.camera.lookAt(this.overviewCenter);
    // A drone that connects later should ease in from here, not snap from a
    // stale chase position.
    this.camInitialized = false;
  }

  private updateChaseCamera(drone: DroneRuntime): void {
    const forward =
      drone.forward.lengthSq() > 0.0001 ? drone.forward : new THREE.Vector3(0, 0, 1);
    const desired = drone.pos
      .clone()
      .addScaledVector(forward, -CHASE_BACK * this.droneScale)
      .add(new THREE.Vector3(0, CHASE_UP * this.droneScale, 0));

    if (!this.camInitialized) {
      this.camPos.copy(desired);
      this.camTarget.copy(drone.pos);
      this.camInitialized = true;
    } else {
      this.camPos.lerp(desired, CAMERA_LERP);
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
    this.bldPoints?.geometry.dispose();
    (this.bldPoints?.material as THREE.Material | undefined)?.dispose();
    this.bldMesh?.geometry.dispose();
    this.bldBlackMat?.dispose();
    this.bldAerialMat?.dispose();
    this.bldEdges?.geometry.dispose();
    (this.bldEdges?.material as THREE.Material | undefined)?.dispose();
    this.mosaic?.dispose();
    this.grid.geometry.dispose();
    (this.grid.material as THREE.Material).dispose();
    this.pruneRigs(new Set());
    this.renderer.dispose();
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** UVs are 2 floats per vertex against 3 floats of position; null when absent
 *  or mismatched (a mismatch would silently garble the aerial drape). */
function matchingUvs(v: BuildingVisual): Float32Array | null {
  if (!v.uvs) return null;
  return v.uvs.length * 3 === v.positions.length * 2 ? v.uvs : null;
}

function meshGeometry(v: TerrainVisual): THREE.BufferGeometry {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(v.positions, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(v.uvs, 2));
  geom.setIndex(new THREE.BufferAttribute(v.indices, 1));
  return geom;
}

function makeTexture(bitmap: ImageBitmap): THREE.Texture {
  const tex = new THREE.Texture(bitmap);
  // The mosaic is built top-left origin, so the usual GL flip must be off.
  tex.flipY = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** 검정 텍스처 건물: near-black, but LIT, so faces separate by shading. */
function blackBuildingMaterial(): THREE.Material {
  return new THREE.MeshLambertMaterial({
    color: 0x0b0e13,
    emissive: 0x05070b,
    side: THREE.DoubleSide,
  });
}

/** 실사 항공뷰: the aerial mosaic itself, addressed by the prisms' own UVs. */
function aerialBuildingMaterial(map: THREE.Texture): THREE.Material {
  return new THREE.MeshBasicMaterial({ map, side: THREE.DoubleSide });
}

/** Restrained tactical linework. Mission objects stay brighter than this. */
function buildEdges(edges: Float32Array): THREE.LineSegments {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(edges, 3));
  return new THREE.LineSegments(
    geom,
    new THREE.LineBasicMaterial({
      color: 0x7dd8ec,
      transparent: true,
      opacity: 0.5,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
}

/**
 * Height-ramped cyan for the 점 option. The building layer hands over neutral
 * daylight grays (it stays palette-free on purpose), so the tactical coloring
 * is applied HERE — at the point the display option is actually chosen.
 */
function tacticalColors(data: SceneData): Float32Array {
  const out = new Float32Array(data.count * 3);
  const near = new THREE.Color(CONFIG.color.controlPoint);
  const far = new THREE.Color(CONFIG.color.controlPointFar);
  const c = new THREE.Color();
  const minY = data.bounds.min.y;
  const span = Math.max(data.bounds.max.y - minY, 0.001);
  for (let i = 0; i < data.count; i++) {
    const t = THREE.MathUtils.clamp((data.positions[i * 3 + 1] - minY) / span, 0, 1);
    c.copy(far).lerp(near, t);
    out[i * 3] = c.r;
    out[i * 3 + 1] = c.g;
    out[i * 3 + 2] = c.b;
  }
  return out;
}
