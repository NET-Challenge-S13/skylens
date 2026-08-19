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
  /** Which drape this mode shows: the cold grade, or the photograph itself. */
  gradedDrape: boolean;
  /** Ambient/sun pair. The tactical modes are lit cold so the prisms read as
   *  part of the map instead of a daylight render dropped onto it. */
  lightSky: number;
  lightGround: number;
  lightIntensity: number;
  sunColor: number;
  sunIntensity: number;
  /** Lifts shadow-side faces so a dark building stays a volume, not a hole. */
  buildingEmissive: number;
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
    gradedDrape: true,
    lightSky: 0x80b9cc,
    lightGround: 0x071019,
    lightIntensity: 1.35,
    sunColor: 0xa4d9e8,
    sunIntensity: 0.85,
    buildingEmissive: 0x152a33,
    buildingPoints: true,
    buildingPrisms: false,
    buildingEdges: false,
  },
  // Default (COMPONENTS.md §4). The map is graded cold and stays dim so the
  // drones and mission overlays own the brightest values; the prisms carry
  // their own baked wall/roof colours, lit, with linework on top. This is the
  // look the interim report figure shows (res/docs/figures/sim_map_view.jpg).
  black: {
    background: CONFIG.color.controlBg,
    fogNear: 70,
    fogFar: 320,
    gradedDrape: true,
    lightSky: 0x80b9cc,
    lightGround: 0x071019,
    lightIntensity: 1.35,
    sunColor: 0xa4d9e8,
    sunIntensity: 0.85,
    buildingEmissive: 0x152a33,
    buildingPoints: false,
    buildingPrisms: true,
    buildingEdges: true,
  },
  // Photoreal: one continuous aerial image across terrain AND buildings.
  aerial: {
    background: 0x4a515c,
    fogNear: 95,
    fogFar: 360,
    gradedDrape: false,
    lightSky: 0xe8eef5,
    lightGround: 0x5f676f,
    lightIntensity: 2.4,
    sunColor: 0xffffff,
    sunIntensity: 1.7,
    buildingEmissive: 0x000000,
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

/** A planned waypoint in scene space: where the aircraft flies, and the ground
 *  under it. */
export interface RouteLeg {
  air: THREE.Vector3;
  ground: THREE.Vector3;
}

/** One streamed cell's building geometry, kept so mode changes reach it too. */
interface CellBuildings {
  mesh: THREE.Mesh;
  edges: THREE.LineSegments | null;
  prism: THREE.MeshLambertMaterial;
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
  /** The assigned route, drawn over the terrain. Null until one is assigned. */
  private routeGroup: THREE.Group | null = null;
  /** Set by debugTopDown: the camera is being aimed from outside and NO
   *  automatic camera may move it — not the overview spin, and not the chase
   *  rig, which runs every frame once a drone is connected and silently undid
   *  every top-down a check tried to take. */
  private cameraFrozen = false;

  private mode: DisplayMode;

  // --- Terrain ---
  private terrainMat: THREE.MeshBasicMaterial | null = null;
  private ringMat: THREE.MeshBasicMaterial | null = null;
  /** [photograph, cold grade] per draped surface, so a mode change swaps the
   *  map instead of re-grading pixels while the operator waits. */
  private terrainTex: [THREE.Texture, THREE.Texture] | null = null;
  private ringTex: [THREE.Texture, THREE.Texture] | null = null;
  private readonly streamedDrapes = new Map<
    THREE.MeshBasicMaterial,
    [THREE.Texture, THREE.Texture]
  >();
  private readonly hemi: THREE.HemisphereLight;
  private readonly sun: THREE.DirectionalLight;
  /** The aerial mosaic, shared by the terrain drape and the building drape. */
  private mosaic: THREE.Texture | null = null;
  /** Every prism material on screen, so a mode change reaches the streamed
   *  cells too and the whole city keeps one look. */
  private readonly prismMats = new Set<THREE.MeshLambertMaterial>();

  // --- Buildings (core scene) ---
  private bldPoints: THREE.Points | null = null;
  private bldMesh: THREE.Mesh | null = null;
  private bldEdges: THREE.LineSegments | null = null;
  private bldPrismMat: THREE.MeshLambertMaterial | null = null;
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
      this.terrainTex = [this.mosaic, makeTexture(input.terrainVisual.textureGraded)];
      this.terrainMat = new THREE.MeshBasicMaterial({
        map: this.terrainTex[style.gradedDrape ? 1 : 0],
        side: THREE.DoubleSide,
      });
      this.scene.add(new THREE.Mesh(meshGeometry(input.terrainVisual), this.terrainMat));
    }

    if (input.surroundVisual) {
      this.ringTex = [
        makeTexture(input.surroundVisual.texture),
        makeTexture(input.surroundVisual.textureGraded),
      ];
      this.ringMat = new THREE.MeshBasicMaterial({
        map: this.ringTex[style.gradedDrape ? 1 : 0],
        side: THREE.DoubleSide,
      });
      this.scene.add(new THREE.Mesh(meshGeometry(input.surroundVisual), this.ringMat));
    }

    this.buildBuildings(input);

    // Lambert prisms need light; Basic materials (drape, points) ignore it. The
    // values look high because three's physical units render legacy ones dark.
    // Colour and intensity are per-mode: the tactical look is lit cold.
    this.hemi = new THREE.HemisphereLight(style.lightSky, style.lightGround, style.lightIntensity);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(style.sunColor, style.sunIntensity);
    this.sun.position.set(40, 70, -30);
    this.scene.add(this.sun);

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

    // The drape is exchanged, not darkened. Multiplying a tint onto the aerial
    // photograph was what made the tactical modes look like a dimmed satellite
    // view instead of the graded map they are supposed to be.
    const drape = style.gradedDrape ? 1 : 0;
    if (this.terrainMat && this.terrainTex) {
      this.terrainMat.map = this.terrainTex[drape];
      this.terrainMat.needsUpdate = true;
    }
    if (this.ringMat && this.ringTex) {
      this.ringMat.map = this.ringTex[drape];
      this.ringMat.needsUpdate = true;
    }
    for (const [mat, tex] of this.streamedDrapes) {
      mat.map = tex[drape];
      mat.needsUpdate = true;
    }

    this.hemi.color.setHex(style.lightSky);
    this.hemi.groundColor.setHex(style.lightGround);
    this.hemi.intensity = style.lightIntensity;
    this.sun.color.setHex(style.sunColor);
    this.sun.intensity = style.sunIntensity;
    for (const mat of this.prismMats) mat.emissive.setHex(style.buildingEmissive);

    // Aerial falls back to the black look when no mosaic loaded, rather than
    // rendering untextured white slabs the operator cannot read.
    const wantAerial = mode === 'aerial' && this.bldAerialMat != null;

    if (this.bldPoints) this.bldPoints.visible = style.buildingPoints;
    if (this.bldMesh) {
      this.bldMesh.visible = style.buildingPrisms;
      const mat = wantAerial ? this.bldAerialMat : this.bldPrismMat;
      if (mat) this.bldMesh.material = mat;
    }
    if (this.bldEdges) this.bldEdges.visible = style.buildingEdges;

    for (const cell of this.cells) {
      cell.mesh.visible = style.buildingPrisms;
      cell.mesh.material = mode === 'aerial' && cell.aerial ? cell.aerial : cell.prism;
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
    // The wall/roof colours buildingSource bakes per vertex. Dropping them is
    // what turned the city into one flat silhouette.
    geom.setAttribute('color', new THREE.BufferAttribute(visual.colors, 3));
    const uvs = matchingUvs(visual);
    if (uvs) geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geom.computeVertexNormals();

    this.bldPrismMat = prismMaterial(STYLES[this.mode].buildingEmissive);
    this.prismMats.add(this.bldPrismMat);
    if (this.mosaic && uvs) {
      this.bldAerialMat = aerialBuildingMaterial(this.mosaic);
    }
    this.bldMesh = new THREE.Mesh(geom, this.bldPrismMat);
    // Named so a check can tell buildings from everything else above the
    // ground — drone rigs, the route line and the danger arcs are meshes too,
    // and counting their corners as building corners hides a real answer.
    this.bldMesh.name = 'buildings';
    this.scene.add(this.bldMesh);

    if (visual.edges && visual.edges.length > 0) {
      this.bldEdges = buildEdges(visual.edges);
      this.scene.add(this.bldEdges);
    }
  }

  /** Streamed terrain cell (world streamer). Returns a disposer. */
  addStreamedTerrain(visual: TerrainVisual): () => void {
    const geom = meshGeometry(visual);
    const tex: [THREE.Texture, THREE.Texture] = [
      makeTexture(visual.texture),
      makeTexture(visual.textureGraded),
    ];
    const mat = new THREE.MeshBasicMaterial({
      map: tex[STYLES[this.mode].gradedDrape ? 1 : 0],
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geom, mat);
    this.scene.add(mesh);
    this.streamedDrapes.set(mat, tex);
    return () => {
      this.streamedDrapes.delete(mat);
      this.scene.remove(mesh);
      geom.dispose();
      mat.dispose();
      tex[0].dispose();
      tex[1].dispose();
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
    // The wall/roof colours buildingSource bakes per vertex. Dropping them is
    // what turned the city into one flat silhouette.
    geom.setAttribute('color', new THREE.BufferAttribute(visual.colors, 3));
    const uvs = matchingUvs(visual);
    if (uvs) geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geom.computeVertexNormals();

    const prism = prismMaterial(STYLES[this.mode].buildingEmissive);
    this.prismMats.add(prism);
    let aerial: THREE.Material | null = null;
    let cellTex: THREE.Texture | null = null;
    if (cellTexture && uvs) {
      cellTex = makeTexture(cellTexture);
      aerial = aerialBuildingMaterial(cellTex);
    }
    const mesh = new THREE.Mesh(geom, this.mode === 'aerial' && aerial ? aerial : prism);
    mesh.name = 'buildings-streamed';
    mesh.visible = style.buildingPrisms;
    this.scene.add(mesh);

    let edges: THREE.LineSegments | null = null;
    if (visual.edges && visual.edges.length > 0) {
      edges = buildEdges(visual.edges);
      edges.visible = style.buildingEdges;
      this.scene.add(edges);
    }

    const cell: CellBuildings = { mesh, edges, prism, aerial };
    this.cells.add(cell);

    return () => {
      this.cells.delete(cell);
      this.scene.remove(mesh);
      geom.dispose();
      this.prismMats.delete(prism);
      prism.dispose();
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

  /**
   * Draw the route the fleet was given. Without it the operator plans a line in
   * one window and then watches aircraft move over terrain in another with
   * nothing to check them against — which is indistinguishable from the drones
   * ignoring the plan.
   *
   * Positions arrive already in scene space: the tower converts GPS in exactly
   * one place (geoFrame.ts), and the viewer is downstream of it like the drone
   * rigs are.
   */
  setRoute(legs: RouteLeg[] | null, loop: boolean): void {
    if (this.routeGroup) {
      this.scene.remove(this.routeGroup);
      this.routeGroup.traverse((o) => {
        if (o instanceof THREE.Line || o instanceof THREE.Mesh) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
      this.routeGroup = null;
    }
    if (!legs || legs.length < 2) return;

    const group = new THREE.Group();

    // The flown line, at the altitudes the operator set.
    const air = legs.map((l) => l.air);
    const path = loop ? [...air, ...[...air].reverse().slice(1)] : air;
    const lineGeom = new THREE.BufferGeometry().setFromPoints(path);
    group.add(
      new THREE.Line(
        lineGeom,
        new THREE.LineBasicMaterial({ color: 0xffd27f, transparent: true, opacity: 0.9 }),
      ),
    );

    // A dropped line and a pad per waypoint: altitude is part of the plan, and a
    // line hanging in space cannot be read against the ground without them.
    for (const leg of legs) {
      const drop = new THREE.BufferGeometry().setFromPoints([leg.air, leg.ground]);
      group.add(
        new THREE.Line(
          drop,
          new THREE.LineBasicMaterial({ color: 0xffd27f, transparent: true, opacity: 0.35 }),
        ),
      );
      const pad = new THREE.Mesh(
        new THREE.RingGeometry(0.25, 0.42, 20),
        new THREE.MeshBasicMaterial({
          color: 0xffd27f,
          transparent: true,
          opacity: 0.75,
          side: THREE.DoubleSide,
        }),
      );
      pad.position.copy(leg.ground);
      pad.rotation.x = -Math.PI / 2;
      group.add(pad);
    }

    this.routeGroup = group;
    this.scene.add(group);
  }

  /**
   * Park the camera straight above a scene point and hold it there. Only the
   * checks use it: comparing the planner's map against the 3D view means
   * looking at the same ground from the same angle, and the overview camera
   * orbits.
   */
  /**
   * Look straight down from `height` above `at`. `fovDeg` narrows the lens: at
   * the default 55 deg a tall roof is much nearer the camera than the street
   * beside it and swells to hide half the block, which is honest perspective
   * but useless for comparing shapes against a map. A narrow lens from far up
   * is nearly orthographic.
   */
  debugTopDown(at: THREE.Vector3, height: number, fovDeg?: number): void {
    this.camInitialized = true;
    if (fovDeg !== undefined) {
      this.camera.fov = fovDeg;
      this.camera.updateProjectionMatrix();
    }
    this.camPos.set(at.x, at.y + height, at.z + 0.001);
    this.camTarget.copy(at);
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camTarget);
    this.cameraFrozen = true;
  }

  /** The scene graph, so a check can measure what is actually in it rather
   *  than judge a screenshot. */
  debugScene(): THREE.Scene {
    return this.scene;
  }

  /** Vertices of the drawn route, for the checks that close the loop between
   *  what was planned and what the operator is looking at. */
  debugRoute(): THREE.Vector3[] | null {
    if (!this.routeGroup) return null;
    const line = this.routeGroup.children.find((c) => c instanceof THREE.Line) as
      | THREE.Line
      | undefined;
    if (!line) return null;
    const pos = line.geometry.getAttribute('position');
    const out: THREE.Vector3[] = [];
    for (let i = 0; i < pos.count; i++) {
      out.push(new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)));
    }
    return out;
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
    if (!this.cameraFrozen) this.updateChaseCamera(active);
    this.renderer.render(this.scene, this.camera);
  }

  private updateOverviewCamera(dt: number): void {
    if (this.cameraFrozen) return;
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
    this.bldPrismMat?.dispose();
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

/**
 * 검정 텍스처 건물: the prisms carry per-vertex wall/roof colours from
 * buildingSource, lit so faces separate by shading. A flat near-black colour
 * was tried instead and it flattened every block into one silhouette — only
 * the linework carried volume, and the baked colours went unused.
 *
 * `emissive` is set per mode (STYLES.buildingEmissive): the cold floor keeps
 * shadow-side faces readable against an equally dark map.
 */
function prismMaterial(emissive: number): THREE.MeshLambertMaterial {
  return new THREE.MeshLambertMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    emissive,
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
