// Splat layer for the STATUS viewer: ONE static scene + arrival bookkeeping.
//
// Wraps @mkkellogg/gaussian-splats-3d's DropInViewer, which is a THREE.Group
// that renders itself inside our existing renderer/camera/loop (it hooks
// onBeforeRender), so we just add it to the scene. `sharedMemoryForWorkers:false`
// keeps it working without COOP/COEP cross-origin-isolation headers.
//
// WHAT CHANGED AND WHY. This used to accumulate the delay-pattern stream as
// separate viewer scenes — one per segment, each replaced as its refinement
// levels landed. That meant a dynamic mesh being rebuilt on every removal,
// which raced the library's async splat-tree build (`visitLeaves` null
// crashes) and left the board rendering nothing. The stream's PLYs are cuts of
// one file we already have, so the geometry is now loaded ONCE from the final
// export, static, and the stream only drives VISIBILITY (splatReveal.ts) and
// the bookkeeping here: which segment has arrived at which level, where its
// piece sits (minimap, camera), how many chunks were delivered (panels).

import * as THREE from 'three';
import { DropInViewer } from '@mkkellogg/gaussian-splats-3d';

export type SplatStatus = 'idle' | 'loading' | 'ready' | 'error';

/** Placement of the scene, in scene units. Structurally a protocol SplatAlign
 *  minus the GPS anchor, which the caller has already resolved. */
export interface ChunkPlacement {
  position: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
}

/** One delivered chunk of the delay-pattern stream. */
export interface SplatChunkInput {
  align: ChunkPlacement;
  segment: number;
  level: number;
  /** No further level will arrive for this segment. */
  final: boolean;
  /**
   * Where this chunk's geometry sits, in scene units.
   *
   * NOT `align.position`: that is the origin of the chunk's own coordinate
   * space. Anything asking "where is this piece of the reconstruction" (the
   * minimap, the camera) means this.
   */
  center: [number, number, number];
}

type StatusCb = (status: SplatStatus, detail?: string) => void;

/** One arrived segment. The geometry for all of them is the single static
 *  scene; these entries are the record of what the stream has delivered. */
export interface LoadedScene {
  segment: number;
  level: number;
  /** The core has delivered this segment's LAST level — it is done refining.
   *  Detections wait for this: a finding stamped on a half-built segment
   *  claims more certainty than the reconstruction has earned. */
  final: boolean;
  /** Where this piece sits, in scene units — the board's own frame. */
  center: [number, number, number];
  /** performance.now() when this segment FIRST arrived. Refinements keep it. */
  arrivedAt: number;
}

export class SplatScene {
  private readonly viewer: DropInViewer;
  private _status: SplatStatus = 'idle';
  private _progress = 0;
  private _chunks = 0;
  private _refined = 0;
  private readonly cbs: StatusCb[] = [];
  private readonly parent: THREE.Scene;
  private readonly order: LoadedScene[] = [];
  private readonly bySegment = new Map<number, LoadedScene>();
  private loadStarted = false;

  constructor(parent: THREE.Scene) {
    this.parent = parent;
    this.viewer = new DropInViewer({
      sharedMemoryForWorkers: false,
      // CPU sort in a worker — more compatible across GPUs/headless than the
      // GPU-accelerated path, and fine for a single static scene.
      gpuAcceleratedSort: false,
      // ONE scene, loaded once, never swapped: the static path is the
      // library's best-trodden one and bakes the placement into the mesh.
      dynamicScene: false,
      // Clamp very large gaussians so they don't render as screen-filling fog
      // (photo splats) or spikes (heavy scenes). A moderate value keeps normal
      // ground/structure splats intact.
      maxScreenSpaceSplatSize: 256,
    });
    this.parent.add(this.viewer);
  }

  get status(): SplatStatus {
    return this._status;
  }

  /** The splat mesh's ShaderMaterial once loaded, else null. */
  get material(): THREE.ShaderMaterial | null {
    return this.viewer.splatMesh?.material ?? null;
  }

  /** Download progress percent (0..100) of the final scene. */
  get progress(): number {
    return this._progress;
  }

  /** Chunk messages delivered by the stream (levels included). */
  get chunks(): number {
    return this._chunks;
  }

  /** Refinements of already-arrived segments (the old `replaced` counter). */
  get replaced(): number {
    return this._refined;
  }

  /**
   * Load the final reconstruction, placed like the stream's chunks are placed.
   * Called on the FIRST chunk arrival; later calls are no-ops. Rendering one
   * final file instead of the stream's own cuts is a demo decision — the cuts
   * are slices of this exact file, so what is shown per segment is identical.
   */
  loadFinal(url: string, placement: ChunkPlacement): Promise<void> {
    if (this.loadStarted) return Promise.resolve();
    this.loadStarted = true;
    this.set('loading');
    return this.viewer
      .addSplatScene(url, {
        position: [...placement.position],
        rotation: [...placement.rotation],
        scale: [...placement.scale],
        showLoadingUI: false,
        progressiveLoad: false,
        onProgress: (percent: number) => {
          if (typeof percent === 'number') this._progress = percent;
        },
      })
      .then(() => {
        this.set('ready');
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        console.error('[splat] final scene failed', url, msg);
        this.set('error', msg);
      });
  }

  /** Record one delivered chunk. Pure bookkeeping — geometry is the final
   *  scene; this feeds the minimap, the camera framing and the panels. */
  noteChunk(chunk: SplatChunkInput): void {
    this._chunks += 1;
    const prev = this.bySegment.get(chunk.segment);
    if (prev) {
      if (chunk.level > prev.level) {
        prev.level = chunk.level;
        this._refined += 1;
      }
      prev.final = prev.final || chunk.final;
      return;
    }
    const entry: LoadedScene = {
      segment: chunk.segment,
      level: chunk.level,
      final: chunk.final,
      center: [...chunk.center],
      arrivedAt: performance.now(),
    };
    this.order.push(entry);
    this.bySegment.set(chunk.segment, entry);
  }

  /**
   * World-space box of the splats actually on screen, in scene units.
   *
   * This library keeps splat positions in data textures and draws them from a
   * single instanced quad, so Box3.setFromObject returns the quad. Ask the
   * mesh for the centres instead, sampling rather than reading a million.
   */
  worldBounds(
    sampleLimit = 20_000,
  ): { min: [number, number, number]; max: [number, number, number]; splats: number } | null {
    const mesh = this.viewer.splatMesh;
    const count = mesh?.getSplatCount?.() ?? 0;
    if (!mesh || count === 0) return null;
    const step = Math.max(1, Math.floor(count / sampleLimit));
    const c = new THREE.Vector3();
    const box = new THREE.Box3();
    for (let i = 0; i < count; i += step) {
      mesh.getSplatCenter(i, c, true);
      box.expandByPoint(c);
    }
    if (box.isEmpty()) return null;
    return {
      min: [box.min.x, box.min.y, box.min.z],
      max: [box.max.x, box.max.y, box.max.z],
      splats: count,
    };
  }

  /**
   * The splat centre nearest to a ray (world space), for click-measuring.
   * Prefers the FIRST hit along the ray over the globally closest centre, so
   * clicking a person picks the person and not the wall behind them.
   */
  pick(
    origin: THREE.Vector3,
    dir: THREE.Vector3,
  ): { point: [number, number, number]; rayDist: number } | null {
    const mesh = this.viewer.splatMesh;
    const count = mesh?.getSplatCount?.() ?? 0;
    if (!mesh || count === 0) return null;
    const c = new THREE.Vector3();
    const rel = new THREE.Vector3();
    let bestT = Infinity;
    let bestD = Infinity;
    const best = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      mesh.getSplatCenter(i, c, true);
      rel.subVectors(c, origin);
      const t = rel.dot(dir);
      if (t < 0.5 || t > bestT) continue;
      const d2 = rel.lengthSq() - t * t;
      // Within 25 cm of the ray counts as "clicked"; take the nearest such.
      if (d2 < 0.0625) {
        bestT = t;
        bestD = Math.sqrt(Math.max(0, d2));
        best.copy(c);
      }
    }
    if (!Number.isFinite(bestT)) return null;
    return { point: [best.x, best.y, best.z], rayDist: bestD };
  }

  /** A sample of splat centres in world space, for the checks. */
  sampleCenters(limit = 4000): Array<[number, number, number]> {
    const mesh = this.viewer.splatMesh;
    const count = mesh?.getSplatCount?.() ?? 0;
    if (!mesh || count === 0) return [];
    const step = Math.max(1, Math.floor(count / limit));
    const c = new THREE.Vector3();
    const out: Array<[number, number, number]> = [];
    for (let i = 0; i < count; i += step) {
      mesh.getSplatCenter(i, c, true);
      out.push([c.x, c.y, c.z]);
    }
    return out;
  }

  /** What has arrived and where. The minimap draws these so the operator can
   *  see how far the reconstruction has got along the track. */
  loadedChunks(): Array<{ segment: number; level: number; center: [number, number, number] }> {
    return this.order.map((e) => ({
      segment: e.segment,
      level: e.level,
      center: [...e.center],
    }));
  }

  /** Arrived segments in arrival order. */
  get scenes(): readonly LoadedScene[] {
    return this.order;
  }

  /** Highest level delivered per segment. */
  get segmentLevels(): Record<number, number> {
    const out: Record<number, number> = {};
    for (const [seg, e] of this.bySegment) out[seg] = e.level;
    return out;
  }

  onStatus(cb: StatusCb): void {
    this.cbs.push(cb);
    cb(this._status);
  }

  private set(status: SplatStatus, detail?: string): void {
    this._status = status;
    for (const cb of this.cbs) cb(status, detail);
  }

  dispose(): void {
    try {
      void this.viewer.dispose();
    } catch {
      /* ignore */
    }
    this.parent.remove(this.viewer);
  }
}
