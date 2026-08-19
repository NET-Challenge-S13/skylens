// Real Gaussian-splat layer for the STATUS viewer.
//
// Wraps @mkkellogg/gaussian-splats-3d's DropInViewer, which is a THREE.Group
// that renders itself inside our existing renderer/camera/loop (it hooks
// onBeforeRender), so we just add it to the scene. `sharedMemoryForWorkers:false`
// keeps it working without COOP/COEP cross-origin-isolation headers.
//
// The scene is not one file: it accumulates the DELAY-PATTERN stream (see
// protocol.ts). Each chunk carries a segment and a refinement level, so this
// class keeps at most one scene per segment — a higher level for a segment
// replaces the lower one already rendering, while other segments stay put.

import type * as THREE from 'three';
import { DropInViewer } from '@mkkellogg/gaussian-splats-3d';

export type SplatStatus = 'idle' | 'loading' | 'ready' | 'error';

/** Placement of one chunk, in scene units. Structurally a protocol SplatAlign
 *  minus the GPS anchor, which the caller has already resolved. */
export interface ChunkPlacement {
  position: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
}

/** What the viewer needs to place one delivered chunk. */
export interface SplatChunkInput {
  url: string;
  align: ChunkPlacement;
  segment: number;
  level: number;
}

type StatusCb = (status: SplatStatus, detail?: string) => void;

/** One loaded scene, in the order the underlying viewer holds it. The splat
 *  shader reads this array positionally: viewer scene i is order[i], which is
 *  what lets the reveal fade a SEGMENT rather than the whole mesh. */
export interface LoadedScene {
  segment: number;
  level: number;
  /** Where this chunk was placed, in scene units — the board's own frame, the
   *  same one the aircraft and the detection pins live in. Kept so a check can
   *  ask whether the reconstruction landed on the flight. */
  position: [number, number, number];
  /** performance.now() when this segment FIRST became visible. A refinement of
   *  an already-visible segment inherits it, so replacing level 1 with level 2
   *  does not restart the fade (which would read as a flicker). */
  arrivedAt: number;
}

export class SplatScene {
  private readonly viewer: DropInViewer;
  private _status: SplatStatus = 'idle';
  private _progress = 0;
  private _chunks = 0;
  private _replaced = 0;
  private readonly cbs: StatusCb[] = [];
  private readonly parent: THREE.Scene;
  private _skipped = 0;
  private _removeFailed = 0;
  /** Loaded scenes in viewer scene-index order (index i = viewer scene i). */
  private readonly order: LoadedScene[] = [];
  private readonly bySegment = new Map<number, LoadedScene>();
  /** First-arrival time per segment, so refinements keep the original fade. */
  private readonly segmentArrival = new Map<number, number>();
  /** Highest level queued OR loaded per segment, so a level that has already
   *  been overtaken never gets loaded just to be replaced a moment later. */
  private readonly desired = new Map<number, number>();
  private readonly queue: SplatChunkInput[] = [];
  private chain: Promise<void> = Promise.resolve();

  constructor(parent: THREE.Scene) {
    this.parent = parent;
    this.viewer = new DropInViewer({
      sharedMemoryForWorkers: false,
      // CPU sort in a worker — more compatible across GPUs/headless than the
      // GPU-accelerated path, and fine for a single static scene.
      gpuAcceleratedSort: false,
      // Segments are added and swapped while the board runs, so the mesh is not
      // a one-shot static build.
      dynamicScene: true,
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

  /** The splat mesh's ShaderMaterial once loaded, else null. Note this changes
   *  identity whenever a scene is removed: the library rebuilds the mesh. */
  get material(): THREE.ShaderMaterial | null {
    return this.viewer.splatMesh?.material ?? null;
  }

  /** Download progress percent (0..100). */
  get progress(): number {
    return this._progress;
  }

  /** Number of splat chunks successfully added (levels included). */
  get chunks(): number {
    return this._chunks;
  }

  /** What is on screen and where. The minimap draws these so the operator can
   *  see how far the reconstruction has got along the track. */
  loadedChunks(): Array<{ segment: number; level: number; position: [number, number, number] }> {
    return this.order.map((e) => ({
      segment: e.segment,
      level: e.level,
      position: [...e.position],
    }));
  }

  /** Lower-level scenes dropped because a refinement replaced them. */
  get replaced(): number {
    return this._replaced;
  }

  /** Levels dropped from the queue because a better one had already arrived. */
  get skipped(): number {
    return this._skipped;
  }

  /** Superseded levels the renderer refused to unload (see addChunk). */
  get removeFailed(): number {
    return this._removeFailed;
  }

  /** Loaded scenes in viewer scene-index order — the reveal maps a splat's
   *  sceneIndex to its segment's arrival time through this. */
  get scenes(): readonly LoadedScene[] {
    return this.order;
  }

  /** Highest level currently rendering per segment. */
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

  /**
   * Queue one chunk of the delay-pattern stream. Ingestion runs one at a time
   * (the underlying loader refuses overlapping loads/unloads), so this returns
   * when the chunk has been ingested — or dropped, if a better level for the
   * same segment arrived while it waited.
   */
  addChunk(chunk: SplatChunkInput): Promise<void> {
    // Out-of-order, duplicate, or already-overtaken delivery.
    if (chunk.level <= (this.desired.get(chunk.segment) ?? 0)) return Promise.resolve();
    this.desired.set(chunk.segment, chunk.level);
    this.queue.push(chunk);
    this.chain = this.chain.then(() => this.pump());
    return this.chain;
  }

  private async pump(): Promise<void> {
    const chunk = this.queue.shift();
    if (!chunk) return;

    const prev = this.bySegment.get(chunk.segment);
    if (prev && prev.level >= chunk.level) return;
    // A refinement of the same segment landed while this one sat in the queue.
    // Loading it now would only be undone by the next queue entry.
    if (chunk.level < (this.desired.get(chunk.segment) ?? 0)) {
      this._skipped += 1;
      return;
    }

    if (this._status === 'idle') this.set('loading');
    try {
      await this.idle();
      await this.viewer.addSplatScene(chunk.url, {
        position: [...chunk.align.position],
        rotation: [...chunk.align.rotation],
        scale: [...chunk.align.scale],
        showLoadingUI: false,
        // Whole-chunk load. Progressive load inside a chunk would resolve the
        // promise while the file is still streaming, and the next add/remove
        // would be rejected as "another load in progress". Progressiveness is
        // the delay pattern's job here, at segment × level granularity.
        progressiveLoad: false,
        onProgress: (percent) => {
          if (typeof percent === 'number') this._progress = percent;
        },
      });
      const arrivedAt = this.segmentArrival.get(chunk.segment) ?? performance.now();
      this.segmentArrival.set(chunk.segment, arrivedAt);
      const entry: LoadedScene = {
        segment: chunk.segment,
        level: chunk.level,
        arrivedAt,
        position: [...chunk.align.position],
      };
      this.order.push(entry);
      this.bySegment.set(chunk.segment, entry);
      this._chunks += 1;

      if (prev) {
        const idx = this.order.indexOf(prev);
        if (idx >= 0) {
          await this.idle();
          try {
            await this.viewer.removeSplatScene(idx, false);
            this.order.splice(idx, 1);
            this._replaced += 1;
          } catch (e: unknown) {
            // The library rebuilds its splat tree asynchronously, so a removal
            // can race a sort still referencing the old mesh. The refinement is
            // already rendering at this point; failing to drop the superseded
            // level costs memory, not correctness, and must not break the queue.
            this._removeFailed += 1;
            console.warn('[splat] superseded level not removed', e);
          }
        }
      }
      this.set('ready');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('[splat] chunk failed', chunk.url, msg);
      this.set('error', msg);
    }
  }

  /** Wait out any load/unload still settling; the viewer rejects overlapping ones. */
  private async idle(): Promise<void> {
    for (let i = 0; i < 400 && this.viewer.viewer.isLoadingOrUnloading(); i++) {
      await new Promise((r) => setTimeout(r, 25));
    }
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
