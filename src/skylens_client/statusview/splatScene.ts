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
import type { SplatAlign } from '../../skylens_core/protocol.ts';

export type SplatStatus = 'idle' | 'loading' | 'ready' | 'error';

/** What the viewer needs to place one delivered chunk. */
export interface SplatChunkInput {
  url: string;
  align: SplatAlign;
  segment: number;
  level: number;
}

type StatusCb = (status: SplatStatus, detail?: string) => void;

/** One loaded scene, in the order the underlying viewer holds it. */
interface Loaded {
  segment: number;
  level: number;
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
  /** Loaded scenes in viewer scene-index order (index i = viewer scene i). */
  private readonly order: Loaded[] = [];
  private readonly bySegment = new Map<number, Loaded>();
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

  /** Lower-level scenes dropped because a refinement replaced them. */
  get replaced(): number {
    return this._replaced;
  }

  /** Levels dropped from the queue because a better one had already arrived. */
  get skipped(): number {
    return this._skipped;
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
      const entry: Loaded = { segment: chunk.segment, level: chunk.level };
      this.order.push(entry);
      this.bySegment.set(chunk.segment, entry);
      this._chunks += 1;

      if (prev) {
        const idx = this.order.indexOf(prev);
        if (idx >= 0) {
          await this.idle();
          await this.viewer.removeSplatScene(idx, false);
          this.order.splice(idx, 1);
          this._replaced += 1;
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
