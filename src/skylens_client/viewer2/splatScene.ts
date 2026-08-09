// Real Gaussian-splat layer for the RECON viewer.
//
// Wraps @mkkellogg/gaussian-splats-3d's DropInViewer, which is a THREE.Group
// that renders itself inside our existing renderer/camera/loop (it hooks
// onBeforeRender), so we just add it to the scene. `sharedMemoryForWorkers:false`
// keeps it working without COOP/COEP cross-origin-isolation headers.
//
// This is the TEST-asset path: a public sample splat loaded from a CDN stands in
// for a real capture. The reveal/marker/camera choreography is unchanged — the
// splat is fit into the same coordinate frame as the procedural point cloud.

import type * as THREE from 'three';
import { DropInViewer } from '@mkkellogg/gaussian-splats-3d';
import type { SplatAlign } from '../../skylens_core/protocol.ts';

export type SplatStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface SplatOptions {
  url: string;
  position: number[];
  rotation: number[];
  scale: number[];
}

type StatusCb = (status: SplatStatus, detail?: string) => void;

export class SplatScene {
  private readonly viewer: DropInViewer;
  private _status: SplatStatus = 'idle';
  private _progress = 0;
  private _chunks = 0;
  private readonly cbs: StatusCb[] = [];
  private readonly parent: THREE.Scene;

  constructor(parent: THREE.Scene) {
    this.parent = parent;
    this.viewer = new DropInViewer({
      sharedMemoryForWorkers: false,
      // CPU sort in a worker — more compatible across GPUs/headless than the
      // GPU-accelerated path, and fine for a single static scene.
      gpuAcceleratedSort: false,
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

  /** Download progress percent (0..100). */
  get progress(): number {
    return this._progress;
  }

  onStatus(cb: StatusCb): void {
    this.cbs.push(cb);
    cb(this._status);
  }

  private set(status: SplatStatus, detail?: string): void {
    this._status = status;
    for (const cb of this.cbs) cb(status, detail);
  }

  load(opts: SplatOptions): void {
    this.set('loading');
    this.viewer
      .addSplatScene(opts.url, {
        position: [...opts.position],
        rotation: [...opts.rotation],
        scale: [...opts.scale],
        showLoadingUI: false,
        progressiveLoad: true,
        onProgress: (percent) => {
          if (typeof percent === 'number') this._progress = percent;
        },
      })
      .then(() => this.set('ready'))
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        this.set('error', msg);
      });
  }

  /** Number of splat chunks successfully added. */
  get chunks(): number {
    return this._chunks;
  }

  /**
   * Progressive ingestion: add one splat chunk placed by its align transform.
   * Call repeatedly as the server streams chunks. Returns when the chunk is in.
   * Callers should serialize (await) additions to avoid loader races.
   */
  addChunk(url: string, align: SplatAlign): Promise<void> {
    if (this._status === 'idle') this.set('loading');
    return this.viewer
      .addSplatScene(url, {
        position: [...align.position],
        rotation: [...align.rotation],
        scale: [...align.scale],
        showLoadingUI: false,
        progressiveLoad: true,
        onProgress: (percent) => {
          if (typeof percent === 'number') this._progress = percent;
        },
      })
      .then(() => {
        this._chunks += 1;
        this.set('ready');
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        this.set('error', msg);
      });
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
