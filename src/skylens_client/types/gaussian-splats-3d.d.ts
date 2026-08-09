// Minimal ambient types for @mkkellogg/gaussian-splats-3d (ships no types).
// Only the surface we use: DropInViewer added into an existing THREE.Scene.

declare module '@mkkellogg/gaussian-splats-3d' {
  import type { Group } from 'three';

  export interface AddSplatSceneOptions {
    position?: number[];
    /** Quaternion [x, y, z, w]. */
    rotation?: number[];
    scale?: number[];
    splatAlphaRemovalThreshold?: number;
    showLoadingUI?: boolean;
    progressiveLoad?: boolean;
    onProgress?: (percent: number, percentLabel?: string, status?: unknown) => void;
  }

  export interface DropInViewerOptions {
    sharedMemoryForWorkers?: boolean;
    gpuAcceleratedSort?: boolean;
    dynamicScene?: boolean;
    antialiased?: boolean;
    [key: string]: unknown;
  }

  /** A THREE.Group you add to your scene; renders in your own loop. */
  export class DropInViewer extends Group {
    constructor(options?: DropInViewerOptions);
    addSplatScene(path: string, options?: AddSplatSceneOptions): Promise<void>;
    getSceneCount(): number;
    dispose(): Promise<void> | void;
    /** The mesh rendering the splats (present after a scene loads). */
    splatMesh: { material: import('three').ShaderMaterial } | null;
  }

  export class Viewer {
    constructor(options?: Record<string, unknown>);
  }

  /** A parsed splat buffer. We only read centers/colors to derive a point cloud. */
  export interface SplatBuffer {
    getSplatCount(): number;
    getSplatCenter(
      index: number,
      outCenter: { x: number; y: number; z: number },
      transform?: unknown,
    ): void;
    getSplatColor(index: number, outColor: { set(r: number, g: number, b: number, a: number): unknown }): void;
  }

  export class SplatLoader {
    static loadFromURL(
      fileName: string,
      onProgress?: (percent: number, percentLabel?: string, chunk?: unknown) => void,
      progressiveLoadToSplatBuffer?: boolean,
      onSectionBuilt?: ((...a: unknown[]) => void) | undefined,
      splatAlphaRemovalThreshold?: number,
      inMemoryCompressionLevel?: number,
      optimizeSplatData?: boolean,
      headers?: unknown,
    ): Promise<SplatBuffer>;
  }

  export const SceneRevealMode: { Instant: number; Gradual: number; Default: number };
}
