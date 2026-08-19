// Progressive REVEAL over the single final splat scene.
//
// The board renders ONE static scene — the final-quality reconstruction — and
// the delay-pattern stream drives what of it is VISIBLE: a segment of the
// scene fades in when the core reports that segment reconstructed, and
// brightens as refinement levels land. The geometry itself never changes;
// arrival messages only move opacity. (The previous design loaded each
// segment × level PLY into a dynamic multi-scene mesh; the constant
// add/remove churn is what kept breaking rendering, and the geometry it
// streamed was just cuts of the very file this renders.)
//
// Segments are SLABS of the scene along its principal axis — exactly how
// split_segments.py cut the assets — so a splat's segment is recovered in the
// vertex shader from its world position: project onto the axis, compare with
// the cut boundaries. Axis/origin/boundaries come from the manifest, mapped
// into world space by the same placement that positioned the scene.

import * as THREE from 'three';
import { CONFIG } from '../../shared/viewer/config';

/** Uniform array sizes. The demo cuts the scene into 4 slabs; leave headroom. */
const MAX_REGIONS = 8;

/** How solid a slab renders per refinement level — the "delay pattern feel":
 *  a fresh segment appears translucent and firms up as levels land. */
function alphaForLevel(level: number, final: boolean): number {
  if (final || level >= 4) return 1.0;
  if (level >= 3) return 0.95;
  if (level >= 2) return 0.8;
  return 0.62;
}

export class SplatReveal {
  private readonly fades = new Float32Array(MAX_REGIONS);
  private readonly targets = new Float32Array(MAX_REGIONS);
  /** What each slab WANTS to show (from arrivals). A slab only gets to fade
   *  once the slab ahead of it is substantially done — the delay pattern
   *  reads as the reconstruction advancing down the corridor, not as the
   *  whole strip flickering in at assorted opacities. */
  private readonly pending = new Float32Array(MAX_REGIONS);
  private regionCount = 1;

  private readonly uniforms = {
    uFade: { value: this.fades },
    uRevealEnabled: { value: 1 },
    uAxis: { value: new THREE.Vector3(1, 0, 0) },
    uOrigin: { value: new THREE.Vector3() },
    uBounds: { value: new Float32Array(MAX_REGIONS - 1) },
    uRegionCount: { value: 1 },
  };

  /**
   * Anchor the reveal regions to the placed scene.
   *
   * `origin`/`axis` are the manifest's split geometry brought into WORLD space
   * by the scene placement; `boundaries` are the manifest's cut positions —
   * projections along the axis RELATIVE to the origin, which a rigid placement
   * leaves untouched (only a scale multiplies them, and the caller applies it).
   */
  setFrame(origin: THREE.Vector3, axis: THREE.Vector3, boundaries: number[]): void {
    this.uniforms.uOrigin.value.copy(origin);
    this.uniforms.uAxis.value.copy(axis).normalize();
    const n = Math.min(boundaries.length, MAX_REGIONS - 1);
    for (let i = 0; i < n; i++) this.uniforms.uBounds.value[i] = boundaries[i];
    this.regionCount = n + 1;
    this.uniforms.uRegionCount.value = this.regionCount;
  }

  /** Number of slabs the scene is cut into (from the manifest). */
  get regions(): number {
    return this.regionCount;
  }

  /** A segment's chunk has landed: queue that slab's fade. Core segments
   *  beyond the asset count wrap, mirroring the server's asset pick. */
  noteArrival(coreSegment: number, level: number, final: boolean): void {
    const region = coreSegment % this.regionCount;
    const target = alphaForLevel(level, final);
    if (target > this.pending[region]) this.pending[region] = target;
  }

  /** `?reveal=off` renders the whole arrived scene at full opacity. */
  setRevealEnabled(on: boolean): void {
    this.uniforms.uRevealEnabled.value = on ? 1 : 0;
  }

  /** Ease each slab toward its target opacity. `dt` in seconds. */
  update(dt: number): void {
    // Chain: a slab's queued reveal is released only once the nearest slab
    // AHEAD of it (lower index = earlier in the flight) has faded most of the
    // way in. Arrivals are not delayed — only their appearance is ordered.
    for (let i = 0; i < this.regionCount; i++) {
      if (this.pending[i] <= this.targets[i]) continue;
      let prev = i - 1;
      while (prev >= 0 && this.pending[prev] <= 0) prev--;
      if (prev < 0 || this.fades[prev] >= CONFIG.reveal.chainGate) {
        this.targets[i] = this.pending[i];
      }
    }
    const step = dt / Math.max(0.05, CONFIG.reveal.fadeSeconds);
    for (let i = 0; i < MAX_REGIONS; i++) {
      const diff = this.targets[i] - this.fades[i];
      if (diff <= 0) continue;
      this.fades[i] = Math.min(this.targets[i], this.fades[i] + step);
    }
  }

  /** Per-slab opacity now, for the checks. */
  get fadeState(): { fades: number[]; targets: number[] } {
    const n = this.regionCount;
    return {
      fades: Array.from(this.fades.slice(0, n)),
      targets: Array.from(this.targets.slice(0, n)),
    };
  }

  /**
   * Patch the splat ShaderMaterial with the reveal.
   *
   * The patch is string replacement against the library's built shader, so it
   * can silently rot when the library changes. Guard: if any anchor fails to
   * match, DON'T patch at all — an unpatched board shows the full scene, a
   * half-patched one links no program and shows nothing. Only the VERTEX
   * shader is touched (the fade multiplies into vColor.a, which the fragment
   * already applies), halving the surface that can go stale.
   */
  attachTo(material: THREE.ShaderMaterial): void {
    const store = material as THREE.ShaderMaterial & { userData: { splatPatched?: boolean } };
    if (store.userData.splatPatched) return;
    store.userData.splatPatched = true;

    const DECL_ANCHOR = 'attribute uint splatIndex;';
    const COLOR_ANCHOR = 'vColor = uintToRGBAVec(sampledCenterColor.r);';

    const prev = material.onBeforeCompile;
    material.onBeforeCompile = (shader, renderer) => {
      prev?.(shader, renderer);
      if (
        !shader.vertexShader.includes(DECL_ANCHOR) ||
        !shader.vertexShader.includes(COLOR_ANCHOR)
      ) {
        console.error(
          '[reveal] splat shader anchors not found — rendering without the reveal',
        );
        return;
      }
      shader.uniforms.uFade = this.uniforms.uFade;
      shader.uniforms.uRevealEnabled = this.uniforms.uRevealEnabled;
      shader.uniforms.uAxis = this.uniforms.uAxis;
      shader.uniforms.uOrigin = this.uniforms.uOrigin;
      shader.uniforms.uBounds = this.uniforms.uBounds;
      shader.uniforms.uRegionCount = this.uniforms.uRegionCount;

      shader.vertexShader = shader.vertexShader
        .replace(
          DECL_ANCHOR,
          `${DECL_ANCHOR}
           uniform float uFade[${MAX_REGIONS}];
           uniform float uRevealEnabled;
           uniform vec3 uAxis;
           uniform vec3 uOrigin;
           uniform float uBounds[${MAX_REGIONS - 1}];
           uniform int uRegionCount;`,
        )
        .replace(
          COLOR_ANCHOR,
          `${COLOR_ANCHOR}
           {
             // Which slab of the scene this splat is in: its world position
             // projected on the split axis vs the manifest's cut boundaries.
             vec3 worldPos = (modelMatrix * vec4(splatCenter, 1.0)).xyz;
             float proj = dot(worldPos - uOrigin, uAxis);
             int region = 0;
             for (int i = 0; i < ${MAX_REGIONS - 1}; i++) {
               if (i < uRegionCount - 1 && proj > uBounds[i]) region = i + 1;
             }
             float vis = uRevealEnabled > 0.5 ? uFade[region] : 1.0;
             if (vis < 0.004) {
               gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
               return;
             }
             vColor.a *= vis;
           }`,
        );
    };
    material.needsUpdate = true;
  }
}
