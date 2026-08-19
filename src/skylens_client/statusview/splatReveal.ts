// Splat shader effects for the situation board: (1) a floater CLIP that discards
// splats outside the robust scene box — photo/SfM reconstructions carry a lot of
// far background gaussians that otherwise fog up the view — and (2) the
// progressive REVEAL, which is now driven by ARRIVAL.
//
// WHAT CHANGED AND WHY. This used to open a top-down coverage mask from
// `state.visited`, the control tower's simulated drone trail. That made two
// independent things decide whether a piece of the scene was visible: the trail
// said "the drone has been here" while the stream said "this segment has been
// reconstructed". They could disagree in both directions — a chunk could land
// inside a still-masked area (reconstructed geometry held back by a simulation),
// or the mask could open over a segment that had not arrived (a promise of
// geometry that is not there). COMPONENTS.md §8 settles it: "보인다 = 복원되어
// 도착했다"가 하나의 진실이 된다.
//
// So there is no mask any more. The splat mesh only ever contains chunks that
// HAVE arrived, which makes visibility automatic; what is left for this class is
// the nicety of fading each segment in over CONFIG.reveal.fadeSeconds instead of
// popping it. The fade is per SEGMENT, keyed on the splat's `sceneIndex` — the
// library exposes it in the vertex shader for dynamic scenes, and SplatScene
// keeps its scene order aligned with it.

import * as THREE from 'three';
import { CONFIG } from '../../shared/viewer/config';
import type { LoadedScene } from './splatScene.ts';

/** Uniform array size. The delay pattern holds one scene per segment plus at
 *  most one in-flight refinement, so this is far above anything real. */
const MAX_SCENES = 64;

export class SplatReveal {
  private readonly fades = new Float32Array(MAX_SCENES);

  private readonly uniforms: {
    uSceneFade: { value: Float32Array };
    uRevealEnabled: { value: number };
    uClipMin: { value: THREE.Vector3 };
    uClipMax: { value: THREE.Vector3 };
    uClipEnabled: { value: number };
  };

  constructor(bounds: THREE.Box3) {
    const size = new THREE.Vector3();
    bounds.getSize(size);
    // Clip box: the robust building bounds with a little headroom.
    const pad = new THREE.Vector3(size.x, size.y, size.z).multiplyScalar(0.06);

    this.uniforms = {
      uSceneFade: { value: this.fades },
      uRevealEnabled: { value: 1 },
      uClipMin: { value: bounds.min.clone().sub(pad) },
      uClipMax: { value: bounds.max.clone().add(pad) },
      // OFF by default. This filter exists to drop the far background gaussians
      // a photo reconstruction carries, and it was written against a scene that
      // sat near the origin. Once chunks are placed on the ground the aircraft
      // flew, it has thrown away real geometry — an empty board is far worse
      // than a few floaters, so it stays off until it can be shown to cut only
      // what it is meant to. `?clip=on` turns it back on for that work.
      uClipEnabled: { value: 0 },
    };
  }

  /**
   * Move the floater clip to the ground the mission actually covers.
   *
   * The box starts from the placeholder cloud, which is all there is before any
   * geometry arrives. Once chunks are placed along a route they can span
   * hundreds of metres, and a box sized for the placeholder discards every one
   * of them — the board then holds a quarter of a million splats and draws
   * nothing, which looks exactly like geometry that never arrived.
   */
  setClip(min: THREE.Vector3, max: THREE.Vector3): void {
    this.uniforms.uClipMin.value.copy(min);
    this.uniforms.uClipMax.value.copy(max);
  }

  /** Turn the floater clip on (?clip=on). Off by default — see the constructor. */
  setClipEnabled(on: boolean): void {
    this.uniforms.uClipEnabled.value = on ? 1 : 0;
  }

  /** The clip box in force, for working out why a full scene draws nothing. */
  debugClip(): [THREE.Vector3, THREE.Vector3] {
    return [this.uniforms.uClipMin.value, this.uniforms.uClipMax.value];
  }

  /** `?reveal=off` renders every arrived chunk at full opacity immediately. */
  setRevealEnabled(on: boolean): void {
    this.uniforms.uRevealEnabled.value = on ? 1 : 0;
  }

  /**
   * Advance the per-segment fades. `scenes` is SplatScene's scene-index-ordered
   * list, so entry i is the splat mesh's scene i; `now` is performance.now().
   */
  update(scenes: readonly LoadedScene[], now: number): void {
    const fadeMs = Math.max(1, CONFIG.reveal.fadeSeconds * 1000);
    const n = Math.min(scenes.length, MAX_SCENES);
    for (let i = 0; i < n; i++) {
      this.fades[i] = Math.min(1, Math.max(0, (now - scenes[i].arrivedAt) / fadeMs));
    }
    // Scenes that have been removed (a superseded level) must not leave a stale
    // fade behind for whatever lands at that index next.
    for (let i = n; i < MAX_SCENES; i++) this.fades[i] = 0;
  }

  /** True once `segment`'s chunk has landed — the board's single visibility
   *  truth, reused to gate detection markers on the segment they were found in. */
  isSegmentRevealed(scenes: readonly LoadedScene[], segment: number): boolean {
    return scenes.some((s) => s.segment === segment);
  }

  /** Patch a splat ShaderMaterial: clip floaters (always) + per-segment fade. */
  attachTo(material: THREE.ShaderMaterial): void {
    const store = material as THREE.ShaderMaterial & { userData: { splatPatched?: boolean } };
    if (store.userData.splatPatched) return;
    store.userData.splatPatched = true;

    const prev = material.onBeforeCompile;
    material.onBeforeCompile = (shader, renderer) => {
      prev?.(shader, renderer);
      shader.uniforms.uSceneFade = this.uniforms.uSceneFade;
      shader.uniforms.uRevealEnabled = this.uniforms.uRevealEnabled;
      shader.uniforms.uClipMin = this.uniforms.uClipMin;
      shader.uniforms.uClipMax = this.uniforms.uClipMax;
      shader.uniforms.uClipEnabled = this.uniforms.uClipEnabled;

      shader.vertexShader = shader.vertexShader
        .replace(
          'attribute uint splatIndex;',
          `attribute uint splatIndex;
           uniform float uSceneFade[${MAX_SCENES}];
           uniform vec3 uClipMin;
           uniform vec3 uClipMax;
           uniform float uClipEnabled;
           varying float vReveal;`,
        )
        .replace(
          'vColor = uintToRGBAVec(sampledCenterColor.r);',
          // `sceneIndex` is declared by the library just above this line for
          // dynamic scenes, so the fade can be looked up per segment.
          `vColor = uintToRGBAVec(sampledCenterColor.r);
           {
             // Where this splat really is. NOT modelMatrix: the mesh is built
             // with dynamicScene, so the library places each scene with its own
             // transforms[sceneIndex] and leaves the mesh matrix at identity.
             // Testing modelMatrix compared a splat's LOCAL coordinates against
             // a box in world metres, which passed only while chunks happened
             // to sit near the origin — and discarded every splat the moment
             // they were placed on the ground the aircraft flew.
             if (uClipEnabled > 0.5) {
               vec4 wc = transform * vec4(splatCenter, 1.0);
               if (wc.x < uClipMin.x || wc.x > uClipMax.x ||
                   wc.y < uClipMin.y || wc.y > uClipMax.y ||
                   wc.z < uClipMin.z || wc.z > uClipMax.z) {
                 gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                 return;
               }
             }
             int si = int(min(sceneIndex, uint(${MAX_SCENES - 1})));
             vReveal = uSceneFade[si];
           }`,
        );

      shader.fragmentShader = shader.fragmentShader
        .replace(
          'varying vec2 vPosition;',
          `varying vec2 vPosition;
           varying float vReveal;
           uniform float uRevealEnabled;`,
        )
        .replace(
          'gl_FragColor = vec4(color.rgb, opacity);',
          // Not-yet-arrived geometry is absent, not ghosted: there is nothing
          // honest to draw for a segment that has not been reconstructed.
          `float vis = uRevealEnabled > 0.5 ? clamp(vReveal, 0.0, 1.0) : 1.0;
           gl_FragColor = vec4(color.rgb, opacity * vis);`,
        );
    };
    material.needsUpdate = true;
  }
}
