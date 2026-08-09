// Splat shader effects for RECON: (1) a floater CLIP that discards splats outside
// the robust scene box — photo/SfM splats have lots of far background gaussians
// that otherwise fog up the view — and (2) an optional progressive REVEAL mask
// driven by a top-down coverage texture (drones "scan" the building in).
//
// Both are injected into the splat material's shader via onBeforeCompile. The
// clip is always on; the reveal is toggled by a uniform.

import * as THREE from 'three';
import type { Visited } from '../../skylens_core/types';
import { CONFIG } from '../../skylens_core/config';

const RES = 256;

export class SplatReveal {
  private readonly progress: Float32Array;
  private readonly started: Uint8Array;
  private readonly tex: THREE.DataTexture;
  private readonly worldMin = new THREE.Vector2();
  private readonly worldSize = new THREE.Vector2();
  private consumed = 0;

  private readonly uniforms: {
    uRevealTex: { value: THREE.Texture };
    uRevealMin: { value: THREE.Vector2 };
    uRevealSize: { value: THREE.Vector2 };
    uRevealGhost: { value: number };
    uRevealEnabled: { value: number };
    uClipMin: { value: THREE.Vector3 };
    uClipMax: { value: THREE.Vector3 };
  };

  constructor(bounds: THREE.Box3) {
    const size = new THREE.Vector3();
    bounds.getSize(size);
    // Clip box: the robust building bounds with a little headroom.
    const pad = new THREE.Vector3(size.x, size.y, size.z).multiplyScalar(0.06);
    const clipMin = bounds.min.clone().sub(pad);
    const clipMax = bounds.max.clone().add(pad);

    // Coverage grid spans the XZ footprint, padded by the reveal radius.
    const cpad = CONFIG.reveal.radius;
    this.worldMin.set(bounds.min.x - cpad, bounds.min.z - cpad);
    this.worldSize.set(
      Math.max(1, bounds.max.x - bounds.min.x + cpad * 2),
      Math.max(1, bounds.max.z - bounds.min.z + cpad * 2),
    );

    this.progress = new Float32Array(RES * RES);
    this.started = new Uint8Array(RES * RES);
    this.tex = new THREE.DataTexture(this.progress, RES, RES, THREE.RedFormat, THREE.FloatType);
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.needsUpdate = true;

    this.uniforms = {
      uRevealTex: { value: this.tex },
      uRevealMin: { value: this.worldMin },
      uRevealSize: { value: this.worldSize },
      uRevealGhost: { value: CONFIG.reveal.ghostOpacity },
      uRevealEnabled: { value: 0 },
      uClipMin: { value: clipMin },
      uClipMax: { value: clipMax },
    };
  }

  setRevealEnabled(on: boolean): void {
    this.uniforms.uRevealEnabled.value = on ? 1 : 0;
  }

  private texel(x: number, z: number): { tx: number; tz: number } {
    const u = (x - this.worldMin.x) / this.worldSize.x;
    const v = (z - this.worldMin.y) / this.worldSize.y;
    return {
      tx: Math.min(RES - 1, Math.max(0, Math.floor(u * RES))),
      tz: Math.min(RES - 1, Math.max(0, Math.floor(v * RES))),
    };
  }

  update(visitedLagged: Visited[], dt: number): void {
    let dirty = false;
    if (this.consumed < visitedLagged.length) {
      const rTexX = Math.ceil((CONFIG.reveal.radius / this.worldSize.x) * RES);
      const rTexZ = Math.ceil((CONFIG.reveal.radius / this.worldSize.y) * RES);
      const rSq = Math.max(rTexX, rTexZ) ** 2;
      for (let vi = this.consumed; vi < visitedLagged.length; vi++) {
        const spot = visitedLagged[vi];
        const { tx, tz } = this.texel(spot.pos[0], spot.pos[2]);
        for (let dz = -rTexZ; dz <= rTexZ; dz++) {
          const z = tz + dz;
          if (z < 0 || z >= RES) continue;
          for (let dx = -rTexX; dx <= rTexX; dx++) {
            const x = tx + dx;
            if (x < 0 || x >= RES) continue;
            if (dx * dx + dz * dz > rSq) continue;
            this.started[z * RES + x] = 1;
          }
        }
      }
      this.consumed = visitedLagged.length;
      dirty = true;
    }
    const rate = dt / Math.max(1e-6, CONFIG.reveal.fadeSeconds);
    for (let i = 0; i < this.progress.length; i++) {
      if (this.started[i] && this.progress[i] < 1) {
        this.progress[i] = Math.min(1, this.progress[i] + rate);
        dirty = true;
      }
    }
    if (dirty) this.tex.needsUpdate = true;
  }

  isAreaRevealed(pos: [number, number, number]): boolean {
    const { tx, tz } = this.texel(pos[0], pos[2]);
    return this.progress[tz * RES + tx] > 0.02;
  }

  /** Patch a splat ShaderMaterial: clip floaters (always) + reveal (toggled). */
  attachTo(material: THREE.ShaderMaterial): void {
    const store = material as THREE.ShaderMaterial & { userData: { splatPatched?: boolean } };
    if (store.userData.splatPatched) return;
    store.userData.splatPatched = true;

    const prev = material.onBeforeCompile;
    material.onBeforeCompile = (shader, renderer) => {
      prev?.(shader, renderer);
      shader.uniforms.uRevealTex = this.uniforms.uRevealTex;
      shader.uniforms.uRevealMin = this.uniforms.uRevealMin;
      shader.uniforms.uRevealSize = this.uniforms.uRevealSize;
      shader.uniforms.uRevealGhost = this.uniforms.uRevealGhost;
      shader.uniforms.uRevealEnabled = this.uniforms.uRevealEnabled;
      shader.uniforms.uClipMin = this.uniforms.uClipMin;
      shader.uniforms.uClipMax = this.uniforms.uClipMax;

      shader.vertexShader = shader.vertexShader
        .replace(
          'attribute uint splatIndex;',
          `attribute uint splatIndex;
           uniform sampler2D uRevealTex;
           uniform vec2 uRevealMin;
           uniform vec2 uRevealSize;
           uniform vec3 uClipMin;
           uniform vec3 uClipMax;
           varying float vReveal;`,
        )
        .replace(
          'vColor = uintToRGBAVec(sampledCenterColor.r);',
          `vColor = uintToRGBAVec(sampledCenterColor.r);
           {
             vec4 wc = modelMatrix * vec4(splatCenter, 1.0);
             if (wc.x < uClipMin.x || wc.x > uClipMax.x ||
                 wc.y < uClipMin.y || wc.y > uClipMax.y ||
                 wc.z < uClipMin.z || wc.z > uClipMax.z) {
               gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
               return;
             }
             vec2 ruv = (wc.xz - uRevealMin) / uRevealSize;
             float rev = 0.0;
             if (ruv.x >= 0.0 && ruv.x <= 1.0 && ruv.y >= 0.0 && ruv.y <= 1.0) {
               rev = texture(uRevealTex, ruv).r;
             }
             vReveal = rev;
           }`,
        );

      shader.fragmentShader = shader.fragmentShader
        .replace(
          'varying vec2 vPosition;',
          `varying vec2 vPosition;
           varying float vReveal;
           uniform float uRevealGhost;
           uniform float uRevealEnabled;`,
        )
        .replace(
          'gl_FragColor = vec4(color.rgb, opacity);',
          `float vis = uRevealEnabled > 0.5
             ? mix(uRevealGhost, 1.0, clamp(vReveal, 0.0, 1.0))
             : 1.0;
           gl_FragColor = vec4(color.rgb, opacity * vis);`,
        );
    };
    material.needsUpdate = true;
  }
}
