// The point layer's own small view, under the main board.
//
// The main view stays what it is (splat scene, markers, drone follow). The
// point layer lives in a SEPARATE scene with its own camera and is drawn into
// a small viewport of the same renderer, inside a panel frame with a height
// legend. Nothing here touches the main scene, camera or controls.
//
// Camera: frames every delivered point, obliquely from the south-east, and
// widens as segments arrive, until the operator drags inside the panel; a
// double click hands the framing back.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { GeoAnchor } from '../../shared/geo.ts';
import type { PointChunk } from '../../shared/protocol.ts';
import { PointLayer } from './pointLayer.ts';
import type { PointColorMode } from './pointLayer.ts';

/** Same ramp as the point shader (pointLayer.ts), for the legend. */
function ramp(t: number): [number, number, number] {
  const c = (v: number): number => Math.min(1, Math.max(0, v));
  return [c(1.5 - Math.abs(4 * t - 3)), c(1.5 - Math.abs(4 * t - 2)), c(1.5 - Math.abs(4 * t - 1))];
}

function legendGradient(): string {
  const stops: string[] = [];
  for (let i = 0; i <= 8; i++) {
    const [r, g, b] = ramp(i / 8).map((v) => Math.round(v * 255));
    stops.push(`rgb(${r},${g},${b}) ${(i / 8) * 100}%`);
  }
  return `linear-gradient(to top, ${stops.join(', ')})`;
}

export interface PointInsetOptions {
  geoAnchor: GeoAnchor;
  enabled: boolean;
  mode?: PointColorMode;
}

export class PointInset {
  readonly layer: PointLayer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(45, 1.5, 0.5, 5000);
  private readonly controls: OrbitControls;
  private readonly frameEl: HTMLDivElement;
  private readonly viewEl: HTMLDivElement;
  private readonly legendLow: HTMLSpanElement;
  private readonly legendHigh: HTMLSpanElement;
  private readonly emptyEl: HTMLDivElement;
  /** Follow the delivered bounds until the operator drags in the panel. */
  private auto = true;
  /** Jump (not ease) to the next framing: nothing was framed yet. */
  private snap = true;
  private lastHeightPx = 0;

  constructor(host: HTMLElement, opts: PointInsetOptions) {
    this.scene.background = new THREE.Color(0x06090e);
    this.layer = new PointLayer(this.scene, {
      geoAnchor: opts.geoAnchor,
      enabled: opts.enabled,
      mode: opts.mode,
    });

    this.frameEl = document.createElement('div');
    this.frameEl.className = 'point-inset';
    this.frameEl.id = 'point-inset';
    const title = document.createElement('div');
    title.className = 'point-inset__title';
    title.textContent = '점 · 고도색';
    this.viewEl = document.createElement('div');
    this.viewEl.className = 'point-inset__view';
    this.viewEl.title = '드래그: 회전 · 휠: 확대 · 더블클릭: 전체 보기';
    this.emptyEl = document.createElement('div');
    this.emptyEl.className = 'point-inset__empty';
    this.emptyEl.textContent = '불러오는 중';
    this.viewEl.appendChild(this.emptyEl);

    const legend = document.createElement('div');
    legend.className = 'point-inset__legend';
    this.legendHigh = document.createElement('span');
    this.legendHigh.className = 'point-inset__tick point-inset__tick--high';
    const bar = document.createElement('div');
    bar.className = 'point-inset__bar';
    bar.style.background = legendGradient();
    this.legendLow = document.createElement('span');
    this.legendLow.className = 'point-inset__tick point-inset__tick--low';
    const caption = document.createElement('span');
    caption.className = 'point-inset__caption';
    caption.textContent = '하위 1% 높이 기준';
    legend.append(this.legendHigh, bar, this.legendLow, caption);

    this.frameEl.append(title, this.viewEl, legend);
    host.appendChild(this.frameEl);

    // The panel's own controls: the main board's camera is never involved.
    this.controls = new OrbitControls(this.camera, this.viewEl);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.addEventListener('start', () => {
      this.auto = false;
    });
    this.viewEl.addEventListener('dblclick', () => this.frame());

    this.setEnabled(opts.enabled);
  }

  offer(chunk: PointChunk): 'accepted' | 'stale' {
    return this.layer.offer(chunk);
  }

  setEnabled(on: boolean): void {
    this.layer.setEnabled(on);
    this.frameEl.classList.toggle('is-open', on);
  }

  get enabled(): boolean {
    return this.layer.enabled;
  }

  setMode(mode: PointColorMode): void {
    this.layer.setMode(mode);
  }

  /** Frame everything delivered and keep following new segments. */
  frame(): boolean {
    this.auto = true;
    this.snap = true;
    return !!this.layer.bounds();
  }

  /** Where the panel's camera is, for checks. */
  get debug(): { camPos: number[]; target: number[]; auto: boolean } {
    const r = (v: THREE.Vector3): number[] => v.toArray().map((n) => Math.round(n * 10) / 10);
    return { camPos: r(this.camera.position), target: r(this.controls.target), auto: this.auto };
  }

  /** Scratch camera for fitting, so the live one is only ever eased. */
  private readonly fitCam = new THREE.PerspectiveCamera();
  private fitKey = '';
  private fitPose: { pos: THREE.Vector3; target: THREE.Vector3 } | null = null;

  /**
   * A view that holds every delivered point, obliquely from the south-east
   * (reads both footprint and height; straight down flattens the colouring).
   * Seen at an angle, the near side of the capture projects larger than the
   * far side, and a round capture leaves its bounding box mostly empty, so
   * neither the box centre nor its corners frame it well. Instead an even
   * sample of the delivered points is projected, and the aim and distance are
   * corrected until their 2nd..98th percentile sits centred in the panel.
   */
  private pose(): { pos: THREE.Vector3; target: THREE.Vector3 } | null {
    const box = this.layer.bounds();
    if (!box || box.isEmpty()) return null;
    // Refit only when what is shown changes (box) or the panel's shape does.
    const key = `${box.min.toArray().join()}|${box.max.toArray().join()}|${this.layer.stats.points}|${this.camera.aspect.toFixed(3)}`;
    if (key === this.fitKey && this.fitPose) return this.fitPose;

    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const r = Math.max(1, sphere.radius);
    const vHalf = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const dir = new THREE.Vector3(0.45, 0.7, 0.6).normalize();
    const cam = this.fitCam;
    cam.fov = this.camera.fov;
    cam.aspect = this.camera.aspect;
    cam.near = 0.1;
    cam.far = r * 100;
    cam.updateProjectionMatrix();

    const sample = this.layer.samplePoints(400);
    if (sample.length === 0) return null;
    const xs = new Float32Array(sample.length);
    const ys = new Float32Array(sample.length);
    const pct = (a: Float32Array, q: number): number => a[Math.min(a.length - 1, Math.floor(q * a.length))];
    const target = sphere.center.clone();
    let dist = r / Math.sin(vHalf);
    const ndc = new THREE.Vector3();
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    for (let iter = 0; iter < 4; iter++) {
      cam.position.copy(target).addScaledVector(dir, dist);
      cam.lookAt(target);
      cam.updateMatrixWorld(true);
      sample.forEach((p, i) => {
        ndc.copy(p).project(cam);
        xs[i] = ndc.x;
        ys[i] = ndc.y;
      });
      xs.sort();
      ys.sort();
      const x0 = pct(xs, 0.02), x1 = pct(xs, 0.98), y0 = pct(ys, 0.02), y1 = pct(ys, 0.98);
      const halfH = Math.tan(vHalf) * dist;
      const halfW = halfH * cam.aspect;
      right.setFromMatrixColumn(cam.matrixWorld, 0);
      up.setFromMatrixColumn(cam.matrixWorld, 1);
      target.addScaledVector(right, ((x0 + x1) / 2) * halfW).addScaledVector(up, ((y0 + y1) / 2) * halfH);
      const extent = Math.max((x1 - x0) / 2, (y1 - y0) / 2);
      dist *= Math.max(0.2, extent / 0.9);
    }

    const near = Math.max(0.1, r / 2000);
    const far = (dist + r) * 3;
    if (Math.abs(this.camera.near - near) > near * 0.05 || this.camera.far < far) {
      this.camera.near = near;
      this.camera.far = Math.max(this.camera.far, far);
      this.camera.updateProjectionMatrix();
    }
    this.controls.maxDistance = Math.max(dist * 3, 10);
    this.fitKey = key;
    this.fitPose = { pos: target.clone().addScaledVector(dir, dist), target };
    return this.fitPose;
  }

  /**
   * Draw the panel after the main frame. The viewport is wherever the panel's
   * view box sits over the board canvas; only that rectangle is cleared and
   * drawn (scissor), so the main frame around it is untouched.
   */
  render(renderer: THREE.WebGLRenderer, canvas: HTMLCanvasElement, dt: number): void {
    const st = this.layer.stats;
    const has = st.points > 0;
    // The panel appears once the feed has offered anything at all.
    this.frameEl.classList.toggle('has-data', st.segments.length > 0);
    this.emptyEl.style.display = has ? 'none' : '';
    if (st.heightRange) {
      const span = st.heightRange[1] - st.heightRange[0];
      this.legendHigh.textContent = `+${span.toFixed(1)} m`;
      this.legendLow.textContent = '0 m';
    } else {
      this.legendHigh.textContent = '';
      this.legendLow.textContent = '';
    }
    if (!this.layer.enabled || !has) return;

    const c = canvas.getBoundingClientRect();
    const v = this.viewEl.getBoundingClientRect();
    const w = Math.round(v.width);
    const h = Math.round(v.height);
    if (w < 8 || h < 8) return;
    const x = Math.round(v.left - c.left);
    const y = Math.round(c.height - (v.top - c.top) - h);

    if (this.camera.aspect !== w / h) {
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
    const heightPx = h * renderer.getPixelRatio();
    if (heightPx !== this.lastHeightPx) {
      this.lastHeightPx = heightPx;
      this.layer.setViewport(heightPx, this.camera.fov);
    }

    const desired = this.auto ? this.pose() : null;
    if (desired) {
      if (this.snap) {
        this.snap = false;
        this.camera.position.copy(desired.pos);
        this.controls.target.copy(desired.target);
      } else {
        const k = 1 - Math.exp(-2.5 * dt);
        this.camera.position.lerp(desired.pos, k);
        this.controls.target.lerp(desired.target, k);
      }
    }
    this.controls.update();

    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = true;
    renderer.setScissorTest(true);
    renderer.setViewport(x, y, w, h);
    renderer.setScissor(x, y, w, h);
    renderer.render(this.scene, this.camera);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, Math.round(c.width), Math.round(c.height));
    renderer.autoClear = prevAutoClear;
    this.layer.afterRender();
  }

  dispose(): void {
    this.controls.dispose();
    this.layer.dispose();
    this.frameEl.remove();
  }
}
