// Point layer: delay-pattern point segments, coloured by height.
//
// Fed by PointChunk (shared/protocol.ts). Each segment is one THREE.Points
// managed by SegmentSlots, so a refined pass REPLACES the first pass of the
// same segment instead of piling on top of it. Only what arrived is drawn:
// no gap filling, no interpolation between points.
//
// FRAME. Point files are ENU metres about the chunk's GPS anchor. Each segment
// sits in a group that carries the anchor offset (gpsToScene against the
// board's own anchor) and the fixed ENU→scene axis turn (x east, y north,
// z up  →  x east, y up, z south), so the 27-byte positions are uploaded
// untouched and the turn costs one matrix, not a pass over the points.
//
// COLOUR. Height ramp over the range the delivered points actually span: the
// 1st..99th percentile of z per segment, unioned over what is on screen, so
// neither a hard-coded ground level nor a few stray points set the scale. The
// range is a uniform; replacing a segment never touches another one's buffers.
// Shading uses the stored normals; a point whose normal was rejected by the
// decoder is drawn flat.

import * as THREE from 'three';
import type { GeoAnchor } from '../../shared/geo.ts';
import { gpsToScene } from '../../shared/geo.ts';
import type { PointChunk } from '../../shared/protocol.ts';
import type { PointCloud } from '../../shared/pointPly.ts';
import { parsePointPly, shufflePoints } from '../../shared/pointPly.ts';
import { SegmentSlots } from './segmentSlots.ts';
import type { SlotState } from './segmentSlots.ts';
import type { PointWorkerReply, PointWorkerRequest } from './pointPly.worker.ts';

const POINT_VERT = /* glsl */ `
  uniform float uZMin;
  uniform float uZMax;
  uniform float uSizeM;
  uniform float uPxPerM;
  uniform int uMode;
  varying vec3 vC;
  vec3 ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)), 0.0, 1.0);
  }
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    // Light from the south-east, high; same direction texture_lab used (ENU
    // (0.4, -0.5, 0.8)) brought into scene axes.
    vec3 L = normalize(vec3(0.4, 0.8, 0.5));
    vec3 n = mat3(modelMatrix) * normal;
    float lit = dot(n, n) > 0.01 ? 0.35 + 0.65 * abs(dot(normalize(n), L)) : 0.8;
    float h = (world.y - uZMin) / max(0.01, uZMax - uZMin);
    vC = uMode == 1 ? color * (0.6 + 0.4 * lit) : ramp(h) * (0.55 + 0.45 * lit);
    vec4 mv = viewMatrix * world;
    gl_PointSize = clamp(uSizeM * uPxPerM / max(0.001, -mv.z), 1.0, 48.0);
    gl_Position = projectionMatrix * mv;
  }
`;

const POINT_FRAG = /* glsl */ `
  varying vec3 vC;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    if (dot(d, d) > 0.25) discard;
    gl_FragColor = vec4(vC, 1.0);
  }
`;

/** ENU (x east, y north, z up) → scene (x east, y up, z = -north). */
const ENU_TO_SCENE = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);

export type PointColorMode = 'height' | 'rgb';

/** One segment arrival, from the moment its load started to its first frame. */
export interface PointTiming {
  segment: number;
  level: number;
  points: number;
  bytes: number;
  /** Network (or disk cache) time inside the worker. */
  fetchMs: number;
  /** Decode/repack time inside the worker. */
  parseMs: number;
  /** Main thread: geometry built and added to the scene. */
  buildMs: number;
  /** Mount → end of the first render that drew it (includes the GPU upload). */
  firstFrameMs: number;
  /** Load start → end of the first render that drew it. */
  totalMs: number;
}

interface Loaded {
  group: THREE.Group;
  points: THREE.Points;
  cloud: Pick<PointCloud, 'count' | 'zLow' | 'zHigh'>;
  /** Scene-space box of the 1st..99th percentile core, for framing. */
  box: THREE.Box3;
  /** Height range in scene y. */
  yLow: number;
  yHigh: number;
  timing: Omit<PointTiming, 'buildMs' | 'firstFrameMs' | 'totalMs'> & {
    startedAt: number;
    buildMs: number;
  };
  mountedAt: number;
}

export interface PointLayerOptions {
  /** The board's ENU origin (CONFIG.geo.anchor). */
  geoAnchor: GeoAnchor;
  enabled: boolean;
  mode?: PointColorMode;
  /** Most points drawn per frame; see `applyBudget`. */
  budget?: number;
}

/**
 * Points drawn per frame, at most. Measured on an Apple M5: 2.9 M points draw
 * at 60 fps in the panel, 8.6 M fall to 30 fps and drag the whole board with
 * them. Above the budget every segment draws the same leading fraction of its
 * (shuffled) points, i.e. an even thinning, with the point size grown to keep
 * the surface covered. Nothing is dropped from memory.
 */
export const POINT_BUDGET = 3_000_000;

export class PointLayer {
  /** Everything this layer draws hangs off this group. */
  readonly root = new THREE.Group();
  private readonly geoAnchor: GeoAnchor;
  private readonly material: THREE.ShaderMaterial;
  private readonly slots: SegmentSlots<PointChunk, Loaded>;
  private worker: Worker | null = null;
  private nextId = 1;
  private readonly waiting = new Map<
    number,
    { resolve: (r: Extract<PointWorkerReply, { type: 'done' }>) => void; reject: (e: Error) => void }
  >();
  private readonly awaitingFrame: Loaded[] = [];
  private readonly timingLog: PointTiming[] = [];
  private readonly errors: Array<{ segment: number; level: number; message: string }> = [];
  private onChangeCbs: Array<() => void> = [];
  private _enabled: boolean;
  private readonly range = { low: 0, high: 1, valid: false };
  private budget: number;
  /** Size per point at full density, metres (from the chunk's voxel). */
  private baseSizeM = 0.15;
  /** Fraction of each segment drawn now (1 = all). */
  private drawFraction = 1;

  constructor(parent: THREE.Object3D, opts: PointLayerOptions) {
    this.geoAnchor = opts.geoAnchor;
    this._enabled = opts.enabled;
    this.budget = Math.max(1, opts.budget ?? POINT_BUDGET);
    this.root.name = 'point-layer';
    this.root.visible = opts.enabled;
    parent.add(this.root);

    this.material = new THREE.ShaderMaterial({
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      vertexColors: true,
      uniforms: {
        uZMin: { value: 0 },
        uZMax: { value: 1 },
        uSizeM: { value: 0.15 },
        uPxPerM: { value: 800 },
        uMode: { value: opts.mode === 'rgb' ? 1 : 0 },
      },
    });

    this.slots = new SegmentSlots<PointChunk, Loaded>(
      {
        load: (msg, signal) => this.load(msg, signal),
        mount: (msg, obj) => this.mount(msg, obj),
        release: (obj) => this.release(obj),
        onError: (msg, err) => {
          const message = err instanceof Error ? err.message : String(err);
          this.errors.push({ segment: msg.segment, level: msg.level, message });
          console.warn(`[points] segment ${msg.segment} level ${msg.level} failed: ${message}`);
        },
      },
      opts.enabled,
    );
  }

  /** A PointChunk arrived. */
  offer(chunk: PointChunk): 'accepted' | 'stale' {
    if (chunk.voxel !== null) {
      this.baseSizeM = Math.max(0.05, chunk.voxel * 1.5);
      this.applyBudget();
    }
    return this.slots.offer(chunk);
  }

  /** Off: hidden and nothing is fetched; arrivals are still remembered, so
   *  turning it on loads the newest level of every segment at once. */
  setEnabled(on: boolean): void {
    this._enabled = on;
    this.root.visible = on;
    this.slots.setActive(on);
  }

  get enabled(): boolean {
    return this._enabled;
  }

  setMode(mode: PointColorMode): void {
    this.material.uniforms.uMode.value = mode === 'rgb' ? 1 : 0;
  }

  /** Point size is in metres; this converts to pixels for the current view. */
  setViewport(heightPx: number, fovDeg: number): void {
    this.material.uniforms.uPxPerM.value =
      heightPx / (2 * Math.tan(THREE.MathUtils.degToRad(fovDeg) / 2));
  }

  onChange(cb: () => void): void {
    this.onChangeCbs.push(cb);
  }

  /** Call right after the frame was rendered: stamps first-frame timings. */
  afterRender(): void {
    if (this.awaitingFrame.length === 0 || !this.root.visible) return;
    const now = performance.now();
    for (const l of this.awaitingFrame.splice(0)) {
      const t = l.timing;
      const entry: PointTiming = {
        segment: t.segment,
        level: t.level,
        points: t.points,
        bytes: t.bytes,
        fetchMs: round1(t.fetchMs),
        parseMs: round1(t.parseMs),
        buildMs: round1(t.buildMs),
        firstFrameMs: round1(now - l.mountedAt),
        totalMs: round1(now - t.startedAt),
      };
      this.timingLog.push(entry);
    }
  }

  /**
   * A uniform sample of what is on screen, in scene coordinates: the first
   * `perSegment` points of each segment (points are shuffled on load, so a
   * prefix is an even sample).
   */
  samplePoints(perSegment = 400): THREE.Vector3[] {
    const out: THREE.Vector3[] = [];
    for (const { obj } of this.slots.shownEntries()) {
      const pos = obj.points.geometry.getAttribute('position') as THREE.BufferAttribute;
      const n = Math.min(perSegment, obj.cloud.count);
      obj.points.updateMatrixWorld(true);
      for (let i = 0; i < n; i++) {
        out.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(obj.points.matrixWorld));
      }
    }
    return out;
  }

  /** Scene-space box of everything on screen, or null. */
  bounds(): THREE.Box3 | null {
    const shown = this.slots.shownEntries();
    if (shown.length === 0) return null;
    const box = new THREE.Box3();
    for (const { obj } of shown) box.union(obj.box);
    return box;
  }

  get stats(): {
    enabled: boolean;
    segments: SlotState[];
    points: number;
    heightRange: [number, number] | null;
    timings: PointTiming[];
    errors: Array<{ segment: number; level: number; message: string }>;
    stale: number;
    cancelled: number;
    drawObjects: number;
    /** Points actually drawn per frame (after the budget). */
    drawn: number;
    drawFraction: number;
  } {
    const shown = this.slots.shownEntries();
    const c = this.slots.counters;
    return {
      enabled: this._enabled,
      segments: this.slots.states(),
      points: shown.reduce((a, s) => a + s.obj.cloud.count, 0),
      heightRange: this.range.valid ? [round1(this.range.low), round1(this.range.high)] : null,
      timings: [...this.timingLog],
      errors: [...this.errors],
      stale: c.stale,
      cancelled: c.cancelled,
      drawObjects: this.root.children.length,
      drawn: shown.reduce((a, s) => a + s.obj.points.geometry.drawRange.count, 0),
      drawFraction: Math.round(this.drawFraction * 1000) / 1000,
    };
  }

  // -------------------------------------------------------------------------

  private ensureWorker(): Worker | null {
    if (this.worker) return this.worker;
    if (typeof Worker === 'undefined') return null;
    try {
      this.worker = new Worker(new URL('./pointPly.worker.ts', import.meta.url), {
        type: 'module',
      });
    } catch (err) {
      console.warn('[points] worker unavailable, decoding on the main thread:', err);
      return null;
    }
    this.worker.onmessage = (ev: MessageEvent<PointWorkerReply>) => {
      const r = ev.data;
      const w = this.waiting.get(r.id);
      if (!w) return;
      this.waiting.delete(r.id);
      if (r.type === 'done') w.resolve(r);
      else w.reject(r.aborted ? new DOMException('aborted', 'AbortError') : new Error(r.message));
    };
    return this.worker;
  }

  private fetchAndDecode(
    url: string,
    signal: AbortSignal,
  ): Promise<Extract<PointWorkerReply, { type: 'done' }>> {
    const abs = new URL(url, window.location.href).href;
    const worker = this.ensureWorker();
    if (!worker) {
      return (async () => {
        const t0 = performance.now();
        const res = await fetch(abs, { signal });
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${abs}`);
        const buf = await res.arrayBuffer();
        const t1 = performance.now();
        const cloud = parsePointPly(buf);
        shufflePoints(cloud);
        return {
          type: 'done' as const,
          id: 0,
          cloud,
          bytes: buf.byteLength,
          fetchMs: t1 - t0,
          parseMs: performance.now() - t1,
        };
      })();
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      signal.addEventListener(
        'abort',
        () => worker.postMessage({ type: 'cancel', id } satisfies PointWorkerRequest),
        { once: true },
      );
      worker.postMessage({ type: 'load', id, url: abs } satisfies PointWorkerRequest);
    });
  }

  private async load(msg: PointChunk, signal: AbortSignal): Promise<Loaded> {
    const startedAt = performance.now();
    const r = await this.fetchAndDecode(msg.url, signal);
    const t0 = performance.now();
    const cloud = r.cloud;

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(cloud.positions, 3));
    if (cloud.colors) geom.setAttribute('color', new THREE.BufferAttribute(cloud.colors, 3, true));
    else geom.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(cloud.count * 3).fill(200), 3, true));
    if (cloud.normals) geom.setAttribute('normal', new THREE.BufferAttribute(cloud.normals, 3, true));
    else geom.setAttribute('normal', new THREE.BufferAttribute(new Int8Array(cloud.count * 3), 3, true));
    // Bounds come from the decoder; skip three's O(n) pass over the points.
    const min = new THREE.Vector3(...cloud.bbox.min);
    const max = new THREE.Vector3(...cloud.bbox.max);
    geom.boundingBox = new THREE.Box3(min, max);
    geom.boundingSphere = geom.boundingBox.getBoundingSphere(new THREE.Sphere());

    const points = new THREE.Points(geom, this.material);
    points.name = `points-seg${msg.segment}-l${msg.level}`;
    // ENU-frame placement from the chunk.
    points.position.set(...msg.align.position);
    points.quaternion.set(...msg.align.rotation);
    points.scale.set(...msg.align.scale);

    const group = new THREE.Group();
    group.name = `point-seg${msg.segment}`;
    const base = msg.align.anchor
      ? gpsToScene(msg.align.anchor, this.geoAnchor)
      : ([0, 0, 0] as [number, number, number]);
    group.position.set(...base);
    group.quaternion.copy(ENU_TO_SCENE);
    group.add(points);
    group.updateMatrixWorld(true);

    const box = new THREE.Box3(
      new THREE.Vector3(...cloud.core.min),
      new THREE.Vector3(...cloud.core.max),
    ).applyMatrix4(points.matrixWorld);
    // Height range in scene y. The rotation in `align` is identity today
    // (protocol.ts), so z maps straight onto y through scale and offset.
    const s = msg.align.scale[2];
    const yOff = base[1] + msg.align.position[2];
    const yLow = cloud.zLow * s + yOff;
    const yHigh = cloud.zHigh * s + yOff;

    return {
      group,
      points,
      cloud: { count: cloud.count, zLow: cloud.zLow, zHigh: cloud.zHigh },
      box,
      yLow,
      yHigh,
      timing: {
        segment: msg.segment,
        level: msg.level,
        points: cloud.count,
        bytes: r.bytes,
        fetchMs: r.fetchMs,
        parseMs: r.parseMs,
        startedAt,
        buildMs: performance.now() - t0,
      },
      mountedAt: 0,
    };
  }

  private mount(_msg: PointChunk, obj: Loaded): void {
    obj.mountedAt = performance.now();
    this.root.add(obj.group);
    this.awaitingFrame.push(obj);
    this.recomputeRange(obj);
    this.applyBudget(obj);
    for (const cb of this.onChangeCbs) cb();
  }

  private release(obj: Loaded): void {
    this.root.remove(obj.group);
    obj.points.geometry.dispose();
    const i = this.awaitingFrame.indexOf(obj);
    if (i >= 0) this.awaitingFrame.splice(i, 1);
    this.recomputeRange(null);
    this.applyBudget();
  }

  /** Change the per-frame point budget (tuning and checks). */
  setBudget(points: number): void {
    this.budget = Math.max(1, points);
    this.applyBudget();
  }

  /** Thin every segment evenly when the total is over budget (see POINT_BUDGET). */
  private applyBudget(adding: Loaded | null = null): void {
    const all = this.slots.shownEntries().map((e) => e.obj);
    if (adding && !all.includes(adding)) all.push(adding);
    const live = all.filter((o) => o.group.parent);
    const total = live.reduce((a, o) => a + o.cloud.count, 0);
    this.drawFraction = total > this.budget ? this.budget / total : 1;
    for (const o of live) {
      o.points.geometry.setDrawRange(0, Math.ceil(o.cloud.count * this.drawFraction));
    }
    // Fewer points over the same area: grow each so the surface stays closed.
    this.material.uniforms.uSizeM.value = this.baseSizeM / Math.sqrt(this.drawFraction);
  }

  /** Union of the height ranges on screen. `adding` is not in the slot map yet
   *  while it is being mounted, so it is folded in explicitly. */
  private recomputeRange(adding: Loaded | null): void {
    let low = Infinity;
    let high = -Infinity;
    const all = this.slots.shownEntries().map((e) => e.obj);
    if (adding) all.push(adding);
    for (const o of all) {
      if (!o.group.parent) continue;
      low = Math.min(low, o.yLow);
      high = Math.max(high, o.yHigh);
    }
    this.range.valid = Number.isFinite(low) && Number.isFinite(high);
    if (!this.range.valid) return;
    this.range.low = low;
    this.range.high = Math.max(high, low + 0.5);
    this.material.uniforms.uZMin.value = this.range.low;
    this.material.uniforms.uZMax.value = this.range.high;
  }

  dispose(): void {
    this.slots.clear();
    this.material.dispose();
    this.worker?.terminate();
    this.worker = null;
    this.root.parent?.remove(this.root);
  }
}

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}
