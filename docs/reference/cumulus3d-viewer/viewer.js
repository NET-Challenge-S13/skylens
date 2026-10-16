// Progressive point-stream viewer, format v3 (see ../FORMAT_v3.md): per-event add/move/remove diffs addressed by dense
// slot id. No hash map, no diffing: records are written straight into GPU-backed typed arrays at index = id.
//  - packs are fetched + parsed in a Web Worker (ArrayBuffers transferred); the main thread only copies typed-array ranges
//  - move animation is entirely on the GPU (aFrom -> position, eased by uTime - tm.x); add/remove are a 150 ms dither fade
//  - uploads are dirty-range only (addUpdateRange) and budgeted per frame so a 500k-point event never stalls a frame
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MOVE_S = 0.7, FADE_S = 0.15, HL_SEC = 2, UPLOAD_BUDGET = 200000, T_LONG = -1e3, T_NEVER = 1e9;
const q = new URLSearchParams(location.search);
let base = q.get('data') || 'data';
if (!base.endsWith('/')) base += '/';
// ?live=1: follow a stream that is still being written — apply each event as soon as it shows up in the manifest.
const LIVE = q.get('live') === '1';
let lastArrival = null;
const $ = id => document.getElementById(id);

// ---------------- renderer / scene ----------------
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x14141a);
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 1e5);
camera.up.set(0, 0, 1);                       // data: x east, y north, z up
camera.position.set(0, -50, 40);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
});

// ---------------- point storage (index = server slot id) ----------------
let cap = 0, used = 0;
let aPos, aFrm, aTm, aCol, aZon;               // position(=target) f32x3, from f32x3, tm f32x3 (tMove,tAdd,tDie), colour u8x3, zone u8
const geom = new THREE.BufferGeometry();
const mat = new THREE.ShaderMaterial({
  uniforms: { uSize: { value: 2 }, uRef: { value: 50 }, uMode: { value: 0 }, uTime: { value: 0 }, uFadeOut: { value: FADE_S } },
  vertexShader: `
    attribute vec3 aFrom; attribute vec3 tm; attribute float zone;
    uniform float uSize, uRef, uMode, uTime, uFadeOut; varying vec3 vCol; varying float vA;
    vec3 hsv(float h){ return clamp(abs(mod(h*6.0+vec3(0,4,2),6.0)-3.0)-1.0,0.0,1.0); }
    void main(){
      float u = clamp((uTime - tm.x)/${MOVE_S.toFixed(2)}, 0.0, 1.0);
      float e = u < 0.5 ? 2.0*u*u : 1.0 - pow(-2.0*u + 2.0, 2.0)/2.0;     // GPU ease-in-out glide old -> new
      vec3 p = mix(aFrom, position, e);
      float dieK = (uTime - tm.z)/uFadeOut;
      float a = clamp((uTime - tm.y)/${FADE_S.toFixed(2)}, 0.0, 1.0) * clamp(1.0 - dieK, 0.0, 1.0);
      if (a <= 0.01) { gl_Position = vec4(2.0,2.0,2.0,1.0); gl_PointSize = 0.0; vCol = vec3(0.0); vA = 0.0; return; }
      vec3 c = color;
      if (uMode > 0.5 && uMode < 1.5) c = 0.25 + 0.75*hsv(fract(zone*0.618034));
      if (uMode > 1.5) {                       // highlight changes: new = green, moved = orange, removed = magenta
        float kn = clamp(1.0 - (uTime - tm.y)/${HL_SEC.toFixed(1)}, 0.0, 1.0);
        float km = clamp(1.0 - (uTime - tm.x)/${HL_SEC.toFixed(1)}, 0.0, 1.0);
        c = mix(c*0.55, vec3(0.2,1.0,0.3), kn); c = mix(c, vec3(1.0,0.25,0.1), km);
        if (tm.z < 1.0e8) c = vec3(1.0,0.1,0.6);
      }
      vCol = c; vA = a;
      vec4 mv = modelViewMatrix * vec4(p,1.0);
      gl_PointSize = clamp(uSize * uRef / -mv.z, 1.0, 48.0);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying vec3 vCol; varying float vA;
    void main(){
      if (vA < 0.02) discard;
      // screen-door dither instead of blending: opaque, depth-correct, no transparent sorting
      float h = fract(52.9829189*fract(dot(gl_FragCoord.xy, vec2(0.06711056,0.00583715))));
      if (h > vA) discard;
      vec2 d = gl_PointCoord - 0.5; if (dot(d,d) > 0.25) discard;
      gl_FragColor = vec4(vCol,1.0);
    }`,
  vertexColors: true,
});
const points = new THREE.Points(geom, mat);
points.frustumCulled = false;                  // no bounding-sphere recompute; the cloud is always in view
scene.add(points);

function growTo(need) {
  if (need <= cap) return;
  const n = Math.max(need, Math.ceil(cap * 1.3), 65536);
  const re = (A, old, k, init) => { const a = new A(n * k); if (init) init(a); if (old) a.set(old); return a; };
  aPos = re(Float32Array, aPos, 3); aFrm = re(Float32Array, aFrm, 3);
  aTm = re(Float32Array, aTm, 3, a => { for (let i = 0; i < n; i++) { a[3 * i] = T_LONG; a[3 * i + 1] = T_NEVER; a[3 * i + 2] = T_NEVER; } });
  aCol = re(Uint8Array, aCol, 3); aZon = re(Uint8Array, aZon, 1);
  cap = n;
  geom.dispose();                              // free the old GPU buffers; new ones are created whole on the next render
  const dyn = (a) => a.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', dyn(new THREE.BufferAttribute(aPos, 3)));
  geom.setAttribute('aFrom', dyn(new THREE.BufferAttribute(aFrm, 3)));
  geom.setAttribute('tm', dyn(new THREE.BufferAttribute(aTm, 3)));
  geom.setAttribute('color', dyn(new THREE.BufferAttribute(aCol, 3, true)));
  geom.setAttribute('zone', dyn(new THREE.BufferAttribute(aZon, 1)));
  dq.length = 0;
}
// dirty queue of [lo,hi) slot ranges, flushed with a per-frame budget
const dq = [];
function markDirty(lo, hi) {
  const l = dq[dq.length - 1];
  if (l && lo >= l[0] && lo <= l[1] + 64) { if (hi > l[1]) l[1] = hi; }
  else dq.push([lo, hi]);
  if (dq.length > 4096) { let a = 1e12, b = 0; for (const r of dq) { if (r[0] < a) a = r[0]; if (r[1] > b) b = r[1]; } dq.length = 0; dq.push([a, b]); }
}
const ATTRS = ['position', 'aFrom', 'tm', 'color', 'zone'];
let lastUploadPts = 0;
function flushUpload() {
  lastUploadPts = 0;
  if (!dq.length) return;
  let budget = UPLOAD_BUDGET; const take = [];
  while (dq.length && budget > 0) {
    const r = dq[0], len = r[1] - r[0];
    if (len <= budget) { take.push([r[0], r[1]]); budget -= len; dq.shift(); }
    else { take.push([r[0], r[0] + budget]); r[0] += budget; budget = 0; }
  }
  for (const n of ATTRS) {
    const a = geom.getAttribute(n); a.clearUpdateRanges();
    for (const r of take) a.addUpdateRange(r[0] * a.itemSize, (r[1] - r[0]) * a.itemSize);
    a.needsUpdate = true;
  }
  for (const r of take) lastUploadPts += r[1] - r[0];
}

// ---------------- worker: fetch + parse packs ----------------
function workerMain() {
  self.onmessage = async e => {
    const { rid, url, na, nm, nr, ms } = e.data;      // ms = move record size (16 = v3, 20 = v4 with colour)
    try {
      const t0 = performance.now();
      const r = await fetch(url); if (!r.ok) throw new Error(r.status + ' ' + url);
      const buf = await r.arrayBuffer(); const t1 = performance.now();
      if (buf.byteLength !== 20 * na + ms * nm + 4 * nr) throw new Error('pack size mismatch ' + url);
      const u32 = new Uint32Array(buf), f32 = new Float32Array(buf), u8 = new Uint8Array(buf);
      const ids = new Uint32Array(na), pos = new Float32Array(3 * na), rgb = new Uint8Array(3 * na), zone = new Uint8Array(na);
      const bb = [1e9, 1e9, 1e9, -1e9, -1e9, -1e9];
      for (let i = 0; i < na; i++) {
        const w = 5 * i, o = 20 * i + 16, x = f32[w + 1], y = f32[w + 2], z = f32[w + 3];
        ids[i] = u32[w]; pos[3 * i] = x; pos[3 * i + 1] = y; pos[3 * i + 2] = z;
        rgb[3 * i] = u8[o]; rgb[3 * i + 1] = u8[o + 1]; rgb[3 * i + 2] = u8[o + 2]; zone[i] = u8[o + 3];
        if (x < bb[0]) bb[0] = x; if (y < bb[1]) bb[1] = y; if (z < bb[2]) bb[2] = z;
        if (x > bb[3]) bb[3] = x; if (y > bb[4]) bb[4] = y; if (z > bb[5]) bb[5] = z;
      }
      const w0 = 5 * na, mw = ms / 4, mids = new Uint32Array(nm), mpos = new Float32Array(3 * nm), mrgb = ms === 20 ? new Uint8Array(3 * nm) : null;
      for (let i = 0; i < nm; i++) {
        const w = w0 + mw * i; mids[i] = u32[w]; mpos[3 * i] = f32[w + 1]; mpos[3 * i + 1] = f32[w + 2]; mpos[3 * i + 2] = f32[w + 3];
        if (mrgb) { const o = 4 * (w + 4); mrgb[3 * i] = u8[o]; mrgb[3 * i + 1] = u8[o + 1]; mrgb[3 * i + 2] = u8[o + 2]; }
      }
      const w1 = w0 + mw * nm, rids = new Uint32Array(nr);
      for (let i = 0; i < nr; i++) rids[i] = u32[w1 + i];
      const t2 = performance.now();
      self.postMessage({ rid, add: { ids, pos, rgb, zone }, move: { ids: mids, pos: mpos, rgb: mrgb }, remove: rids, bbox: na ? bb : null,
        bytes: buf.byteLength, fetchMs: t1 - t0, parseMs: t2 - t1 }, [ids.buffer, pos.buffer, rgb.buffer, zone.buffer, mids.buffer, mpos.buffer, rids.buffer].concat(mrgb ? [mrgb.buffer] : []));
    } catch (err) { self.postMessage({ rid, error: String(err && err.message || err) }); }
  };
}
const worker = new Worker(URL.createObjectURL(new Blob(['(' + workerMain.toString() + ')()'], { type: 'text/javascript' })));
const pendingRpc = new Map(); let rpcId = 0;
worker.onmessage = e => { const p = pendingRpc.get(e.data.rid); pendingRpc.delete(e.data.rid); if (e.data.error) p.rej(new Error(e.data.error)); else p.res(e.data); };
function parsePack(ev) {
  if (!ev.pack) return Promise.resolve(null);
  return new Promise((res, rej) => {
    const rid = ++rpcId; pendingRpc.set(rid, { res, rej });
    worker.postMessage({ rid, url: new URL(base + ev.pack, location.href).href, na: ev.n_add | 0, nm: ev.n_move | 0, nr: ev.n_remove | 0, ms: (manifest && manifest.version >= 4) ? 20 : 16 });
  });
}
// parsed packs are kept (ev._P) so step-back + step-forward re-applies without refetching
const getParsed = ev => ev._P !== undefined ? Promise.resolve(ev._P) : (ev._p || (ev._p = parsePack(ev).then(P => { ev._P = P; ev._p = null; return P; })));
function prefetchAhead() { for (let i = applied; i < Math.min(events.length, applied + 3); i++) getParsed(events[i]); }

// ---------------- apply an event ----------------
const ease = u => u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
let bmin = [1e9, 1e9, 1e9], bmax = [-1e9, -1e9, -1e9];
// Primitive writers (animate=false -> snap). Shared by forward apply and the inverse (step back).
function doRemove(ids, animate, now) {
  const t = animate ? now : T_LONG;
  for (let i = 0; i < ids.length; i++) { const id = ids[i]; aTm[3 * id + 2] = t; markDirty(id, id + 1); }
}
function doAdd(ids, pos, rgb, zone, animate, now) {
  const na = ids.length, tAdd = animate ? now : T_LONG;
  for (let i = 0; i < na;) {
    let j = i + 1; const id0 = ids[i];
    while (j < na && ids[j] === id0 + (j - i)) j++;
    const id1 = id0 + (j - i);
    if (id1 > used) { used = id1; geom.setDrawRange(0, used); }
    const ps = pos.subarray(3 * i, 3 * j);
    aPos.set(ps, 3 * id0); aFrm.set(ps, 3 * id0); aCol.set(rgb.subarray(3 * i, 3 * j), 3 * id0); aZon.set(zone.subarray(i, j), id0);
    for (let k = 3 * id0, ke = 3 * id1; k < ke; k += 3) { aTm[k] = T_LONG; aTm[k + 1] = tAdd; aTm[k + 2] = T_NEVER; }
    markDirty(id0, id1);
    i = j;
  }
}
function doMove(ids, mpos, mrgb, animate, now) {
  const nm = ids.length;
  for (let i = 0; i < nm; i++) {
    const id = ids[i], j = 3 * id, x = mpos[3 * i], y = mpos[3 * i + 1], z = mpos[3 * i + 2];
    if (animate) {
      const u = (now - aTm[j]) / MOVE_S;       // a glide still running continues from its in-flight position
      if (u >= 0 && u < 1) { const e = ease(u); aFrm[j] += (aPos[j] - aFrm[j]) * e; aFrm[j + 1] += (aPos[j + 1] - aFrm[j + 1]) * e; aFrm[j + 2] += (aPos[j + 2] - aFrm[j + 2]) * e; }
      else { aFrm[j] = aPos[j]; aFrm[j + 1] = aPos[j + 1]; aFrm[j + 2] = aPos[j + 2]; }
      aTm[j] = now;
    } else { aFrm[j] = x; aFrm[j + 1] = y; aFrm[j + 2] = z; aTm[j] = T_LONG; }
    aPos[j] = x; aPos[j + 1] = y; aPos[j + 2] = z;
    if (mrgb) { aCol[j] = mrgb[3 * i]; aCol[j + 1] = mrgb[3 * i + 1]; aCol[j + 2] = mrgb[3 * i + 2]; }   // colour switches instantly
    markDirty(id, id + 1);
  }
}
// Forward apply of a parsed pack; returns stats + an undo log (previous state of everything the event touches).
function applyParsed(P, animate) {
  const t0 = performance.now(), now = t0 / 1000;
  const st = { added: 0, moved: 0, removed: 0, undo: null };
  if (!P) return st;
  const rem = P.remove, { ids, pos, rgb, zone } = P.add, mids = P.move.ids, mpos = P.move.pos, mrgb = P.move.rgb;
  const undo = { addIds: ids, rmIds: rem, rmPos: new Float32Array(3 * rem.length), rmRgb: new Uint8Array(3 * rem.length), rmZone: new Uint8Array(rem.length),
    mvIds: mids, mvPos: new Float32Array(3 * mids.length), mvRgb: mrgb ? new Uint8Array(3 * mids.length) : null };
  for (let i = 0; i < rem.length; i++) {
    const j = 3 * rem[i];
    undo.rmPos[3 * i] = aPos[j]; undo.rmPos[3 * i + 1] = aPos[j + 1]; undo.rmPos[3 * i + 2] = aPos[j + 2];
    undo.rmRgb[3 * i] = aCol[j]; undo.rmRgb[3 * i + 1] = aCol[j + 1]; undo.rmRgb[3 * i + 2] = aCol[j + 2]; undo.rmZone[i] = aZon[rem[i]];
  }
  for (let i = 0; i < mids.length; i++) {
    const j = 3 * mids[i];
    undo.mvPos[3 * i] = aPos[j]; undo.mvPos[3 * i + 1] = aPos[j + 1]; undo.mvPos[3 * i + 2] = aPos[j + 2];
    if (mrgb) { undo.mvRgb[3 * i] = aCol[j]; undo.mvRgb[3 * i + 1] = aCol[j + 1]; undo.mvRgb[3 * i + 2] = aCol[j + 2]; }
  }
  doRemove(rem, animate, now); doAdd(ids, pos, rgb, zone, animate, now); doMove(mids, mpos, mrgb, animate, now);
  st.removed = rem.length; st.added = ids.length; st.moved = mids.length; st.undo = undo;
  if (P.bbox) { for (let d = 0; d < 3; d++) { if (P.bbox[d] < bmin[d]) bmin[d] = P.bbox[d]; if (P.bbox[d + 3] > bmax[d]) bmax[d] = P.bbox[d + 3]; } }
  st.applyMs = performance.now() - t0;
  return st;
}
// Inverse of an event: moved points glide back (and get their old colour), added points fade out, removed points fade back in.
function applyUndo(u) {
  const t0 = performance.now(), now = t0 / 1000;
  doMove(u.mvIds, u.mvPos, u.mvRgb, true, now); doRemove(u.addIds, true, now); doAdd(u.rmIds, u.rmPos, u.rmRgb, u.rmZone, true, now);
  return performance.now() - t0;
}
async function applyEvent(ev, animate) {
  const tw = performance.now();
  const P = await getParsed(ev);
  const waitMs = performance.now() - tw;
  const st = applyParsed(P, animate);
  st.waitMs = waitMs; st.bytes = P ? P.bytes : 0; st.fetchMs = P ? P.fetchMs : 0; st.parseMs = P ? P.parseMs : 0;
  st.workerMs = st.fetchMs + st.parseMs; st.applyMs = st.applyMs || 0;
  ev._undo = st.undo; delete st.undo;
  return st;
}
// headless-test helpers
window.__dbg = {
  alive() { let n = 0; for (let i = 0; i < used; i++) if (aTm[3 * i + 1] < 1e8 && aTm[3 * i + 2] >= 1e8) n++; return n; },
  sample(ids) { const t = performance.now() / 1000; return ids.map(id => { const j = 3 * id, u = Math.min(1, Math.max(0, (t - aTm[j]) / MOVE_S)), e = ease(u);
    return { id, u: +u.toFixed(2), cur: [0, 1, 2].map(k => +(aFrm[j + k] + (aPos[j + k] - aFrm[j + k]) * e).toFixed(3)), to: [0, 1, 2].map(k => +aPos[j + k].toFixed(3)), rgb: [aCol[j], aCol[j + 1], aCol[j + 2]] }; }); },
  undo(i) { const u = events[i]._undo; return u ? { mv: Array.from(u.mvIds.slice(0, 4)), n: [u.addIds.length, u.rmIds.length, u.mvIds.length] } : null; },
};
function fitView() {
  if (bmax[0] < bmin[0]) return;
  const c = new THREE.Vector3((bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2);
  const r = Math.max(1, new THREE.Vector3(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]).length() / 2);
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0, -0.9, 0.7).normalize().multiplyScalar(r * 2.2));
  camera.near = r / 500; camera.far = r * 50; camera.updateProjectionMatrix();
  mat.uniforms.uRef.value = r * 1.5;
}
let manifest = null, events = [], applied = 0;

// ---------------- "generating" UI: waiting screen, top-right monitor ----------------
const pageOpened = performance.now();
let statusInfo = null, statusAt = 0, everApplied = false, waitFade = LIVE ? 1 : 0, lastApplyAt = 0, frontZone = -1, flashUntil = 0;
const STAGE_KO = { registration: '등록', triangulation: '삼각측량', ba: '번들 조정', dense: '조밀화', done: '완료' };
const mmss = (sec, rnd) => { sec = Math.max(0, rnd ? Math.round(sec) : Math.floor(sec)); return Math.floor(sec / 60) + ':' + String(sec % 60).padStart(2, '0'); };
function progressText() {
  const s = statusInfo; if (!s) return '';
  const parts = [];
  if (s.positions != null) parts.push('위치 ' + s.positions + (s.positions_total != null ? '/' + s.positions_total : ''));
  const zs = (s.zone != null ? '구역 ' + s.zone : '') + (s.stage ? (s.zone != null ? ' ' : '') + (STAGE_KO[s.stage] || s.stage) : '');
  if (zs) parts.push(zs);
  return parts.join(' · ');
}
async function pollStatus() {
  try {
    const r = await fetch(base + 'status.json?_=' + Date.now());
    statusInfo = r.ok ? await r.json() : null; statusAt = performance.now();
  } catch { statusInfo = null; }
}
// waiting-screen wave field (rendered by the main renderer, additive, on top of the empty scene)
const waitScene = new THREE.Scene();
const waitCam = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 50);
waitCam.up.set(0, 0, 1);
addEventListener('resize', () => { waitCam.aspect = innerWidth / innerHeight; waitCam.updateProjectionMatrix(); waitMat.uniforms.uPx.value = pxScale(); });
const pxScale = () => Math.min(devicePixelRatio, 2) * Math.max(1.0, innerHeight / 520) * 2.2;
const waitMat = new THREE.ShaderMaterial({
  uniforms: { uT: { value: 0 }, uOpacity: { value: 1 }, uPx: { value: pxScale() } },
  transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
  vertexShader: `
    uniform float uT, uPx; varying vec3 vC; varying float vK;
    void main(){
      vec2 p = position.xy; float r = length(p);
      float h = 0.11*sin(r*15.0 - uT*2.4)/(1.0 + r*2.2) + 0.05*sin(r*6.5 - uT*1.2 + 1.3)
              + 0.035*sin(p.x*6.0 + uT*0.9)*sin(p.y*5.0 - uT*0.7);
      float k = clamp(h*4.5 + 0.45, 0.0, 1.0);
      vec3 lo = vec3(0.05,0.28,0.85), mid = vec3(0.0,0.78,1.0), hi = vec3(0.85,1.0,1.0);
      vC = k < 0.6 ? mix(lo, mid, k/0.6) : mix(mid, hi, (k-0.6)/0.4);
      vK = 1.0 - smoothstep(0.75, 1.25, r);
      vec4 mv = modelViewMatrix*vec4(p, h, 1.0);
      gl_PointSize = uPx*(0.8 + 1.3*k)*2.3/ -mv.z;
      gl_Position = projectionMatrix*mv;
    }`,
  fragmentShader: `
    uniform float uT, uOpacity; varying vec3 vC; varying float vK;
    void main(){
      vec2 d = gl_PointCoord - 0.5; float a = 1.0 - dot(d,d)*4.0; if (a <= 0.0) discard;
      float pulse = 0.85 + 0.15*sin(uT*1.7);
      gl_FragColor = vec4(vC*pulse, a*a*vK*uOpacity*0.9);
    }`,
});
{
  const N = 190, ext = 1.3, arr = new Float32Array(N * N * 3);
  for (let i = 0, k = 0; i < N; i++) for (let j = 0; j < N; j++, k += 3) {
    arr[k] = ((i + Math.random() * 0.5) / (N - 1) - 0.5) * 2 * ext; arr[k + 1] = ((j + Math.random() * 0.5) / (N - 1) - 0.5) * 2 * ext; arr[k + 2] = 0;
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  const wp = new THREE.Points(g, waitMat); wp.frustumCulled = false; waitScene.add(wp);
}
function updateWait(now, dt) {
  if (everApplied && waitFade > 0) waitFade = Math.max(0, waitFade - dt / 0.8);
  const el = $('wait');
  if (waitFade <= 0) { if (el.style.display !== 'none') el.style.display = 'none'; return false; }
  el.style.opacity = waitFade;
  const t = (now - pageOpened) / 1000, a = t * 0.07;
  waitCam.position.set(Math.sin(a) * 2.3, -Math.cos(a) * 2.3, 1.15 + 0.12 * Math.sin(t * 0.2)); waitCam.lookAt(0, 0, -0.05);
  waitMat.uniforms.uT.value = t; waitMat.uniforms.uOpacity.value = waitFade;
  const pr = progressText();
  $('waitSub').textContent = (pr ? pr + '  ·  ' : '') + '경과 ' + mmss(t);
  return true;
}
// top-right monitor (2D canvas, ~1500 dots)
const monEl = $('monitor'), monCv = $('monCanvas'), monCtx = monCv.getContext('2d');
const MON_W = 220, MON_H = 140, MC = 50, MR = 30;
{ const d = Math.min(devicePixelRatio, 2); monCv.width = MON_W * d; monCv.height = MON_H * d; monCtx.scale(d, d); }
const monPal = []; for (let i = 0; i < 10; i++) {
  const k = i / 9, c = k < 0.6 ? [13 + (0 - 13) * k / 0.6, 72 + (200 - 72) * k / 0.6, 217 + (255 - 217) * k / 0.6] : [0 + 215 * (k - 0.6) / 0.4, 200 + 55 * (k - 0.6) / 0.4, 255];
  monPal.push(`rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${0.35 + 0.65 * k})`);
}
let monLastDraw = 0, monDoneDrawn = false;
function drawMonitor(now, calm) {
  const t = now / 1000, ca = Math.cos(t * 0.18), sa = Math.sin(t * 0.18), ctx = monCtx;
  ctx.clearRect(0, 0, MON_W, MON_H);
  const cx = MON_W / 2, cy = MON_H * 0.66;
  for (let b = 0; b < 10; b++) {            // bucket by colour to avoid per-dot style strings
    ctx.fillStyle = monPal[b]; ctx.beginPath();
    for (let i = 0; i < MC; i++) for (let j = 0; j < MR; j++) {
      const x = (i / (MC - 1) - 0.5) * 2, y = (j / (MR - 1) - 0.5) * 2;
      const r = Math.hypot(x, y);
      const h = calm ? 0 : 0.16 * Math.sin(r * 9 - t * 3.2) / (1 + r * 2) + 0.05 * Math.sin(r * 4 - t * 1.4 + 1);
      const k = Math.min(1, Math.max(0, h * 4.5 + 0.45)), bk = Math.round(k * 9); if (bk !== b) continue;
      const xr = x * ca - y * sa, yr = x * sa + y * ca, sc = 1 / (1 + yr * 0.22);
      const sx = cx + xr * sc * 104, sy = cy + yr * sc * 26 - h * sc * 80;
      ctx.rect(sx - 0.8, sy - 0.8, 1.7 * sc + 0.3, 1.7 * sc + 0.3);
    }
    ctx.fill();
  }
}
function updateMonitor(now) {
  const show = applied > 0;
  if (!show) { if (monEl.style.display !== 'none') monEl.style.display = 'none'; return; }
  monEl.style.display = 'block';
  const last = events[applied - 1], fin = last && last.kind === 'final';
  monEl.classList.toggle('done', fin);
  monEl.classList.toggle('flash', now < flashUntil);
  const secAgo = Math.max(0, Math.floor((now - lastApplyAt) / 1000));
  if (fin) {
    $('monLabel').textContent = `생성 완료 · 총 ${last.points.toLocaleString()}점 · ${mmss(last.t, true)}`;
    $('monSub').textContent = progressText() || `구역 ${frontZone + 1}개`;
    $('monAge').textContent = '';
    if (!monDoneDrawn) { drawMonitor(now, true); monDoneDrawn = true; }
    return;
  }
  monDoneDrawn = false;
  $('monLabel').textContent = `생성 중 · 구역 ${frontZone + 1}`;
  $('monSub').textContent = (LIVE ? progressText() : '') || `이벤트 ${applied}/${events.length}`;
  $('monAge').textContent = `마지막 갱신 ${secAgo}초 전`;
  if (now - monLastDraw > 33) { monLastDraw = now; drawMonitor(now, false); }
}
function onApplied(st, flash) {   // call after applied changed
  everApplied = true; lastApplyAt = performance.now();
  frontZone = -1; for (let i = 0; i < applied; i++) if (events[i].kind !== 'final' && events[i].zone > frontZone) frontZone = events[i].zone;
  if (flash) flashUntil = lastApplyAt + 1200;
}


// ---------------- replay control ----------------
let animMs = 0, frameMs = 0, playing = true, speed = 20, simT = 0, busy = false, lastFrame = performance.now(), autoFitted = false;
const statLines = { cur: '' };
function resetAll() {
  dq.length = 0;
  for (let i = 0; i < used; i++) { aTm[3 * i] = T_LONG; aTm[3 * i + 1] = T_NEVER; aTm[3 * i + 2] = T_NEVER; }
  if (used) markDirty(0, used);
  applied = 0;
}
async function applyNext(animate = true) {
  const ev = events[applied];
  growTo(manifest.capacity || 0);
  const tm = performance.now();
  const st = await applyEvent(ev, animate);
  st.mainMs = st.applyMs;
  ev._st = st; applied++; onApplied(st, true);
  if (!autoFitted) { fitView(); autoFitted = true; }
  statLines.cur = `event ${ev.seq} (zone ${ev.zone}, ${ev.kind})\n  worker fetch ${st.fetchMs.toFixed(0)} + parse ${st.parseMs.toFixed(0)} ms (${(st.bytes / 1e6).toFixed(2)} MB), main apply ${st.applyMs.toFixed(1)} ms`;
  renderList(); prefetchAhead();
}
// jump to "state after event n" without animation: replay the diffs of events 0..n-1 in order
async function seek(n) {
  busy = true;
  try {
    growTo(manifest.capacity || 0);
    resetAll();
    for (let i = 0; i < n; i++) getParsed(events[i]);
    const t0 = performance.now(); let tot = 0;
    for (let i = 0; i < events.length; i++) events[i]._undo = null;
    for (let i = 0; i < n; i++) { const P = await getParsed(events[i]); events[i]._undo = applyParsed(P, false).undo; }
    markDirty(0, used);
    applied = n; simT = n ? events[n - 1].t : (events[0]?.t ?? 0);
    if (n) onApplied(null, false);
    statLines.cur = `seek to event ${n}: ${n ? events[n - 1].points : 0} points, ${(performance.now() - t0).toFixed(0)} ms`;
    renderList(); prefetchAhead();
  } finally { busy = false; }
}
async function stepBack() {
  if (busy || applied <= 0) return; busy = true;
  try {
    const ev = events[applied - 1], u = ev._undo;
    if (!u && ev.pack) { await seek(applied - 1); return; }      // no log (should not happen): instant rebuild
    const ms = u ? applyUndo(u) : 0;
    applied--; simT = applied ? events[applied - 1].t : (events[0]?.t ?? 0) - 0.01;
    onApplied(null, false);
    statLines.cur = `step back from event ${ev.seq}: moved ${ev._st ? fmt(ev._st.moved) : '?'} glide back, ${ev._st ? fmt(ev._st.added) : '?'} fade out, ${ev._st ? fmt(ev._st.removed) : '?'} fade in (${ms.toFixed(1)} ms)`;
    ev._undoMs = ms; renderList(); prefetchAhead();
  } finally { busy = false; }
}
async function step(dir) {
  if (busy) return; busy = true;
  try {
    if (dir > 0 && applied < events.length) { await applyNext(true); simT = events[applied - 1].t; }
  } finally { busy = false; }
}
const fmt = n => Number(n).toLocaleString('en-US');
function renderList() {
  const el = $('events');
  if (el.childElementCount !== events.length) {
    el.innerHTML = '';
    events.forEach((ev, i) => {
      const d = document.createElement('div'); d.className = 'ev'; d.onclick = () => seek(i + 1);
      const gap = i ? ev.t - events[i - 1].t : ev.t;
      d.innerHTML = `<span class="dot"></span><span>${ev.seq}</span><span class="z">Z${ev.zone}</span><span class="k k-${ev.kind}">${ev.kind === 'refined' ? '&#9670; refined' : ev.kind === 'replace' ? '&#8635; replace' : ev.kind === 'final' ? '&#10003; final' : ev.kind}</span><span class="tt">T+${mmss(ev.t)}</span><span class="dl">+${Math.round(gap)}s</span><span class="p">${fmt(ev.points)}</span>`;
      el.appendChild(d);
    });
  }
  [...el.children].forEach((d, i) => { d.className = 'ev' + (i < applied ? ' done' : '') + (i === applied - 1 ? ' cur' : ''); });
  const cur = el.children[applied - 1]; if (cur && cur !== renderList.last) { renderList.last = cur; cur.scrollIntoView({ block: 'nearest' }); }
}
function reconClock() {   // reconstruction clock: live = status.t (else last event t + time since); replay = replay clock
  if (!LIVE) return simT;
  if (statusInfo && statusInfo.t != null) return statusInfo.t + (performance.now() - statusAt) / 1000;
  if (events.length && lastArrival) return events[events.length - 1].t + (Date.now() - lastArrival) / 1000;
  return (performance.now() - pageOpened) / 1000;
}
let statsAt = 0;
function updateStats() {
  const nowT = performance.now(); if (nowT - statsAt < 200) return; statsAt = nowT;
  const kv = (l, v, c = '') => `<div class="kv ${c}"><span class="lbl">${l}</span><b>${v}</b></div>`;
  const cur = events[applied - 1], st = cur && cur._st;
  const fin = cur && cur.kind === 'final';
  let h = '';
  if (LIVE) h += `<div class="live"><i></i>LIVE${lastArrival ? `  <span class="lbl">last event ${((Date.now() - lastArrival) / 1000).toFixed(0)} s ago</span>` : `  <span class="lbl">waiting for stream...</span>`}</div>`;
  h += `<div class="hud">${fin ? kv('총 소요', mmss(cur.t, true), 'tot') : kv('경과 T+', mmss(reconClock()))}${kv('PTS', fmt(cur ? cur.points : 0))}${kv('FRAME', frameMs.toFixed(0) + ' ms')}${kv('SPEED', LIVE ? 'live' : speed + 'x')}${kv('EVT', applied + '/' + events.length + (busy ? ' ...' : ''))}</div>`;
  let frac = events.length ? applied / events.length : 0, lab = `EVENTS ${applied}/${events.length}`;
  if (LIVE && statusInfo && statusInfo.positions_total) { frac = statusInfo.positions / statusInfo.positions_total; lab = `POSITIONS ${statusInfo.positions}/${statusInfo.positions_total}`; }
  h += `<div><div class="barl"><span class="lbl">${lab}</span><span class="lbl">${(frac * 100).toFixed(0)}%</span></div><div class="bar"><i style="width:${(Math.min(1, frac) * 100).toFixed(1)}%"></i></div></div>`;
  if (cur && st && st.moved != null) h += `<div class="cnt"><span class="c-mv">MOVED<b>${fmt(st.moved)}</b></span><span class="c-ap">NEW<b>${fmt(st.added)}</b></span><span class="c-ds">GONE<b>${fmt(st.removed)}</b></span></div>`;
  h += `<div class="sm">${statLines.cur.replace(/</g, '&lt;').replace(/\n/g, '<br>')}</div>`;
  $('stats').innerHTML = h;
}

async function loadManifest() {
  const r = await fetch(base + 'manifest.json?_=' + Date.now());
  if (!r.ok) throw new Error('manifest ' + r.status);
  const m = await r.json();
  if (m.version !== 3 && m.version !== 4) throw new Error('unsupported manifest version ' + m.version + ' (viewer needs v3/v4; see conv_v2_to_v3.py)');
  manifest = m;
  if (manifest.events.length > events.length) lastArrival = Date.now();
  const old = new Map(events.map(e => [e.seq, e]));
  events = manifest.events.slice().sort((a, b) => a.seq - b.seq).map(e => old.get(e.seq) || e);
  if (manifest.capacity) growTo(manifest.capacity);
  renderList(); prefetchAhead();
}
$('play').onclick = () => { playing = !playing; $('play').innerHTML = playing ? '&#10074;&#10074;' : '&#9654;'; };
$('next').onclick = () => { playing = false; $('play').innerHTML = '&#9654;'; step(1); };
$('prev').onclick = () => { playing = false; $('play').innerHTML = '&#9654;'; stepBack(); };
$('speed').onchange = e => { speed = +e.target.value; };
$('size').oninput = e => { mat.uniforms.uSize.value = +e.target.value; };
$('mode').onchange = e => { mat.uniforms.uMode.value = +e.target.value; mat.uniforms.uFadeOut.value = +e.target.value === 2 ? 1.2 : FADE_S; };
$('fit').onclick = fitView;
$('evhead').onclick = () => { const c = $('events').classList.toggle('collapsed'); $('evhead').innerHTML = (c ? '&#9656;' : '&#9662;') + ' EVENTS'; };

function frame() {
  const now = performance.now(), dt = (now - lastFrame) / 1000; lastFrame = now;
  if (playing && !busy) {
    simT += dt * speed;
    if (applied < events.length && (LIVE || events[applied].t <= simT)) {
      if (LIVE) simT = events[applied].t;
      busy = true;
      applyNext(true).catch(e => { console.error(e); playing = false; }).finally(() => { busy = false; });
    }
  }
  flushUpload();                               // the only per-frame JS work for the cloud; animation runs on the GPU
  const tAnim = performance.now() - now; animMs = animMs * 0.9 + tAnim * 0.1; frameMs = frameMs * 0.9 + dt * 100;
  mat.uniforms.uTime.value = now / 1000;
  updateMonitor(now);
  const showWait = updateWait(now, dt);
  { const k = LIVE ? waitFade : 0; renderer.setClearColor(new THREE.Color().setRGB((4 + 16 * (1 - k)) / 255, (8 + 12 * (1 - k)) / 255, (15 + 11 * (1 - k)) / 255, THREE.SRGBColorSpace)); }
  controls.update(); renderer.render(scene, camera);
  if (showWait) { renderer.autoClear = false; renderer.render(waitScene, waitCam); renderer.autoClear = true; }
  updateStats();
  window.__viewer = { animMs, frameMs, uploadPts: lastUploadPts, pendingUpload: dq.length, applied, total: events.length, shown: applied ? events[applied - 1].points : 0, busy, events, waitFade, monitor: monEl.style.display, monText: monEl.innerText, front: frontZone };   // for headless tests
  requestAnimationFrame(frame);
}
try {
  growTo(65536);
  if (LIVE) { pollStatus(); setInterval(pollStatus, 1000); }
  if (LIVE) await loadManifest().catch(() => {});   // the stream may not exist yet; the poll below picks it up
  else await loadManifest();
  simT = events.length ? events[0].t - 0.01 : 0;
  setInterval(() => loadManifest().catch(e => { if (!/manifest 404/.test(e.message)) console.error(e); $('stats').dataset.err = e.message; }), LIVE ? 1000 : 3000);
  frame();
} catch (e) { $('stats').textContent = 'error: ' + e.message; console.error(e); }
