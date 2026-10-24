// Point-segment PLY decoding (protocol.ts PointChunk).
//
// Pure data — no DOM, no Three.js, no Node built-ins — so the board's parse
// worker, the demo asset script and the tests all run the SAME decoder.
//
// The wire record is 27 bytes (x y z f32, r g b u8, nx ny nz f32, little
// endian). 27 is not a multiple of 4, and WebGL refuses float attributes at an
// unaligned stride, so the record cannot be uploaded as-is: it is repacked into
// one array per attribute here. Header type names come in both spellings
// (`float`/`uchar` from the voxel pass, `float32`/`uint8` from the pipeline).
//
// Nothing is invented: a point whose position is not finite is dropped and
// counted, and a normal that is not unit length is written as zero (the shader
// then shades it flat) rather than guessed.

const TYPE_SIZE: Record<string, number> = {
  char: 1,
  int8: 1,
  uchar: 1,
  uint8: 1,
  short: 2,
  int16: 2,
  ushort: 2,
  uint16: 2,
  int: 4,
  int32: 4,
  uint: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
};

const FLOAT_TYPES = new Set(['float', 'float32']);
const BYTE_TYPES = new Set(['uchar', 'uint8']);

export interface PlyLayout {
  count: number;
  /** Byte offset of the first vertex record. */
  dataOffset: number;
  stride: number;
  /** Byte offset of each property inside a record. */
  offsets: Record<string, number>;
  types: Record<string, string>;
}

export interface PointCloud {
  /** Points kept (non-finite positions dropped). */
  count: number;
  /** Points in the file. */
  declared: number;
  positions: Float32Array;
  /** RGB, 0..255. Null when the file has no colour. */
  colors: Uint8Array | null;
  /** Unit normals ×127. Null when the file has no normals; a zero triple marks
   *  a point whose stored normal was not unit length. */
  normals: Int8Array | null;
  /** How many stored normals were rejected (not finite, not unit length). */
  badNormals: number;
  bbox: { min: [number, number, number]; max: [number, number, number] };
  /** The same box over the 1st..99th percentile on each axis: a few stray
   *  points far off the capture move `bbox`, not this. For framing. */
  core: { min: [number, number, number]; max: [number, number, number] };
  /** Robust height range (1st / 99th percentile of z), metres. */
  zLow: number;
  zHigh: number;
}

/** Read the ASCII header. Throws on anything this decoder cannot honour. */
export function readPlyLayout(buf: ArrayBuffer): PlyLayout {
  const head = new Uint8Array(buf, 0, Math.min(buf.byteLength, 4096));
  let text = '';
  for (let i = 0; i < head.length; i++) text += String.fromCharCode(head[i]);
  const end = text.indexOf('end_header');
  if (!text.startsWith('ply') || end < 0) throw new Error('not a PLY file (no header)');
  const nl = text.indexOf('\n', end);
  if (nl < 0) throw new Error('truncated PLY header');
  const lines = text.slice(0, end).split(/\r?\n/).map((l) => l.trim());

  const format = lines.find((l) => l.startsWith('format '));
  if (!format || !format.includes('binary_little_endian')) {
    throw new Error(`unsupported PLY format: ${format ?? 'none'}`);
  }

  let count = -1;
  let inVertex = false;
  let sawOtherElementFirst = false;
  let stride = 0;
  const offsets: Record<string, number> = {};
  const types: Record<string, string> = {};
  for (const line of lines) {
    const parts = line.split(/\s+/);
    if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
      if (inVertex) count = Number(parts[2]);
      else if (count < 0) sawOtherElementFirst = true;
      continue;
    }
    if (parts[0] === 'property' && inVertex) {
      if (parts[1] === 'list') throw new Error('list properties on vertex are not supported');
      const size = TYPE_SIZE[parts[1]];
      if (!size) throw new Error(`unknown PLY type ${parts[1]}`);
      offsets[parts[2]] = stride;
      types[parts[2]] = parts[1];
      stride += size;
    }
  }
  if (sawOtherElementFirst) throw new Error('vertex must be the first PLY element');
  if (!Number.isInteger(count) || count < 0) throw new Error('PLY has no vertex element');
  for (const k of ['x', 'y', 'z']) {
    if (!FLOAT_TYPES.has(types[k])) throw new Error(`PLY ${k} must be float32`);
  }
  const dataOffset = nl + 1;
  if (dataOffset + count * stride > buf.byteLength) {
    throw new Error(
      `PLY body truncated: ${count} × ${stride} B expected, ${buf.byteLength - dataOffset} B present`,
    );
  }
  return { count, dataOffset, stride, offsets, types };
}

/** Percentile of a value set via a histogram (exact enough for a colour ramp). */
function histPercentiles(
  zs: Float32Array,
  n: number,
  lo: number,
  hi: number,
  qs: number[],
): number[] {
  if (n === 0) return qs.map(() => 0);
  if (!(hi > lo)) return qs.map(() => lo);
  const BINS = 2048;
  const h = new Uint32Array(BINS);
  const k = (BINS - 1) / (hi - lo);
  for (let i = 0; i < n; i++) h[Math.floor((zs[i] - lo) * k)]++;
  return qs.map((q) => {
    const target = q * n;
    let acc = 0;
    for (let b = 0; b < BINS; b++) {
      acc += h[b];
      if (acc >= target) return lo + (b + 0.5) / k;
    }
    return hi;
  });
}

export function parsePointPly(buf: ArrayBuffer): PointCloud {
  const L = readPlyLayout(buf);
  const dv = new DataView(buf, L.dataOffset, L.count * L.stride);
  const ox = L.offsets.x;
  const oy = L.offsets.y;
  const oz = L.offsets.z;
  const hasColor = ['red', 'green', 'blue'].every((k) => BYTE_TYPES.has(L.types[k]));
  const hasNormal = ['nx', 'ny', 'nz'].every((k) => FLOAT_TYPES.has(L.types[k]));
  const or = L.offsets.red;
  const og = L.offsets.green;
  const ob = L.offsets.blue;
  const onx = L.offsets.nx;
  const ony = L.offsets.ny;
  const onz = L.offsets.nz;

  const positions = new Float32Array(L.count * 3);
  const colors = hasColor ? new Uint8Array(L.count * 3) : null;
  const normals = hasNormal ? new Int8Array(L.count * 3) : null;
  const xs = new Float32Array(L.count);
  const ys = new Float32Array(L.count);
  const zs = new Float32Array(L.count);

  let n = 0;
  let badNormals = 0;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0, base = 0; i < L.count; i++, base += L.stride) {
    const x = dv.getFloat32(base + ox, true);
    const y = dv.getFloat32(base + oy, true);
    const z = dv.getFloat32(base + oz, true);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    const j = n * 3;
    positions[j] = x;
    positions[j + 1] = y;
    positions[j + 2] = z;
    xs[n] = x;
    ys[n] = y;
    zs[n] = z;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
    if (colors) {
      colors[j] = dv.getUint8(base + or);
      colors[j + 1] = dv.getUint8(base + og);
      colors[j + 2] = dv.getUint8(base + ob);
    }
    if (normals) {
      const nx = dv.getFloat32(base + onx, true);
      const ny = dv.getFloat32(base + ony, true);
      const nz = dv.getFloat32(base + onz, true);
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (Number.isFinite(len) && Math.abs(len - 1) < 0.05) {
        // Renormalise first: 1.02 × 127 would wrap to a negative Int8.
        const k = 127 / len;
        normals[j] = Math.round(nx * k);
        normals[j + 1] = Math.round(ny * k);
        normals[j + 2] = Math.round(nz * k);
      } else {
        badNormals++;
      }
    }
    n++;
  }

  const [zLow, zHigh] = histPercentiles(zs, n, minZ, maxZ, [0.01, 0.99]);
  const [xLow, xHigh] = histPercentiles(xs, n, minX, maxX, [0.01, 0.99]);
  const [yLow, yHigh] = histPercentiles(ys, n, minY, maxY, [0.01, 0.99]);
  const empty = n === 0;
  return {
    count: n,
    declared: L.count,
    positions: n === L.count ? positions : positions.slice(0, n * 3),
    colors: colors && (n === L.count ? colors : colors.slice(0, n * 3)),
    normals: normals && (n === L.count ? normals : normals.slice(0, n * 3)),
    badNormals,
    bbox: empty
      ? { min: [0, 0, 0], max: [0, 0, 0] }
      : { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
    core: empty
      ? { min: [0, 0, 0], max: [0, 0, 0] }
      : { min: [xLow, yLow, zLow], max: [xHigh, yHigh, zHigh] },
    zLow,
    zHigh,
  };
}

/**
 * Put the points in a random (seeded, so repeatable) order, in place.
 *
 * Any prefix of a shuffled cloud is a uniform sample of the whole, so a
 * renderer over its point budget can draw the first N of each segment and
 * thin everything evenly instead of dropping whole regions.
 */
export function shufflePoints(pc: PointCloud, seed = 1): void {
  let s = seed >>> 0 || 1;
  const rand = (): number => {
    // xorshift32
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
  const p = pc.positions;
  const c = pc.colors;
  const nm = pc.normals;
  for (let i = pc.count - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    if (i === j) continue;
    const a = i * 3;
    const b = j * 3;
    for (let k = 0; k < 3; k++) {
      const tp = p[a + k];
      p[a + k] = p[b + k];
      p[b + k] = tp;
      if (c) {
        const tc = c[a + k];
        c[a + k] = c[b + k];
        c[b + k] = tc;
      }
      if (nm) {
        const tn = nm[a + k];
        nm[a + k] = nm[b + k];
        nm[b + k] = tn;
      }
    }
  }
}

/** Encode points in the 27-byte wire record. Used by tests and fixtures. */
export function encodePointPly(
  positions: ArrayLike<number>,
  colors?: ArrayLike<number> | null,
  normals?: ArrayLike<number> | null,
): ArrayBuffer {
  const count = Math.floor(positions.length / 3);
  const header =
    'ply\nformat binary_little_endian 1.0\n' +
    `element vertex ${count}\n` +
    'property float x\nproperty float y\nproperty float z\n' +
    'property uchar red\nproperty uchar green\nproperty uchar blue\n' +
    'property float nx\nproperty float ny\nproperty float nz\n' +
    'end_header\n';
  const buf = new ArrayBuffer(header.length + count * 27);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < header.length; i++) u8[i] = header.charCodeAt(i);
  const dv = new DataView(buf, header.length);
  for (let i = 0; i < count; i++) {
    const b = i * 27;
    dv.setFloat32(b, positions[i * 3], true);
    dv.setFloat32(b + 4, positions[i * 3 + 1], true);
    dv.setFloat32(b + 8, positions[i * 3 + 2], true);
    dv.setUint8(b + 12, colors ? colors[i * 3] : 200);
    dv.setUint8(b + 13, colors ? colors[i * 3 + 1] : 200);
    dv.setUint8(b + 14, colors ? colors[i * 3 + 2] : 200);
    dv.setFloat32(b + 15, normals ? normals[i * 3] : 0, true);
    dv.setFloat32(b + 19, normals ? normals[i * 3 + 1] : 0, true);
    dv.setFloat32(b + 23, normals ? normals[i * 3 + 2] : 1, true);
  }
  return buf;
}
