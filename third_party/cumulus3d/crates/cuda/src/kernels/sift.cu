/*
 * sift.cu
 *
 * Copyright (c) 2026 ParkSangWoo
 *
 * Author(s):
 *
 *      ParkSangWoo <dev.parksangwoo@gmail.com>
 *
 *
 * This file is part of cumulus3d.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Third-party notices: parts of the algorithms, default parameters and data
 * formats in this file follow other open-source projects. Their copyright
 * notices and licenses are reproduced in THIRD_PARTY_NOTICES.md.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// The whole SIFT. The front part (normalization, upsampling, blur, downsampling) uses the same formulas and operation order as
// the CPU pyramid of cumulus3d-features and is compiled with fused multiply-add off (--fmad=false), so it matches the CPU bit for
// bit. The back part (DoG, detection, orientation, selection, descriptors) uses the same formulas as the CPU detect.rs /
// orient.rs / mod.rs, but the summation order (warp split) and transcendental implementations differ, so the result is not
// bit-identical (same quality). The summation order is fixed, so every run gives the same result.

typedef unsigned char u8;

// dst[y·w + x] = src[y·sw + x] / 255
extern "C" __global__ void sift_normalize(const u8* __restrict__ src, int sw, int w, int h, float* __restrict__ dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dst[(size_t)y * w + x] = (float)src[(size_t)y * sw + x] / 255.0f;
}

// 2x upsampling: output (2r+a, 2c+b), edge replication.
extern "C" __global__ void sift_upsample2(const float* __restrict__ src, int w, int h, float* __restrict__ dst) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= w || yy >= 2 * h) return;
    int r = yy / 2;
    float a = (float)(yy % 2) * 0.5f;
    int r1 = min(r + 1, h - 1);
    int c1 = min(c + 1, w - 1);
    const float* row0 = src + (size_t)r * w;
    const float* row1 = src + (size_t)r1 * w;
    float top0 = row0[c], bot0 = row1[c];
    float v0 = top0 + a * (bot0 - top0);
    float v1 = row0[c1] + a * (row1[c1] - row0[c1]);
    float* out = dst + (size_t)yy * (2 * w);
    out[2 * c] = v0;
    out[2 * c + 1] = v0 + 0.5f * (v1 - v0);
}

#define TILE_W 128
#define MAX_R 16

// Horizontal blur: out = k_r·p + Σ_{i<r} k_i·(row[x+i−r] + row[x+r−i]) (edge replication), i ascending.
extern "C" __global__ void sift_blur_h(const float* __restrict__ src, float* __restrict__ dst, int w, int h, const float* __restrict__ k, int r) {
    __shared__ float s[TILE_W + 2 * MAX_R];
    __shared__ float kk[2 * MAX_R + 1];
    int y = blockIdx.y;
    int x0 = blockIdx.x * TILE_W;
    const float* row = src + (size_t)y * w;
    for (int i = threadIdx.x; i < TILE_W + 2 * r; i += blockDim.x) {
        int xx = min(max(x0 + i - r, 0), w - 1);
        s[i] = row[xx];
    }
    for (int i = threadIdx.x; i <= 2 * r; i += blockDim.x) kk[i] = k[i];
    __syncthreads();
    int x = x0 + threadIdx.x;
    if (x >= w) return;
    int t = threadIdx.x;
    float acc = kk[r] * s[t + r];
    for (int i = 0; i < r; ++i) acc += kk[i] * (s[t + i] + s[t + 2 * r - i]);
    dst[(size_t)y * w + x] = acc;
}

// Vertical blur: the same formula along the row direction (row index fixed).
extern "C" __global__ void sift_blur_v(const float* __restrict__ src, float* __restrict__ dst, int w, int h, const float* __restrict__ k, int r) {
    __shared__ float kk[2 * MAX_R + 1];
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i <= 2 * r; i += blockDim.x * blockDim.y) kk[i] = k[i];
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float acc = kk[r] * __ldg(src + (size_t)y * w + x);
    for (int i = 0; i < r; ++i) {
        int ya = min(max(y + i - r, 0), h - 1);
        int yb = min(max(y + r - i, 0), h - 1);
        acc += kk[i] * (__ldg(src + (size_t)ya * w + x) + __ldg(src + (size_t)yb * w + x));
    }
    dst[(size_t)y * w + x] = acc;
}

// Take only even rows and columns for 1/2 downscaling.
extern "C" __global__ void sift_decimate2(const float* __restrict__ src, int w, float* __restrict__ dst, int w2, int h2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    dst[(size_t)y * w2 + x] = src[(size_t)(2 * y) * w + 2 * x];
}

// ---------------------------------------------------------------------------------------------
// Detection, orientation, selection, descriptors
// ---------------------------------------------------------------------------------------------

#define PI_F 3.14159265358979f
#define TWO_PI_F (2.0f * PI_F)
#define SQRT2_F 1.41421356237f
#define ORI_NB 36
#define ORI_WARPS 4
#define DESC_WARPS 2
#define RANK_T 256

// Rust f32::rem_euclid(r) (r > 0).
__device__ __forceinline__ float sift_rem_euclid(float a, float r) {
    float m = fmodf(a, r);
    return m < 0.0f ? m + r : m;
}

// DoG: dog[d·n + i] = hi[i] − lo[i] (CPU detect::dog).
extern "C" __global__ void sift_dog(const float* __restrict__ hi, const float* __restrict__ lo, float* __restrict__ dog, int d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dog[(size_t)d * n + i] = hi[i] - lo[i];
}

// First decide maximum vs minimum from the left and right neighbors on the same level; the remaining 24 neighbors allow
// equal values (CPU is_extremum).
__device__ bool sift_is_extremum(const float* dm, const float* d0, const float* dp, int w, int i, float v) {
    float l = d0[i - 1], rt = d0[i + 1];
    bool is_max;
    if (v > fmaxf(l, rt)) {
        is_max = true;
    } else if (v < fminf(l, rt)) {
        is_max = false;
    } else {
        return false;
    }
    const float* imgs[3] = {d0, dm, dp};
    for (int k = 0; k < 3; ++k) {
        const float* img = imgs[k];
        for (int dy = -1; dy <= 1; ++dy) {
            if (k == 0 && dy == 0) continue;
            int base = i + dy * w;
            for (int dx = -1; dx <= 1; ++dx) {
                float n = img[base + dx];
                if (is_max) {
                    if (v < n) return false;
                } else if (v > n) {
                    return false;
                }
            }
        }
    }
    return true;
}

// 3×3 partial-pivot Gaussian elimination (CPU solve3). False if the pivot magnitude < 1e−10.
__device__ bool sift_solve3(float a[3][3], float b[3], float x[3]) {
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int r = col + 1; r < 3; ++r)
            if (fabsf(a[r][col]) > fabsf(a[piv][col])) piv = r;
        if (fabsf(a[piv][col]) < 1e-10f) return false;
        if (piv != col) {
            for (int c = 0; c < 3; ++c) {
                float t = a[col][c];
                a[col][c] = a[piv][c];
                a[piv][c] = t;
            }
            float t = b[col];
            b[col] = b[piv];
            b[piv] = t;
        }
        for (int r = col + 1; r < 3; ++r) {
            float f = a[r][col] / a[col][col];
            for (int c = col; c < 3; ++c) a[r][c] -= f * a[col][c];
            b[r] -= f * b[col];
        }
    }
    for (int r = 2; r >= 0; --r) {
        float s = b[r];
        for (int c = r + 1; c < 3; ++c) s -= a[r][c] * x[c];
        x[r] = s / a[r][r];
    }
    return true;
}

// Sub-pixel refinement (CPU refine). On success out = (x, y, σ_o, response) (in octave pixel units).
__device__ bool sift_refine(const float* dm, const float* d0, const float* dp, int w, int h, int r, int c, int j, int iters,
                            int reject, float peak, float sigma0, float kk, float out[4]) {
    if (iters < 1) iters = 1;
    for (int it = 0; it < iters; ++it) {
        int i = r * w + c;
        float v = d0[i];
        float fx = 0.5f * (d0[i + 1] - d0[i - 1]);
        float fy = 0.5f * (d0[i + w] - d0[i - w]);
        float fs = 0.5f * (dp[i] - dm[i]);
        float fxx = d0[i - 1] + d0[i + 1] - 2.0f * v;
        float fyy = d0[i - w] + d0[i + w] - 2.0f * v;
        float fss = dp[i] + dm[i] - 2.0f * v;
        float fxy = 0.25f * (d0[i + w + 1] + d0[i - w - 1] - d0[i + w - 1] - d0[i - w + 1]);
        float fxs = 0.25f * (dp[i + 1] + dm[i - 1] - dp[i - 1] - dm[i + 1]);
        float fys = 0.25f * (dp[i + w] + dm[i - w] - dp[i - w] - dm[i + w]);
        float hm[3][3] = {{fxx, fxy, fxs}, {fxy, fyy, fys}, {fxs, fys, fss}};
        float rhs[3] = {-fx, -fy, -fs};
        float d[3] = {0.0f, 0.0f, 0.0f};
        float resp;
        if (!sift_solve3(hm, rhs, d)) {
            if (reject) return false;
            d[0] = 0.0f;
            d[1] = 0.0f;
            d[2] = 0.0f;
            resp = v;
        } else {
            if (it + 1 < iters) {
                bool moved = false;
                if (d[0] > 0.6f && c + 2 < w) {
                    c += 1;
                    moved = true;
                } else if (d[0] < -0.6f && c > 1) {
                    c -= 1;
                    moved = true;
                }
                if (d[1] > 0.6f && r + 2 < h) {
                    r += 1;
                    moved = true;
                } else if (d[1] < -0.6f && r > 1) {
                    r -= 1;
                    moved = true;
                }
                if (moved) continue;
            }
            resp = v + 0.5f * (fx * d[0] + fy * d[1] + fs * d[2]);
            if (fabsf(resp) <= peak || fabsf(d[0]) >= 1.0f || fabsf(d[1]) >= 1.0f || fabsf(d[2]) >= 1.0f) return false;
        }
        out[0] = (float)c + 0.5f + d[0];
        out[1] = (float)r + 0.5f + d[1];
        out[2] = sigma0 * powf(kk, (float)j + d[2]);
        out[3] = resp;
        return true;
    }
    return false;
}

// Extremum detection + edge suppression + sub-pixel refinement at detection level j = blockIdx.z (0..S) of one octave
// (CPU detect_level). dog is D_0..D_{S+1} concatenated as [d·w·h]. Candidates are appended atomically:
// cf[4·k] = (x, y, σ_o, response), ci[2·k] = (detection pixel index r·w + c, level index lv_base + j).
// counters[0] = total candidate count, counters[1 + level] = per-level candidate count (counted even beyond the capacity).
extern "C" __global__ void sift_detect(const float* __restrict__ dog, int w, int h, int lv_base, float peak, float edge, int iters,
                                       int reject, float sigma0, float kk, float* __restrict__ cf, int* __restrict__ ci, int cap,
                                       int* __restrict__ counters) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z;
    if (c < 1 || r < 1 || c >= w - 1 || r >= h - 1) return;
    size_t n = (size_t)w * h;
    const float* dm = dog + (size_t)j * n;
    const float* d0 = dm + n;
    const float* dp = d0 + n;
    int i = r * w + c;
    float v = d0[i];
    if (fabsf(v) <= 0.8f * peak) return;
    if (!sift_is_extremum(dm, d0, dp, w, i, v)) return;
    // Edge suppression (3×3 at the same level).
    float fxx = d0[i - 1] + d0[i + 1] - 2.0f * v;
    float fyy = d0[i - w] + d0[i + w] - 2.0f * v;
    float fxy = 0.25f * (d0[i + w + 1] + d0[i - w - 1] - d0[i + w - 1] - d0[i - w + 1]);
    float det = fxx * fyy - fxy * fxy;
    float tr = fxx + fyy;
    float edge_ratio = (edge + 1.0f) * (edge + 1.0f) / edge;
    if (det <= 0.0f || tr * tr > edge_ratio * det) return;
    float o[4];
    if (!sift_refine(dm, d0, dp, w, h, r, c, j, iters, reject, peak, sigma0, kk, o)) return;
    int L = lv_base + j;
    int k = atomicAdd(&counters[0], 1);
    atomicAdd(&counters[1 + L], 1);
    if (k < cap) {
        cf[4 * k] = o[0];
        cf[4 * k + 1] = o[1];
        cf[4 * k + 2] = o[2];
        cf[4 * k + 3] = o[3];
        ci[2 * k] = i;
        ci[2 * k + 1] = L;
    }
}

// Gradient (CPU grad_at): magnitude ½‖∇‖, angle atan2(dy, dx). Assumes 1 ≤ x ≤ w−2, 1 ≤ y ≤ h−2.
__device__ __forceinline__ void sift_grad(const float* g, int w, int x, int y, float* m, float* th) {
    size_t i = (size_t)y * w + x;
    float dx = g[i + 1] - g[i - 1];
    float dy = g[i + w] - g[i - w];
    *m = 0.5f * sqrtf(dx * dx + dy * dy);
    *th = atan2f(dy, dx);
}

// 16-bit quantized angle (CPU quantize_angle).
__device__ __forceinline__ int sift_quantize_angle(float theta) {
    float f = sift_rem_euclid(theta / TWO_PI_F, 1.0f);
    unsigned int q = (unsigned int)floorf(f * 65535.0f);
    return (int)min(q, 65534u);
}

// Orientation assignment (CPU orient::orientations): one warp per candidate. Each lane accumulates into its own histogram
// column (shared memory [bin][lane]), the columns are summed in a fixed order, then lane 0 does the smoothing and peak
// selection. Items are appended atomically: items[3·k] = (candidate index, quantized angle, orientation slot).
// counters[0] total, counters[1 + level] per level.
// lv_ptr[level] = device address of that level's Gaussian image, lv_i[4·level] = (w, h, octave, j).
extern "C" __global__ void sift_orient(const float* __restrict__ cf, const int* __restrict__ ci, int ncand,
                                       const unsigned long long* __restrict__ lv_ptr, const int* __restrict__ lv_i, int max_ori,
                                       int interp, int upright, int* __restrict__ items, int cap, int* __restrict__ counters) {
    __shared__ float hs[ORI_WARPS][ORI_NB * 32];
    __shared__ float hf[ORI_WARPS][ORI_NB];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int cid = blockIdx.x * ORI_WARPS + warp;
    if (cid >= ncand) return;  // same decision for the whole warp
    int L = ci[2 * cid + 1];
    if (upright) {
        if (lane == 0) {
            int k = atomicAdd(&counters[0], 1);
            atomicAdd(&counters[1 + L], 1);
            if (k < cap) {
                items[3 * k] = cid;
                items[3 * k + 1] = 0;
                items[3 * k + 2] = 0;
            }
        }
        return;
    }
    const float* g = (const float*)lv_ptr[L];
    int w = lv_i[4 * L], h = lv_i[4 * L + 1];
    float x = cf[4 * cid], y = cf[4 * cid + 1], sigma = cf[4 * cid + 2];
    float sw = 1.5f * sigma;
    float rad = 2.0f * sw;
    int x0 = max((int)floorf(x - rad), 1);
    int x1 = min((int)floorf(x + rad), w - 2);
    int y0 = max((int)floorf(y - rad), 1);
    int y1 = min((int)floorf(y + rad), h - 2);
    float r2 = rad * rad + 0.5f;
    float inv = 1.0f / (2.0f * sw * sw);
    float bscale = (float)ORI_NB / TWO_PI_F;
    float* hl = hs[warp];
    for (int b = 0; b < ORI_NB; ++b) hl[b * 32 + lane] = 0.0f;
    int nx = x1 - x0 + 1, ny = y1 - y0 + 1;
    if (nx > 0 && ny > 0) {
        int tot = nx * ny;
        for (int p = lane; p < tot; p += 32) {
            int xi = x0 + p % nx;
            int yi = y0 + p / nx;
            float dy = (float)yi + 0.5f - y;
            float dx = (float)xi + 0.5f - x;
            float d2 = dx * dx + dy * dy;
            if (d2 >= r2) continue;
            float m, th;
            sift_grad(g, w, xi, yi, &m, &th);
            float wgt = m * expf(-d2 * inv);
            if (interp) {
                float t = th * bscale - 0.5f;
                float b0f = floorf(t);
                float f = t - b0f;
                int b0 = ((int)b0f % ORI_NB + ORI_NB) % ORI_NB;
                hl[b0 * 32 + lane] += wgt * (1.0f - f);
                hl[((b0 + 1) % ORI_NB) * 32 + lane] += wgt * f;
            } else {
                int b = (int)floorf(th * bscale);
                if (b < 0) b += ORI_NB;
                b = min(b, ORI_NB - 1);
                hl[b * 32 + lane] += wgt;
            }
        }
    }
    __syncwarp();
    for (int b = lane; b < ORI_NB; b += 32) {
        float s = 0.0f;
        for (int t = 0; t < 32; ++t) s += hl[b * 32 + t];
        hf[warp][b] = s;
    }
    __syncwarp();
    if (lane != 0) return;
    float hist[ORI_NB];
    for (int i = 0; i < ORI_NB; ++i) hist[i] = hf[warp][i];
    // Circular 3-tap average, 6 times.
    for (int it = 0; it < 6; ++it) {
        float prev[ORI_NB];
        for (int i = 0; i < ORI_NB; ++i) prev[i] = hist[i];
        for (int i = 0; i < ORI_NB; ++i) hist[i] = (prev[(i + ORI_NB - 1) % ORI_NB] + prev[i] + prev[(i + 1) % ORI_NB]) / 3.0f;
    }
    float hmax = 0.0f;
    for (int i = 0; i < ORI_NB; ++i) hmax = fmaxf(hmax, hist[i]);
    int nsel = max(max_ori, 1);
    unsigned long long taken = 0ull;
    int outq[ORI_NB];
    int nout = 0;
    // Peaks in descending magnitude (ties by ascending bin index), nsel of them, only one per quantized angle.
    for (int t = 0; t < nsel; ++t) {
        int best = -1;
        float bv = 0.0f;
        for (int i = 0; i < ORI_NB; ++i) {
            if ((taken >> i) & 1ull) continue;
            float hl_ = hist[(i + ORI_NB - 1) % ORI_NB], hc = hist[i], hr = hist[(i + 1) % ORI_NB];
            if (hc > 0.8f * hmax && hc > hl_ && hc > hr) {
                if (best < 0 || hc > bv) {
                    best = i;
                    bv = hc;
                }
            }
        }
        if (best < 0) break;
        taken |= 1ull << best;
        float hl_ = hist[(best + ORI_NB - 1) % ORI_NB], hc = hist[best], hr = hist[(best + 1) % ORI_NB];
        float di = 0.5f * (hr - hl_) / (2.0f * hc - hr - hl_);
        float th = ((float)best + di + 0.5f) / bscale;
        int q = sift_quantize_angle(th);
        bool dup = false;
        for (int u = 0; u < nout; ++u) dup = dup || outq[u] == q;
        if (dup) continue;
        int slot = nout;
        outq[nout++] = q;
        int k = atomicAdd(&counters[0], 1);
        atomicAdd(&counters[1 + L], 1);
        if (k < cap) {
            items[3 * k] = cid;
            items[3 * k + 1] = q;
            items[3 * k + 2] = slot;
        }
    }
}

// Assigns the selection groups. lv_state[level]: 0 dropped, 1 kept whole, 2 picked by rank.
// If grid is false (TopK), group 0; if true (SpatialGrid), the grid cell index in input image coordinates + per-cell counts.
extern "C" __global__ void sift_group(const int* __restrict__ items, int n, const float* __restrict__ cf, const int* __restrict__ ci,
                                      const int* __restrict__ lv_i, const int* __restrict__ lv_state, int grid, int cells, float iw,
                                      float ih, int* __restrict__ group, int* __restrict__ bucket_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cid = items[3 * i];
    int L = ci[2 * cid + 1];
    if (lv_state[L] != 2) {
        group[i] = -1;
        return;
    }
    if (!grid) {
        group[i] = 0;
        return;
    }
    float so = ldexpf(1.0f, lv_i[4 * L + 2]);
    float x = so * (cf[4 * cid] - 0.5f) + 0.5f;
    float y = so * (cf[4 * cid + 1] - 0.5f) + 0.5f;
    // Rust `as usize` (negative, NaN -> 0), then clamp to cells − 1.
    float tx = x / iw * (float)cells;
    float ty = y / ih * (float)cells;
    int cx = tx > 0.0f ? (tx >= (float)cells ? cells - 1 : (int)tx) : 0;
    int cy = ty > 0.0f ? (ty >= (float)cells ? cells - 1 : (int)ty) : 0;
    int b = cy * cells + cx;
    group[i] = b;
    atomicAdd(&bucket_count[b], 1);
}

// Rank within a group: descending |response|, ties by (coarser level first, detection pixel order, orientation slot) = CPU
// list order.
extern "C" __global__ void sift_rank(const int* __restrict__ items, int n, const float* __restrict__ cf, const int* __restrict__ ci,
                                     const int* __restrict__ group, int* __restrict__ rank) {
    __shared__ float sr[RANK_T];
    __shared__ int sg[RANK_T], sl[RANK_T], sp[RANK_T], ss[RANK_T];
    int i = blockIdx.x * RANK_T + threadIdx.x;
    int gi = -1, li = 0, pi = 0, si = 0;
    float ri = 0.0f;
    if (i < n) {
        gi = group[i];
        int cid = items[3 * i];
        ri = fabsf(cf[4 * cid + 3]);
        pi = ci[2 * cid];
        li = ci[2 * cid + 1];
        si = items[3 * i + 2];
    }
    int cnt = 0;
    for (int base = 0; base < n; base += RANK_T) {
        int k = base + threadIdx.x;
        if (k < n) {
            int cid = items[3 * k];
            sg[threadIdx.x] = group[k];
            sr[threadIdx.x] = fabsf(cf[4 * cid + 3]);
            sp[threadIdx.x] = ci[2 * cid];
            sl[threadIdx.x] = ci[2 * cid + 1];
            ss[threadIdx.x] = items[3 * k + 2];
        } else {
            sg[threadIdx.x] = -2;
        }
        __syncthreads();
        if (gi >= 0) {
            int m = min(RANK_T, n - base);
            for (int t = 0; t < m; ++t) {
                if (sg[t] != gi) continue;
                float rt = sr[t];
                bool better = rt > ri || (rt == ri && (sl[t] > li || (sl[t] == li && (sp[t] < pi || (sp[t] == pi && ss[t] < si)))));
                cnt += better ? 1 : 0;
            }
        }
        __syncthreads();
    }
    if (i < n) rank[i] = cnt;
}

// List of items to keep. State 1 keeps all; state 2 keeps, in mode 1 (TopK), rank < limit,
// and in mode 2 (SpatialGrid), rank < rcut, or rank = rcut and cell index < bcut.
extern "C" __global__ void sift_keep(const int* __restrict__ items, int n, const int* __restrict__ ci, const int* __restrict__ lv_state,
                                     const int* __restrict__ group, const int* __restrict__ rank, int mode, int limit, int rcut, int bcut,
                                     int* __restrict__ keep, int cap, int* __restrict__ counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int L = ci[2 * items[3 * i] + 1];
    int st = lv_state[L];
    bool k = false;
    if (st == 1) {
        k = true;
    } else if (st == 2) {
        int rk = rank[i];
        if (mode == 1) {
            k = rk < limit;
        } else if (mode == 2) {
            k = rk < rcut || (rk == rcut && group[i] < bcut);
        }
    }
    if (!k) return;
    int idx = atomicAdd(counter, 1);
    if (idx < cap) keep[idx] = i;
}

__device__ __forceinline__ float sift_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void sift_l2(float d[4]) {
    float n = sqrtf(sift_warp_sum(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]));
    if (n > 0.0f) {
        for (int q = 0; q < 4; ++q) d[q] /= n;
    }
}

// 128-dimensional descriptor (CPU orient::descriptor + normalize_quantize): one warp per kept item.
// Each lane accumulates into its own histogram column ([bin][lane]), and lane l sums bins l, l+32, l+64, l+96 in a fixed order.
// Output k: fo[5·k] = (x, y, scale, output orientation, response) (input image pixel units), io[4·k] = (octave, level,
// detection pixel, orientation slot), desc[128·k..] = uint8 descriptor. Processed count = min(*kcount, cap).
extern "C" __global__ void sift_descriptor(const int* __restrict__ keep, const int* __restrict__ kcount, int cap,
                                           const int* __restrict__ items, const float* __restrict__ cf, const int* __restrict__ ci,
                                           const unsigned long long* __restrict__ lv_ptr, const int* __restrict__ lv_i, int l1root,
                                           float* __restrict__ fo, int* __restrict__ io, u8* __restrict__ desc) {
    __shared__ float hs[DESC_WARPS][128 * 32];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int k = blockIdx.x * DESC_WARPS + warp;
    int nk = min(*kcount, cap);
    if (k >= nk) return;  // same decision for the whole warp
    int it = keep[k];
    int cid = items[3 * it];
    int q = items[3 * it + 1];
    int slot = items[3 * it + 2];
    int pix = ci[2 * cid];
    int L = ci[2 * cid + 1];
    const float* g = (const float*)lv_ptr[L];
    int w = lv_i[4 * L], h = lv_i[4 * L + 1], o = lv_i[4 * L + 2], j = lv_i[4 * L + 3];
    float x = cf[4 * cid], y = cf[4 * cid + 1], sigma = cf[4 * cid + 2], resp = cf[4 * cid + 3];
    float theta = (float)q / 65535.0f * TWO_PI_F;
    float s = 3.0f * sigma;
    float sn, c;
    sincosf(theta, &sn, &c);
    float rad = s * 2.5f * SQRT2_F + 1.0f;
    int x0 = max((int)floorf(x - rad), 1);
    int x1 = min((int)ceilf(x + rad), w - 2);
    int y0 = max((int)floorf(y - rad), 1);
    int y1 = min((int)ceilf(y + rad), h - 2);
    float inv_s = 1.0f / s;
    float bpr = 4.0f / PI_F;
    float* hl = hs[warp];
    for (int b = 0; b < 128; ++b) hl[b * 32 + lane] = 0.0f;
    int nx = x1 - x0 + 1, ny = y1 - y0 + 1;
    if (nx > 0 && ny > 0) {
        int tot = nx * ny;
        for (int p = lane; p < tot; p += 32) {
            int xi = x0 + p % nx;
            int yi = y0 + p / nx;
            float dy = (float)yi + 0.5f - y;
            float dx = (float)xi + 0.5f - x;
            // Rotated cell coordinates (relative to the keypoint center).
            float rx = (c * dx + sn * dy) * inv_s;
            float ry = (c * dy - sn * dx) * inv_s;
            float fx = rx + 1.5f;
            float fy = ry + 1.5f;
            if (fx <= -1.0f || fx >= 4.0f || fy <= -1.0f || fy >= 4.0f) continue;
            float m, th;
            sift_grad(g, w, xi, yi, &m, &th);
            float gw = expf(-(rx * rx + ry * ry) / 8.0f) * m;
            float t = sift_rem_euclid((theta - th) * bpr, 8.0f);
            float tb = floorf(t);
            float tf = t - tb;
            int b0 = ((int)tb) % 8;
            int b1 = (b0 + 1) % 8;
            int ix0 = (int)floorf(fx);
            int iy0 = (int)floorf(fy);
            float wx1 = fx - (float)ix0;
            float wy1 = fy - (float)iy0;
            for (int a = 0; a < 2; ++a) {
                int iy = iy0 + a;
                float wy = a == 0 ? 1.0f - wy1 : wy1;
                if (iy < 0 || iy >= 4 || wy <= 0.0f) continue;
                for (int bb = 0; bb < 2; ++bb) {
                    int ix = ix0 + bb;
                    float wx = bb == 0 ? 1.0f - wx1 : wx1;
                    if (ix < 0 || ix >= 4 || wx <= 0.0f) continue;
                    int base = (iy * 4 + ix) * 8;
                    float ww = gw * wx * wy;
                    hl[(base + b0) * 32 + lane] += ww * (1.0f - tf);
                    hl[(base + b1) * 32 + lane] += ww * tf;
                }
            }
        }
    }
    __syncwarp();
    float d[4];
    for (int qq = 0; qq < 4; ++qq) {
        int b = lane + 32 * qq;
        float acc = 0.0f;
        // Rotate each lane's starting column to avoid shared memory bank conflicts (the order per lane is fixed).
        for (int t = 0; t < 32; ++t) acc += hl[b * 32 + ((t + lane) & 31)];
        d[qq] = acc;
    }
    // L2 -> clip at 0.2 -> L2, then L1-ROOT (or L2), round after ×512 and saturate to [0,255].
    sift_l2(d);
    for (int qq = 0; qq < 4; ++qq) d[qq] = fminf(d[qq], 0.2f);
    sift_l2(d);
    if (l1root) {
        float sum = sift_warp_sum(d[0] + d[1] + d[2] + d[3]);
        if (sum > 0.0f) {
            for (int qq = 0; qq < 4; ++qq) d[qq] = sqrtf(d[qq] / sum);
        }
    } else {
        sift_l2(d);
    }
    u8* out = desc + (size_t)k * 128;
    for (int qq = 0; qq < 4; ++qq) out[lane + 32 * qq] = (u8)fminf(fmaxf(roundf(512.0f * d[qq]), 0.0f), 255.0f);
    if (lane == 0) {
        float so = ldexpf(1.0f, o);
        fo[5 * k] = so * (x - 0.5f) + 0.5f;
        fo[5 * k + 1] = so * (y - 0.5f) + 0.5f;
        fo[5 * k + 2] = so * sigma;
        fo[5 * k + 3] = sift_rem_euclid(TWO_PI_F - theta, TWO_PI_F);
        fo[5 * k + 4] = resp;
        io[4 * k] = o;
        io[4 * k + 1] = j;
        io[4 * k + 2] = pix;
        io[4 * k + 3] = slot;
    }
}
