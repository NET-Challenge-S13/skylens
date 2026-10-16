/*
 * patchmatch.cu
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

// 다중 가설 뷰 선택 PatchMatch 커널(적·흑 반쪽 단계, 단일 가설 평가, 최종 판독).
// 규칙은 cumulus3d-dense 의 kernel 모듈 문서와 같다. 컴파일 상수:
//   WR (창 반경), WSTEP (표본 간격), HW_INTERP (1 = 텍스처 하드웨어 쌍선형, 0 = 소프트웨어 쌍선형)

typedef unsigned long long u64;
typedef unsigned char u8;

#define WN ((2 * WR) / WSTEP + 1)
#define MAXSRC 32
#define BX 32
#define BY 8
#define TILE_W (2 * BX + 2 * WR + 1)
#define TILE_H (BY + 2 * WR)
#define INF_ERR 1e30f
// 창 표본이 적으면 픽셀별 양방향 가중치를 공유 메모리에 한 번 계산해 둔다.
#define SHW (WN * WN <= 36)
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 3
#endif

struct LevelDesc {
    u64 tex;
    u64 off;
    int pitch, w, h, pad;
    float fx, fy, cx, cy;
};

struct PairRec {
    int src;
    float r[9];
    float t[3];
    float c[3];
};

// 스케일별 상대 기하의 호모그래피 상수부: A = K_j R K⁻¹, b = K_j t.
struct LevelPair {
    float A[9];
    float b[3];
};

struct ViewDesc {
    u64 key;
    float dmin, dmax;
    int nsrc, pad;
};

struct BatchItem {
    long long off;
    int view, w, h, pad;
};

struct KParams {
    u64 seed;
    const u8* img;
    const LevelDesc* lv;
    const PairRec* pairs;
    const ViewDesc* views;
    const BatchItem* batch;
    const float* snap;
    const long long* snap_off;
    float4* plane;
    float* cost;
    u8* bview;
    u8* count;
    int nlevels, level, geometric, run_id;
    int t, color, random_init, topk;
    float tau_good, tau_bad, inv2beta2, prev_w;
    int n1, n2;
    float inv2s, inv2c;
    float cos_tri_min, inv2inc, lambda, gmax;
    float eps, phi;
    float cos_filter_tri, q_min, ncc_norm, inv2ncc;
    float filter_gmax, pad0;
    const float4* prior;
    int use_prior;
    float weak_var;
    const LevelPair* lpairs;
};

// ---------------------------------------------------------------- 난수
__device__ __forceinline__ u64 mix64(u64 z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float rnd(u64& s) {
    s += 0x9e3779b97f4a7c15ULL;
    u64 z = mix64(s);
    return ((float)(z >> 40) + 1.0f) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ u64 rng_state(u64 seed, u64 key, int level, int run, int t, int half, u64 pixel) {
    u64 tag = ((u64)level << 48) ^ ((u64)run << 32) ^ ((u64)t << 16) ^ (u64)half;
    return mix64(seed ^ mix64(key ^ mix64(tag)) ^ pixel);
}

// ---------------------------------------------------------------- 벡터
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross3(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ __forceinline__ float3 nrm3(float3 a) {
    float l = rsqrtf(fmaxf(dot3(a, a), 1e-30f));
    return make_float3(a.x * l, a.y * l, a.z * l);
}

// ---------------------------------------------------------------- 기준 영상 접근
struct RefGlobal {
    const u8* p;
    int pitch, w, h;
    __device__ __forceinline__ float at(int x, int y) const {
        if (x < 0 || y < 0 || x >= w || y >= h) return 0.0f;
        return (float)__ldg(p + (size_t)y * pitch + x) * (1.0f / 255.0f);
    }
    __device__ __forceinline__ void store(int, float) const {}
    __device__ __forceinline__ float wt(int, float w) const { return w; }
    static constexpr bool cached = false;
};
struct RefTile {
    const float* s;
    int x0, y0;
    float* wbuf;
    int tid;
    __device__ __forceinline__ float at(int x, int y) const { return s[(y - y0) * TILE_W + (x - x0)]; }
    __device__ __forceinline__ void store(int k, float v) const {
#if SHW
        wbuf[k * (BX * BY) + tid] = v;
#endif
    }
    __device__ __forceinline__ float wt(int k, float w) const {
#if SHW
        return wbuf[k * (BX * BY) + tid];
#else
        return w;
#endif
    }
    static constexpr bool cached = SHW;
};

// 원천 영상 표본(정수 중심 규약, 밖은 0).
__device__ __forceinline__ float src_sample(const KParams& P, const LevelDesc& L, float u, float v) {
#if HW_INTERP
    return tex2D<float>((cudaTextureObject_t)L.tex, u + 0.5f, v + 0.5f);
#else
    float xf = floorf(u), yf = floorf(v);
    int x0 = (int)xf, y0 = (int)yf;
    float ax = u - xf, ay = v - yf;
    const u8* b = P.img + L.off;
    float p00 = 0.f, p10 = 0.f, p01 = 0.f, p11 = 0.f;
    bool xin0 = x0 >= 0 && x0 < L.w, xin1 = x0 + 1 >= 0 && x0 + 1 < L.w;
    if (y0 >= 0 && y0 < L.h) {
        const u8* r = b + (size_t)y0 * L.pitch;
        if (xin0) p00 = __ldg(r + x0);
        if (xin1) p10 = __ldg(r + x0 + 1);
    }
    if (y0 + 1 >= 0 && y0 + 1 < L.h) {
        const u8* r = b + (size_t)(y0 + 1) * L.pitch;
        if (xin0) p01 = __ldg(r + x0);
        if (xin1) p11 = __ldg(r + x0 + 1);
    }
    float top = p00 + ax * (p10 - p00), bot = p01 + ax * (p11 - p01);
    return (top + ay * (bot - top)) * (1.0f / 255.0f);
#endif
}

// 기준 패치 통계.
struct RefStats {
    float ic, W, mu, var;
};

template <class REF>
__device__ RefStats ref_stats(const KParams& P, const REF& ref, int x, int y) {
    RefStats s;
    s.ic = ref.at(x, y);
    float W = 0.f, S = 0.f, SS = 0.f;
#pragma unroll
    for (int a = 0; a < WN; ++a) {
        int dy = -WR + a * WSTEP;
#pragma unroll
        for (int b = 0; b < WN; ++b) {
            int dx = -WR + b * WSTEP;
            float ik = ref.at(x + dx, y + dy);
            float d = s.ic - ik;
            float w = __expf(-(float)(dx * dx + dy * dy) * P.inv2s - d * d * P.inv2c);
            ref.store(a * WN + b, w);
            W += w;
            S += w * ik;
            SS += w * ik * ik;
        }
    }
    s.W = W;
    s.mu = S / W;
    s.var = SS / W - s.mu * s.mu;
    return s;
}

// 평면 유도 호모그래피 H = K_j (R + t nᵀ/δ) K⁻¹ = A + b (nᵀK⁻¹)/δ.
__device__ __forceinline__ bool homography(const LevelDesc& R, const LevelPair& L, float x, float y, float d, float3 n, float* H) {
    float ifx = 1.0f / R.fx, ify = 1.0f / R.fy;
    float3 ray = make_float3((x - R.cx) * ifx, (y - R.cy) * ify, 1.0f);
    float delta = d * dot3(n, ray);
    if (!(delta < -1e-12f)) return false;
    float id = 1.0f / delta;
    float u0 = n.x * ifx, u1 = n.y * ify;
    float u2 = n.z - u0 * R.cx - u1 * R.cy;
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        float br = L.b[r] * id;
        H[r * 3 + 0] = L.A[r * 3 + 0] + br * u0;
        H[r * 3 + 1] = L.A[r * 3 + 1] + br * u1;
        H[r * 3 + 2] = L.A[r * 3 + 2] + br * u2;
    }
    return true;
}

// 비용 1 − ρ (양방향 가중 NCC). 실패는 2.
template <class REF>
__device__ float ncc_cost(const KParams& P, const REF& ref, const RefStats& rs, const LevelDesc& S, const float* H, int x, int y) {
    float fx = (float)x, fy = (float)y;
    float z = H[6] * fx + H[7] * fy + H[8];
    if (!(z > 1e-8f)) return 2.0f;
    float iz = 1.0f / z;
    float u = (H[0] * fx + H[1] * fy + H[2]) * iz, v = (H[3] * fx + H[4] * fy + H[5]) * iz;
    if (!(u >= 0.f && v >= 0.f && u <= (float)(S.w - 1) && v <= (float)(S.h - 1))) return 2.0f;
    if (rs.var < 1e-5f) return 2.0f;
    float Ss = 0.f, SSs = 0.f, IS = 0.f;
#pragma unroll
    for (int a = 0; a < WN; ++a) {
        int dy = -WR + a * WSTEP;
        float sy = fy + (float)dy;
#pragma unroll
        for (int b = 0; b < WN; ++b) {
            int dx = -WR + b * WSTEP;
            float sx = fx + (float)dx;
            float ik = ref.at(x + dx, y + dy);
            float w;
            if (REF::cached) {
                w = ref.wt(a * WN + b, 0.f);
            } else {
                float d = rs.ic - ik;
                w = __expf(-(float)(dx * dx + dy * dy) * P.inv2s - d * d * P.inv2c);
            }
            float zz = H[6] * sx + H[7] * sy + H[8];
            float sv = 0.f;
            if (zz > 1e-8f) {
                float izz = 1.0f / zz;
                sv = src_sample(P, S, (H[0] * sx + H[1] * sy + H[2]) * izz, (H[3] * sx + H[4] * sy + H[5]) * izz);
            }
            float ws = w * sv;
            Ss += ws;
            SSs += ws * sv;
            IS += ws * ik;
        }
    }
    float iW = 1.0f / rs.W;
    float mus = Ss * iW;
    float vars = SSs * iW - mus * mus;
    if (vars < 1e-5f) return 2.0f;
    float cov = IS * iW - rs.mu * mus;
    float c = 1.0f - cov * rsqrtf(rs.var * vars);
    return fminf(fmaxf(c, 0.0f), 2.0f);
}

// 순·역 재투영 오차(실패는 INF_ERR).
__device__ float geo_error(const KParams& P, const LevelDesc& R, const LevelDesc& S, const PairRec& pr, int x, int y, float d) {
    long long off = P.snap_off[pr.src];
    if (off < 0) return INF_ERR;
    float3 X = make_float3((x - R.cx) / R.fx * d, (y - R.cy) / R.fy * d, d);
    float xj = pr.r[0] * X.x + pr.r[1] * X.y + pr.r[2] * X.z + pr.t[0];
    float yj = pr.r[3] * X.x + pr.r[4] * X.y + pr.r[5] * X.z + pr.t[1];
    float zj = pr.r[6] * X.x + pr.r[7] * X.y + pr.r[8] * X.z + pr.t[2];
    if (!(zj > 1e-8f)) return INF_ERR;
    float u = S.fx * xj / zj + S.cx, v = S.fy * yj / zj + S.cy;
    float uf = floorf(u + 0.5f), vf = floorf(v + 0.5f);
    if (!(uf >= 0.f && vf >= 0.f && uf < (float)S.w && vf < (float)S.h)) return INF_ERR;
    float dj = __ldg(P.snap + off + (long long)vf * S.w + (long long)uf);
    if (!(dj > 0.f)) return INF_ERR;
    float3 Yj = make_float3((u - S.cx) / S.fx * dj, (v - S.cy) / S.fy * dj, dj);
    float3 q = make_float3(Yj.x - pr.t[0], Yj.y - pr.t[1], Yj.z - pr.t[2]);
    float Yx = pr.r[0] * q.x + pr.r[3] * q.y + pr.r[6] * q.z;
    float Yy = pr.r[1] * q.x + pr.r[4] * q.y + pr.r[7] * q.z;
    float Yz = pr.r[2] * q.x + pr.r[5] * q.y + pr.r[8] * q.z;
    if (!(Yz > 1e-8f)) return INF_ERR;
    float cx = R.fx * Yx / Yz + R.cx - (float)x, cy = R.fy * Yy / Yz + R.cy - (float)y;
    return sqrtf(cx * cx + cy * cy);
}

// 가설 하나 × 원천 하나의 (광도 비용, 절단 기하 오차).
template <class REF>
__device__ __forceinline__ float pair_cost(const KParams& P, const REF& ref, const RefStats& rs, const LevelDesc& R, const LevelDesc& S, const PairRec& pr, const LevelPair& lp, int x, int y, float4 h, float* geo) {
    float H[9];
    float c = 2.0f;
    if (homography(R, lp, (float)x, (float)y, h.w, make_float3(h.x, h.y, h.z), H)) c = ncc_cost(P, ref, rs, S, H, x, y);
    if (P.geometric) *geo = fminf(geo_error(P, R, S, pr, x, y, h.w), P.gmax);
    return c;
}

// 상위 K 평균(기하 실행이면 m + λ·Δe).
template <class REF>
__device__ float topk_cost(const KParams& P, const REF& ref, const RefStats& rs, const LevelDesc& R, const PairRec* prs, const LevelPair* lps, int nsrc, int x, int y, float4 h) {
    float b0 = 1e30f, b1 = 1e30f, b2 = 1e30f;
    for (int j = 0; j < nsrc; ++j) {
        const PairRec& pr = prs[j];
        const LevelDesc& S = P.lv[pr.src * P.nlevels + P.level];
        float g = 0.f;
        float c = pair_cost(P, ref, rs, R, S, pr, lps[j], x, y, h, &g);
        if (P.geometric) c += P.lambda * g;
        if (c < b0) { b2 = b1; b1 = b0; b0 = c; }
        else if (c < b1) { b2 = b1; b1 = c; }
        else if (c < b2) { b2 = c; }
    }
    int k = min(P.topk, nsrc);
    float s = b0;
    if (k >= 2) s += b1;
    if (k >= 3) s += b2;
    return k > 0 ? s / (float)k : 2.0f;
}

// 무작위 법선(구면 균등, 카메라 쪽).
__device__ float3 random_normal(u64& st, float3 ray) {
    float3 n = make_float3(0.f, 0.f, -1.f);
    for (int k = 0; k < 64; ++k) {
        float v1 = 2.f * rnd(st) - 1.f, v2 = 2.f * rnd(st) - 1.f;
        float s = v1 * v1 + v2 * v2;
        if (s < 1.f && s > 0.f) {
            float r = 2.f * sqrtf(1.f - s);
            n = make_float3(v1 * r, v2 * r, 1.f - 2.f * s);
            break;
        }
    }
    if (dot3(n, ray) > 0.f) n = make_float3(-n.x, -n.y, -n.z);
    return n;
}

__device__ float random_depth(u64& st, float dmin, float dmax) {
    float a = 1.f / dmax, b = 1.f / dmin;
    return 1.f / (a + rnd(st) * (b - a));
}

__device__ float3 perturb_normal(u64& st, float3 n, float3 ray, float phi) {
    float3 a = random_normal(st, make_float3(0.f, 0.f, 0.f));
    float3 ax = cross3(a, n);
    float l2 = dot3(ax, ax);
    float ang = rnd(st) * phi;
    if (l2 < 1e-12f) return n;
    ax = nrm3(ax);
    float3 bn = cross3(ax, n);
    for (int k = 0; k < 4; ++k) {
        float sa, ca;
        __sincosf(ang, &sa, &ca);
        float3 m = make_float3(n.x * ca + bn.x * sa, n.y * ca + bn.y * sa, n.z * ca + bn.z * sa);
        if (dot3(m, ray) < 0.f) return nrm3(m);
        ang *= 0.5f;
    }
    return n;
}

__device__ __forceinline__ float3 pixel_ray(const LevelDesc& R, int x, int y) { return make_float3((x - R.cx) / R.fx, (y - R.cy) / R.fy, 1.0f); }

// 사전확률(삼각측량·입사각·해상도).
__device__ float view_prior(const KParams& P, const LevelDesc& R, const PairRec& pr, const float* H, bool hok, int x, int y, float4 h) {
    if (!hok) return 0.f;
    float3 ray = pixel_ray(R, x, y);
    float3 X = make_float3(ray.x * h.w, ray.y * h.w, h.w);
    float3 S = make_float3(pr.c[0] - X.x, pr.c[1] - X.y, pr.c[2] - X.z);
    float ls = sqrtf(dot3(S, S)), lx = sqrtf(dot3(X, X));
    if (ls <= 0.f || lx <= 0.f) return 0.f;
    float cost = -dot3(S, X) / (ls * lx);
    float tri = 1.f;
    if (cost > P.cos_tri_min) {
        float s = 1.f - (1.f - cost) / (1.f - P.cos_tri_min);
        tri = fminf(fmaxf(1.f - s * s, 0.f), 1.f);
    }
    float cphi = dot3(S, make_float3(h.x, h.y, h.z)) / ls;
    float xi = 1.f - fmaxf(0.f, cphi);
    float inc = __expf(-xi * xi * P.inv2inc);
    float r = (float)WR;
    float cxs[4] = {x - r, x - r, x + r, x + r};
    float cys[4] = {y - r, y + r, y + r, y - r};
    float px[4], py[4];
    for (int k = 0; k < 4; ++k) {
        float z = H[6] * cxs[k] + H[7] * cys[k] + H[8];
        if (!(z > 1e-8f)) return 0.f;
        px[k] = (H[0] * cxs[k] + H[1] * cys[k] + H[2]) / z;
        py[k] = (H[3] * cxs[k] + H[4] * cys[k] + H[5]) / z;
    }
    float area = 0.f;
    for (int k = 0; k < 4; ++k) area += px[k] * py[(k + 1) & 3] - px[(k + 1) & 3] * py[k];
    area = 0.5f * fabsf(area);
    float aref = (2.f * r + 1.f) * (2.f * r + 1.f);
    if (!(area > 0.f)) return 0.f;
    float res = fminf(area / aref, aref / area);
    return tri * inc * res;
}

// 영역 오프셋(위 방향 기준; 나머지는 90° 회전).
__constant__ int V_OFF[7][2] = {{0, -1}, {-1, -2}, {1, -2}, {-2, -3}, {2, -3}, {-3, -4}, {3, -4}};

__device__ __forceinline__ void rot_dir(int dir, int dx, int dy, int& ox, int& oy) {
    // 위 → 오른쪽 → 아래 → 왼쪽: (dx,dy) → (−dy, dx)
    ox = dx;
    oy = dy;
    for (int k = 0; k < dir; ++k) {
        int t = ox;
        ox = -oy;
        oy = t;
    }
}

// ================================================================ 반쪽 단계
extern "C" __global__ void __launch_bounds__(BX * BY, MIN_BLOCKS) pm_half(KParams P) {
    __shared__ float tile[TILE_W * TILE_H];
    __shared__ float wsh[SHW ? WN * WN * BX * BY : 1];
    const BatchItem bi = P.batch[blockIdx.z];
    const LevelDesc R = P.lv[bi.view * P.nlevels + P.level];
    const int w = bi.w, h = bi.h;
    const int x0 = blockIdx.x * (2 * BX) - WR, y0 = blockIdx.y * BY - WR;
    if (x0 + WR >= w || y0 + WR >= h) return;  // 블록 전체가 밖
    RefGlobal g;
    g.p = P.img + R.off;
    g.pitch = R.pitch;
    g.w = w;
    g.h = h;
    for (int i = threadIdx.y * BX + threadIdx.x; i < TILE_W * TILE_H; i += BX * BY) {
        int tx = i % TILE_W, ty = i / TILE_W;
        tile[i] = g.at(x0 + tx, y0 + ty);
    }
    __syncthreads();
    const int y = blockIdx.y * BY + threadIdx.y;
    const int x = 2 * (blockIdx.x * BX + threadIdx.x) + ((y + P.color) & 1);
    if (x >= w || y >= h) return;
    RefTile ref;
    ref.s = tile;
    ref.x0 = x0;
    ref.y0 = y0;
    ref.wbuf = wsh;
    ref.tid = threadIdx.y * BX + threadIdx.x;

    const ViewDesc vd = P.views[bi.view];
    const PairRec* prs = P.pairs + (size_t)bi.view * MAXSRC;
    const LevelPair* lps = P.lpairs + ((size_t)bi.view * P.nlevels + P.level) * MAXSRC;
    const int nsrc = vd.nsrc;
    const long long base = bi.off;
    const int pix = y * w + x;
    float4* pl = P.plane + base;
    float* co = P.cost + base;
    const float3 ray = pixel_ray(R, x, y);

    // 1. 후보: 현재 + 영역 8개.
    float4 hyp[9];
    hyp[0] = pl[pix];
    for (int r = 0; r < 8; ++r) {
        int dir = r & 3;
        bool strip = r >= 4;
        int cnt = strip ? 11 : 7;
        float best = 1e30f;
        int bq = -1, bx = 0, by = 0;
        for (int k = 0; k < cnt; ++k) {
            int dx, dy;
            if (strip) rot_dir(dir, 0, -(3 + 2 * k), dx, dy);
            else rot_dir(dir, V_OFF[k][0], V_OFF[k][1], dx, dy);
            int qx = x + dx, qy = y + dy;
            if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
            float c = co[qy * w + qx];
            if (c < best) {
                best = c;
                bq = qy * w + qx;
                bx = qx;
                by = qy;
            }
        }
        hyp[1 + r] = hyp[0];
        if (bq >= 0) {
            float4 q = pl[bq];
            float3 n = make_float3(q.x, q.y, q.z);
            float3 rq = pixel_ray(R, bx, by);
            float den = dot3(n, ray);
            if (den < -1e-6f && q.w > 0.f) {
                float d = q.w * dot3(n, rq) / den;
                if (d > 0.f && isfinite(d)) hyp[1 + r] = make_float4(n.x, n.y, n.z, d);
            }
        }
    }

    // 2. 뷰 순회: 투표·가중·집계.
    RefStats rs = ref_stats(P, ref, x, y);
    const int vprev = P.bview[base + pix];
    float num[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) num[i] = 0.f;
    float den = 0.f, wbest = 0.f;
    int vbest = 255;
    float wv[MAXSRC];
    // 대체 집계(가중 합이 0 일 때): 사전확률 가중 평균. 재계산 없이 같은 순회에서 모은다.
    float numf[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) numf[i] = 0.f;
    float denf = 0.f;
    for (int j = 0; j < nsrc; ++j) {
        const PairRec& pr = prs[j];
        const LevelDesc S = P.lv[pr.src * P.nlevels + P.level];
        float m[9], gg[9];
        float prior;
        {
            float H[9];
            bool ok = homography(R, lps[j], (float)x, (float)y, hyp[0].w, make_float3(hyp[0].x, hyp[0].y, hyp[0].z), H);
            prior = view_prior(P, R, pr, H, ok, x, y, hyp[0]);
            // 사전확률이 0 인 뷰는 가중치도 0 이라 비용을 계산하지 않는다(상위 K 대체 경로는 따로 전부 계산).
            wv[j] = 0.f;
            if (!(prior > 0.f)) continue;
            m[0] = ok ? ncc_cost(P, ref, rs, S, H, x, y) : 2.f;
            gg[0] = P.geometric ? fminf(geo_error(P, R, S, pr, x, y, hyp[0].w), P.gmax) : 0.f;
        }
        for (int i = 1; i < 9; ++i) {
            float H[9];
            bool ok = homography(R, lps[j], (float)x, (float)y, hyp[i].w, make_float3(hyp[i].x, hyp[i].y, hyp[i].z), H);
            m[i] = ok ? ncc_cost(P, ref, rs, S, H, x, y) : 2.f;
            gg[i] = P.geometric ? fminf(geo_error(P, R, S, pr, x, y, hyp[i].w), P.gmax) : 0.f;
        }
        int good = 0, bad = 0;
        float conf = 0.f;
#pragma unroll
        for (int i = 1; i < 9; ++i) {
            if (m[i] < P.tau_good) {
                good++;
                conf += __expf(-m[i] * m[i] * P.inv2beta2);
            }
            if (m[i] > P.tau_bad) bad++;
        }
        bool adopt = good > P.n1 && bad < P.n2;
        bool isprev = j == vprev;
        float wp = adopt ? (isprev ? 2.f : 1.f) * conf / (float)good : (isprev ? P.prev_w : 0.f);
        float w2 = wp * prior;
        wv[j] = w2 > 0.f ? w2 : -prior;
#pragma unroll
        for (int i = 0; i < 9; ++i) numf[i] += prior * (m[i] + P.lambda * gg[i]);
        denf += prior;
        if (w2 > 0.f) {
#pragma unroll
            for (int i = 0; i < 9; ++i) num[i] += w2 * (m[i] + P.lambda * gg[i]);
            den += w2;
            if (w2 > wbest) {
                wbest = w2;
                vbest = j;
            }
        }
    }
    float agg[9];
    int bi9 = 0;
    if (den > 0.f) {
#pragma unroll
        for (int i = 0; i < 9; ++i) agg[i] = num[i] / den;
        for (int i = 1; i < 9; ++i)
            if (agg[i] < agg[bi9]) bi9 = i;
    } else if (denf > 0.f) {
        for (int i = 0; i < 9; ++i) agg[i] = numf[i] / denf;
        for (int i = 1; i < 9; ++i)
            if (agg[i] < agg[bi9]) bi9 = i;
        // 정제도 같은 가중(사전확률)으로 비교한다.
        for (int j = 0; j < nsrc; ++j) wv[j] = -wv[j];
        den = denf;
    } else {
        for (int i = 0; i < 9; ++i) agg[i] = 2.f;
    }
    float4 best = hyp[bi9];
    float bcost = agg[bi9];

    // 3. 정제 6가설.
    u64 st = rng_state(P.seed, vd.key, P.level, P.run_id, P.t, P.color, (u64)pix);
    float d_r = random_depth(st, vd.dmin, vd.dmax);
    float3 n_r = random_normal(st, ray);
    float u = 2.f * rnd(st) - 1.f;
    float inv = best.w > 0.f ? (1.f / best.w) * (1.f + u * P.eps) : -1.f;
    float d_p = inv > 0.f ? 1.f / inv : d_r;
    if (!(best.w > 0.f)) best.w = d_r;
    float3 n0 = make_float3(best.x, best.y, best.z);
    float3 n_p = perturb_normal(st, n0, ray, P.phi);
    float4 cand[6] = {make_float4(n0.x, n0.y, n0.z, d_r), make_float4(n_r.x, n_r.y, n_r.z, best.w), make_float4(n_r.x, n_r.y, n_r.z, d_r),
                      make_float4(n0.x, n0.y, n0.z, d_p), make_float4(n_p.x, n_p.y, n_p.z, best.w), make_float4(n_p.x, n_p.y, n_p.z, d_p)};
    for (int c = 0; c < 6; ++c) {
        float cc;
        if (den > 0.f) {
            float nm = 0.f;
            for (int j = 0; j < nsrc; ++j) {
                if (!(wv[j] > 0.f)) continue;
                const PairRec& pr = prs[j];
                const LevelDesc S = P.lv[pr.src * P.nlevels + P.level];
                float g = 0.f;
                float m = pair_cost(P, ref, rs, R, S, pr, lps[j], x, y, cand[c], &g);
                nm += wv[j] * (m + P.lambda * g);
            }
            cc = nm / den;
        } else {
            cc = 2.f;
        }
        if (cc < bcost) {
            bcost = cc;
            best = cand[c];
        }
    }
    // 4. 무늬 약한 픽셀: 거친 스케일 가설을 후보로만 추가.
    if (P.use_prior && rs.var < P.weak_var) {
        float4 q = P.prior[base + pix];
        if (q.w > 0.f && dot3(make_float3(q.x, q.y, q.z), ray) < 0.f) {
            float cc;
            if (den > 0.f) {
                float nm = 0.f;
                for (int j = 0; j < nsrc; ++j) {
                    if (!(wv[j] > 0.f)) continue;
                    const PairRec& pr = prs[j];
                    const LevelDesc S = P.lv[pr.src * P.nlevels + P.level];
                    float g = 0.f;
                    float m = pair_cost(P, ref, rs, R, S, pr, lps[j], x, y, q, &g);
                    nm += wv[j] * (m + P.lambda * g);
                }
                cc = nm / den;
            } else {
                cc = 2.f;
            }
            if (cc < bcost) {
                bcost = cc;
                best = q;
            }
        }
    }
    pl[pix] = best;
    co[pix] = bcost;
    P.bview[base + pix] = (u8)vbest;
}

// ================================================================ 초기화·평가(전체 픽셀)
// mode 0: 비용만 평가해 cost 에 기록. mode 1: 실행 시작(무작위 초기화 선택 + 비용 + 최중요 뷰 비움).
extern "C" __global__ void __launch_bounds__(BX * BY, MIN_BLOCKS) pm_eval(KParams P, int mode) {
    const BatchItem bi = P.batch[blockIdx.z];
    const int x = blockIdx.x * BX + threadIdx.x, y = blockIdx.y * BY + threadIdx.y;
    if (x >= bi.w || y >= bi.h) return;
    const LevelDesc R = P.lv[bi.view * P.nlevels + P.level];
    const ViewDesc vd = P.views[bi.view];
    const PairRec* prs = P.pairs + (size_t)bi.view * MAXSRC;
    const LevelPair* lps = P.lpairs + ((size_t)bi.view * P.nlevels + P.level) * MAXSRC;
    const int pix = y * bi.w + x;
    const long long base = bi.off;
    float3 ray = pixel_ray(R, x, y);
    float4 h = P.plane[base + pix];
    if (mode == 1 && P.random_init) {
        u64 st = rng_state(P.seed, vd.key, P.level, P.run_id, 0xFFFF, 0, (u64)pix);
        float d = random_depth(st, vd.dmin, vd.dmax);
        float3 n = random_normal(st, ray);
        h = make_float4(n.x, n.y, n.z, d);
        P.plane[base + pix] = h;
    }
    RefGlobal g;
    g.p = P.img + R.off;
    g.pitch = R.pitch;
    g.w = bi.w;
    g.h = bi.h;
    float c = 2.f;
    if (h.w > 0.f) {
        RefStats rs = ref_stats(P, g, x, y);
        c = topk_cost(P, g, rs, R, prs, lps, vd.nsrc, x, y, h);
    }
    P.cost[base + pix] = c;
    if (mode == 1) P.bview[base + pix] = 255;
}

// ================================================================ 최종 판독
extern "C" __global__ void __launch_bounds__(BX * BY, MIN_BLOCKS) pm_filter(KParams P) {
    const BatchItem bi = P.batch[blockIdx.z];
    const int x = blockIdx.x * BX + threadIdx.x, y = blockIdx.y * BY + threadIdx.y;
    if (x >= bi.w || y >= bi.h) return;
    const LevelDesc R = P.lv[bi.view * P.nlevels + P.level];
    const ViewDesc vd = P.views[bi.view];
    const PairRec* prs = P.pairs + (size_t)bi.view * MAXSRC;
    const LevelPair* lps = P.lpairs + ((size_t)bi.view * P.nlevels + P.level) * MAXSRC;
    const int pix = y * bi.w + x;
    const long long base = bi.off;
    float4 h = P.plane[base + pix];
    int count = 0;
    if (h.w > 0.f) {
        RefGlobal g;
        g.p = P.img + R.off;
        g.pitch = R.pitch;
        g.w = bi.w;
        g.h = bi.h;
        RefStats rs = ref_stats(P, g, x, y);
        float3 ray = pixel_ray(R, x, y);
        float3 X = make_float3(ray.x * h.w, ray.y * h.w, h.w);
        float lx = sqrtf(dot3(X, X));
        for (int j = 0; j < vd.nsrc; ++j) {
            const PairRec& pr = prs[j];
            const LevelDesc S = P.lv[pr.src * P.nlevels + P.level];
            float3 Sv = make_float3(pr.c[0] - X.x, pr.c[1] - X.y, pr.c[2] - X.z);
            float ls = sqrtf(dot3(Sv, Sv));
            if (!(ls > 0.f)) continue;
            float cost = -dot3(Sv, X) / (ls * lx);
            if (cost > P.cos_filter_tri) continue;
            float cphi = dot3(Sv, make_float3(h.x, h.y, h.z)) / ls;
            if (cphi <= 0.f) continue;
            float H[9];
            if (!homography(R, lps[j], (float)x, (float)y, h.w, make_float3(h.x, h.y, h.z), H)) continue;
            float m = ncc_cost(P, g, rs, S, H, x, y);
            float e = P.ncc_norm * __expf(-m * m * P.inv2ncc);
            if (e / (e + 0.5f) < P.q_min) continue;
            if (geo_error(P, R, S, pr, x, y, h.w) > P.filter_gmax) continue;
            count++;
        }
    }
    P.count[base + pix] = (u8)min(count, 255);
}

// ================================================================ 스케일 사이 상향 표본(결합 양방향)
// 입력: prior 버퍼의 저해상 평면(항목별 시작 = batch.pad), 출력: plane 버퍼.
extern "C" __global__ void __launch_bounds__(BX * BY) pm_upsample(KParams P, float inv2s, float inv2c) {
    const BatchItem bi = P.batch[blockIdx.z];
    const int x = blockIdx.x * BX + threadIdx.x, y = blockIdx.y * BY + threadIdx.y;
    if (x >= bi.w || y >= bi.h) return;
    const LevelDesc R = P.lv[bi.view * P.nlevels + P.level];
    const LevelDesc L = P.lv[bi.view * P.nlevels + P.level - 1];
    const float4* low = P.prior + bi.pad;
    const float3 rp = pixel_ray(R, x, y);
    const float ip = (float)__ldg(P.img + R.off + (size_t)y * R.pitch + x) * (1.0f / 255.0f);
    const float px = ((float)x - 0.5f) * 0.5f, py = ((float)y - 0.5f) * 0.5f;
    const int qx0 = (int)roundf(px), qy0 = (int)roundf(py);
    float sw = 0.f, sd = 0.f;
    float3 sn = make_float3(0.f, 0.f, 0.f);
    float nd = 1e30f, ndep = 0.f;
    float3 nn = make_float3(0.f, 0.f, 0.f);
    for (int qy = qy0 - 2; qy <= qy0 + 2; ++qy) {
        if (qy < 0 || qy >= L.h) continue;
        for (int qx = qx0 - 2; qx <= qx0 + 2; ++qx) {
            if (qx < 0 || qx >= L.w) continue;
            float4 q = low[qy * L.w + qx];
            if (!(q.w > 0.f)) continue;
            float3 n = make_float3(q.x, q.y, q.z);
            float den = dot3(n, rp);
            if (!(den < -1e-6f)) continue;
            float d = q.w * dot3(n, pixel_ray(L, qx, qy)) / den;
            if (!(d > 0.f) || !isfinite(d)) continue;
            float ds = ((float)qx - px) * ((float)qx - px) + ((float)qy - py) * ((float)qy - py);
            if (ds < nd) {
                nd = ds;
                ndep = d;
                nn = n;
            }
            float iq = (float)__ldg(P.img + L.off + (size_t)qy * L.pitch + qx) * (1.0f / 255.0f);
            float w = __expf(-ds * inv2s - (ip - iq) * (ip - iq) * inv2c);
            sw += w;
            sd += w * d;
            sn.x += w * n.x;
            sn.y += w * n.y;
            sn.z += w * n.z;
        }
    }
    float4 out = make_float4(0.f, 0.f, 0.f, 0.f);
    float d = 0.f;
    float3 n = make_float3(0.f, 0.f, 0.f);
    bool ok = false;
    if (sw > 1e-12f) {
        d = sd / sw;
        n = sn;
        ok = true;
    } else if (nd < 1e29f) {
        d = ndep;
        n = nn;
        ok = true;
    }
    if (ok && dot3(n, n) > 1e-24f) {
        n = nrm3(n);
        if (dot3(n, rp) >= 0.f) n = make_float3(-n.x, -n.y, -n.z);
        out = make_float4(n.x, n.y, n.z, d);
    }
    P.plane[bi.off + y * bi.w + x] = out;
}

// ================================================================ 5×5 중앙값 평면 필터
// 입력: prior 버퍼(항목별 시작 = batch.off), 출력: plane 버퍼.
extern "C" __global__ void __launch_bounds__(BX * BY) pm_median(KParams P) {
    const BatchItem bi = P.batch[blockIdx.z];
    const int x = blockIdx.x * BX + threadIdx.x, y = blockIdx.y * BY + threadIdx.y;
    if (x >= bi.w || y >= bi.h) return;
    const LevelDesc R = P.lv[bi.view * P.nlevels + P.level];
    const float4* in = P.prior + bi.off;
    const int w = bi.w, h = bi.h;
    float4 self = in[y * w + x];
    if (!(self.w > 0.f)) {
        P.plane[bi.off + y * w + x] = self;
        return;
    }
    const float3 rp = pixel_ray(R, x, y);
    float ds[25];
    int qs[25];
    int c = 0;
    for (int qy = max(y - 2, 0); qy < min(y + 3, h); ++qy) {
        for (int qx = max(x - 2, 0); qx < min(x + 3, w); ++qx) {
            float4 q = in[qy * w + qx];
            if (!(q.w > 0.f)) continue;
            float3 n = make_float3(q.x, q.y, q.z);
            float den = dot3(n, rp);
            if (!(den < -1e-6f)) continue;
            float d = q.w * dot3(n, pixel_ray(R, qx, qy)) / den;
            if (!(d > 0.f) || !isfinite(d)) continue;
            // 삽입 정렬(깊이, 같으면 색인).
            int qi = qy * w + qx;
            int k = c++;
            while (k > 0 && (ds[k - 1] > d || (ds[k - 1] == d && qs[k - 1] > qi))) {
                ds[k] = ds[k - 1];
                qs[k] = qs[k - 1];
                --k;
            }
            ds[k] = d;
            qs[k] = qi;
        }
    }
    float4 out = self;
    if (c > 0) {
        int m = (c - 1) / 2;
        float4 q = in[qs[m]];
        out = make_float4(q.x, q.y, q.z, ds[m]);
    }
    P.plane[bi.off + y * w + x] = out;
}
