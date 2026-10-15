/*
 * sparse.cu
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

// 희소 단계 커널: 두 뷰 기하·절대 자세 RANSAC 가설 채점, RANSAC 삼각측량, 관측 검사.
// fmad 를 끄고(곱셈-덧셈 융합 금지) CPU 코드와 같은 순서로 계산해 반올림까지 맞춘다.
// 지지도의 잔차 합은 블록의 0번 스레드가 자료 순서대로 더한다(CPU Support::measure 와 같은 합).

#define BLOCK 128
#define DBL_EPS 2.220446049250313e-16
#define DBL_MAXV 1.7976931348623157e308
#define PI_D 3.141592653589793
#define USIZE_MAX_F 1.8446744073709552e19
#define MAX_TRI_OBS 64
#define OBS_STRIDE 18

typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ double inf_d() { return __longlong_as_double(0x7ff0000000000000LL); }

// Sampson 제곱 오차. m 은 행 우선 3×3, a·b 는 동차 좌표.
// l = M a, bm = Mᵀ b, 분자 = (b·l)², 분모 = bm_x² + bm_y² + l_x² + l_y².
__device__ double sampson(const double* m, double a0, double a1, double a2, double b0, double b1, double b2) {
    double l0 = m[0] * a0 + m[1] * a1 + m[2] * a2;
    double l1 = m[3] * a0 + m[4] * a1 + m[5] * a2;
    double l2 = m[6] * a0 + m[7] * a1 + m[8] * a2;
    double bm0 = m[0] * b0 + m[3] * b1 + m[6] * b2;
    double bm1 = m[1] * b0 + m[4] * b1 + m[7] * b2;
    double num = b0 * l0 + b1 * l1 + b2 * l2;
    double den = bm0 * bm0 + bm1 * bm1 + l0 * l0 + l1 * l1;
    if (den == 0.0) return inf_d();
    return num * num / den;
}

// 호모그래피 단방향 전이 제곱 오차. 유한하지 않으면 +∞.
__device__ double transfer(const double* h, double x1, double y1, double x2, double y2) {
    double px = h[0] * x1 + h[1] * y1 + h[2] * 1.0;
    double py = h[3] * x1 + h[4] * y1 + h[5] * 1.0;
    double pz = h[6] * x1 + h[7] * y1 + h[8] * 1.0;
    double dx = x2 - px / pz;
    double dy = y2 - py / pz;
    double r = dx * dx + dy * dy;
    return isfinite(r) ? r : inf_d();
}

// 카메라 투영(CPU Camera::cam_to_img). c = [모델, fx, fy, cx, cy, k1, k2, p1, p2].
__device__ bool cam_to_img(const double* c, double x, double y, double z, double* px, double* py) {
    if (z < DBL_EPS) return false;
    double u = x / z;
    double v = y / z;
    int model = (int)c[0];
    double du, dv;
    if (model <= 1) {
        du = 0.0;
        dv = 0.0;
    } else if (model <= 3) {
        double k1 = c[5], k2 = c[6];
        double r2 = u * u + v * v;
        double radial = k1 * r2 + k2 * r2 * r2;
        du = u * radial;
        dv = v * radial;
    } else {
        double k1 = c[5], k2 = c[6], p1 = c[7], p2 = c[8];
        double r2 = u * u + v * v;
        double radial = k1 * r2 + k2 * r2 * r2;
        du = u * radial + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u);
        dv = v * radial + 2.0 * p2 * u * v + p1 * (r2 + 2.0 * v * v);
    }
    *px = c[1] * (u + du) + c[3];
    *py = c[2] * (v + dv) + c[4];
    return true;
}

// 픽셀 재투영 제곱 오차. pose = [R 행 우선 9, t 3]. 투영 실패면 DBL_MAX.
__device__ double reproj_sq(const double* cam, const double* pose, double qx, double qy, double X, double Y, double Z, double* depth) {
    double xc = pose[0] * X + pose[1] * Y + pose[2] * Z + pose[9];
    double yc = pose[3] * X + pose[4] * Y + pose[5] * Z + pose[10];
    double zc = pose[6] * X + pose[7] * Y + pose[8] * Z + pose[11];
    if (depth) *depth = zc;
    double px, py;
    if (!cam_to_img(cam, xc, yc, zc, &px, &py)) return DBL_MAXV;
    double dx = px - qx;
    double dy = py - qy;
    return dx * dx + dy * dy;
}

// 블록 하나 = 가설 하나. 스레드마다 잔차, 0번 스레드가 자료 순서대로 인라이어 수·합을 센다.
// ent_i[3e..] = (자료 시작 오프셋(double 단위), 자료 수, 종류 0=E 1=F 2=H), ent_m[10e..] = (모델 9, 제곱 임계).
// 자료: E = (a.xyz, b.xyz) 6개, F/H = (x1.xy, x2.xy) 4개.
extern "C" __global__ void score_two_view(const double* __restrict__ data, const long long* __restrict__ ent_i,
                                          const double* __restrict__ ent_m, u32* out_n, double* out_s) {
    __shared__ double res[BLOCK];
    __shared__ double hm[10];
    int e = blockIdx.x;
    long long off = ent_i[3 * (long long)e];
    int n = (int)ent_i[3 * (long long)e + 1];
    int kind = (int)ent_i[3 * (long long)e + 2];
    if (threadIdx.x < 10) hm[threadIdx.x] = ent_m[10 * (long long)e + threadIdx.x];
    __syncthreads();
    double thr = hm[9];
    u32 cnt = 0;
    double sum = 0.0;
    for (int c0 = 0; c0 < n; c0 += BLOCK) {
        int i = c0 + threadIdx.x;
        double r = 0.0;
        if (i < n) {
            if (kind == 0) {
                const double* d = data + off + 6 * (long long)i;
                r = sampson(hm, d[0], d[1], d[2], d[3], d[4], d[5]);
            } else if (kind == 1) {
                const double* d = data + off + 4 * (long long)i;
                r = sampson(hm, d[0], d[1], 1.0, d[2], d[3], 1.0);
            } else {
                const double* d = data + off + 4 * (long long)i;
                r = transfer(hm, d[0], d[1], d[2], d[3]);
            }
        }
        res[threadIdx.x] = r;
        __syncthreads();
        if (threadIdx.x == 0) {
            int m = min(BLOCK, n - c0);
            for (int k = 0; k < m; ++k) {
                double v = res[k];
                if (v <= thr) {
                    cnt += 1;
                    sum += v;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_n[e] = cnt;
        out_s[e] = sum;
    }
}

// 절대 자세 가설 채점. data = 대응마다 (x, y, X, Y, Z), cam = 카메라 9개,
// poses = 가설마다 (R 9, t 3), 블록 하나 = 가설 하나.
extern "C" __global__ void score_pose(const double* __restrict__ data, int n, const double* __restrict__ cam,
                                      const double* __restrict__ poses, double thr, u32* out_n, double* out_s) {
    __shared__ double res[BLOCK];
    __shared__ double pm[12];
    __shared__ double cm[9];
    int e = blockIdx.x;
    if (threadIdx.x < 12) pm[threadIdx.x] = poses[12 * (long long)e + threadIdx.x];
    if (threadIdx.x < 9) cm[threadIdx.x] = cam[threadIdx.x];
    __syncthreads();
    u32 cnt = 0;
    double sum = 0.0;
    for (int c0 = 0; c0 < n; c0 += BLOCK) {
        int i = c0 + threadIdx.x;
        double r = 0.0;
        if (i < n) {
            const double* d = data + 5 * (long long)i;
            r = reproj_sq(cm, pm, d[0], d[1], d[2], d[3], d[4], 0);
        }
        res[threadIdx.x] = r;
        __syncthreads();
        if (threadIdx.x == 0) {
            int m = min(BLOCK, n - c0);
            for (int k = 0; k < m; ++k) {
                double v = res[k];
                if (v <= thr) {
                    cnt += 1;
                    sum += v;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_n[e] = cnt;
        out_s[e] = sum;
    }
}

// 관측 검사. q_i = 관측마다 (카메라 번호, 자세 번호), q_d = (x, y, X, Y, Z),
// cams = 카메라마다 9개, poses = 자세마다 (R 9, t 3). out = (재투영 제곱 오차, 깊이).
extern "C" __global__ void check_obs(const int* __restrict__ q_i, const double* __restrict__ q_d, int n,
                                     const double* __restrict__ cams, const double* __restrict__ poses, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double* d = q_d + 5 * (long long)i;
    const double* c = cams + 9 * (long long)q_i[2 * (long long)i];
    const double* p = poses + 12 * (long long)q_i[2 * (long long)i + 1];
    double depth = 0.0;
    double r = reproj_sq(c, p, d[0], d[1], d[2], d[3], d[4], &depth);
    out[2 * (long long)i] = r;
    out[2 * (long long)i + 1] = depth;
}

// ---------------- RANSAC 삼각측량 ----------------
// 관측 하나 = (P 행 우선 12, 중심 3, 광선 3) 18개.

// 4×4 대칭이 아닌 행렬 A 의 최소 특이값 오른쪽 특이 벡터(단측 야코비).
__device__ bool null_vector4(double a[4][4], double out[4]) {
    double v[4][4];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) v[r][c] = (r == c) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 40; ++sweep) {
        bool rotated = false;
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int k = 0; k < 4; ++k) {
                    alpha += a[k][p] * a[k][p];
                    beta += a[k][q] * a[k][q];
                    gamma += a[k][p] * a[k][q];
                }
                if (gamma == 0.0 || fabs(gamma) <= 1e-15 * sqrt(alpha * beta)) continue;
                rotated = true;
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0.0 ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                double cs = 1.0 / sqrt(1.0 + t * t);
                double sn = cs * t;
                for (int k = 0; k < 4; ++k) {
                    double ap = a[k][p], aq = a[k][q];
                    a[k][p] = cs * ap - sn * aq;
                    a[k][q] = sn * ap + cs * aq;
                    double vp = v[k][p], vq = v[k][q];
                    v[k][p] = cs * vp - sn * vq;
                    v[k][q] = sn * vp + cs * vq;
                }
            }
        }
        if (!rotated) break;
    }
    int best = 0;
    double bn = -1.0;
    for (int c = 0; c < 4; ++c) {
        double s = 0.0;
        for (int k = 0; k < 4; ++k) s += a[k][c] * a[k][c];
        if (bn < 0.0 || s < bn) {
            bn = s;
            best = c;
        }
    }
    for (int k = 0; k < 4; ++k) out[k] = v[k][best];
    return isfinite(bn);
}

__device__ bool dehom(const double h[4], double x[3]) {
    if (h[3] == 0.0) return false;
    x[0] = h[0] / h[3];
    x[1] = h[1] / h[3];
    x[2] = h[2] / h[3];
    return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]);
}

// 두 뷰 DLT(정규화 평면 좌표 = 광선.xy / 광선.z).
__device__ bool tri_dlt(const double* oa, const double* ob, double x[3]) {
    const double* ra = oa + 15;
    const double* rb = ob + 15;
    if (ra[2] <= 0.0 || rb[2] <= 0.0) return false;
    double ax = ra[0] / ra[2], ay = ra[1] / ra[2];
    double bx = rb[0] / rb[2], by = rb[1] / rb[2];
    double a[4][4];
    for (int c = 0; c < 4; ++c) {
        a[0][c] = ax * oa[8 + c] - oa[c];
        a[1][c] = ay * oa[8 + c] - oa[4 + c];
        a[2][c] = bx * ob[8 + c] - ob[c];
        a[3][c] = by * ob[8 + c] - ob[4 + c];
    }
    double h[4];
    if (!null_vector4(a, h)) return false;
    return dehom(h, x);
}

// 다중 뷰: M = Σ Qᵀ Q, Q = P − b (bᵀ P), M 의 최소 고유 벡터(대칭 반정치라 특이 벡터와 같음).
__device__ bool tri_multi(const double* obs, u64 mask, int n, double x[3]) {
    double m[4][4];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) m[r][c] = 0.0;
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        if (!((mask >> i) & 1ULL)) continue;
        cnt++;
        const double* o = obs + OBS_STRIDE * i;
        const double* b = o + 15;
        double bp[4];
        for (int c = 0; c < 4; ++c) bp[c] = b[0] * o[c] + b[1] * o[4 + c] + b[2] * o[8 + c];
        double q[3][4];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) q[r][c] = o[4 * r + c] - b[r] * bp[c];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) m[r][c] += q[0][r] * q[0][c] + q[1][r] * q[1][c] + q[2][r] * q[2][c];
    }
    if (cnt < 2) return false;
    double h[4];
    if (!null_vector4(m, h)) return false;
    return dehom(h, x);
}

__device__ double angle_between(double ax, double ay, double az, double bx, double by, double bz) {
    double n = sqrt(ax * ax + ay * ay + az * az) * sqrt(bx * bx + by * by + bz * bz);
    if (n == 0.0) return 0.0;
    double c = (ax * bx + ay * by + az * bz) / n;
    c = fmin(fmax(c, -1.0), 1.0);
    return acos(c);
}

// 모든 뷰 깊이 양수 + 뷰 쌍 최대 삼각측량 각 ≥ 최소각.
__device__ bool valid_point(const double* obs, u64 mask, int n, const double x[3], double min_angle) {
    for (int i = 0; i < n; ++i) {
        if (!((mask >> i) & 1ULL)) continue;
        const double* p = obs + OBS_STRIDE * i;
        if (!(p[8] * x[0] + p[9] * x[1] + p[10] * x[2] + p[11] >= DBL_EPS)) return false;
    }
    for (int i = 0; i < n; ++i) {
        if (!((mask >> i) & 1ULL)) continue;
        const double* ci = obs + OBS_STRIDE * i + 12;
        for (int j = 0; j < i; ++j) {
            if (!((mask >> j) & 1ULL)) continue;
            const double* cj = obs + OBS_STRIDE * j + 12;
            double th = angle_between(x[0] - ci[0], x[1] - ci[1], x[2] - ci[2], x[0] - cj[0], x[1] - cj[1], x[2] - cj[2]);
            double ang = fmin(th, PI_D - th);
            if (ang >= min_angle) return true;
        }
    }
    return false;
}

// 각도 오차 제곱.
__device__ double ang_res(const double* o, const double x[3]) {
    double xc0 = o[0] * x[0] + o[1] * x[1] + o[2] * x[2] + o[3] * 1.0;
    double xc1 = o[4] * x[0] + o[5] * x[1] + o[6] * x[2] + o[7] * 1.0;
    double xc2 = o[8] * x[0] + o[9] * x[1] + o[10] * x[2] + o[11] * 1.0;
    double nn = sqrt(xc0 * xc0 + xc1 * xc1 + xc2 * xc2);
    double e;
    if (nn == 0.0) {
        e = PI_D;
    } else {
        const double* r = o + 15;
        double c = (r[0] * xc0 + r[1] * xc1 + r[2] * xc2) / nn;
        c = fmin(fmax(c, -1.0), 1.0);
        e = acos(c);
    }
    return e * e;
}

__device__ void support(const double* obs, int n, const double x[3], double thr, u32* cnt, double* sum, u64* mask) {
    u32 c = 0;
    double s = 0.0;
    u64 m = 0;
    for (int i = 0; i < n; ++i) {
        double r = ang_res(obs + OBS_STRIDE * i, x);
        if (r <= thr) {
            c++;
            s += r;
            m |= (1ULL << i);
        }
    }
    *cnt = c;
    *sum = s;
    *mask = m;
}

__device__ bool better(u32 n1, double s1, u32 n2, double s2) {
    if (n1 != n2) return n1 > n2;
    return s1 < s2;
}

// CPU compute_num_trials(k = 2, 배수 3). usize::MAX 는 ~0ULL.
__device__ u64 num_trials(u32 nin, int n, double conf) {
    const int k = 2;
    if ((int)nin < k || n < k) return ~0ULL;
    double p = 1.0;
    for (int i = 0; i < k; ++i) p *= (double)((int)nin - i) / (double)(n - i);
    if (p >= 1.0) return 1ULL;
    if (p <= 0.0) return ~0ULL;
    double nom = 1.0 - conf;
    if (nom <= 0.0) return ~0ULL;
    double denom = 1.0 - p;
    if (denom <= 0.0) return 1ULL;
    double v = ceil(log(nom) / log(denom) * 3.0);
    if (isnan(v) || v >= USIZE_MAX_F) return ~0ULL;
    return (u64)v;
}

// 스레드 하나 = 문제 하나(CPU estimate_triangulation: 조합 표본 2점 DLT + 다중 뷰 LO).
// desc_i[4p..] = (관측 시작 번호, 관측 수, 최소 반복, 최대 반복),
// desc_d[3p..] = (제곱 임계, 최소 삼각측량 각(라디안), 신뢰도).
// 출력: out_x[3p..] 점, out_mask[p] 인라이어 비트, out_ok[p] 성공 여부.
extern "C" __global__ void tri_ransac(const double* __restrict__ obs_all, const long long* __restrict__ desc_i,
                                      const double* __restrict__ desc_d, int np, double* out_x, u64* out_mask, int* out_ok) {
    int pi = blockIdx.x * blockDim.x + threadIdx.x;
    if (pi >= np) return;
    long long o0 = desc_i[4 * (long long)pi];
    int n = (int)desc_i[4 * (long long)pi + 1];
    u64 min_trials = (u64)desc_i[4 * (long long)pi + 2];
    u64 max_trials = (u64)desc_i[4 * (long long)pi + 3];
    double thr = desc_d[3 * (long long)pi];
    double min_angle = desc_d[3 * (long long)pi + 1];
    double conf = desc_d[3 * (long long)pi + 2];
    out_ok[pi] = 0;
    out_mask[pi] = 0;
    if (n < 2 || n > MAX_TRI_OBS) return;
    const double* obs = obs_all + OBS_STRIDE * o0;

    bool have = false;
    double bx[3] = {0.0, 0.0, 0.0};
    u32 bn = 0;
    double bs = inf_d();
    u64 dyn_max = max_trials;
    int si = 0, sj = 1;
    bool first = true;
    u64 trials = 0;
    while (trials < max_trials) {
        u64 t = trials;
        trials++;
        if (!first) {
            if (sj < n - 1) {
                sj++;
            } else if (si < n - 2) {
                si++;
                sj = si + 1;
            } else {
                si = 0;
                sj = 1;
            }
        }
        first = false;
        double x[3];
        u64 smask = (1ULL << si) | (1ULL << sj);
        if (!tri_dlt(obs + OBS_STRIDE * si, obs + OBS_STRIDE * sj, x)) continue;
        if (!valid_point(obs, smask, n, x, min_angle)) continue;
        u32 cnt;
        double sum;
        u64 mask;
        support(obs, n, x, thr, &cnt, &sum, &mask);
        if (better(cnt, sum, bn, bs)) {
            double cx[3] = {x[0], x[1], x[2]};
            u32 cn = cnt;
            double cs = sum;
            if (cnt > 2) {
                // 국소 최적화: 인라이어 전체 다중 뷰.
                u64 cur = mask;
                for (int it = 0; it < 10; ++it) {
                    u32 prev = cn;
                    double lx[3];
                    if (tri_multi(obs, cur, n, lx) && valid_point(obs, cur, n, lx, min_angle)) {
                        u32 ln;
                        double ls;
                        u64 lm;
                        support(obs, n, lx, thr, &ln, &ls, &lm);
                        if (better(ln, ls, cn, cs)) {
                            cx[0] = lx[0];
                            cx[1] = lx[1];
                            cx[2] = lx[2];
                            cn = ln;
                            cs = ls;
                            cur = lm;
                        }
                    }
                    if (cn <= prev) break;
                }
            }
            have = true;
            bx[0] = cx[0];
            bx[1] = cx[1];
            bx[2] = cx[2];
            bn = cn;
            bs = cs;
            dyn_max = num_trials(bn, n, conf);
        }
        if (t >= dyn_max && t >= min_trials) break;
    }
    if (!have || bn < 2) return;
    u32 fc;
    double fs;
    u64 fm;
    support(obs, n, bx, thr, &fc, &fs, &fm);
    out_x[3 * (long long)pi] = bx[0];
    out_x[3 * (long long)pi + 1] = bx[1];
    out_x[3 * (long long)pi + 2] = bx[2];
    out_mask[pi] = fm;
    out_ok[pi] = 1;
}
