/*
 * ba.cu
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

// 번들 조정 GPU 커널(전부 f64). cumulus3d-ba 의 problem.rs 와 같은 수식이다.
//
// - ba_eval: 관측별 잔차·견고 손실 배율·야코비안(자세 2×6, 점 2×3, 카메라 2×nv), 비용 부분합.
// - ba_img_normal: 영상별 J_Eᵀ J_E, J_Eᵀ r 를 모아 밀집 H(n_e×n_e)·g_E 에 더한다.
// - ba_point_normal: 점별 V = J_xᵀ J_x, g_p, 점-E 블록 W = J_Eᵀ J_x.
// - ba_vinv: 감쇠한 V 의 역행렬(3×3 촐레스키).
// - ba_s_init / ba_schur / ba_symmetrize / ba_fix_mask: 축소 계통 S = H + D − W V⁻¹ Wᵀ, 우변 −g_E + W V⁻¹ g_p.
// - ba_matvec / ba_precond / ba_dot* / ba_axpy2 / ba_xpby: 블록 야코비 PCG.
// - ba_backsub / ba_model_cost: 점 역대입과 모형 비용 변화.
//
// 부분합은 고정 격자(블록 256)의 블록별 합을 호스트가 블록 순서대로 더한다(같은 입력이면 같은 값).
// H·S·우변 누적은 원자적 덧셈이라 합산 순서가 실행마다 다를 수 있다(마지막 자리 수준 차이).

#define BS 256
#define MAXC 8
#define DBL_EPS 2.220446049250313e-16
#define DBL_MINPOS 2.2250738585072014e-308

__device__ __forceinline__ void atomic_add_f64(double* addr, double v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed, __double_as_longlong(v + __longlong_as_double(assumed)));
    } while (assumed != old);
#else
    atomicAdd(addr, v);
#endif
}

// 블록 합을 partial[blockIdx.x] 에 쓴다(커널마다 한 번만 부른다).
__device__ void block_sum(double v, double* partial) {
    __shared__ double sh[BS];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

__device__ void block_max(double v, double* partial) {
    __shared__ double shm[BS];
    shm[threadIdx.x] = v;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) shm[threadIdx.x] = fmax(shm[threadIdx.x], shm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = shm[0];
}

// 견고 손실: s = 잔차 제곱노름 → (ρ(s), √ρ'(s)). kind 0 없음, 1 SoftL1, 2 Cauchy, 3 Huber.
__device__ void loss_weight(int kind, double a, double s, double* rho, double* w) {
    if (kind == 1) {
        double b = a * a, c = 1.0 / b;
        double sum = 1.0 + s * c;
        double tmp = sqrt(sum);
        double r1 = fmax(1.0 / tmp, DBL_MINPOS);
        *rho = 2.0 * b * (tmp - 1.0);
        *w = sqrt(r1);
    } else if (kind == 2) {
        double b = a * a, c = 1.0 / b;
        double sum = 1.0 + s * c;
        double inv = 1.0 / sum;
        *rho = b * log(sum);
        *w = sqrt(fmax(inv, DBL_MINPOS));
    } else if (kind == 3) {
        double b = a * a;
        if (s > b) {
            double r = sqrt(s);
            double r1 = fmax(a / r, DBL_MINPOS);
            *rho = 2.0 * a * r - b;
            *w = sqrt(r1);
        } else {
            *rho = s;
            *w = 1.0;
        }
    } else {
        *rho = s;
        *w = 1.0;
    }
}

// 카메라 투영과 야코비안(cumulus3d-core Camera::cam_to_img_with_jacobian 과 같은 수식).
// 모델: 0 SIMPLE_PINHOLE(f,cx,cy) 1 PINHOLE(fx,fy,cx,cy) 2 SIMPLE_RADIAL(f,cx,cy,k)
//       3 RADIAL(f,cx,cy,k1,k2) 4 OPENCV(fx,fy,cx,cy,k1,k2,p1,p2).
// J = ∂픽셀/∂카메라좌표(2×3 행 우선). dp != 0 이면 ∂픽셀/∂파라미터(행 우선 2×n). z < ε 면 거짓.
__device__ bool project(int model, int n, const double* p, const double* xc, double* pix, double* J, double* dp) {
    if (xc[2] < DBL_EPS) return false;
    double iz = 1.0 / xc[2];
    double u = xc[0] * iz, v = xc[1] * iz;
    double fx, fy, cx, cy;
    if (model == 1 || model == 4) {
        fx = p[0]; fy = p[1]; cx = p[2]; cy = p[3];
    } else {
        fx = p[0]; fy = p[0]; cx = p[1]; cy = p[2];
    }
    double du = 0.0, dv = 0.0, j00 = 0.0, j01 = 0.0, j10 = 0.0, j11 = 0.0;
    double r2 = u * u + v * v;
    if (model == 2 || model == 3) {
        double k1 = p[3], k2 = (model == 3) ? p[4] : 0.0;
        double radial = k1 * r2 + k2 * r2 * r2;
        double g = 2.0 * k1 + 4.0 * k2 * r2;
        du = u * radial;
        dv = v * radial;
        j00 = radial + u * g * u;
        j01 = u * g * v;
        j10 = v * g * u;
        j11 = radial + v * g * v;
    } else if (model == 4) {
        double k1 = p[4], k2 = p[5], p1 = p[6], p2 = p[7];
        double radial = k1 * r2 + k2 * r2 * r2;
        du = u * radial + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u);
        dv = v * radial + 2.0 * p2 * u * v + p1 * (r2 + 2.0 * v * v);
        j00 = radial + u * (2.0 * k1 * u + 4.0 * k2 * r2 * u) + 2.0 * p1 * v + 6.0 * p2 * u;
        j01 = u * (2.0 * k1 * v + 4.0 * k2 * r2 * v) + 2.0 * p1 * u + 2.0 * p2 * v;
        j10 = v * (2.0 * k1 * u + 4.0 * k2 * r2 * u) + 2.0 * p2 * v + 2.0 * p1 * u;
        j11 = radial + v * (2.0 * k1 * v + 4.0 * k2 * r2 * v) + 2.0 * p2 * u + 6.0 * p1 * v;
    }
    double xd = u + du, yd = v + dv;
    pix[0] = fx * xd + cx;
    pix[1] = fy * yd + cy;
    // D = I + jd, nm = [iz 0 −x·iz²; 0 iz −y·iz²], J = diag(fx, fy)·D·nm.
    double d00 = 1.0 + j00, d01 = j01, d10 = j10, d11 = 1.0 + j11;
    double n02 = -xc[0] * iz * iz, n12 = -xc[1] * iz * iz;
    J[0] = fx * (d00 * iz);
    J[1] = fx * (d01 * iz);
    J[2] = fx * (d00 * n02 + d01 * n12);
    J[3] = fy * (d10 * iz);
    J[4] = fy * (d11 * iz);
    J[5] = fy * (d10 * n02 + d11 * n12);
    if (dp) {
        double* rx = dp;
        double* ry = dp + n;
        for (int i = 0; i < n; i++) {
            rx[i] = 0.0;
            ry[i] = 0.0;
        }
        if (model == 0) {
            rx[0] = u; rx[1] = 1.0;
            ry[0] = v; ry[2] = 1.0;
        } else if (model == 1) {
            rx[0] = u; rx[2] = 1.0;
            ry[1] = v; ry[3] = 1.0;
        } else if (model == 2 || model == 3) {
            double f = fx;
            rx[0] = xd; rx[1] = 1.0; rx[3] = f * u * r2;
            ry[0] = yd; ry[2] = 1.0; ry[3] = f * v * r2;
            if (model == 3) {
                rx[4] = f * u * r2 * r2;
                ry[4] = f * v * r2 * r2;
            }
        } else {
            rx[0] = xd; rx[2] = 1.0;
            rx[4] = fx * u * r2;
            rx[5] = fx * u * r2 * r2;
            rx[6] = fx * 2.0 * u * v;
            rx[7] = fx * (r2 + 2.0 * u * u);
            ry[1] = yd; ry[3] = 1.0;
            ry[4] = fy * v * r2;
            ry[5] = fy * v * r2 * r2;
            ry[6] = fy * (r2 + 2.0 * v * v);
            ry[7] = fy * 2.0 * u * v;
        }
    }
    return true;
}

// 관측별 잔차·야코비안. want_jac = 0 이면 비용만(후보 상태 평가). 투영 실패면 잔차·야코비안 0, 비용 0.
// 영상 i: img_pose, img_cam, img_hs(비기준 센서 여부), img_sr(3×3)·img_st(3) = rig_to_sensor.
// 자세 p: rot(3×3 행 우선)·trans, pose_eoff = E 스칼라 오프셋(−1 상수), pose_mask 비트(1 = 성분 상수).
// 카메라 c: cam_model, cam_np, cam_params(8 간격), cam_eoff(−1 상수), cam_nv, cam_var(8 간격).
extern "C" __global__ void ba_eval(int nobs, int want_jac, int cs,
                                   const int* __restrict__ obs_img, const int* __restrict__ obs_pt, const double* __restrict__ obs_xy,
                                   const int* __restrict__ img_pose, const int* __restrict__ img_cam, const int* __restrict__ img_hs,
                                   const double* __restrict__ img_sr, const double* __restrict__ img_st,
                                   const double* __restrict__ rot, const double* __restrict__ trans,
                                   const int* __restrict__ pose_eoff, const int* __restrict__ pose_mask,
                                   const int* __restrict__ cam_model, const int* __restrict__ cam_np, const double* __restrict__ cam_params,
                                   const int* __restrict__ cam_eoff, const int* __restrict__ cam_nv, const int* __restrict__ cam_var,
                                   const double* __restrict__ pts, const int* __restrict__ pt_v,
                                   const double* __restrict__ scale_e, const double* __restrict__ scale_p,
                                   int loss_kind, double loss_a,
                                   double* jp, double* jx, double* jc, double* res, double* partial) {
    double acc = 0.0;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nobs; k += gridDim.x * blockDim.x) {
        int i = obs_img[k], p = img_pose[i], c = img_cam[i], j = obs_pt[k];
        const double* R = rot + 9 * p;
        const double* t = trans + 3 * p;
        const double* X = pts + 3 * j;
        double xrr[3], xr[3], xc[3];
        for (int r = 0; r < 3; r++) {
            xrr[r] = R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2];
            xr[r] = xrr[r] + t[r];
        }
        const int hs = img_hs[i];
        const double* Rs = img_sr + 9 * i;
        if (hs) {
            const double* ts = img_st + 3 * i;
            for (int r = 0; r < 3; r++) xc[r] = Rs[3 * r] * xr[0] + Rs[3 * r + 1] * xr[1] + Rs[3 * r + 2] * xr[2] + ts[r];
        } else {
            xc[0] = xr[0];
            xc[1] = xr[1];
            xc[2] = xr[2];
        }
        const int n = cam_np[c];
        double dp[2 * MAXC];
        double J[6], pix[2];
        bool ok = project(cam_model[c], n, cam_params + MAXC * c, xc, pix, J, want_jac ? dp : 0);
        if (!ok) {
            if (want_jac) {
                for (int q = 0; q < 12; q++) jp[12 * k + q] = 0.0;
                for (int q = 0; q < 6; q++) jx[6 * k + q] = 0.0;
                for (int q = 0; q < cs; q++) jc[(size_t)cs * k + q] = 0.0;
                res[2 * k] = 0.0;
                res[2 * k + 1] = 0.0;
            }
            continue;
        }
        double r0 = pix[0] - obs_xy[2 * k], r1 = pix[1] - obs_xy[2 * k + 1];
        double rho, w;
        loss_weight(loss_kind, loss_a, r0 * r0 + r1 * r1, &rho, &w);
        acc += rho;
        if (!want_jac) continue;
        res[2 * k] = r0 * w;
        res[2 * k + 1] = r1 * w;
        // M = J·Rs (센서가 있으면).
        double M[6];
        if (hs) {
            for (int r = 0; r < 2; r++)
                for (int q = 0; q < 3; q++) M[3 * r + q] = J[3 * r] * Rs[q] + J[3 * r + 1] * Rs[3 + q] + J[3 * r + 2] * Rs[6 + q];
        } else {
            for (int q = 0; q < 6; q++) M[q] = J[q];
        }
        double* JP = jp + 12 * k;
        const int pe = pose_eoff[p];
        if (pe >= 0) {
            // jrot = M·skew(xrr)·(−2), skew(v) = [0 −v2 v1; v2 0 −v0; −v1 v0 0].
            double S[9] = {0.0, -xrr[2], xrr[1], xrr[2], 0.0, -xrr[0], -xrr[1], xrr[0], 0.0};
            const int mask = pose_mask[p];
            for (int r = 0; r < 2; r++) {
                for (int q = 0; q < 3; q++) {
                    double jr = (M[3 * r] * S[q] + M[3 * r + 1] * S[3 + q] + M[3 * r + 2] * S[6 + q]) * (-2.0);
                    JP[r * 6 + q] = ((mask >> q) & 1) ? 0.0 : jr * w;
                    JP[r * 6 + 3 + q] = ((mask >> (3 + q)) & 1) ? 0.0 : M[3 * r + q] * w;
                }
            }
            for (int q = 0; q < 6; q++) {
                double s = scale_e[pe + q];
                JP[q] *= s;
                JP[6 + q] *= s;
            }
        } else {
            for (int q = 0; q < 12; q++) JP[q] = 0.0;
        }
        double* JX = jx + 6 * k;
        const int vi = pt_v[j];
        if (vi >= 0) {
            const double* sp = scale_p + 3 * vi;
            for (int r = 0; r < 2; r++)
                for (int q = 0; q < 3; q++) {
                    double v = M[3 * r] * R[q] + M[3 * r + 1] * R[3 + q] + M[3 * r + 2] * R[6 + q];
                    JX[r * 3 + q] = v * w * sp[q];
                }
        } else {
            for (int q = 0; q < 6; q++) JX[q] = 0.0;
        }
        double* JC = jc + (size_t)cs * k;
        const int ce = cam_eoff[c];
        if (ce >= 0) {
            const int nv = cam_nv[c];
            for (int q = 0; q < nv; q++) {
                int pi = cam_var[MAXC * c + q];
                double s = scale_e[ce + q];
                JC[q] = dp[pi] * w * s;
                JC[nv + q] = dp[n + pi] * w * s;
            }
            for (int q = 2 * nv; q < cs; q++) JC[q] = 0.0;
        } else {
            for (int q = 0; q < cs; q++) JC[q] = 0.0;
        }
    }
    block_sum(acc, partial);
}

// 관측 k 의 E 국소 야코비안 행 row, 열 l(0..5 자세, 6.. 카메라).
__device__ __forceinline__ double jloc(const double* jp, const double* jc, int cs, int nv, int k, int row, int l) {
    return l < 6 ? jp[12 * k + row * 6 + l] : jc[(size_t)cs * k + row * nv + (l - 6)];
}

// 영상 하나 = 블록 하나. 스레드가 국소 (6+nv)² 항목과 기울기 항목을 나눠 영상의 관측을 차례로 합하고
// 전역 H·g_E 에 원자적으로 더한다(rig 프레임·공유 카메라가 여러 영상에서 겹치므로).
extern "C" __global__ void ba_img_normal(int n_e, int cs,
                                         const int* __restrict__ img_obs_ptr, const int* __restrict__ img_obs_idx,
                                         const int* __restrict__ img_pose, const int* __restrict__ img_cam,
                                         const int* __restrict__ pose_eoff, const int* __restrict__ cam_eoff, const int* __restrict__ cam_nv,
                                         const double* __restrict__ jp, const double* __restrict__ jc, const double* __restrict__ res,
                                         double* H, double* g_e) {
    const int i = blockIdx.x;
    const int pe = pose_eoff[img_pose[i]];
    const int c = img_cam[i];
    const int ce = cam_eoff[c];
    const int nv = ce >= 0 ? cam_nv[c] : 0;
    if (pe < 0 && ce < 0) return;
    const int L = 6 + nv;
    const int nent = L * L + L;
    const int k0 = img_obs_ptr[i], k1 = img_obs_ptr[i + 1];
    for (int e = threadIdx.x; e < nent; e += blockDim.x) {
        const bool isg = e >= L * L;
        int r, cc = 0;
        if (isg) {
            r = e - L * L;
        } else {
            r = e / L;
            cc = e % L;
        }
        int gr = r < 6 ? (pe >= 0 ? pe + r : -1) : ce + (r - 6);
        if (gr < 0) continue;
        int gc = 0;
        if (!isg) {
            gc = cc < 6 ? (pe >= 0 ? pe + cc : -1) : ce + (cc - 6);
            if (gc < 0) continue;
        }
        double acc = 0.0;
        for (int q = k0; q < k1; q++) {
            int k = img_obs_idx[q];
            double a0 = jloc(jp, jc, cs, nv, k, 0, r), a1 = jloc(jp, jc, cs, nv, k, 1, r);
            if (isg) {
                acc += a0 * res[2 * k] + a1 * res[2 * k + 1];
            } else {
                acc += a0 * jloc(jp, jc, cs, nv, k, 0, cc) + a1 * jloc(jp, jc, cs, nv, k, 1, cc);
            }
        }
        if (isg) {
            atomic_add_f64(&g_e[gr], acc);
        } else {
            atomic_add_f64(&H[(size_t)gr * n_e + gc], acc);
        }
    }
}

// 가변 점별 V(3×3), g_p(3), 점이 닿는 E 블록별 W = J_Eᵀ J_x(dim×3 행 우선).
// obs_qp / obs_qc = 관측의 자세·카메라 블록이 점의 W 목록에서 차지하는 항목(−1 없음).
extern "C" __global__ void ba_point_normal(int nvp, int cs, const int* __restrict__ v_pt, const int* __restrict__ pt_obs,
                                           const int* __restrict__ obs_qp, const int* __restrict__ obs_qc,
                                           const int* __restrict__ pw_ptr, const int* __restrict__ pw_dim, const int* __restrict__ pw_voff,
                                           const double* __restrict__ jp, const double* __restrict__ jx, const double* __restrict__ jc,
                                           const double* __restrict__ res, double* V, double* gp, double* pw_vals) {
    for (int vi = blockIdx.x * blockDim.x + threadIdx.x; vi < nvp; vi += gridDim.x * blockDim.x) {
        const int j = v_pt[vi];
        for (int q = pw_ptr[vi]; q < pw_ptr[vi + 1]; q++) {
            double* W = pw_vals + pw_voff[q];
            for (int e = 0; e < 3 * pw_dim[q]; e++) W[e] = 0.0;
        }
        double m[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double g[3] = {0.0, 0.0, 0.0};
        for (int k = pt_obs[j]; k < pt_obs[j + 1]; k++) {
            const double* JX = jx + 6 * k;
            const double r0 = res[2 * k], r1 = res[2 * k + 1];
            for (int a = 0; a < 3; a++) {
                g[a] += JX[a] * r0 + JX[3 + a] * r1;
                for (int b = 0; b < 3; b++) m[3 * a + b] += JX[a] * JX[b] + JX[3 + a] * JX[3 + b];
            }
            const int qp = obs_qp[k];
            if (qp >= 0) {
                const double* JP = jp + 12 * k;
                double* W = pw_vals + pw_voff[qp];
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 3; c++) W[3 * r + c] += JP[r] * JX[c] + JP[6 + r] * JX[3 + c];
            }
            const int qc = obs_qc[k];
            if (qc >= 0) {
                const int nv = pw_dim[qc];
                const double* JC = jc + (size_t)cs * k;
                double* W = pw_vals + pw_voff[qc];
                for (int r = 0; r < nv; r++)
                    for (int c = 0; c < 3; c++) W[3 * r + c] += JC[r] * JX[c] + JC[nv + r] * JX[3 + c];
            }
        }
        for (int e = 0; e < 9; e++) V[9 * vi + e] = m[e];
        for (int e = 0; e < 3; e++) gp[3 * vi + e] = g[e];
    }
}

// 첫 선형화 뒤 야코비 열 배율 1/(1+√대각).
extern "C" __global__ void ba_set_scales(int n_e, int nvp, const double* __restrict__ H, const double* __restrict__ V, double* scale_e,
                                         double* scale_p) {
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n_e + nvp; t += gridDim.x * blockDim.x) {
        if (t < n_e) {
            scale_e[t] = 1.0 / (1.0 + sqrt(H[(size_t)t * n_e + t]));
        } else {
            int vi = t - n_e;
            for (int q = 0; q < 3; q++) scale_p[3 * vi + q] = 1.0 / (1.0 + sqrt(V[9 * vi + 4 * q]));
        }
    }
}

// 감쇠 V 의 역행렬. 양정치가 아니거나 비유한이면 fail[0] = 1.
extern "C" __global__ void ba_vinv(int nvp, double radius, const double* __restrict__ V, double* vinv, int* fail) {
    for (int vi = blockIdx.x * blockDim.x + threadIdx.x; vi < nvp; vi += gridDim.x * blockDim.x) {
        const double* v = V + 9 * vi;
        double m[9];
        for (int e = 0; e < 9; e++) m[e] = v[e];
        for (int q = 0; q < 3; q++) m[4 * q] += fmin(fmax(v[4 * q], 1e-6), 1e32) / radius;
        bool ok = true;
        double l00 = 0.0, l10 = 0.0, l20 = 0.0, l11 = 0.0, l21 = 0.0, l22 = 0.0;
        if (m[0] > 0.0) {
            l00 = sqrt(m[0]);
            l10 = m[3] / l00;
            l20 = m[6] / l00;
            double d1 = m[4] - l10 * l10;
            if (d1 > 0.0) {
                l11 = sqrt(d1);
                l21 = (m[7] - l20 * l10) / l11;
                double d2 = m[8] - l20 * l20 - l21 * l21;
                if (d2 > 0.0) {
                    l22 = sqrt(d2);
                } else {
                    ok = false;
                }
            } else {
                ok = false;
            }
        } else {
            ok = false;
        }
        double* o = vinv + 9 * vi;
        if (!ok) {
            fail[0] = 1;
            for (int e = 0; e < 9; e++) o[e] = 0.0;
            continue;
        }
        // L⁻¹ (아래삼각) → A⁻¹ = L⁻ᵀ L⁻¹.
        double i00 = 1.0 / l00, i11 = 1.0 / l11, i22 = 1.0 / l22;
        double i10 = -l10 * i00 * i11;
        double i21 = -l21 * i11 * i22;
        double i20 = -(l20 * i00 + l21 * i10) * i22;
        double a00 = i00 * i00 + i10 * i10 + i20 * i20;
        double a01 = i10 * i11 + i20 * i21;
        double a02 = i20 * i22;
        double a11 = i11 * i11 + i21 * i21;
        double a12 = i21 * i22;
        double a22 = i22 * i22;
        o[0] = a00; o[1] = a01; o[2] = a02;
        o[3] = a01; o[4] = a11; o[5] = a12;
        o[6] = a02; o[7] = a12; o[8] = a22;
        for (int e = 0; e < 9; e++)
            if (!isfinite(o[e])) fail[0] = 1;
    }
}

// S = H + diag(clamp(H_ii)/radius), 우변 = −g_E.
extern "C" __global__ void ba_s_init(int n_e, double radius, const double* __restrict__ H, const double* __restrict__ g_e, double* S,
                                     double* rhs) {
    const size_t nn = (size_t)n_e * n_e;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < nn; t += (size_t)gridDim.x * blockDim.x) {
        double h = H[t];
        size_t r = t / n_e, c = t % n_e;
        if (r == c) h += fmin(fmax(h, 1e-6), 1e32) / radius;
        S[t] = h;
        if (t < (size_t)n_e) rhs[t] = -g_e[t];
    }
}

// 점-E 항목 q 하나 = 스레드 하나: T = W_q V⁻¹, 우변 += T g_p, 같은 점의 항목 q2(오프셋 ≥) 에 대해
// S[q 행, q2 열] −= T W_q2ᵀ (위 블록 삼각과 대각 블록만; 아래는 ba_symmetrize 가 채운다).
extern "C" __global__ void ba_schur(int nent, int n_e, const int* __restrict__ pw_pt, const int* __restrict__ pw_ptr,
                                    const int* __restrict__ pw_eoff, const int* __restrict__ pw_dim, const int* __restrict__ pw_voff,
                                    const double* __restrict__ pw_vals, const double* __restrict__ vinv, const double* __restrict__ gp,
                                    double* S, double* rhs) {
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < nent; q += gridDim.x * blockDim.x) {
        const int vi = pw_pt[q];
        const int d1 = pw_dim[q], o1 = pw_eoff[q];
        const double* W1 = pw_vals + pw_voff[q];
        const double* Vi = vinv + 9 * vi;
        const double* g = gp + 3 * vi;
        double T[3 * MAXC];
        for (int r = 0; r < d1; r++) {
            for (int m = 0; m < 3; m++) T[3 * r + m] = W1[3 * r] * Vi[m] + W1[3 * r + 1] * Vi[3 + m] + W1[3 * r + 2] * Vi[6 + m];
            atomic_add_f64(&rhs[o1 + r], T[3 * r] * g[0] + T[3 * r + 1] * g[1] + T[3 * r + 2] * g[2]);
        }
        for (int q2 = pw_ptr[vi]; q2 < pw_ptr[vi + 1]; q2++) {
            const int o2 = pw_eoff[q2];
            if (o2 < o1) continue;
            const int d2 = pw_dim[q2];
            const double* W2 = pw_vals + pw_voff[q2];
            for (int r = 0; r < d1; r++) {
                double* row = S + (size_t)(o1 + r) * n_e + o2;
                for (int c = 0; c < d2; c++) {
                    double v = T[3 * r] * W2[3 * c] + T[3 * r + 1] * W2[3 * c + 1] + T[3 * r + 2] * W2[3 * c + 2];
                    atomic_add_f64(&row[c], -v);
                }
            }
        }
    }
}

// 블록 행 번호가 큰 쪽(아래 블록 삼각)을 위에서 복사한다.
extern "C" __global__ void ba_symmetrize(int n_e, const int* __restrict__ row_blk, double* S) {
    const size_t nn = (size_t)n_e * n_e;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < nn; t += (size_t)gridDim.x * blockDim.x) {
        int r = (int)(t / n_e), c = (int)(t % n_e);
        if (row_blk[r] > row_blk[c]) S[t] = S[(size_t)c * n_e + r];
    }
}

// 상수 자세 성분: 대각 1, 우변 0 (행·열은 이미 0).
extern "C" __global__ void ba_fix_mask(int nm, int n_e, const int* __restrict__ idx, double* S, double* rhs) {
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < nm; t += gridDim.x * blockDim.x) {
        int i = idx[t];
        S[(size_t)i * n_e + i] = 1.0;
        rhs[i] = 0.0;
    }
}

// 대각 블록 모으기(블록 야코비 전처리용, 호스트가 역행렬을 만든다).
extern "C" __global__ void ba_diag_blocks(int nb, int n_e, const int* __restrict__ blk_off, const int* __restrict__ blk_dim,
                                          const int* __restrict__ blk_poff, const double* __restrict__ S, double* out) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        int o = blk_off[b], d = blk_dim[b], po = blk_poff[b];
        for (int r = 0; r < d; r++)
            for (int c = 0; c < d; c++) out[po + r * d + c] = S[(size_t)(o + r) * n_e + o + c];
    }
}

// y = S x (워프 하나가 행 하나).
extern "C" __global__ void ba_matvec(int n, const double* __restrict__ S, const double* __restrict__ x, double* y) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarp = (gridDim.x * blockDim.x) >> 5;
    for (int r = warp; r < n; r += nwarp) {
        const double* row = S + (size_t)r * n;
        double acc = 0.0;
        for (int c = lane; c < n; c += 32) acc += row[c] * x[c];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) y[r] = acc;
    }
}

// z = P⁻¹ r (블록 야코비). row_off/row_dim/row_poff = 스칼라 행이 속한 블록의 시작·크기·역행렬 오프셋.
extern "C" __global__ void ba_precond(int n, const int* __restrict__ row_off, const int* __restrict__ row_dim,
                                      const int* __restrict__ row_poff, const double* __restrict__ pinv, const double* __restrict__ r,
                                      double* z) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int o = row_off[i], d = row_dim[i], po = row_poff[i];
        int li = i - o;
        double acc = 0.0;
        for (int c = 0; c < d; c++) acc += pinv[po + li * d + c] * r[o + c];
        z[i] = acc;
    }
}

extern "C" __global__ void ba_dot(int n, const double* __restrict__ a, const double* __restrict__ b, double* partial) {
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += a[i] * b[i];
    block_sum(acc, partial);
}

// Σ x (b + r).
extern "C" __global__ void ba_dot_xbr(int n, const double* __restrict__ x, const double* __restrict__ b, const double* __restrict__ r,
                                      double* partial) {
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += x[i] * (b[i] + r[i]);
    block_sum(acc, partial);
}

// x += α p, r −= α ap.
extern "C" __global__ void ba_axpy2(int n, double alpha, const double* __restrict__ p, const double* __restrict__ ap, double* x,
                                    double* r) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
    }
}

// p = z + β p.
extern "C" __global__ void ba_xpby(int n, double beta, const double* __restrict__ z, double* p) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = z[i] + beta * p[i];
}

// J_E,k δE (2).
__device__ __forceinline__ void e_product(int k, int cs, const int* obs_img, const int* img_pose, const int* img_cam,
                                          const int* pose_eoff, const int* cam_eoff, const int* cam_nv, const double* jp,
                                          const double* jc, const double* de, double* y) {
    const int i = obs_img[k];
    y[0] = 0.0;
    y[1] = 0.0;
    const int pe = pose_eoff[img_pose[i]];
    if (pe >= 0) {
        const double* JP = jp + 12 * k;
        for (int c = 0; c < 6; c++) {
            y[0] += JP[c] * de[pe + c];
            y[1] += JP[6 + c] * de[pe + c];
        }
    }
    const int cam = img_cam[i];
    const int ce = cam_eoff[cam];
    if (ce >= 0) {
        const int nv = cam_nv[cam];
        const double* JC = jc + (size_t)cs * k;
        for (int c = 0; c < nv; c++) {
            y[0] += JC[c] * de[ce + c];
            y[1] += JC[nv + c] * de[ce + c];
        }
    }
}

// 점 역대입 δp = V⁻¹(−g_p − Σ J_xᵀ J_E δE), 후보 점 = 현재 + 배율·δp. 부분합 = ‖배율·δp‖².
extern "C" __global__ void ba_backsub(int nvp, int cs, const int* __restrict__ v_pt, const int* __restrict__ pt_obs,
                                      const int* __restrict__ obs_img, const int* __restrict__ img_pose, const int* __restrict__ img_cam,
                                      const int* __restrict__ pose_eoff, const int* __restrict__ cam_eoff, const int* __restrict__ cam_nv,
                                      const double* __restrict__ jp, const double* __restrict__ jx, const double* __restrict__ jc,
                                      const double* __restrict__ gp, const double* __restrict__ vinv, const double* __restrict__ de,
                                      const double* __restrict__ scale_p, const double* __restrict__ pts_cur, double* pts_cand,
                                      double* delta_p, double* partial, int* fail) {
    double acc = 0.0;
    for (int vi = blockIdx.x * blockDim.x + threadIdx.x; vi < nvp; vi += gridDim.x * blockDim.x) {
        const int j = v_pt[vi];
        double b[3] = {-gp[3 * vi], -gp[3 * vi + 1], -gp[3 * vi + 2]};
        for (int k = pt_obs[j]; k < pt_obs[j + 1]; k++) {
            double y[2];
            e_product(k, cs, obs_img, img_pose, img_cam, pose_eoff, cam_eoff, cam_nv, jp, jc, de, y);
            const double* JX = jx + 6 * k;
            for (int q = 0; q < 3; q++) b[q] -= JX[q] * y[0] + JX[3 + q] * y[1];
        }
        const double* M = vinv + 9 * vi;
        const double* sp = scale_p + 3 * vi;
        for (int q = 0; q < 3; q++) {
            double d = M[3 * q] * b[0] + M[3 * q + 1] * b[1] + M[3 * q + 2] * b[2];
            if (!isfinite(d)) fail[0] = 1;
            delta_p[3 * vi + q] = d;
            double dd = d * sp[q];
            pts_cand[3 * j + q] = pts_cur[3 * j + q] + dd;
            acc += dd * dd;
        }
    }
    block_sum(acc, partial);
}

// 모형 비용 변화 −(rᵀy + ½‖y‖²), y = J_E δE + J_x δp.
extern "C" __global__ void ba_model_cost(int nobs, int cs, const int* __restrict__ obs_img, const int* __restrict__ obs_pt,
                                         const int* __restrict__ img_pose, const int* __restrict__ img_cam,
                                         const int* __restrict__ pose_eoff, const int* __restrict__ cam_eoff, const int* __restrict__ cam_nv,
                                         const int* __restrict__ pt_v, const double* __restrict__ jp, const double* __restrict__ jx,
                                         const double* __restrict__ jc, const double* __restrict__ res, const double* __restrict__ de,
                                         const double* __restrict__ delta_p, double* partial) {
    double acc = 0.0;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nobs; k += gridDim.x * blockDim.x) {
        double y[2];
        e_product(k, cs, obs_img, img_pose, img_cam, pose_eoff, cam_eoff, cam_nv, jp, jc, de, y);
        const int v = pt_v[obs_pt[k]];
        if (v >= 0) {
            const double* d = delta_p + 3 * v;
            const double* JX = jx + 6 * k;
            y[0] += JX[0] * d[0] + JX[1] * d[1] + JX[2] * d[2];
            y[1] += JX[3] * d[0] + JX[4] * d[1] + JX[5] * d[2];
        }
        const double r0 = res[2 * k], r1 = res[2 * k + 1];
        acc += -(r0 * y[0] + r1 * y[1] + 0.5 * (y[0] * y[0] + y[1] * y[1]));
    }
    block_sum(acc, partial);
}

// max |g_p / 배율| (기울기 무한노름의 점 부분).
extern "C" __global__ void ba_gp_max(int nvp, const double* __restrict__ gp, const double* __restrict__ scale_p, double* partial) {
    double m = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 3 * nvp; i += gridDim.x * blockDim.x) m = fmax(m, fabs(gp[i] / scale_p[i]));
    block_max(m, partial);
}
