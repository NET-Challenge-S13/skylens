/*
 * undistort.cu
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

// Fused undistortion + area-average downscale, matching the CPU path in cumulus3d_dense::undistort:
//   1. For each input-resolution pixel (x, y) of the auxiliary PINHOLE camera, map to the distorted source position
//      (double precision, same formulas as Camera::normalized_to_img), then bilinear-sample the source (f32, rounded to u8,
//      black when the 2x2 footprint leaves the image or the position is not finite).
//   2. Each output pixel is the exact area average (fractional box weights) of those warped input-resolution pixels.
// The warped intermediate image is never materialised: the footprint of an output pixel (about 2x2 .. 4x4 input pixels for
// 2048 -> 960) is warped on the fly. When the output has the input size the footprint is exactly one pixel.
// cam[0..4]  = auxiliary pinhole fx, fy, cx, cy (at input resolution)
// cam[4..12] = source fx, fy, cx, cy, k1, k2, p1, p2 (OPENCV form; RADIAL models use p1 = p2 = 0, SIMPLE_RADIAL also k2 = 0)

#define MAX_CH 4

__device__ __forceinline__ bool warp_pos(const double* cam, int x, int y, float* ox, float* oy) {
    double u = ((double)x + 0.5 - cam[2]) / cam[0];
    double v = ((double)y + 0.5 - cam[3]) / cam[1];
    double r2 = u * u + v * v;
    double radial = cam[8] * r2 + cam[9] * r2 * r2;
    double du = u * radial + 2.0 * cam[10] * u * v + cam[11] * (r2 + 2.0 * u * u);
    double dv = v * radial + 2.0 * cam[11] * u * v + cam[10] * (r2 + 2.0 * v * v);
    double px = cam[4] * (u + du) + cam[6];
    double py = cam[5] * (v + dv) + cam[7];
    if (!(isfinite(px) && isfinite(py))) return false;
    *ox = (float)(px - 0.5);
    *oy = (float)(py - 0.5);
    return true;
}

// Warped, rounded value of input-resolution pixel (x, y) for all channels. Returns false (all zero) when outside.
__device__ __forceinline__ bool warp_pixel(const unsigned char* __restrict__ src, int sw, int sh, int ch, const double* cam, int x, int y,
                                           float* out) {
    float fx, fy;
    for (int c = 0; c < MAX_CH; ++c) out[c] = 0.0f;
    if (!warp_pos(cam, x, y, &fx, &fy)) return false;
    float x0f = floorf(fx), y0f = floorf(fy);
    if (x0f < 0.0f || y0f < 0.0f || x0f + 1.0f >= (float)sw || y0f + 1.0f >= (float)sh) return false;
    float ax = fx - x0f, ay = fy - y0f;
    int x0 = (int)x0f, y0 = (int)y0f;
    for (int c = 0; c < ch; ++c) {
        float p00 = (float)src[((size_t)y0 * sw + x0) * ch + c];
        float p10 = (float)src[((size_t)y0 * sw + x0 + 1) * ch + c];
        float p01 = (float)src[((size_t)(y0 + 1) * sw + x0) * ch + c];
        float p11 = (float)src[((size_t)(y0 + 1) * sw + x0 + 1) * ch + c];
        float top = p00 * (1.0f - ax) + p10 * ax;
        float bot = p01 * (1.0f - ax) + p11 * ax;
        float v = top * (1.0f - ay) + bot * ay;
        out[c] = fminf(fmaxf(roundf(v), 0.0f), 255.0f);
    }
    return true;
}

extern "C" __global__ void undistort_resample(const unsigned char* __restrict__ src, int sw, int sh, int ch, unsigned char* dst, int ow,
                                              int oh, const double* __restrict__ cam) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= ow || oy >= oh) return;
    double sx = (double)sw / (double)ow, sy = (double)sh / (double)oh;
    double bx0 = (double)ox * sx, bx1 = (double)(ox + 1) * sx;
    double by0 = (double)oy * sy, by1 = (double)(oy + 1) * sy;
    double area = (bx1 - bx0) * (by1 - by0);
    int ix0 = max((int)floor(bx0), 0), ix1 = min((int)ceil(bx1), sw);
    int iy0 = max((int)floor(by0), 0), iy1 = min((int)ceil(by1), sh);
    double acc[MAX_CH] = {0.0, 0.0, 0.0, 0.0};
    for (int j = iy0; j < iy1; ++j) {
        double wy = fmin((double)(j + 1), by1) - fmax((double)j, by0);
        if (wy <= 0.0) continue;
        for (int i = ix0; i < ix1; ++i) {
            double wx = fmin((double)(i + 1), bx1) - fmax((double)i, bx0);
            if (wx <= 0.0) continue;
            float v[MAX_CH];
            if (!warp_pixel(src, sw, sh, ch, cam, i, j, v)) continue;
            double w = wx * wy;
            for (int c = 0; c < ch; ++c) acc[c] += w * (double)v[c];
        }
    }
    for (int c = 0; c < ch; ++c) {
        float m = (float)(area > 0.0 ? acc[c] / area : 0.0);
        dst[((size_t)oy * ow + ox) * ch + c] = (unsigned char)fminf(fmaxf(roundf(m), 0.0f), 255.0f);
    }
}
