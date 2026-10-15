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

// SIFT 가우시안 스케일 공간. cumulus3d-features 의 CPU 피라미드와 같은 식·같은 연산 순서.
// 곱셈-덧셈 융합을 끈 채(--fmad=false) 컴파일해 CPU 결과와 비트 단위로 맞춘다.

typedef unsigned char u8;

// dst[y·w + x] = src[y·sw + x] / 255
extern "C" __global__ void sift_normalize(const u8* __restrict__ src, int sw, int w, int h, float* __restrict__ dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dst[(size_t)y * w + x] = (float)src[(size_t)y * sw + x] / 255.0f;
}

// 2배 업샘플: 출력 (2r+a, 2c+b), 가장자리 복제.
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

// 가로 블러: out = k_r·p + Σ_{i<r} k_i·(row[x+i−r] + row[x+r−i]) (가장자리 복제), i 오름차순.
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

// 세로 블러: 같은 식을 행 방향으로(행 색인 고정).
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

// 짝수 행·열만 취해 1/2 축소.
extern "C" __global__ void sift_decimate2(const float* __restrict__ src, int w, float* __restrict__ dst, int w2, int h2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    dst[(size_t)y * w2 + x] = src[(size_t)(2 * y) * w + 2 * x];
}
