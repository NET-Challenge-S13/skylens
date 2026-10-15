/*
 * synthetic.rs
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
 * SPDX-License-Identifier: Apache-2.0
 */

//! 시험용 합성 장면: 레이 캐스팅으로 그린 무늬 바닥(z = 0) + 상자들, 격자 배치 카메라, 참 깊이·법선, 희소점.
//! 정확도 시험(이 크레이트와 GPU 백엔드)이 함께 쓴다.

use crate::image::ImageBuffer;
use crate::scene::{gray_of, DenseScene, DenseView, ScenePoint};
use std::sync::Arc;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: V3) -> V3 {
    let n = dot(a, a).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

fn hash2(ix: i64, iy: i64, seed: u64) -> f64 {
    let z = (ix as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (iy as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f) ^ seed;
    (crate::math::mix64(z) >> 11) as f64 / (1u64 << 53) as f64
}

fn value_noise(x: f64, y: f64, seed: u64) -> f64 {
    let (x0, y0) = (x.floor(), y.floor());
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(x - x0), s(y - y0));
    let (ix, iy) = (x0 as i64, y0 as i64);
    let (a, b, c, d) = (hash2(ix, iy, seed), hash2(ix + 1, iy, seed), hash2(ix, iy + 1, seed), hash2(ix + 1, iy + 1, seed));
    let top = a + (b - a) * sx;
    let bot = c + (d - c) * sx;
    top + (bot - top) * sy
}

/// 고주파 무늬(평면 좌표 미터): 잡음 블록 + 부드러운 그라디언트.
pub fn texture(u: f64, v: f64, seed: u64) -> f64 {
    let blocks = hash2((u / 0.18).floor() as i64, (v / 0.18).floor() as i64, seed ^ 0x55);
    let fine = value_noise(u / 0.09, v / 0.09, seed);
    let coarse = value_noise(u / 1.5, v / 1.5, seed + 7);
    (0.45 * blocks + 0.3 * fine + 0.25 * coarse).clamp(0.0, 1.0)
}

/// 상자(축 정렬).
#[derive(Clone, Copy, Debug)]
pub struct SynthBox {
    /// 최소 꼭짓점.
    pub min: V3,
    /// 최대 꼭짓점.
    pub max: V3,
}

/// 합성 장면 설정.
#[derive(Clone, Debug)]
pub struct SynthConfig {
    /// 영상 너비(픽셀).
    pub width: usize,
    /// 영상 높이(픽셀).
    pub height: usize,
    /// 초점 거리(픽셀).
    pub focal: f64,
    /// 카메라 격자(가로 × 세로)와 간격(m), 높이(m).
    pub grid: (usize, usize),
    /// 카메라 격자 간격(m).
    pub spacing: f64,
    /// 카메라 높이(m).
    pub altitude: f64,
    /// 장면의 상자들.
    pub boxes: Vec<SynthBox>,
    /// 무늬·희소점 난수 시드.
    pub seed: u64,
    /// 희소점 수.
    pub num_points: usize,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            width: 384,
            height: 288,
            focal: 300.0,
            grid: (3, 3),
            spacing: 3.0,
            altitude: 20.0,
            boxes: vec![SynthBox { min: [-3.0, -2.0, 0.0], max: [1.0, 2.5, 4.0] }, SynthBox { min: [3.0, 2.0, 0.0], max: [6.0, 5.0, 2.0] }],
            seed: 11,
            num_points: 3000,
        }
    }
}

/// 합성 장면과 뷰별 참 깊이·법선(카메라 좌표, 카메라 쪽).
pub struct SynthScene {
    /// 조밀화 장면.
    pub scene: DenseScene,
    /// 뷰별 참 깊이.
    pub depth: Vec<Vec<f32>>,
    /// 뷰별 참 법선(카메라 좌표).
    pub normal: Vec<Vec<[f32; 3]>>,
    /// 만들 때 쓴 설정.
    pub config: SynthConfig,
}

/// 광선과 장면의 첫 교점: (거리, 세계 법선, 무늬 값).
fn cast(o: V3, d: V3, cfg: &SynthConfig) -> Option<(f64, V3, f64)> {
    let mut best: Option<(f64, V3, f64)> = None;
    if d[2] < -1e-12 {
        let t = -o[2] / d[2];
        if t > 0.0 {
            let p = [o[0] + t * d[0], o[1] + t * d[1], 0.0];
            best = Some((t, [0.0, 0.0, 1.0], texture(p[0], p[1], cfg.seed)));
        }
    }
    for (bi, b) in cfg.boxes.iter().enumerate() {
        let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
        let mut axis = 0;
        let mut sign = 0.0;
        let mut ok = true;
        for a in 0..3 {
            if d[a].abs() < 1e-12 {
                if o[a] < b.min[a] || o[a] > b.max[a] {
                    ok = false;
                }
                continue;
            }
            let (mut ta, mut tb) = ((b.min[a] - o[a]) / d[a], (b.max[a] - o[a]) / d[a]);
            let mut s = -1.0;
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
                s = 1.0;
            }
            if ta > t0 {
                t0 = ta;
                axis = a;
                sign = s;
            }
            t1 = t1.min(tb);
        }
        if !ok || t0 > t1 || t0 <= 0.0 {
            continue;
        }
        if best.is_some_and(|bb| bb.0 <= t0) {
            continue;
        }
        let p = [o[0] + t0 * d[0], o[1] + t0 * d[1], o[2] + t0 * d[2]];
        let mut n = [0.0; 3];
        n[axis] = sign;
        let (u, v) = match axis {
            0 => (p[1], p[2]),
            1 => (p[0], p[2]),
            _ => (p[0], p[1]),
        };
        best = Some((t0, n, texture(u + 17.0 * bi as f64, v - 5.0 * axis as f64, cfg.seed + 3)));
    }
    best
}

/// 장면 만들기.
pub fn make_scene(cfg: &SynthConfig) -> SynthScene {
    let (w, h, f) = (cfg.width, cfg.height, cfg.focal);
    let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
    let mut views = Vec::new();
    let mut depths = Vec::new();
    let mut normals = Vec::new();
    let (gx, gy) = cfg.grid;
    for j in 0..gy {
        for i in 0..gx {
            let c = [(i as f64 - (gx as f64 - 1.0) / 2.0) * cfg.spacing, (j as f64 - (gy as f64 - 1.0) / 2.0) * cfg.spacing, cfg.altitude];
            // 장면 중심 쪽으로 살짝 기울인 시선.
            let target = [c[0] * 0.5, c[1] * 0.5, 0.0];
            let zc = unit(sub(target, c));
            let xc = unit(cross(zc, [0.0, 1.0, 0.0]));
            let yc = cross(zc, xc);
            let r = [xc, yc, zc];
            let t = [-dot(xc, c), -dot(yc, c), -dot(zc, c)];
            let mut img = ImageBuffer::new(w, h, 3);
            let mut dm = vec![0.0f32; w * h];
            let mut nm = vec![[0.0f32; 3]; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0;
                    for (sx, sy) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25), (0.0, 0.0)] {
                        let rc = [(x as f64 + sx - cx) / f, (y as f64 + sy - cy) / f, 1.0];
                        let dw = unit([0, 1, 2].map(|k| xc[k] * rc[0] + yc[k] * rc[1] + zc[k] * rc[2]));
                        let hit = cast(c, dw, cfg);
                        let val = hit.map_or(0.0, |hh| hh.2);
                        if sx == 0.0 && sy == 0.0 {
                            if let Some((tt, nw, _)) = hit {
                                let p = [c[0] + tt * dw[0], c[1] + tt * dw[1], c[2] + tt * dw[2]];
                                dm[y * w + x] = (dot(zc, p) + t[2]) as f32;
                                let ncam = [dot(xc, nw), dot(yc, nw), dot(zc, nw)];
                                nm[y * w + x] = [ncam[0] as f32, ncam[1] as f32, ncam[2] as f32];
                            }
                        } else {
                            acc += val;
                        }
                    }
                    let v = (acc / 4.0 * 230.0 + 15.0).round().clamp(0.0, 255.0) as u8;
                    let p = (y * w + x) * 3;
                    img.data[p] = v;
                    img.data[p + 1] = v;
                    img.data[p + 2] = (v as f64 * 0.8) as u8;
                }
            }
            let color = Arc::new(img);
            views.push(DenseView {
                image_id: (views.len() + 1) as u32,
                name: format!("synth_{j}_{i}.png"),
                width: w,
                height: h,
                k: [f, f, cx, cy],
                r,
                t,
                gray: Arc::new(gray_of(&color)),
                color,
            });
            depths.push(dm);
            normals.push(nm);
        }
    }
    // 희소점: 첫 뷰들의 참 깊이에서 고른 점, 다른 뷰에서 참 깊이와 맞으면 관측으로 넣는다.
    let mut points = Vec::new();
    let mut s = cfg.seed ^ 0xabc;
    let mut rnd = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        (crate::math::mix64(s) >> 11) as f64 / (1u64 << 53) as f64
    };
    let nv = views.len();
    for k in 0..cfg.num_points {
        let v = k % nv;
        let (x, y) = ((rnd() * (w - 1) as f64).round() as usize, (rnd() * (h - 1) as f64).round() as usize);
        let d = depths[v][y * w + x] as f64;
        if d <= 0.0 {
            continue;
        }
        let vw = &views[v];
        let p = vw.to_world(&[(x as f64 - cx) / f * d, (y as f64 - cy) / f * d, d]);
        let mut obs = Vec::new();
        for (o, ov) in views.iter().enumerate() {
            let c = ov.to_cam(&p);
            if c[2] <= 0.0 {
                continue;
            }
            let (u, vv, z) = ov.project_cam(&c);
            let (ui, vi) = (u.round(), vv.round());
            if ui < 0.0 || vi < 0.0 || ui >= w as f64 || vi >= h as f64 {
                continue;
            }
            let dt = depths[o][vi as usize * w + ui as usize] as f64;
            if (dt - z).abs() < 0.01 * z {
                obs.push(o as u32);
            }
        }
        if obs.len() >= 2 {
            points.push(ScenePoint { xyz: p, views: obs });
        }
    }
    SynthScene { scene: DenseScene { views, points }, depth: depths, normal: normals, config: cfg.clone() }
}

impl SynthScene {
    /// 세계 점에서 가장 가까운 장면 표면(바닥, 상자 면)까지 거리.
    pub fn surface_distance(&self, p: [f64; 3]) -> f64 {
        let mut best = p[2].abs();
        for b in &self.config.boxes {
            // 상자 표면까지 거리(안쪽이면 가장 가까운 면).
            let inside = (0..3).all(|a| p[a] >= b.min[a] && p[a] <= b.max[a]);
            let dd = if inside {
                (0..3).map(|a| (p[a] - b.min[a]).min(b.max[a] - p[a])).fold(f64::INFINITY, f64::min)
            } else {
                let q: Vec<f64> = (0..3).map(|a| (b.min[a] - p[a]).max(0.0).max(p[a] - b.max[a])).collect();
                (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
            };
            best = best.min(dd);
        }
        best
    }
}
