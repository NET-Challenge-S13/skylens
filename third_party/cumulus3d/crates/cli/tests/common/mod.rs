/*
 * mod.rs
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

//! 테스트 공용 합성 데이터: 레이 캐스팅으로 그린 무늬 바닥 + 건물 상자, 드론 3대 × NPOS 위치, gps_ref.txt.
#![allow(dead_code)]

use cumulus3d_align::EnuFrame;
use image::{Rgb, RgbImage};
use std::io::Write;
use std::path::Path;

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
fn norm(a: V3) -> V3 {
    let n = dot(a, a).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

fn hash2(ix: i64, iy: i64, seed: u64) -> f64 {
    let mut z = (ix as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ (iy as u64).wrapping_mul(0xc2b2ae3d27d4eb4f) ^ seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
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

fn texture(u: f64, v: f64, seed: u64) -> f64 {
    let (mut s, mut amp, mut f) = (0.0, 0.5, 1.0 / 1.2);
    for o in 0..5 {
        s += amp * value_noise(u * f, v * f, seed + o);
        amp *= 0.55;
        f *= 2.2;
    }
    s.clamp(0.0, 1.0)
}

struct Boxo {
    min: V3,
    max: V3,
}

/// 바닥 위 건물 모양 상자들(높이 다양): 평면 장면의 초점-거리 모호성을 없앤다.
fn boxes() -> Vec<Boxo> {
    let mut v = Vec::new();
    for i in 0..8 {
        for j in 0..6 {
            let h = hash2(i, j, 5);
            if h < 0.35 {
                continue;
            }
            let cx = -4.0 + i as f64 * 3.6 + hash2(i, j, 6) * 1.2;
            let cy = -11.0 + j as f64 * 4.2 + hash2(i, j, 7) * 1.2;
            let (sx, sy) = (0.8 + hash2(i, j, 8) * 1.2, 0.8 + hash2(i, j, 9) * 1.5);
            v.push(Boxo { min: [cx - sx, cy - sy, 0.0], max: [cx + sx, cy + sy, 0.8 + 4.0 * hash2(i, j, 10)] });
        }
    }
    v
}

/// 반직선 교차 → 색.
fn shade(o: V3, d: V3, bx: &[Boxo]) -> [f64; 3] {
    let mut best = (f64::MAX, 0usize, 2usize);
    if d[2].abs() > 1e-12 {
        let t = -o[2] / d[2];
        if t > 0.0 {
            best = (t, 0, 2);
        }
    }
    for (bi, b) in bx.iter().enumerate() {
        let (mut tmin, mut tmax, mut axis) = (f64::MIN, f64::MAX, 0);
        let mut ok = true;
        for a in 0..3 {
            if d[a].abs() < 1e-12 {
                if o[a] < b.min[a] || o[a] > b.max[a] {
                    ok = false;
                }
                continue;
            }
            let t1 = (b.min[a] - o[a]) / d[a];
            let t2 = (b.max[a] - o[a]) / d[a];
            let (tn, tf) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            if tn > tmin {
                tmin = tn;
                axis = a;
            }
            tmax = tmax.min(tf);
        }
        if ok && tmin <= tmax && tmin > 0.0 && tmin < best.0 {
            best = (tmin, bi + 1, axis);
        }
    }
    if best.0 == f64::MAX {
        return [0.5, 0.6, 0.8];
    }
    let p = [o[0] + d[0] * best.0, o[1] + d[1] * best.0, o[2] + d[2] * best.0];
    let (u, v) = match best.2 {
        2 => (p[0], p[1]),
        0 => (p[1], p[2]),
        _ => (p[0], p[2]),
    };
    let t = texture(u, v, 17 + best.1 as u64 * 101);
    let g = texture(u * 0.7 + 3.1, v * 0.7, 99 + best.1 as u64);
    let base = 0.1 + 0.85 * t;
    [base, base * (0.7 + 0.3 * g), base * (0.5 + 0.5 * (1.0 - g))]
}

const W: u32 = 320;
const H: u32 = 240;
// 기본 초점 추정(1.2 × 최대 변)과 같게 해 EXIF 없이도 내부값이 맞게 한다.
const F: f64 = 1.2 * 320.0;

/// 카메라 중심 c 에서 target 을 보는 영상.
fn render(c: V3, target: V3, bx: &[Boxo]) -> RgbImage {
    let z = norm(sub(target, c));
    let x = norm(cross(z, [0.0, 0.0, 1.0]));
    let y = cross(z, x);
    let mut img = RgbImage::new(W, H);
    for j in 0..H {
        for i in 0..W {
            let mut acc = [0.0; 3];
            for (sx, sy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let u = (i as f64 + sx - W as f64 / 2.0) / F;
                let v = (j as f64 + sy - H as f64 / 2.0) / F;
                let d = norm([x[0] * u + y[0] * v + z[0], x[1] * u + y[1] * v + z[1], x[2] * u + y[2] * v + z[2]]);
                let s = shade(c, d, bx);
                for k in 0..3 {
                    acc[k] += s[k] * 0.25;
                }
            }
            img.put_pixel(i, j, Rgb([(acc[0] * 255.0) as u8, (acc[1] * 255.0) as u8, (acc[2] * 255.0) as u8]));
        }
    }
    img
}

pub const NPOS: usize = 8;
const STEP: f64 = 1.5;

/// images/camF|camR|camL/camX_NNNN.jpg (프레임 번호 = 위치 × 3, 실제 촬영 데이터와 같은 배치) + gps_ref.txt.
pub fn make_dataset(root: &Path) {
    let bx = boxes();
    let enu = EnuFrame::new(36.0, 127.0, 30.0);
    let mut gps = String::new();
    for cam in ["camF", "camR", "camL"] {
        std::fs::create_dir_all(root.join("images").join(cam)).unwrap();
    }
    for p in 0..NPOS {
        let x = p as f64 * STEP;
        let specs = [
            ("camF", [x, 0.0, 15.0], [x + 4.0, 0.0, 0.0]),
            ("camR", [x, -3.0, 16.5], [x + 4.0, 1.5, 0.0]),
            ("camL", [x, 3.0, 14.0], [x + 1.0, -1.0, 0.0]),
        ];
        for (cam, c, t) in specs {
            let name = format!("{cam}/{cam}_{:04}.jpg", p * 3);
            let img = render(c, t, &bx);
            let f = std::fs::File::create(root.join("images").join(&name)).unwrap();
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(f), 95);
            enc.encode_image(&img).unwrap();
            let (lat, lon, alt) = enu.enu_to_lla(&cumulus3d_core::Vec3::new(c[0], c[1], c[2]));
            gps.push_str(&format!("{name} {lat:.10} {lon:.10} {alt:.4}\n"));
        }
    }
    // 실제 촬영 데이터처럼 camF, camR, camL 순서로 정렬된 파일.
    let mut lines: Vec<&str> = gps.lines().collect();
    lines.sort_by_key(|l| {
        let n = l.split_whitespace().next().unwrap();
        (["camF", "camR", "camL"].iter().position(|c| n.starts_with(c)).unwrap(), n.to_string())
    });
    let mut f = std::fs::File::create(root.join("gps_ref.txt")).unwrap();
    for l in lines {
        writeln!(f, "{l}").unwrap();
    }
}
