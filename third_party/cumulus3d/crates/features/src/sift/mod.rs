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
 * Third-party notices: parts of the algorithms, default parameters and data
 * formats in this file follow other open-source projects. Their copyright
 * notices and licenses are reproduced in THIRD_PARTY_NOTICES.md.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! SIFT. 무거운 계산은 [`SiftEngine`] 뒤에 둔다.

pub mod detect;
pub mod orient;
pub mod pyramid;

use crate::gray::GrayImage;
use cumulus3d_core::{Descriptors, Error, Keypoint, Result};
use detect::{Candidate, DetectParams};
pub use orient::DescriptorNormalization;
use orient::{Gradient, OrientParams};
use pyramid::{BufferPool, ScaleSpace};
use rayon::prelude::*;

/// 최대 특징 수 제한 방식.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum FeatureSelection {
    /// 레벨 단위 규칙: 거친 레벨부터 통째로, 상한을 넘긴 레벨까지 포함. 기본.
    #[default]
    CompatLevels,
    /// 개선: 거친 레벨 우선 + 레벨 내 |DoG 응답| 내림차순으로 정확히 K개.
    TopK,
    /// 개선: 영상을 `cells × cells` 격자로 나눠 칸마다 |응답| 순으로 번갈아 뽑아 정확히 K개(공간 균등).
    SpatialGrid {
        /// 한 축의 격자 칸 수.
        cells: u32,
    },
}

/// SIFT 옵션. 기본값 = 파이프라인 설정 + 레벨 단위 선택 동작.
#[derive(Clone, Debug, PartialEq)]
pub struct SiftOptions {
    /// 최대 특징 수(0 = 무제한).
    pub max_num_features: usize,
    /// −1(2배 업샘플) 또는 0.
    pub first_octave: i32,
    /// None = 영상 크기로 자동(floor(log2 min) − 3). Some(n) = 고정.
    pub num_octaves: Option<usize>,
    /// 옥타브당 검출 레벨 수 S.
    pub octave_resolution: usize,
    /// DoG 극값 임계값(대비).
    pub peak_threshold: f32,
    /// 엣지 억제 임계값(주곡률 비).
    pub edge_threshold: f32,
    /// 키포인트당 최대 방향 수.
    pub max_num_orientations: usize,
    /// 참이면 방향 할당 없이 θ = 0.
    pub upright: bool,
    /// 기술자 정규화 방식.
    pub normalization: DescriptorNormalization,
    /// 입력 너비를 4의 배수로 내림(기본 동작). 끄면 오른쪽 열을 버리지 않음(개선).
    pub truncate_width_to_4: bool,
    /// 부분화소 보정 반복 수. 1 = 1회(이동 없음, 기본). 5 = 화소 이동하며 반복(개선).
    pub refinement_iterations: usize,
    /// 특이 헤시안이면 버림(개선). 기본 false = δ=0 으로 통과.
    pub reject_singular_refinement: bool,
    /// 방향 히스토그램 인접 빈 선형 보간(개선).
    pub orientation_bin_interpolation: bool,
    /// 최대 특징 수 제한 방식.
    pub selection: FeatureSelection,
}

impl Default for SiftOptions {
    fn default() -> Self {
        Self {
            max_num_features: 8192,
            first_octave: -1,
            num_octaves: None,
            octave_resolution: 3,
            peak_threshold: 0.006667,
            edge_threshold: 10.0,
            max_num_orientations: 2,
            upright: false,
            normalization: DescriptorNormalization::L1Root,
            truncate_width_to_4: true,
            refinement_iterations: 1,
            reject_singular_refinement: false,
            orientation_bin_interpolation: false,
            selection: FeatureSelection::CompatLevels,
        }
    }
}

/// 특징 하나(입력 영상 화소 단위, 좌상단 화소 중심 = (0.5, 0.5)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiftFeature {
    /// x 좌표(화소).
    pub x: f32,
    /// y 좌표(화소).
    pub y: f32,
    /// 스케일 σ(입력 영상 화소 단위).
    pub scale: f32,
    /// 출력 방향 = (2π − θ_int) mod 2π.
    pub orientation: f32,
    /// 검출 옥타브 번호.
    pub octave: i32,
    /// 옥타브 내 검출 레벨 j (0..S).
    pub level: u32,
    /// 보정된 DoG 값.
    pub response: f32,
}

impl SiftFeature {
    /// 아핀 키포인트.
    pub fn keypoint(&self) -> Keypoint {
        Keypoint::from_scale_orientation(self.x, self.y, self.scale, self.orientation)
    }
}

/// 추출 결과. `features[i]` ↔ `descriptors.row(i)`.
#[derive(Clone, Debug, Default)]
pub struct SiftOutput {
    /// 특징 목록.
    pub features: Vec<SiftFeature>,
    /// 128차원 uint8 기술자(행 i ↔ 특징 i).
    pub descriptors: Descriptors,
}

impl SiftOutput {
    /// 특징들을 아핀 키포인트로 변환.
    pub fn keypoints(&self) -> Vec<Keypoint> {
        self.features.iter().map(|f| f.keypoint()).collect()
    }
    /// 특징 수.
    pub fn len(&self) -> usize {
        self.features.len()
    }
    /// 특징이 없으면 참.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// SIFT 계산 백엔드. CUDA 등 다른 구현을 끼울 수 있게 trait 로 둔다.
pub trait SiftEngine: Send + Sync {
    /// 백엔드 이름(로그용).
    fn name(&self) -> &str;
    /// 회색 영상에서 특징 추출. 좌표는 입력 영상 화소 단위.
    fn extract(&self, image: &GrayImage, options: &SiftOptions) -> Result<SiftOutput>;
}

/// CPU(rayon) 백엔드. 피라미드 버퍼를 영상 간에 재사용한다.
#[derive(Default)]
pub struct CpuSift {
    pool: BufferPool,
}

impl CpuSift {
    /// 빈 버퍼 풀로 CPU 백엔드를 만든다.
    pub fn new() -> Self {
        Self::default()
    }
}

/// 레벨 하나의 결과(방향 포함).
struct LevelFeats {
    oi: usize,
    j: usize,
    items: Vec<(Candidate, u16)>,
}

impl SiftEngine for CpuSift {
    fn name(&self) -> &str {
        "cpu"
    }

    fn extract(&self, image: &GrayImage, opts: &SiftOptions) -> Result<SiftOutput> {
        if opts.octave_resolution == 0 {
            return Err(Error::InvalidArgument("octave_resolution 은 1 이상".into()));
        }
        if opts.first_octave != -1 && opts.first_octave != 0 {
            return Err(Error::Unsupported(format!("first_octave {} (−1, 0 만 지원)", opts.first_octave)));
        }
        let w = if opts.truncate_width_to_4 { image.width - image.width % 4 } else { image.width };
        let h = image.height;
        if w < 8 || h < 8 {
            return Ok(SiftOutput::default());
        }
        // 입력 정규화 /255 (+ 너비 절단).
        let mut base = self.pool.take(w * h);
        base.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let src = &image.data[y * image.width..y * image.width + w];
            for (o, &v) in row.iter_mut().zip(src) {
                *o = v as f32 / 255.0;
            }
        });
        let o_min = opts.first_octave;
        let (w0, h0, base) = if o_min == -1 {
            let mut up = self.pool.take(4 * w * h);
            pyramid::upsample2(&base, w, h, &mut up);
            self.pool.give(base);
            (2 * w, 2 * h, up)
        } else {
            (w, h, base)
        };
        let ss = ScaleSpace::new(opts.octave_resolution);
        let n_oct = opts.num_octaves.unwrap_or_else(|| pyramid::auto_num_octaves(w0, h0)).max(1);
        let pyr = pyramid::build_pyramid(base, w0, h0, o_min, n_oct, ss, 0.5, &self.pool);

        let s = ss.s;
        let k_max = if opts.max_num_features == 0 { usize::MAX } else { opts.max_num_features };
        let dp = DetectParams {
            peak_threshold: opts.peak_threshold,
            edge_threshold: opts.edge_threshold,
            refinement_iterations: opts.refinement_iterations,
            reject_singular: opts.reject_singular_refinement,
            sigma0: ss.sigma0,
            k: ss.k,
        };
        let op = OrientParams { max_num_orientations: opts.max_num_orientations, bin_interpolation: opts.orientation_bin_interpolation };

        // 거친 옥타브·높은 레벨부터. 레벨 단위 규칙상 잘릴 레벨은 검출 자체를 생략(결과 동일).
        let mut levels: Vec<LevelFeats> = Vec::new();
        let (mut cum_det, mut cum_or) = (0usize, 0usize);
        'outer: for oi in (0..pyr.len()).rev() {
            let oct = &pyr[oi];
            let (ow, oh) = (oct.w, oct.h);
            let mut dogs: Vec<Vec<f32>> = Vec::new();
            for j in (0..s).rev() {
                let stop = match opts.selection {
                    FeatureSelection::CompatLevels => cum_det > k_max,
                    FeatureSelection::TopK => cum_or >= k_max,
                    FeatureSelection::SpatialGrid { .. } => false,
                };
                if stop {
                    for d in dogs {
                        self.pool.give(d);
                    }
                    break 'outer;
                }
                if dogs.is_empty() {
                    for d in 0..s + 2 {
                        let mut buf = self.pool.take(ow * oh);
                        detect::dog(&oct.gauss[d + 1], &oct.gauss[d], &mut buf);
                        dogs.push(buf);
                    }
                }
                let cands = detect::detect_level(&dogs[j], &dogs[j + 1], &dogs[j + 2], ow, oh, j, &dp);
                cum_det += cands.len();
                let items: Vec<(Candidate, u16)> = if opts.upright {
                    cands.into_iter().map(|c| (c, 0u16)).collect()
                } else {
                    // 방향 창은 작아서 즉석 기울기가 유리.
                    let grad = Gradient::lazy(oct.level(j as i32), ow, oh);
                    cands
                        .par_iter()
                        .flat_map_iter(|c| orient::orientations(&grad, c.x, c.y, c.sigma, &op).into_iter().map(move |q| (*c, q)))
                        .collect()
                };
                cum_or += items.len();
                levels.push(LevelFeats { oi, j, items });
            }
            for d in dogs {
                self.pool.give(d);
            }
        }

        select(&mut levels, opts.selection, k_max, &pyr);

        // 출력 순서: 옥타브 오름차순, 레벨 오름차순, 레벨 내 검출 순서.
        levels.sort_by_key(|l| (l.oi, l.j));
        let total: usize = levels.iter().map(|l| l.items.len()).sum();
        let mut features = Vec::with_capacity(total);
        let mut desc = Descriptors::with_capacity(total);
        for lv in &levels {
            if lv.items.is_empty() {
                continue;
            }
            let oct = &pyr[lv.oi];
            let g = oct.level(lv.j as i32);
            let sig = ss.sigma(lv.j as f32);
            let work = lv.items.len() as f64 * 225.0 * (sig as f64).powi(2);
            let grad = if work > (oct.w * oct.h) as f64 { Gradient::precomputed(g, oct.w, oct.h) } else { Gradient::lazy(g, oct.w, oct.h) };
            let ds: Vec<[u8; 128]> = lv
                .items
                .par_iter()
                .map(|(c, q)| orient::descriptor(&grad, c.x, c.y, c.sigma, orient::dequantize_angle(*q), opts.normalization))
                .collect();
            let scale_o = 2f32.powi(oct.o);
            for ((c, q), d) in lv.items.iter().zip(&ds) {
                let th = orient::dequantize_angle(*q);
                let ori = (std::f32::consts::TAU - th).rem_euclid(std::f32::consts::TAU);
                features.push(SiftFeature {
                    x: scale_o * (c.x - 0.5) + 0.5,
                    y: scale_o * (c.y - 0.5) + 0.5,
                    scale: scale_o * c.sigma,
                    orientation: ori,
                    octave: oct.o,
                    level: lv.j as u32,
                    response: c.response,
                });
                desc.push(d);
            }
        }
        for oct in pyr {
            for g in oct.gauss {
                self.pool.give(g);
            }
        }
        Ok(SiftOutput { features, descriptors: desc })
    }
}

/// 최대 특징 수 제한. `levels` 는 거친 → 미세 순서.
fn select(levels: &mut [LevelFeats], mode: FeatureSelection, k: usize, pyr: &[pyramid::Octave]) {
    let total: usize = levels.iter().map(|l| l.items.len()).sum();
    if total <= k {
        return;
    }
    match mode {
        FeatureSelection::CompatLevels => {
            // 가장 미세한 레벨부터 "전체 − 그 레벨 > K" 인 동안 통째 제거.
            let mut total = total;
            for lv in levels.iter_mut().rev() {
                if total - lv.items.len() > k {
                    total -= lv.items.len();
                    lv.items.clear();
                } else {
                    break;
                }
            }
        }
        FeatureSelection::TopK => {
            let mut remaining = k;
            for lv in levels.iter_mut() {
                if lv.items.len() <= remaining {
                    remaining -= lv.items.len();
                    continue;
                }
                let mut idx: Vec<usize> = (0..lv.items.len()).collect();
                idx.sort_by(|&a, &b| lv.items[b].0.response.abs().total_cmp(&lv.items[a].0.response.abs()).then(a.cmp(&b)));
                let mut keep = vec![false; lv.items.len()];
                for &i in idx.iter().take(remaining) {
                    keep[i] = true;
                }
                let mut it = keep.iter();
                lv.items.retain(|_| *it.next().unwrap_or(&false));
                remaining = 0;
            }
        }
        FeatureSelection::SpatialGrid { cells } => {
            let cells = cells.max(1) as usize;
            // 입력 영상 크기 ≈ 첫 옥타브 크기 × 2^o_min.
            let o0 = &pyr[0];
            let sc = 2f32.powi(o0.o);
            let (iw, ih) = (o0.w as f32 * sc, o0.h as f32 * sc);
            let mut buckets: Vec<Vec<(f32, usize, usize)>> = vec![Vec::new(); cells * cells];
            for (li, lv) in levels.iter().enumerate() {
                let so = 2f32.powi(pyr[lv.oi].o);
                for (ii, (c, _)) in lv.items.iter().enumerate() {
                    let x = so * (c.x - 0.5) + 0.5;
                    let y = so * (c.y - 0.5) + 0.5;
                    let cx = ((x / iw * cells as f32) as usize).min(cells - 1);
                    let cy = ((y / ih * cells as f32) as usize).min(cells - 1);
                    buckets[cy * cells + cx].push((c.response.abs(), li, ii));
                }
            }
            for b in &mut buckets {
                b.sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
            }
            let mut keep: Vec<Vec<bool>> = levels.iter().map(|l| vec![false; l.items.len()]).collect();
            let mut taken = 0;
            let mut round = 0;
            while taken < k {
                let mut any = false;
                for b in &buckets {
                    if let Some(&(_, li, ii)) = b.get(round) {
                        any = true;
                        if taken < k {
                            keep[li][ii] = true;
                            taken += 1;
                        }
                    }
                }
                if !any {
                    break;
                }
                round += 1;
            }
            for (lv, kp) in levels.iter_mut().zip(keep) {
                let mut it = kp.into_iter();
                lv.items.retain(|_| it.next().unwrap_or(false));
            }
        }
    }
}
