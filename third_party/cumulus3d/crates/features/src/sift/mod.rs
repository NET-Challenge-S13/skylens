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

//! SIFT. Heavy computation sits behind [`SiftEngine`].

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

/// Maximum feature count limiting mode.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum FeatureSelection {
    /// Level-wise rule: whole levels from the coarsest, including the level that crosses the cap. Default.
    #[default]
    CompatLevels,
    /// Improvement: coarse levels first + within a level descending |DoG response|, exactly K features.
    TopK,
    /// Improvement: split the image into a `cells x cells` grid and pick alternately per cell in |response| order,
    /// exactly K features (spatially uniform).
    SpatialGrid {
        /// Number of grid cells per axis.
        cells: u32,
    },
}

/// SIFT options. Defaults = pipeline settings + level-wise selection behavior.
#[derive(Clone, Debug, PartialEq)]
pub struct SiftOptions {
    /// Maximum feature count (0 = unlimited).
    pub max_num_features: usize,
    /// -1 (2x upsample) or 0.
    pub first_octave: i32,
    /// None = automatic from image size (floor(log2 min) - 3). Some(n) = fixed.
    pub num_octaves: Option<usize>,
    /// Number of detection levels per octave S.
    pub octave_resolution: usize,
    /// DoG extremum threshold (contrast).
    pub peak_threshold: f32,
    /// Edge suppression threshold (principal curvature ratio).
    pub edge_threshold: f32,
    /// Maximum number of orientations per keypoint.
    pub max_num_orientations: usize,
    /// If true, θ = 0 without orientation assignment.
    pub upright: bool,
    /// Descriptor normalization mode.
    pub normalization: DescriptorNormalization,
    /// Round the input width down to a multiple of 4 (default behavior). If disabled, the right columns are not dropped (improvement).
    pub truncate_width_to_4: bool,
    /// Number of sub-pixel refinement iterations. 1 = once (no shift, default). 5 = iterate while shifting pixels (improvement).
    pub refinement_iterations: usize,
    /// Discard if the Hessian is singular (improvement). Default false = pass with δ=0.
    pub reject_singular_refinement: bool,
    /// Linear interpolation between adjacent orientation histogram bins (improvement).
    pub orientation_bin_interpolation: bool,
    /// Maximum feature count limiting mode.
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

/// One feature (in input-image pixel units, top-left pixel center = (0.5, 0.5)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiftFeature {
    /// x coordinate (pixels).
    pub x: f32,
    /// y coordinate (pixels).
    pub y: f32,
    /// Scale σ (in input-image pixel units).
    pub scale: f32,
    /// Output orientation = (2π - θ_int) mod 2π.
    pub orientation: f32,
    /// Detection octave index.
    pub octave: i32,
    /// Detection level j within the octave (0..S).
    pub level: u32,
    /// Refined DoG value.
    pub response: f32,
}

impl SiftFeature {
    /// Affine keypoint.
    pub fn keypoint(&self) -> Keypoint {
        Keypoint::from_scale_orientation(self.x, self.y, self.scale, self.orientation)
    }
}

/// Extraction result. `features[i]` <-> `descriptors.row(i)`.
#[derive(Clone, Debug, Default)]
pub struct SiftOutput {
    /// Feature list.
    pub features: Vec<SiftFeature>,
    /// 128-dimensional uint8 descriptors (row i <-> feature i).
    pub descriptors: Descriptors,
}

impl SiftOutput {
    /// Converts the features to affine keypoints.
    pub fn keypoints(&self) -> Vec<Keypoint> {
        self.features.iter().map(|f| f.keypoint()).collect()
    }
    /// Number of features.
    pub fn len(&self) -> usize {
        self.features.len()
    }
    /// True if there are no features.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// SIFT compute backend. A trait so that other implementations such as CUDA can be plugged in.
pub trait SiftEngine: Send + Sync {
    /// Backend name (for logging).
    fn name(&self) -> &str;
    /// Extracts features from a grayscale image. Coordinates are in input-image pixel units.
    fn extract(&self, image: &GrayImage, options: &SiftOptions) -> Result<SiftOutput>;
}

/// CPU (rayon) backend. Reuses pyramid buffers across images.
#[derive(Default)]
pub struct CpuSift {
    pool: BufferPool,
}

impl CpuSift {
    /// Creates a CPU backend with an empty buffer pool.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result of one level (including orientations).
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
        // Input normalization /255 (+ width truncation).
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

        // From coarse octaves and high levels first. Under the level-wise rule, levels that would be cut
        // skip detection entirely (same result).
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
                    // The orientation window is small, so on-the-fly gradients are advantageous.
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

        // Output order: octave ascending, level ascending, detection order within a level.
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

/// Limits the maximum feature count. `levels` is in coarse -> fine order.
fn select(levels: &mut [LevelFeats], mode: FeatureSelection, k: usize, pyr: &[pyramid::Octave]) {
    let total: usize = levels.iter().map(|l| l.items.len()).sum();
    if total <= k {
        return;
    }
    match mode {
        FeatureSelection::CompatLevels => {
            // From the finest level, remove whole levels while "total - that level > K".
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
            // Input image size ~ first octave size x 2^o_min.
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
