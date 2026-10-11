//! Rust twin of `utils/collate.py`.
//!
//! Batch assembly and CenterNet target encoding: turns a list of dataset samples (the contract
//! in `datasets/base.rs`) into the tensor batch the SkyLens UNet expects.
//!
//! | key               | shape / dtype                     | notes                                    |
//! |-------------------|-----------------------------------|------------------------------------------|
//! | `pixel_values`    | (B, C, H, W) float32              | C = 4, or 5 with `validity_channel`      |
//! | `modality_mask`   | (B, 2) bool                       | `[rgb_present, thermal_present]`         |
//! | `danger_labels`   | (B, H, W) int64                   | 255 = ignore; `None` if no sample has one|
//! | `person_heatmap`  | (B, 1, H/s, W/s) float32          | CenterNet gaussian splat                 |
//! | `person_wh`       | (B, 2, H/s, W/s) float32          | (w, h) in output-stride units            |
//! | `person_reg_mask` | (B, 1, H/s, W/s) float32          | 1 at valid centres                       |
//! | `person_offset`   | (B, 2, H/s, W/s) float32          | sub-cell centre `c/s - floor(c/s)`       |
//!
//! The four `person_*` fields are `None` together when no sample in the batch carries boxes.
//! That is how per-head training (README §6.3) works: the loop checks presence to decide which
//! loss to compute.
//!
//! Channel layout (README §2.1/§2.3): `0:3` RGB in `[0, 1]`; `3` thermal min-max normalised
//! into `[0.1, 1.0]` with exactly `0.0` meaning "absent"; `4` (optional) validity mask.
//!
//! Differences from Python: the RNG is [`SimpleRng`] (SplitMix64) rather than numpy's
//! generator, so a given seed does not reproduce Python's dropout draws.

use tch::{Kind, Tensor};

use crate::datasets::base::{
    to_vec_f32, to_vec_u8, DatasetError, Result, Sample, SimpleRng, IGNORE_INDEX,
};

/// Value reserved in the thermal channel to mean "this modality is absent".
pub const THERMAL_ABSENT: f32 = 0.0;
/// Lower bound of the live thermal range (README §2.3).
pub const THERMAL_MIN: f32 = 0.1;

/// How the real-valued gaussian radius becomes an integer cell count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiusRounding {
    /// Truncate toward zero (the v4 behaviour). Radius below 1.0 gives a one-hot target.
    Trunc,
    /// Round to nearest (ties to even, like Python's `round`).
    Round,
}

impl RadiusRounding {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "trunc" => Ok(RadiusRounding::Trunc),
            "round" => Ok(RadiusRounding::Round),
            other => Err(DatasetError::InvalidSample(format!(
                "radius_rounding must be one of [\"round\", \"trunc\"]; got {other:?}"
            ))),
        }
    }
}

// --------------------------------------------------------------------------- //
// CenterNet target encoding
// --------------------------------------------------------------------------- //

/// CornerNet/CenterNet gaussian radius for a box of `(height, width)`: the smallest root of the
/// three "IoU >= min_overlap" cases.
pub fn gaussian_radius(det_size: (f64, f64), min_overlap: f64) -> f64 {
    let (height, width) = det_size;

    let (a1, b1) = (1.0, height + width);
    let c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap);
    let sq1 = (b1 * b1 - 4.0 * a1 * c1).max(0.0).sqrt();
    let r1 = (b1 - sq1) / (2.0 * a1);

    let (a2, b2) = (4.0, 2.0 * (height + width));
    let c2 = (1.0 - min_overlap) * width * height;
    let sq2 = (b2 * b2 - 4.0 * a2 * c2).max(0.0).sqrt();
    let r2 = (b2 - sq2) / (2.0 * a2);

    let a3 = 4.0 * min_overlap;
    let b3 = -2.0 * min_overlap * (height + width);
    let c3 = (min_overlap - 1.0) * width * height;
    let sq3 = (b3 * b3 - 4.0 * a3 * c3).max(0.0).sqrt();
    let r3 = (-b3 + sq3) / (2.0 * a3);

    r1.min(r2).min(r3)
}

/// Unnormalised 2-D gaussian kernel (row-major `shape.0 x shape.1`) with peak 1.0 at the centre.
pub fn gaussian2d(shape: (usize, usize), sigma: f64) -> Vec<f64> {
    let m = (shape.0 as f64 - 1.0) / 2.0;
    let n = (shape.1 as f64 - 1.0) / 2.0;
    let mut h = Vec::with_capacity(shape.0 * shape.1);
    for yi in 0..shape.0 {
        let y = yi as f64 - m;
        for xi in 0..shape.1 {
            let x = xi as f64 - n;
            h.push((-(x * x + y * y) / (2.0 * sigma * sigma)).exp());
        }
    }
    let max = h.iter().copied().fold(f64::MIN, f64::max);
    for v in h.iter_mut() {
        if *v < f64::EPSILON * max {
            *v = 0.0;
        }
    }
    h
}

/// Splats a gaussian onto a row-major `(h, w)` heatmap in place, combining with `max` so that
/// nearby people do not erase each other. `center` is `(cx, cy)` in heatmap coordinates.
pub fn draw_gaussian(heatmap: &mut [f32], hw: (usize, usize), center: (i64, i64), radius: i64, k: f32) {
    let radius = radius.max(0);
    let diameter = (2 * radius + 1) as usize;
    let gauss = gaussian2d((diameter, diameter), diameter as f64 / 6.0);

    let (h, w) = (hw.0 as i64, hw.1 as i64);
    let (cx, cy) = center;
    let left = cx.min(radius);
    let right = (w - cx).min(radius + 1);
    let top = cy.min(radius);
    let bottom = (h - cy).min(radius + 1);
    if left + right <= 0 || top + bottom <= 0 {
        return;
    }
    for dy in -top..bottom {
        for dx in -left..right {
            let g = gauss[((radius + dy) as usize) * diameter + (radius + dx) as usize] as f32 * k;
            let idx = ((cy + dy) as usize) * hw.1 + (cx + dx) as usize;
            if g > heatmap[idx] {
                heatmap[idx] = g;
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// collator
// --------------------------------------------------------------------------- //

#[derive(Debug, Clone)]
pub struct CollatorConfig {
    /// Output stride `s` of the point head; targets are rendered at `(H // s, W // s)`.
    pub person_head_stride: usize,
    /// Append a 5th channel that is 1.0 where the thermal plane is real.
    pub validity_channel: bool,
    /// `min_overlap` passed to [`gaussian_radius`]; must be in (0, 1).
    pub min_overlap: f64,
    pub radius_rounding: RadiusRounding,
    /// `(p_drop_thermal, p_drop_rgb)`, symmetric modality dropout (README §2.2). The
    /// recommended training value is `(0.25, 0.25)`, leaving 0.5 for the full 4-channel mode.
    pub modality_dropout: (f64, f64),
    /// Safety cap on boxes encoded per image.
    pub max_objects: usize,
}

impl Default for CollatorConfig {
    fn default() -> Self {
        CollatorConfig {
            person_head_stride: 4,
            validity_channel: false,
            min_overlap: 0.7,
            radius_rounding: RadiusRounding::Trunc,
            modality_dropout: (0.0, 0.0),
            max_objects: 512,
        }
    }
}

/// The batch dict. `None` fields are keys that are absent in Python.
#[derive(Debug)]
pub struct Batch {
    pub pixel_values: Tensor,
    pub modality_mask: Tensor,
    pub danger_labels: Option<Tensor>,
    pub person_heatmap: Option<Tensor>,
    pub person_wh: Option<Tensor>,
    pub person_reg_mask: Option<Tensor>,
    pub person_offset: Option<Tensor>,
}

impl Batch {
    /// Moves every tensor to `device`.
    pub fn to_device(&self, device: tch::Device) -> Batch {
        let mv = |t: &Option<Tensor>| t.as_ref().map(|t| t.to_device(device));
        Batch {
            pixel_values: self.pixel_values.to_device(device),
            modality_mask: self.modality_mask.to_device(device),
            danger_labels: mv(&self.danger_labels),
            person_heatmap: mv(&self.person_heatmap),
            person_wh: mv(&self.person_wh),
            person_reg_mask: mv(&self.person_reg_mask),
            person_offset: mv(&self.person_offset),
        }
    }

    pub fn has_seg_gt(&self) -> bool {
        self.danger_labels.is_some()
    }

    pub fn has_person_gt(&self) -> bool {
        self.person_heatmap.is_some()
    }
}

pub struct SkyLensCollator {
    cfg: CollatorConfig,
    rng: SimpleRng,
}

impl SkyLensCollator {
    pub fn new(cfg: CollatorConfig) -> Result<Self> {
        Self::with_rng(cfg, SimpleRng::from_entropy())
    }

    pub fn with_rng(cfg: CollatorConfig, rng: SimpleRng) -> Result<Self> {
        if cfg.person_head_stride < 1 {
            return Err(DatasetError::InvalidSample("person_head_stride must be >= 1".into()));
        }
        if !(cfg.min_overlap > 0.0 && cfg.min_overlap < 1.0) {
            return Err(DatasetError::InvalidSample(format!(
                "min_overlap must be in (0, 1); got {}",
                cfg.min_overlap
            )));
        }
        Ok(SkyLensCollator { cfg, rng })
    }

    pub fn config(&self) -> &CollatorConfig {
        &self.cfg
    }

    /// Collates samples into the model's batch.
    pub fn collate(&mut self, samples: &[Sample]) -> Result<Batch> {
        if samples.is_empty() {
            return Err(DatasetError::InvalidSample("SkyLensCollator received an empty batch".into()));
        }
        let (h, w) = Self::verify_uniform_size(samples)?;
        let s = self.cfg.person_head_stride;
        let (oh, ow) = ((h / s).max(1), (w / s).max(1));
        let b = samples.len();

        let mut pixels: Vec<f32> = Vec::new();
        let mut modality: Vec<bool> = Vec::with_capacity(b * 2);
        let mut c_out = 0;
        for sample in samples {
            let (px, c, has_rgb, has_thermal) = self.to_pixel_values(sample)?;
            c_out = c;
            pixels.extend_from_slice(&px);
            modality.push(has_rgb);
            modality.push(has_thermal);
        }

        let pixel_values =
            Tensor::from_slice(&pixels).reshape([b as i64, c_out as i64, h as i64, w as i64]);
        let modality_mask = Tensor::from_slice(&modality).reshape([b as i64, 2]);

        // --- segmentation head ---
        let danger_labels = if samples.iter().any(|s| s.danger_mask.is_some()) {
            let mut labels: Vec<i64> = Vec::with_capacity(b * h * w);
            for s_ in samples {
                match &s_.danger_mask {
                    Some(m) => labels.extend(to_vec_u8(m).into_iter().map(i64::from)),
                    None => labels.extend(std::iter::repeat(i64::from(IGNORE_INDEX)).take(h * w)),
                }
            }
            Some(Tensor::from_slice(&labels).reshape([b as i64, h as i64, w as i64]))
        } else {
            None
        };

        // --- point-detection head ---
        let (mut person_heatmap, mut person_wh, mut person_reg_mask, mut person_offset) =
            (None, None, None, None);
        if samples.iter().any(|s| s.person_boxes.is_some()) {
            let plane = oh * ow;
            let mut hm = vec![0f32; b * plane];
            let mut wh = vec![0f32; b * 2 * plane];
            let mut reg = vec![0f32; b * plane];
            let mut off = vec![0f32; b * 2 * plane];
            for (i, s_) in samples.iter().enumerate() {
                let Some(boxes) = &s_.person_boxes else { continue };
                let boxes = to_vec_f32(boxes);
                if boxes.is_empty() {
                    continue;
                }
                self.encode_boxes(
                    &boxes,
                    (oh, ow),
                    &mut hm[i * plane..(i + 1) * plane],
                    &mut wh[i * 2 * plane..(i + 1) * 2 * plane],
                    &mut reg[i * plane..(i + 1) * plane],
                    Some(&mut off[i * 2 * plane..(i + 1) * 2 * plane]),
                );
            }
            let (bi, oh_, ow_) = (b as i64, oh as i64, ow as i64);
            person_heatmap = Some(Tensor::from_slice(&hm).reshape([bi, 1, oh_, ow_]));
            person_wh = Some(Tensor::from_slice(&wh).reshape([bi, 2, oh_, ow_]));
            person_reg_mask = Some(Tensor::from_slice(&reg).reshape([bi, 1, oh_, ow_]));
            person_offset = Some(Tensor::from_slice(&off).reshape([bi, 2, oh_, ow_]));
        }

        Ok(Batch {
            pixel_values,
            modality_mask,
            danger_labels,
            person_heatmap,
            person_wh,
            person_reg_mask,
            person_offset,
        })
    }

    fn verify_uniform_size(samples: &[Sample]) -> Result<(usize, usize)> {
        let shape_of = |s: &Sample| {
            let sz = s.image.size();
            (sz[0] as usize, sz[1] as usize)
        };
        let first = shape_of(&samples[0]);
        if samples.iter().any(|s| shape_of(s) != first) {
            let mut shapes: Vec<_> = samples.iter().map(shape_of).collect();
            shapes.sort();
            shapes.dedup();
            return Err(DatasetError::InvalidSample(format!(
                "SkyLensCollator needs all images in a batch to share (H, W); got {shapes:?}. \
                 Add a resize/crop to the dataset `transforms`."
            )));
        }
        Ok(first)
    }

    /// Builds a flat `(C, H, W)` float32 buffer with the fixed channel layout. Returns
    /// `(buffer, C, has_rgb, has_thermal)` where the flags reflect any modality dropout.
    fn to_pixel_values(&mut self, sample: &Sample) -> Result<(Vec<f32>, usize, bool, bool)> {
        let sz = sample.image.size();
        let (h, w, c) = match sz.len() {
            2 => (sz[0] as usize, sz[1] as usize, 1usize),
            3 => (sz[0] as usize, sz[1] as usize, sz[2] as usize),
            _ => {
                return Err(DatasetError::InvalidSample(format!(
                    "image must be (H, W, C); got shape {sz:?}"
                )))
            }
        };
        let is_u8 = sample.image.kind() == Kind::Uint8;
        let image = to_vec_f32(&sample.image);
        let px = |y: usize, x: usize, ch: usize| image[(y * w + x) * c + ch];

        let (mut has_rgb, mut has_thermal) = (sample.has_rgb, sample.has_thermal);
        let n = h * w;

        let mut rgb = vec![0f32; 3 * n]; // CHW
        let mut thermal = vec![THERMAL_ABSENT; n];

        if has_rgb && c >= 3 {
            let mut max = f32::MIN;
            for y in 0..h {
                for x in 0..w {
                    for ch in 0..3 {
                        let v = px(y, x, ch);
                        max = max.max(v);
                        rgb[ch * n + y * w + x] = v;
                    }
                }
            }
            if is_u8 || max > 1.5 {
                // uint8, or a float image still in 0..255
                for v in rgb.iter_mut() {
                    *v /= 255.0;
                }
            }
        }

        if has_thermal {
            let ch = if c >= 4 { 3 } else { 0 };
            let plane: Vec<f32> = (0..h)
                .flat_map(|y| (0..w).map(move |x| (y, x)))
                .map(|(y, x)| px(y, x, ch))
                .collect();
            thermal = Self::normalize_thermal(&plane, c >= 4);
        }

        // symmetric modality dropout (README §2.2)
        let (p_thermal, p_rgb) = self.cfg.modality_dropout;
        if has_thermal && p_thermal > 0.0 && self.rng.random() < p_thermal {
            thermal = vec![THERMAL_ABSENT; n];
            has_thermal = false;
        } else if has_rgb && p_rgb > 0.0 && self.rng.random() < p_rgb {
            rgb = vec![0f32; 3 * n];
            has_rgb = false;
        }

        let c_out = if self.cfg.validity_channel { 5 } else { 4 };
        let mut out = Vec::with_capacity(c_out * n);
        out.extend_from_slice(&rgb);
        out.extend_from_slice(&thermal);
        if self.cfg.validity_channel {
            out.extend(std::iter::repeat(if has_thermal { 1.0f32 } else { 0.0 }).take(n));
        }
        Ok((out, c_out, has_rgb, has_thermal))
    }

    /// Maps a thermal plane into `[0.1, 1.0]`, keeping 0.0 reserved.
    pub fn normalize_thermal(plane: &[f32], already_scaled: bool) -> Vec<f32> {
        let lo = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if already_scaled && lo >= 0.0 && hi <= 1.0 {
            // dataset already applied the reserved-range normalisation
            return plane.to_vec();
        }
        plane
            .iter()
            .map(|&v| {
                let norm = if hi <= lo { 0.0 } else { (v - lo) / (hi - lo) };
                THERMAL_MIN + (1.0 - THERMAL_MIN) * norm
            })
            .collect()
    }

    /// CenterNet encoding of flat xyxy pixel boxes into the output-stride grids.
    /// `wh` and `offset` are `(2, oh, ow)`; `heatmap` and `reg_mask` are `(oh, ow)`.
    fn encode_boxes(
        &self,
        boxes: &[f32],
        (oh, ow): (usize, usize),
        heatmap: &mut [f32],
        wh: &mut [f32],
        reg_mask: &mut [f32],
        mut offset: Option<&mut [f32]>,
    ) {
        let s = self.cfg.person_head_stride as f32;
        let plane = oh * ow;
        for bx in boxes.chunks_exact(4).take(self.cfg.max_objects) {
            let (x1, y1, x2, y2) = (bx[0], bx[1], bx[2], bx[3]);
            let (bw, bh) = ((x2 - x1) / s, (y2 - y1) / s);
            if bw <= 0.0 || bh <= 0.0 {
                continue;
            }
            let (cx, cy) = (((x1 + x2) / 2.0) / s, ((y1 + y2) / 2.0) / s);
            // int() truncates toward zero, so a centre in (-1, 0) lands in cell 0
            let (cxi, cyi) = (cx as i64, cy as i64);
            if !(0 <= cxi && cxi < ow as i64 && 0 <= cyi && cyi < oh as i64) {
                continue;
            }

            let r = gaussian_radius((f64::from(bh), f64::from(bw)), self.cfg.min_overlap);
            let radius = match self.cfg.radius_rounding {
                RadiusRounding::Round => r.round_ties_even() as i64,
                RadiusRounding::Trunc => r as i64,
            }
            .max(0);
            draw_gaussian(heatmap, (oh, ow), (cxi, cyi), radius, 1.0);

            let pos = cyi as usize * ow + cxi as usize;
            wh[pos] = bw;
            wh[plane + pos] = bh;
            reg_mask[pos] = 1.0;
            if let Some(off) = offset.as_deref_mut() {
                off[pos] = cx - cxi as f32;
                off[plane + pos] = cy - cyi as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::base::{boxes_tensor, mask_tensor, Raster};

    fn rgb_sample(h: usize, w: usize, v: u8) -> Sample {
        let img = Tensor::from_slice(&vec![v; h * w * 3]).reshape([h as i64, w as i64, 3]);
        Sample::new(img, true, false)
    }

    #[test]
    fn gaussian_radius_matches_reference_values() {
        // CornerNet reference: a 100x100 box at IoU 0.7 gives about 14.7 (smallest root)
        let r = gaussian_radius((100.0, 100.0), 0.7);
        assert!(r > 5.0 && r < 20.0, "{r}");
        // tiny boxes collapse to a radius below 1
        assert!(gaussian_radius((2.0, 2.0), 0.7) < 1.0);
        // monotone in size
        assert!(gaussian_radius((40.0, 40.0), 0.7) > gaussian_radius((10.0, 10.0), 0.7));
    }

    #[test]
    fn gaussian_kernel_peaks_at_one_in_the_centre() {
        let g = gaussian2d((5, 5), 5.0 / 6.0);
        assert!((g[12] - 1.0).abs() < 1e-12);
        assert!(g[0] < g[12]);
    }

    #[test]
    fn draw_gaussian_clips_at_borders_and_keeps_max() {
        let mut hm = vec![0f32; 5 * 5];
        draw_gaussian(&mut hm, (5, 5), (0, 0), 2, 1.0);
        assert_eq!(hm[0], 1.0);
        assert!(hm[1] > 0.0 && hm[1] < 1.0);
        draw_gaussian(&mut hm, (5, 5), (1, 0), 2, 1.0);
        // second peak lands without erasing the first (max-combined)
        assert_eq!(hm[0], 1.0);
        assert_eq!(hm[1], 1.0);
    }

    #[test]
    fn batch_shapes_and_dtypes() {
        let mut a = rgb_sample(16, 16, 255);
        a.danger_mask = Some(mask_tensor(&Raster::from_vec(16, 16, 1, vec![1u8; 256])));
        let mut b = rgb_sample(16, 16, 0);
        b.person_boxes = Some(boxes_tensor(&[[4.0, 4.0, 12.0, 12.0]]));

        let mut col = SkyLensCollator::with_rng(CollatorConfig::default(), SimpleRng::new(1)).unwrap();
        let batch = col.collate(&[a, b]).unwrap();

        assert_eq!(batch.pixel_values.size(), vec![2, 4, 16, 16]);
        assert_eq!(batch.modality_mask.size(), vec![2, 2]);
        let labels = batch.danger_labels.as_ref().unwrap();
        assert_eq!(labels.size(), vec![2, 16, 16]);
        assert_eq!(labels.kind(), Kind::Int64);
        // sample without a mask is filled with ignore
        assert_eq!(labels.get(1).max().int64_value(&[]), 255);
        assert_eq!(batch.person_heatmap.as_ref().unwrap().size(), vec![2, 1, 4, 4]);
        assert_eq!(batch.person_wh.as_ref().unwrap().size(), vec![2, 2, 4, 4]);
        assert_eq!(batch.person_reg_mask.as_ref().unwrap().size(), vec![2, 1, 4, 4]);
        assert_eq!(batch.person_offset.as_ref().unwrap().size(), vec![2, 2, 4, 4]);
        // RGB is scaled to [0, 1] and thermal stays at the reserved 0.0
        let px = to_vec_f32(&batch.pixel_values.get(0));
        assert!((px[0] - 1.0).abs() < 1e-6);
        assert_eq!(px[3 * 256], 0.0);
    }

    #[test]
    fn heads_without_gt_have_no_keys() {
        let mut col = SkyLensCollator::with_rng(CollatorConfig::default(), SimpleRng::new(1)).unwrap();
        let batch = col.collate(&[rgb_sample(8, 8, 10)]).unwrap();
        assert!(batch.danger_labels.is_none());
        assert!(!batch.has_person_gt());
    }

    #[test]
    fn centre_cell_gets_peak_wh_and_offset() {
        let mut s = rgb_sample(16, 16, 0);
        // centre (8, 8) px -> cell (2, 2); width 8 px -> 2 cells
        s.person_boxes = Some(boxes_tensor(&[[4.0, 4.0, 12.0, 12.0]]));
        let mut col = SkyLensCollator::with_rng(CollatorConfig::default(), SimpleRng::new(1)).unwrap();
        let batch = col.collate(&[s]).unwrap();
        let hm = to_vec_f32(batch.person_heatmap.as_ref().unwrap());
        assert_eq!(hm[2 * 4 + 2], 1.0);
        let wh = to_vec_f32(batch.person_wh.as_ref().unwrap());
        assert_eq!(wh[2 * 4 + 2], 2.0);
        assert_eq!(wh[16 + 2 * 4 + 2], 2.0);
        let reg = to_vec_f32(batch.person_reg_mask.as_ref().unwrap());
        assert_eq!(reg.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn mixed_sizes_are_rejected() {
        let mut col = SkyLensCollator::with_rng(CollatorConfig::default(), SimpleRng::new(1)).unwrap();
        assert!(col.collate(&[rgb_sample(8, 8, 0), rgb_sample(8, 16, 0)]).is_err());
        assert!(col.collate(&[]).is_err());
    }

    #[test]
    fn validity_channel_follows_thermal_presence() {
        let cfg = CollatorConfig { validity_channel: true, ..Default::default() };
        let mut col = SkyLensCollator::with_rng(cfg, SimpleRng::new(1)).unwrap();
        let batch = col.collate(&[rgb_sample(8, 8, 10)]).unwrap();
        assert_eq!(batch.pixel_values.size(), vec![1, 5, 8, 8]);
        assert_eq!(batch.pixel_values.get(0).get(4).max().double_value(&[]), 0.0);
    }

    #[test]
    fn full_thermal_dropout_clears_the_plane() {
        // 4-channel sample: RGB + thermal in [0.1, 1]
        let mut data = Vec::new();
        for _ in 0..16 {
            data.extend_from_slice(&[0.5f32, 0.5, 0.5, 0.7]);
        }
        let img = Tensor::from_slice(&data).reshape([4, 4, 4]);
        let s = Sample::new(img, true, true);
        let cfg = CollatorConfig { modality_dropout: (1.0, 0.0), ..Default::default() };
        let mut col = SkyLensCollator::with_rng(cfg, SimpleRng::new(3)).unwrap();
        let batch = col.collate(&[s]).unwrap();
        let mask = to_vec_u8(&batch.modality_mask);
        assert_eq!(mask, vec![1, 0]);
        assert_eq!(batch.pixel_values.get(0).get(3).abs().max().double_value(&[]), 0.0);
    }

    #[test]
    fn thermal_normalisation_reserves_zero() {
        let n = SkyLensCollator::normalize_thermal(&[10.0, 20.0, 30.0], false);
        assert!((n[0] - 0.1).abs() < 1e-6 && (n[2] - 1.0).abs() < 1e-6);
        // an already-scaled plane passes through untouched
        let p = SkyLensCollator::normalize_thermal(&[0.1, 0.5, 1.0], true);
        assert_eq!(p, vec![0.1, 0.5, 1.0]);
    }

    #[test]
    fn radius_rounding_parse() {
        assert_eq!(RadiusRounding::parse("round").unwrap(), RadiusRounding::Round);
        assert!(RadiusRounding::parse("ceil").is_err());
    }
}
