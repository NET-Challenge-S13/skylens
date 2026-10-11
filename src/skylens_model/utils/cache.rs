//! Rust twin of `utils/cache.py`.
//!
//! On-disk cache of resized samples. Originals are large (RescueNet 3000x4000, LLVIP
//! 1024x1280) and decoding plus resizing them every epoch starves the GPU, so each sample is
//! resized once and later epochs read small files.
//!
//! Layout per sample:
//!
//! ```text
//! <i>.jpg     RGB (JPEG q95)
//! <i>_t.png   8-bit thermal, only when present
//! <i>_m.png   danger mask, only when present (lossless)
//! index.json  boxes, modality flags, which files exist
//! ```
//!
//! [`ResizedCache`] always returns float32 `[0, 1]` images, matching the collator's
//! normalisation. The JSON index is byte-compatible in shape with the Python one
//! (`{"size": N, "samples": [{has_rgb, has_thermal, has_mask, boxes, has_boxes}, ...]}`), so a
//! cache built by either side can be read by the other.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageEncoder};
use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};

use crate::datasets::base::{
    boxes_from_tensor, boxes_tensor, mask_tensor, read_mask_raster, read_rgb_raster, resize_area_f32,
    resize_nearest, tensor_f32, to_vec_f32, to_vec_u8, DatasetError, Raster, Result, Sample,
    SkyLensDataset, Transform,
};

/// RGB is JPEG q95. The goal is decoding speed, so no lossless format; masks and thermal stay
/// lossless PNG.
const JPEG_QUALITY: u8 = 95;

/// Image tensor `(H, W, C)` as a float raster plus whether it was uint8.
fn image_raster(t: &Tensor) -> (Raster<f32>, bool) {
    let sz = t.size();
    let c = if sz.len() == 3 { sz[2] as usize } else { 1 };
    let r = Raster::from_vec(sz[0] as usize, sz[1] as usize, c, to_vec_f32(t));
    (r, t.kind() == Kind::Uint8)
}

/// Resizes image, mask and boxes to `size x size`.
///
/// The mask uses nearest (labels survive), the image uses area interpolation (suits
/// shrinking). uint8 images stay uint8 (rounded), float images stay float.
pub fn resize_sample(sample: &Sample, size: usize) -> Sample {
    let (raster, was_u8) = image_raster(&sample.image);
    let (h, w) = (raster.h, raster.w);
    let resized = resize_area_f32(&raster, size, size);
    let image = if was_u8 {
        let u = crate::datasets::base::f32_to_u8(&resized);
        crate::datasets::base::tensor_u8(&u)
    } else {
        tensor_f32(&resized)
    };

    let mut out = sample.clone();
    out.image = image;

    if let Some(mask) = &sample.danger_mask {
        let m = Raster::from_vec(h, w, 1, to_vec_u8(mask));
        out.danger_mask = Some(mask_tensor(&resize_nearest(&m, size, size)));
    }

    if let Some(boxes) = &sample.person_boxes {
        let mut b = boxes_from_tensor(boxes);
        let (sx, sy) = (size as f32 / w as f32, size as f32 / h as f32);
        for bx in b.iter_mut() {
            bx[0] *= sx;
            bx[2] *= sx;
            bx[1] *= sy;
            bx[3] *= sy;
        }
        out.person_boxes = Some(boxes_tensor(&b));
    }
    out
}

/// Float `[0,1]` or uint8 raster to uint8 `[0, 255]`. A float raster whose max is at most 1.5
/// is taken to be in `[0, 1]`.
fn to_uint8(r: &Raster<f32>, was_u8: bool) -> Raster<u8> {
    if was_u8 {
        return crate::datasets::base::f32_to_u8(r);
    }
    let max = r.data.iter().copied().fold(f32::MIN, f32::max);
    let scale = if max <= 1.5 { 255.0 } else { 1.0 };
    Raster::from_vec(
        r.h,
        r.w,
        r.c,
        r.data.iter().map(|&v| (v * scale).round_ties_even().clamp(0.0, 255.0) as u8).collect(),
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheRecord {
    pub has_rgb: bool,
    pub has_thermal: bool,
    pub has_mask: bool,
    pub boxes: Vec<Vec<f32>>,
    pub has_boxes: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheIndex {
    size: usize,
    samples: Vec<CacheRecord>,
}

fn sample_path(dir: &Path, i: usize, suffix: &str) -> PathBuf {
    dir.join(format!("{i:06}{suffix}"))
}

fn write_jpeg(path: &Path, rgb: &[u8], w: u32, h: u32) -> Result<()> {
    let file = BufWriter::new(File::create(path)?);
    JpegEncoder::new_with_quality(file, JPEG_QUALITY).write_image(rgb, w, h, ExtendedColorType::Rgb8)?;
    Ok(())
}

/// Walks `dataset` once and writes resized copies into `cache_dir`.
///
/// `dataset` must be the raw one (built without transforms); resizing happens here. A complete
/// cache of the same size and length is left alone unless `overwrite` is set.
pub fn build_resized_cache(
    dataset: &dyn SkyLensDataset,
    cache_dir: &Path,
    size: usize,
    overwrite: bool,
    log_every: usize,
) -> Result<PathBuf> {
    let index_path = cache_dir.join("index.json");

    if index_path.is_file() && !overwrite {
        if let Ok(meta) = serde_json::from_slice::<CacheIndex>(&std::fs::read(&index_path)?) {
            if meta.size == size && meta.samples.len() == dataset.len() {
                return Ok(cache_dir.to_path_buf());
            }
        }
    }

    std::fs::create_dir_all(cache_dir)?;
    let mut records = Vec::with_capacity(dataset.len());

    for i in 0..dataset.len() {
        let s = resize_sample(&dataset.get(i)?, size);
        let (raster, was_u8) = image_raster(&s.image);
        let img = to_uint8(&raster, was_u8);

        // RGB plane: first three channels, or channel 0 repeated for a thermal-only image
        let mut rgb = Vec::with_capacity(size * size * 3);
        for p in 0..size * size {
            for ch in 0..3 {
                let src = if img.c >= 3 { ch } else { 0 };
                rgb.push(img.data[p * img.c + src]);
            }
        }
        write_jpeg(&sample_path(cache_dir, i, ".jpg"), &rgb, size as u32, size as u32)?;

        if s.has_thermal {
            let ch = if img.c >= 4 { 3 } else { 0 };
            let plane: Vec<u8> = (0..size * size).map(|p| img.data[p * img.c + ch]).collect();
            image::save_buffer(
                sample_path(cache_dir, i, "_t.png"),
                &plane,
                size as u32,
                size as u32,
                ExtendedColorType::L8,
            )?;
        }

        if let Some(mask) = &s.danger_mask {
            image::save_buffer(
                sample_path(cache_dir, i, "_m.png"),
                &to_vec_u8(mask),
                size as u32,
                size as u32,
                ExtendedColorType::L8,
            )?;
        }

        let boxes: Vec<Vec<f32>> = match &s.person_boxes {
            Some(b) => boxes_from_tensor(b).iter().map(|x| x.to_vec()).collect(),
            None => Vec::new(),
        };
        records.push(CacheRecord {
            has_rgb: s.has_rgb,
            has_thermal: s.has_thermal,
            has_mask: s.danger_mask.is_some(),
            boxes,
            has_boxes: s.person_boxes.is_some(),
        });

        if log_every > 0 && (i + 1) % log_every == 0 {
            println!("    {}/{}", i + 1, dataset.len());
        }
    }

    let index = CacheIndex { size, samples: records };
    std::fs::write(&index_path, serde_json::to_vec(&index)?)?;
    Ok(cache_dir.to_path_buf())
}

/// Reads the cache written by [`build_resized_cache`].
///
/// `transforms` should carry augmentation only (no resize; the size is already fixed).
pub struct ResizedCache {
    pub root: PathBuf,
    pub size: usize,
    pub records: Vec<CacheRecord>,
    transforms: Option<Transform>,
}

impl ResizedCache {
    pub fn new(cache_dir: impl AsRef<Path>, transforms: Option<Transform>) -> Result<Self> {
        let root = cache_dir.as_ref().to_path_buf();
        let meta: CacheIndex = serde_json::from_slice(&std::fs::read(root.join("index.json"))?)?;
        Ok(ResizedCache { root, size: meta.size, records: meta.samples, transforms })
    }
}

impl SkyLensDataset for ResizedCache {
    fn len(&self) -> usize {
        self.records.len()
    }

    fn get(&self, i: usize) -> Result<Sample> {
        let r = self.records.get(i).ok_or_else(|| {
            DatasetError::InvalidSample(format!("cache index {i} out of range ({})", self.records.len()))
        })?;
        let rgb = read_rgb_raster(&sample_path(&self.root, i, ".jpg"))?;
        let (h, w) = (rgb.h, rgb.w);
        let c = if r.has_thermal { 4 } else { 3 };
        let mut image = Raster::<f32>::new(h, w, c);
        for p in 0..h * w {
            for ch in 0..3 {
                image.data[p * c + ch] = f32::from(rgb.data[p * 3 + ch]) / 255.0;
            }
        }
        if r.has_thermal {
            let t = read_mask_raster(&sample_path(&self.root, i, "_t.png"))?;
            for p in 0..h * w {
                image.data[p * c + 3] = f32::from(t.data[p]) / 255.0;
            }
        }

        let mut s = Sample::new(tensor_f32(&image), r.has_rgb, r.has_thermal);
        if r.has_mask {
            s.danger_mask = Some(mask_tensor(&read_mask_raster(&sample_path(&self.root, i, "_m.png"))?));
        }
        if r.has_boxes {
            let boxes: Vec<[f32; 4]> = r
                .boxes
                .iter()
                .filter(|b| b.len() == 4)
                .map(|b| [b[0], b[1], b[2], b[3]])
                .collect();
            s.person_boxes = Some(boxes_tensor(&boxes));
        }
        Ok(match &self.transforms {
            Some(t) => t(s),
            None => s,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::base::tensor_u8;

    fn sample(h: usize, w: usize) -> Sample {
        let img = Raster::from_vec(h, w, 3, vec![100u8; h * w * 3]);
        let mut s = Sample::new(tensor_u8(&img), true, false);
        s.danger_mask = Some(mask_tensor(&Raster::from_vec(h, w, 1, vec![3u8; h * w])));
        s.person_boxes = Some(boxes_tensor(&[[10.0, 20.0, 30.0, 40.0]]));
        s
    }

    #[test]
    fn resize_scales_boxes_and_keeps_labels() {
        let out = resize_sample(&sample(100, 200), 50);
        assert_eq!(out.image.size(), vec![50, 50, 3]);
        assert_eq!(out.danger_mask.as_ref().unwrap().size(), vec![50, 50]);
        let b = boxes_from_tensor(out.person_boxes.as_ref().unwrap());
        assert_eq!(b, vec![[2.5, 10.0, 7.5, 20.0]]);
        assert!(to_vec_u8(out.danger_mask.as_ref().unwrap()).iter().all(|&v| v == 3));
    }

    #[test]
    fn empty_boxes_stay_empty() {
        let mut s = sample(10, 10);
        s.person_boxes = Some(boxes_tensor(&[]));
        let out = resize_sample(&s, 5);
        assert_eq!(out.person_boxes.unwrap().size(), vec![0, 4]);
    }

    #[test]
    fn uint8_conversion_handles_unit_and_byte_ranges() {
        let unit = Raster::from_vec(1, 2, 1, vec![0.0f32, 1.0]);
        assert_eq!(to_uint8(&unit, false).data, vec![0, 255]);
        let byte = Raster::from_vec(1, 2, 1, vec![0.0f32, 200.0]);
        assert_eq!(to_uint8(&byte, false).data, vec![0, 200]);
    }

    #[test]
    fn index_json_shape_matches_python() {
        let idx = CacheIndex {
            size: 8,
            samples: vec![CacheRecord {
                has_rgb: true,
                has_thermal: false,
                has_mask: false,
                boxes: vec![],
                has_boxes: false,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&idx).unwrap();
        assert_eq!(v["size"], 8);
        assert_eq!(v["samples"][0]["has_boxes"], false);
        assert!(v["samples"][0]["boxes"].as_array().unwrap().is_empty());
    }
}
