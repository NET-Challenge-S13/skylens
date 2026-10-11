//! Rust twin of `datasets/base.py`.
//!
//! Common sample contract, unified class schema, file helpers and image IO shared by every
//! SkyLens dataset. Design references are the same as in the Python file: bbox lives on the
//! label side (the collator turns boxes into heatmaps), a sample declares which modalities it
//! carries (`has_rgb` / `has_thermal`), and `None` is a normal value for `danger_mask` and
//! `person_boxes` because heads are trained separately.
//!
//! Unified label schema:
//!
//! | value | name         | source                                          |
//! |-------|--------------|-------------------------------------------------|
//! | 0     | normal       | everything not otherwise dangerous              |
//! | 1     | fire         | FLAME fire masks, FLAME 3, fire segmentation    |
//! | 2     | collapse     | RescueNet building major-damage / total-destr.  |
//! | 3     | road_blocked | RescueNet road-blocked                          |
//! | 255   | ignore       | unlabeled / void                                |
//!
//! NOT PORTED: the download helpers (`download_url`, `download_hf`, `download_kaggle`,
//! `extract_archive`, `check_integrity`). They wrap torchvision, `huggingface_hub` and the
//! Kaggle CLI, which have no Rust equivalent worth inventing; training data is prepared by the
//! Python side. A dataset whose files are missing fails with
//! [`DatasetError::ManualDownloadRequired`] carrying the same human-actionable message.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use image::DynamicImage;
use tch::{Kind, Tensor};
use thiserror::Error;

// --------------------------------------------------------------------------- //
// class schema
// --------------------------------------------------------------------------- //

/// Unified danger-zone segmentation labels (`DangerClass` in Python).
pub struct DangerClass;

impl DangerClass {
    pub const NORMAL: u8 = 0;
    pub const FIRE: u8 = 1;
    pub const COLLAPSE: u8 = 2;
    pub const ROAD_BLOCKED: u8 = 3;
    pub const IGNORE: u8 = 255;
}

pub const DANGER_CLASS_NAMES: [&str; 4] = ["normal", "fire", "collapse", "road_blocked"];
pub const NUM_DANGER_CLASSES: usize = DANGER_CLASS_NAMES.len();
pub const IGNORE_INDEX: u8 = DangerClass::IGNORE;

/// Image extensions accepted by most datasets. Matching is exact-case, like the Python tuple.
pub const IMG_EXT: [&str; 5] = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"];

// --------------------------------------------------------------------------- //
// errors
// --------------------------------------------------------------------------- //

#[derive(Debug, Error)]
pub enum DatasetError {
    /// The dataset cannot be fetched programmatically (or the files are not under `root`).
    #[error("{0}")]
    ManualDownloadRequired(String),
    #[error("{dataset}: unknown split {split:?}; expected one of {expected:?}")]
    InvalidSplit { dataset: String, split: String, expected: Vec<String> },
    #[error("{0}")]
    InvalidSample(String),
    #[error("could not read {kind}: {path}")]
    Unreadable { kind: &'static str, path: String },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tensor: {0}")]
    Tensor(#[from] tch::TchError),
}

pub type Result<T> = std::result::Result<T, DatasetError>;

// --------------------------------------------------------------------------- //
// sample contract
// --------------------------------------------------------------------------- //

/// The exact contract every `get` must satisfy (Python: the sample dict).
///
/// * `image`        `(H, W, C)` Uint8 or Float; C=3 RGB, 4 RGB+thermal, 1 thermal
/// * `danger_mask`  `(H, W)` Uint8, 0..3 and 255 = ignore, or `None`
/// * `person_boxes` `(N, 4)` Float, xyxy pixels, or `None`
#[derive(Debug)]
pub struct Sample {
    pub image: Tensor,
    pub has_rgb: bool,
    pub has_thermal: bool,
    pub danger_mask: Option<Tensor>,
    pub person_boxes: Option<Tensor>,
    /// Extra key written by `Flame3Pairs` (0 = no fire, 1 = fire) when the folder name says so.
    pub frame_label: Option<i64>,
}

impl Clone for Sample {
    fn clone(&self) -> Self {
        Sample {
            image: self.image.shallow_clone(),
            has_rgb: self.has_rgb,
            has_thermal: self.has_thermal,
            danger_mask: self.danger_mask.as_ref().map(|t| t.shallow_clone()),
            person_boxes: self.person_boxes.as_ref().map(|t| t.shallow_clone()),
            frame_label: self.frame_label,
        }
    }
}

impl Sample {
    pub fn new(image: Tensor, has_rgb: bool, has_thermal: bool) -> Self {
        Sample {
            image,
            has_rgb,
            has_thermal,
            danger_mask: None,
            person_boxes: None,
            frame_label: None,
        }
    }
}

/// Callable applied to the sample before it is returned.
pub type Transform = Box<dyn Fn(Sample) -> Sample + Send + Sync>;

/// A map-style dataset (`__len__` / `__getitem__`).
pub trait SkyLensDataset: Send + Sync {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, idx: usize) -> Result<Sample>;
}

/// How a dataset can be acquired (`availability` in Python).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    Auto,
    Account,
    Manual,
}

/// Class-level metadata (`name`, `splits`, `homepage`, ... in Python).
#[derive(Debug, Clone, Copy)]
pub struct DatasetMeta {
    pub name: &'static str,
    pub splits: &'static [&'static str],
    pub availability: Availability,
    pub homepage: &'static str,
    pub download_note: &'static str,
    pub expected_layout: &'static str,
}

/// Fails with a readable error when `split` is not one of `meta.splits`.
pub fn verify_split(meta: &DatasetMeta, split: &str) -> Result<String> {
    if meta.splits.contains(&split) {
        Ok(split.to_string())
    } else {
        Err(DatasetError::InvalidSplit {
            dataset: meta.name.to_string(),
            split: split.to_string(),
            expected: meta.splits.iter().map(|s| s.to_string()).collect(),
        })
    }
}

/// A message a human can actually act on. Never fail silently.
pub fn manual_download_message(meta: &DatasetMeta, root: &Path) -> String {
    let verdict = match meta.availability {
        Availability::Auto => {
            "This dataset IS auto-downloadable, but the attempt did not produce the expected files."
        }
        Availability::Account => {
            "This dataset requires a (free) account, so it cannot be fetched without credentials."
        }
        Availability::Manual => "This dataset CANNOT be downloaded programmatically.",
    };
    let indent = |s: &str| s.trim().replace('\n', "\n  ");
    let mut out = String::new();
    let _ = write!(out, "{}: data not found under {}\n\n{}\n", meta.name, root.display(), verdict);
    if !meta.homepage.is_empty() {
        let _ = write!(out, "\n  Source: {}\n", meta.homepage);
    }
    if !meta.download_note.is_empty() {
        let _ = write!(out, "\n  {}\n", indent(meta.download_note));
    }
    if !meta.expected_layout.is_empty() {
        let _ = write!(
            out,
            "\n  After downloading, extract so that the tree looks like:\n\n  {}\n",
            indent(meta.expected_layout)
        );
    }
    let _ = write!(out, "\n  Then re-run with root={:?}.", root.display().to_string());
    out
}

/// Resolves `root` like `Path.expanduser().resolve()` (best effort: `~` and canonicalise).
pub fn resolve_root(root: impl AsRef<Path>) -> PathBuf {
    let root = root.as_ref();
    let expanded = match root.strip_prefix("~") {
        Ok(rest) => match std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            Some(home) => PathBuf::from(home).join(rest),
            None => root.to_path_buf(),
        },
        Err(_) => root.to_path_buf(),
    };
    std::fs::canonicalize(&expanded).unwrap_or(expanded)
}

/// Shared constructor tail: a missing dataset raises the actionable message.
pub fn ensure_exists(meta: &DatasetMeta, root: &Path, exists: bool) -> Result<()> {
    if exists {
        Ok(())
    } else {
        Err(DatasetError::ManualDownloadRequired(manual_download_message(meta, root)))
    }
}

/// Applies `transforms` (if any) after the contract check, like `__getitem__`.
pub fn finish_sample(sample: Sample, transforms: Option<&Transform>) -> Result<Sample> {
    let sample = normalize_sample(sample)?;
    Ok(match transforms {
        Some(t) => t(sample),
        None => sample,
    })
}

/// Coerces a sample to the fixed contract, and fails loudly if it can't be.
pub fn normalize_sample(mut sample: Sample) -> Result<Sample> {
    let mut image = sample.image;
    if image.dim() == 2 {
        image = image.unsqueeze(2);
    }
    if image.dim() != 3 {
        return Err(DatasetError::InvalidSample(format!(
            "image must be (H, W, C); got shape {:?}",
            image.size()
        )));
    }
    if !matches!(image.kind(), Kind::Uint8 | Kind::Float) {
        image = image.to_kind(Kind::Float);
    }
    sample.image = image;

    if let Some(mask) = sample.danger_mask.take() {
        if mask.dim() != 2 {
            return Err(DatasetError::InvalidSample(format!(
                "danger_mask must be (H, W); got {:?}",
                mask.size()
            )));
        }
        sample.danger_mask = Some(mask.to_kind(Kind::Uint8));
    }
    if let Some(boxes) = sample.person_boxes.take() {
        sample.person_boxes = Some(boxes.to_kind(Kind::Float).reshape([-1, 4]));
    }
    Ok(sample)
}

// --------------------------------------------------------------------------- //
// tensor <-> vec helpers
// --------------------------------------------------------------------------- //

/// Row-major `(H, W, C)` buffer. The pixel-level helpers work on this and only convert to
/// tensors at the end, which keeps the pure logic testable without libtorch kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct Raster<T> {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<T>,
}

impl<T: Copy + Default> Raster<T> {
    pub fn new(h: usize, w: usize, c: usize) -> Self {
        Raster { h, w, c, data: vec![T::default(); h * w * c] }
    }

    pub fn from_vec(h: usize, w: usize, c: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), h * w * c, "raster buffer does not match its shape");
        Raster { h, w, c, data }
    }

    #[inline]
    pub fn at(&self, y: usize, x: usize, ch: usize) -> T {
        self.data[(y * self.w + x) * self.c + ch]
    }

    #[inline]
    pub fn set(&mut self, y: usize, x: usize, ch: usize, v: T) {
        self.data[(y * self.w + x) * self.c + ch] = v;
    }
}

/// `(H, W, C)` Uint8 tensor from a raster.
pub fn tensor_u8(r: &Raster<u8>) -> Tensor {
    Tensor::from_slice(&r.data).reshape([r.h as i64, r.w as i64, r.c as i64])
}

/// `(H, W, C)` Float tensor from a raster.
pub fn tensor_f32(r: &Raster<f32>) -> Tensor {
    Tensor::from_slice(&r.data).reshape([r.h as i64, r.w as i64, r.c as i64])
}

/// `(H, W)` Uint8 tensor from a single-channel raster.
pub fn mask_tensor(r: &Raster<u8>) -> Tensor {
    Tensor::from_slice(&r.data).reshape([r.h as i64, r.w as i64])
}

/// `(N, 4)` Float tensor from xyxy boxes.
pub fn boxes_tensor(boxes: &[[f32; 4]]) -> Tensor {
    if boxes.is_empty() {
        return Tensor::zeros([0, 4], (Kind::Float, tch::Device::Cpu));
    }
    let flat: Vec<f32> = boxes.iter().flatten().copied().collect();
    Tensor::from_slice(&flat).reshape([boxes.len() as i64, 4])
}

/// Reads a tensor back into a flat `Vec<f32>` (any numeric kind).
pub fn to_vec_f32(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&t.contiguous().to_kind(Kind::Float).flatten(0, -1))
        .expect("tensor to Vec<f32>")
}

/// Reads a tensor back into a flat `Vec<u8>`.
pub fn to_vec_u8(t: &Tensor) -> Vec<u8> {
    Vec::<u8>::try_from(&t.contiguous().to_kind(Kind::Uint8).flatten(0, -1))
        .expect("tensor to Vec<u8>")
}

/// Reads `(N, 4)` boxes back into arrays.
pub fn boxes_from_tensor(t: &Tensor) -> Vec<[f32; 4]> {
    to_vec_f32(t).chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

// --------------------------------------------------------------------------- //
// file discovery
// --------------------------------------------------------------------------- //

/// True when the file name ends with one of `exts` (exact-case, leading dot included).
pub fn has_ext(path: &Path, exts: &[&str]) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(e) => {
            let dotted = format!(".{e}");
            exts.contains(&dotted.as_str())
        }
        None => false,
    }
}

/// Direct children of `dir` (files and dirs), sorted. Empty when `dir` is unreadable.
pub fn list_dir(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

/// Sorted direct children of `dir` with one of the image extensions.
pub fn list_images(dir: &Path, exts: &[&str]) -> Vec<PathBuf> {
    list_dir(dir).into_iter().filter(|p| has_ext(p, exts)).collect()
}

/// `dir.iterdir()` has any entry at all.
pub fn dir_has_entries(dir: &Path) -> bool {
    std::fs::read_dir(dir).map(|mut rd| rd.next().is_some()).unwrap_or(false)
}

/// Recursive file walk (`Path.rglob("*")` restricted to files), sorted.
pub fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for p in list_dir(&d) {
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// First existing file `dir/<stem><ext>` over `exts`.
pub fn find_with_stem(dir: &Path, stem: &str, exts: &[&str]) -> Option<PathBuf> {
    exts.iter().map(|e| dir.join(format!("{stem}{e}"))).find(|p| p.is_file())
}

pub fn file_stem(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

// --------------------------------------------------------------------------- //
// io
// --------------------------------------------------------------------------- //

fn open_image(path: &Path, kind: &'static str) -> Result<DynamicImage> {
    if !path.is_file() {
        return Err(DatasetError::Unreadable { kind, path: path.display().to_string() });
    }
    image::open(path)
        .map_err(|_| DatasetError::Unreadable { kind, path: path.display().to_string() })
}

/// Reads an 8-bit RGB image as an `(H, W, 3)` raster.
pub fn read_rgb_raster(path: &Path) -> Result<Raster<u8>> {
    let img = open_image(path, "image")?.to_rgb8();
    let (w, h) = img.dimensions();
    Ok(Raster::from_vec(h as usize, w as usize, 3, img.into_raw()))
}

/// Reads an 8-bit RGB image as an `(H, W, 3)` uint8 tensor.
pub fn read_rgb(path: &Path) -> Result<Tensor> {
    Ok(tensor_u8(&read_rgb_raster(path)?))
}

/// Reads a single-channel label image as an `(H, W)` raster.
///
/// Python reads with `IMREAD_UNCHANGED`, keeps channel 0 of a colour image (which is blue in
/// OpenCV's BGR order, so channel index 2 here) and wraps 16-bit values with `astype(uint8)`.
pub fn read_mask_raster(path: &Path) -> Result<Raster<u8>> {
    let img = open_image(path, "mask")?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data: Vec<u8> = match img {
        DynamicImage::ImageLuma8(b) => b.into_raw(),
        DynamicImage::ImageLumaA8(b) => b.into_raw().chunks_exact(2).map(|p| p[0]).collect(),
        DynamicImage::ImageLuma16(b) => b.into_raw().into_iter().map(|v| v as u8).collect(),
        DynamicImage::ImageLumaA16(b) => {
            b.into_raw().chunks_exact(2).map(|p| p[0] as u8).collect()
        }
        DynamicImage::ImageRgb16(b) => b.into_raw().chunks_exact(3).map(|p| p[2] as u8).collect(),
        DynamicImage::ImageRgba16(b) => b.into_raw().chunks_exact(4).map(|p| p[2] as u8).collect(),
        other => other.to_rgb8().into_raw().chunks_exact(3).map(|p| p[2]).collect(),
    };
    Ok(Raster::from_vec(h, w, 1, data))
}

/// Reads a single-channel label image as an `(H, W)` uint8 tensor.
pub fn read_mask(path: &Path) -> Result<Tensor> {
    Ok(mask_tensor(&read_mask_raster(path)?))
}

/// Reads a thermal frame as an `(H, W, 1)` float raster.
///
/// NOT PORTED: Python reads radiometric TIFFs with `tifffile` so float32 temperatures survive.
/// Here TIFFs go through the `image` crate, which covers 8/16-bit grayscale (values kept
/// unscaled) but rejects single-channel float32 TIFFs. Swap in the `tiff` crate if FLAME 3
/// frames turn out to be float.
pub fn read_thermal_raster(path: &Path) -> Result<Raster<f32>> {
    let img = open_image(path, "thermal frame")?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data: Vec<f32> = match img {
        DynamicImage::ImageLuma8(b) => b.into_raw().into_iter().map(f32::from).collect(),
        DynamicImage::ImageLumaA8(b) => {
            b.into_raw().chunks_exact(2).map(|p| f32::from(p[0])).collect()
        }
        DynamicImage::ImageLuma16(b) => b.into_raw().into_iter().map(f32::from).collect(),
        DynamicImage::ImageLumaA16(b) => {
            b.into_raw().chunks_exact(2).map(|p| f32::from(p[0])).collect()
        }
        DynamicImage::ImageRgb16(b) => b
            .into_raw()
            .chunks_exact(3)
            .map(|p| (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 3.0)
            .collect(),
        DynamicImage::ImageRgba16(b) => b
            .into_raw()
            .chunks_exact(4)
            .map(|p| (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 3.0)
            .collect(),
        DynamicImage::ImageRgb32F(b) => {
            b.into_raw().chunks_exact(3).map(|p| (p[0] + p[1] + p[2]) / 3.0).collect()
        }
        DynamicImage::ImageRgba32F(b) => {
            b.into_raw().chunks_exact(4).map(|p| (p[0] + p[1] + p[2]) / 3.0).collect()
        }
        other => other
            .to_rgb8()
            .into_raw()
            .chunks_exact(3)
            .map(|p| (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 3.0)
            .collect(),
    };
    Ok(Raster::from_vec(h, w, 1, data))
}

/// Concatenates RGB (uint8) and thermal (float) into one `(H, W, C)` float raster.
///
/// The thermal plane is min-max normalised into `[0.1, 1.0]` (README §2.3), reserving exactly
/// `0.0` for "no thermal". RGB is scaled to `[0, 1]`.
pub fn stack_rgb_thermal(rgb: Option<&Raster<u8>>, thermal: Option<&Raster<f32>>) -> Result<Raster<f32>> {
    let (h, w) = match (rgb, thermal) {
        (Some(r), _) => (r.h, r.w),
        (None, Some(t)) => (t.h, t.w),
        (None, None) => {
            return Err(DatasetError::InvalidSample(
                "at least one of rgb/thermal must be provided".into(),
            ))
        }
    };
    let c = rgb.map_or(0, |_| 3) + thermal.map_or(0, |_| 1);
    let mut out = Raster::<f32>::new(h, w, c);

    let (lo, hi) = match thermal {
        Some(t) => {
            let lo = t.data.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = t.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (lo, hi)
        }
        None => (0.0, 0.0),
    };
    for y in 0..h {
        for x in 0..w {
            let mut ch = 0;
            if let Some(r) = rgb {
                for k in 0..3 {
                    out.set(y, x, ch, f32::from(r.at(y, x, k)) / 255.0);
                    ch += 1;
                }
            }
            if let Some(t) = thermal {
                let v = t.at(y, x, 0);
                let n = if hi <= lo { 0.0 } else { (v - lo) / (hi - lo) };
                out.set(y, x, ch, 0.1 + 0.9 * n);
            }
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------------- //
// resizing (OpenCV conventions)
// --------------------------------------------------------------------------- //

/// Bilinear resize with OpenCV's half-pixel-centre mapping (`INTER_LINEAR`).
pub fn resize_bilinear(src: &Raster<f32>, nh: usize, nw: usize) -> Raster<f32> {
    let mut out = Raster::<f32>::new(nh, nw, src.c);
    if src.h == 0 || src.w == 0 || nh == 0 || nw == 0 {
        return out;
    }
    let sy = src.h as f32 / nh as f32;
    let sx = src.w as f32 / nw as f32;
    for y in 0..nh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(src.h - 1);
        let y1 = (y0 + 1).min(src.h - 1);
        let wy = fy - y0 as f32;
        for x in 0..nw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(src.w - 1);
            let x1 = (x0 + 1).min(src.w - 1);
            let wx = fx - x0 as f32;
            for ch in 0..src.c {
                let top = src.at(y0, x0, ch) * (1.0 - wx) + src.at(y0, x1, ch) * wx;
                let bot = src.at(y1, x0, ch) * (1.0 - wx) + src.at(y1, x1, ch) * wx;
                out.set(y, x, ch, top * (1.0 - wy) + bot * wy);
            }
        }
    }
    out
}

/// Nearest-neighbour resize (`INTER_NEAREST`: `src = floor(dst * scale)`), label safe.
pub fn resize_nearest<T: Copy + Default>(src: &Raster<T>, nh: usize, nw: usize) -> Raster<T> {
    let mut out = Raster::<T>::new(nh, nw, src.c);
    if src.h == 0 || src.w == 0 {
        return out;
    }
    for y in 0..nh {
        let sy = ((y * src.h) / nh).min(src.h - 1);
        for x in 0..nw {
            let sx = ((x * src.w) / nw).min(src.w - 1);
            for ch in 0..src.c {
                out.set(y, x, ch, src.at(sy, sx, ch));
            }
        }
    }
    out
}

/// Area (box-average) resize, the `INTER_AREA` used for shrinking.
///
/// Each destination pixel averages the source cells its footprint touches with fractional
/// edge weights, which matches OpenCV for non-integer ratios up to rounding. Upscaling falls
/// back to bilinear, as OpenCV does.
pub fn resize_area_f32(src: &Raster<f32>, nh: usize, nw: usize) -> Raster<f32> {
    if nh >= src.h || nw >= src.w {
        return resize_bilinear(src, nh, nw);
    }
    let mut out = Raster::<f32>::new(nh, nw, src.c);
    let sy = src.h as f32 / nh as f32;
    let sx = src.w as f32 / nw as f32;
    for y in 0..nh {
        let (ya, yb) = (y as f32 * sy, (y as f32 + 1.0) * sy);
        for x in 0..nw {
            let (xa, xb) = (x as f32 * sx, (x as f32 + 1.0) * sx);
            for ch in 0..src.c {
                let mut acc = 0.0f32;
                let mut wsum = 0.0f32;
                let mut yy = ya.floor() as usize;
                while (yy as f32) < yb && yy < src.h {
                    let wy = (yb.min(yy as f32 + 1.0) - ya.max(yy as f32)).max(0.0);
                    let mut xx = xa.floor() as usize;
                    while (xx as f32) < xb && xx < src.w {
                        let wx = (xb.min(xx as f32 + 1.0) - xa.max(xx as f32)).max(0.0);
                        acc += src.at(yy, xx, ch) * wy * wx;
                        wsum += wy * wx;
                        xx += 1;
                    }
                    yy += 1;
                }
                out.set(y, x, ch, if wsum > 0.0 { acc / wsum } else { 0.0 });
            }
        }
    }
    out
}

pub fn u8_to_f32(r: &Raster<u8>) -> Raster<f32> {
    Raster::from_vec(r.h, r.w, r.c, r.data.iter().map(|&v| f32::from(v)).collect())
}

pub fn f32_to_u8(r: &Raster<f32>) -> Raster<u8> {
    Raster::from_vec(
        r.h,
        r.w,
        r.c,
        r.data.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect(),
    )
}

// --------------------------------------------------------------------------- //
// small deterministic RNG (replaces numpy's Generator; avoids a `rand` dependency)
// --------------------------------------------------------------------------- //

/// SplitMix64: tiny, seedable, good enough for dropout draws, shuffles and crop offsets.
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    /// Seeds from the wall clock (the `np.random.default_rng()` default).
    pub fn from_entropy() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        SimpleRng::new(nanos)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, n)`; `n == 0` returns 0.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const META: DatasetMeta = DatasetMeta {
        name: "Demo",
        splits: &["train", "val"],
        availability: Availability::Account,
        homepage: "https://example.org",
        download_note: "note line 1\nnote line 2",
        expected_layout: "<root>/\n  a/",
    };

    #[test]
    fn class_schema_matches_contract() {
        assert_eq!(DangerClass::NORMAL, 0);
        assert_eq!(DangerClass::FIRE, 1);
        assert_eq!(DangerClass::COLLAPSE, 2);
        assert_eq!(DangerClass::ROAD_BLOCKED, 3);
        assert_eq!(IGNORE_INDEX, 255);
        assert_eq!(NUM_DANGER_CLASSES, 4);
    }

    #[test]
    fn split_verification() {
        assert!(verify_split(&META, "train").is_ok());
        assert!(matches!(verify_split(&META, "test"), Err(DatasetError::InvalidSplit { .. })));
    }

    #[test]
    fn manual_message_has_all_sections() {
        let msg = manual_download_message(&META, Path::new("/data/demo"));
        assert!(msg.contains("Demo: data not found under"));
        assert!(msg.contains("requires a (free) account"));
        assert!(msg.contains("Source: https://example.org"));
        assert!(msg.contains("note line 2"));
        assert!(msg.contains("After downloading"));
    }

    #[test]
    fn extension_match_is_exact_case() {
        assert!(has_ext(Path::new("a.jpg"), &IMG_EXT));
        assert!(has_ext(Path::new("a.JPG"), &IMG_EXT));
        assert!(!has_ext(Path::new("a.Jpg"), &IMG_EXT));
        assert!(!has_ext(Path::new("a"), &IMG_EXT));
    }

    #[test]
    fn stack_reserves_zero_for_missing_thermal() {
        let rgb = Raster::from_vec(1, 2, 3, vec![255, 0, 0, 0, 255, 0]);
        let th = Raster::from_vec(1, 2, 1, vec![10.0, 30.0]);
        let s = stack_rgb_thermal(Some(&rgb), Some(&th)).unwrap();
        assert_eq!(s.c, 4);
        assert!((s.at(0, 0, 0) - 1.0).abs() < 1e-6);
        assert!((s.at(0, 0, 3) - 0.1).abs() < 1e-6);
        assert!((s.at(0, 1, 3) - 1.0).abs() < 1e-6);
        // flat thermal -> all 0.1 (still above the reserved 0.0)
        let flat = Raster::from_vec(1, 1, 1, vec![5.0]);
        let s = stack_rgb_thermal(None, Some(&flat)).unwrap();
        assert!((s.at(0, 0, 0) - 0.1).abs() < 1e-6);
        assert!(stack_rgb_thermal(None, None).is_err());
    }

    #[test]
    fn nearest_resize_preserves_labels() {
        let src = Raster::from_vec(2, 2, 1, vec![0u8, 3, 255, 2]);
        let up = resize_nearest(&src, 4, 4);
        assert_eq!(up.at(0, 0, 0), 0);
        assert_eq!(up.at(0, 3, 0), 3);
        assert_eq!(up.at(3, 0, 0), 255);
        assert!(up.data.iter().all(|v| [0u8, 3, 255, 2].contains(v)));
    }

    #[test]
    fn area_resize_averages_blocks() {
        let src = Raster::from_vec(2, 2, 1, vec![0.0f32, 2.0, 4.0, 6.0]);
        let out = resize_area_f32(&src, 1, 1);
        assert!((out.at(0, 0, 0) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = SimpleRng::new(7);
        let mut b = SimpleRng::new(7);
        for _ in 0..100 {
            let x = a.random();
            assert_eq!(x, b.random());
            assert!((0.0..1.0).contains(&x));
        }
        let mut v: Vec<usize> = (0..10).collect();
        a.shuffle(&mut v);
        v.sort();
        assert_eq!(v, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn normalize_sample_coerces_shapes() {
        let img = Tensor::from_slice(&[1u8, 2, 3, 4]).reshape([2, 2]);
        let mut s = Sample::new(img, true, false);
        s.danger_mask = Some(Tensor::from_slice(&[0i64, 1, 2, 3]).reshape([2, 2]));
        s.person_boxes = Some(Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0]));
        let s = normalize_sample(s).unwrap();
        assert_eq!(s.image.size(), vec![2, 2, 1]);
        assert_eq!(s.danger_mask.as_ref().unwrap().kind(), Kind::Uint8);
        assert_eq!(s.person_boxes.as_ref().unwrap().size(), vec![1, 4]);
    }
}
