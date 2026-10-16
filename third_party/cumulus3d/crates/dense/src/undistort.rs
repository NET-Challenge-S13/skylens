/*
 * undistort.rs
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

//! Undistortion: PINHOLE camera computation, per-camera resampling-map cache, image resampling, sparse model conversion.

use crate::image::{GrayImage, ImageBuffer, Integral};
use cumulus3d_core::interop;
use cumulus3d_core::{Camera, CameraId, CameraModelKind, Error, Image, ImageId, Point3D, Reconstruction, Result, Vec2};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Undistortion options (the pipeline uses `max_image_size = 960`).
#[derive(Clone, Debug, PartialEq)]
pub struct UndistortOptions {
    /// Allowed fraction of blank pixels at the border (0 = no blank pixels, 1 = whole input included).
    pub blank_pixels: f64,
    /// Lower bound of the output size scale.
    pub min_scale: f64,
    /// Upper bound of the output size scale.
    pub max_scale: f64,
    /// No limit if ≤ 0 (conventionally −1).
    pub max_image_size: i64,
    /// Region of interest, left (fraction of image width).
    pub roi_min_x: f64,
    /// Region of interest, top (fraction of image height).
    pub roi_min_y: f64,
    /// Region of interest, right (fraction of image width).
    pub roi_max_x: f64,
    /// Region of interest, bottom (fraction of image height).
    pub roi_max_y: f64,
    /// Number of source (neighbor) images written to patch-match.cfg.
    pub num_patch_match_src_images: usize,
    /// JPEG quality (−1 = 100).
    pub jpeg_quality: i32,
}

impl Default for UndistortOptions {
    fn default() -> Self {
        Self {
            blank_pixels: 0.0,
            min_scale: 0.2,
            max_scale: 2.0,
            max_image_size: -1,
            roi_min_x: 0.0,
            roi_min_y: 0.0,
            roi_max_x: 1.0,
            roi_max_y: 1.0,
            num_patch_match_src_images: 20,
            jpeg_quality: -1,
        }
    }
}

impl UndistortOptions {
    /// User pipeline values (max_image_size 960).
    pub fn pipeline() -> Self {
        Self { max_image_size: 960, ..Self::default() }
    }
    fn roi_enabled(&self) -> bool {
        self.roi_min_x != 0.0 || self.roi_min_y != 0.0 || self.roi_max_x != 1.0 || self.roi_max_y != 1.0
    }
    fn hash_into<H: Hasher>(&self, h: &mut H) {
        for v in [self.blank_pixels, self.min_scale, self.max_scale, self.roi_min_x, self.roi_min_y, self.roi_max_x, self.roi_max_y] {
            v.to_bits().hash(h);
        }
        self.max_image_size.hash(h);
    }
}

fn is_perspective(m: CameraModelKind) -> bool {
    // All models this crate knows (0..4) are perspective models.
    matches!(
        m,
        CameraModelKind::SingleFocalPinhole
            | CameraModelKind::Pinhole
            | CameraModelKind::SingleFocalRadial
            | CameraModelKind::Radial
            | CameraModelKind::OpenCv
    )
}

fn has_no_distortion(cam: &Camera) -> bool {
    cam.model.extra_param_slots().iter().all(|&i| cam.params[i].abs() <= 1e-8)
}

/// Computes the distortion-free PINHOLE camera. The camera id is kept.
pub fn undistorted_camera(cam: &Camera, opts: &UndistortOptions) -> Result<Camera> {
    if !is_perspective(cam.model) {
        return Err(Error::Unsupported(format!("왜곡 보정: 모델 {}", cam.model.name())));
    }
    let (w, h) = (cam.width as f64, cam.height as f64);
    let (fx, fy) = (cam.focal_length_x(), cam.focal_length_y());
    let (mut cx, mut cy) = (cam.principal_point_x(), cam.principal_point_y());
    let mut out_w = cam.width;
    let mut out_h = cam.height;

    // ROI
    let (mut x_min, mut y_min, mut x_max, mut y_max) = (0i64, 0i64, cam.width as i64, cam.height as i64);
    let roi = opts.roi_enabled();
    if roi {
        x_min = (opts.roi_min_x * w).round() as i64;
        y_min = (opts.roi_min_y * h).round() as i64;
        x_max = (opts.roi_max_x * w).round() as i64;
        y_max = (opts.roi_max_y * h).round() as i64;
        x_min = x_min.min(cam.width as i64 - 1);
        y_min = y_min.min(cam.height as i64 - 1);
        x_max = x_max.max(x_min + 1);
        y_max = y_max.max(y_min + 1);
        out_w = (x_max - x_min) as u64;
        out_h = (y_max - y_min) as u64;
        cx -= x_min as f64;
        cy -= y_min as f64;
    }

    let pinhole_proj = |uv: &Vec2, cx: f64, cy: f64| Vec2::new(fx * uv.x + cx, fy * uv.y + cy);

    if roi || !matches!(cam.model, CameraModelKind::SingleFocalPinhole | CameraModelKind::Pinhole) {
        let (mut left_min, mut left_max) = (f64::MAX, f64::MIN);
        let (mut right_min, mut right_max) = (f64::MAX, f64::MIN);
        for y in y_min..y_max {
            let yc = y as f64 + 0.5;
            if let Some(uv) = cam.img_to_normalized(&Vec2::new(0.5, yc)) {
                let p = pinhole_proj(&uv, cx, cy);
                left_min = left_min.min(p.x);
                left_max = left_max.max(p.x);
            }
            if let Some(uv) = cam.img_to_normalized(&Vec2::new(w - 0.5, yc)) {
                let p = pinhole_proj(&uv, cx, cy);
                right_min = right_min.min(p.x);
                right_max = right_max.max(p.x);
            }
        }
        let (mut top_min, mut top_max) = (f64::MAX, f64::MIN);
        let (mut bottom_min, mut bottom_max) = (f64::MAX, f64::MIN);
        for x in x_min..x_max {
            let xc = x as f64 + 0.5;
            if let Some(uv) = cam.img_to_normalized(&Vec2::new(xc, 0.5)) {
                let p = pinhole_proj(&uv, cx, cy);
                top_min = top_min.min(p.y);
                top_max = top_max.max(p.y);
            }
            if let Some(uv) = cam.img_to_normalized(&Vec2::new(xc, h - 0.5)) {
                let p = pinhole_proj(&uv, cx, cy);
                bottom_min = bottom_min.min(p.y);
                bottom_max = bottom_max.max(p.y);
            }
        }
        let (w0, h0) = (out_w as f64, out_h as f64);
        let scale_x_min = (cx / (cx - left_min)).min((w0 - 0.5 - cx) / (right_max - cx));
        let scale_y_min = (cy / (cy - top_min)).min((h0 - 0.5 - cy) / (bottom_max - cy));
        let scale_x_max = (cx / (cx - left_max)).max((w0 - 0.5 - cx) / (right_min - cx));
        let scale_y_max = (cy / (cy - top_max)).max((h0 - 0.5 - cy) / (bottom_min - cy));
        let b = opts.blank_pixels;
        let scale_x = (1.0 / (scale_x_min * b + scale_x_max * (1.0 - b))).clamp(opts.min_scale, opts.max_scale);
        let scale_y = (1.0 / (scale_y_min * b + scale_y_max * (1.0 - b))).clamp(opts.min_scale, opts.max_scale);
        let nw = (scale_x * w0).max(1.0).floor() as u64;
        let nh = (scale_y * h0).max(1.0).floor() as u64;
        cx *= nw as f64 / w0;
        cy *= nh as f64 / h0;
        out_w = nw;
        out_h = nh;
    }

    let mut out = Camera::new(cam.camera_id, CameraModelKind::Pinhole, out_w, out_h, vec![fx, fy, cx, cy])?;
    apply_max_image_size(&mut out, opts.max_image_size);
    Ok(out)
}

/// max_image_size rule: only shrinks; rounding goes away from 0.
fn apply_max_image_size(cam: &mut Camera, max_size: i64) {
    if max_size <= 0 {
        return;
    }
    let m = max_size as f64;
    let s = (m / cam.width as f64).min(m / cam.height as f64);
    if s < 1.0 {
        let nw = (s * cam.width as f64).round() as u64;
        let nh = (s * cam.height as f64).round() as u64;
        cam.rescale_to(nw.max(1), nh.max(1));
    }
}

/// Undistortion result of one camera: PINHOLE camera + resampling map at the input resolution.
#[derive(Debug)]
pub struct CameraUndistortion {
    /// Input (distorted) camera.
    pub source: Camera,
    /// Undistorted PINHOLE camera.
    pub pinhole: Camera,
    /// For each pixel of the auxiliary camera (input size), the input image sample position (sx−0.5, sy−0.5). NaN on failure.
    lut: Vec<[f32; 2]>,
}

impl CameraUndistortion {
    /// Camera computation + LUT construction.
    pub fn new(source: &Camera, opts: &UndistortOptions) -> Result<Self> {
        let pinhole = undistorted_camera(source, opts)?;
        let (sw, sh) = (source.width as usize, source.height as usize);
        let mut aux = pinhole.clone();
        aux.rescale_to(source.width, source.height);
        let (fx, fy, cx, cy) = (aux.focal_length_x(), aux.focal_length_y(), aux.principal_point_x(), aux.principal_point_y());
        let mut lut = vec![[f32::NAN; 2]; sw * sh];
        lut.par_chunks_mut(sw).enumerate().for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                let uv = Vec2::new((x as f64 + 0.5 - cx) / fx, (y as f64 + 0.5 - cy) / fy);
                let p = source.normalized_to_img(&uv);
                if p.x.is_finite() && p.y.is_finite() {
                    *v = [(p.x - 0.5) as f32, (p.y - 0.5) as f32];
                }
            }
        });
        Ok(Self { source: source.clone(), pinhole, lut })
    }

    /// Image resampling: bilinear warp at the input resolution (black outside the range) → area-average downscale.
    pub fn undistort_image(&self, img: &ImageBuffer) -> Result<ImageBuffer> {
        let (sw, sh) = (self.source.width as usize, self.source.height as usize);
        if img.width != sw || img.height != sh {
            return Err(Error::InvalidArgument(format!("왜곡 보정: 영상 크기 {}x{} != 카메라 {}x{}", img.width, img.height, sw, sh)));
        }
        let ch = img.channels;
        let mut warped = ImageBuffer::new(sw, sh, ch);
        warped.data.par_chunks_mut(sw * ch).enumerate().for_each(|(y, row)| {
            for x in 0..sw {
                let [fx, fy] = self.lut[y * sw + x];
                if !fx.is_finite() {
                    continue;
                }
                let x0f = fx.floor();
                let y0f = fy.floor();
                let (x0, y0) = (x0f as i64, y0f as i64);
                if x0 < 0 || y0 < 0 || x0 + 1 >= sw as i64 || y0 + 1 >= sh as i64 {
                    continue;
                }
                let (ax, ay) = (fx - x0f, fy - y0f);
                let (x0, y0) = (x0 as usize, y0 as usize);
                for c in 0..ch {
                    let p = |xx: usize, yy: usize| img.data[(yy * sw + xx) * ch + c] as f32;
                    let top = p(x0, y0) * (1.0 - ax) + p(x0 + 1, y0) * ax;
                    let bot = p(x0, y0 + 1) * (1.0 - ax) + p(x0 + 1, y0 + 1) * ax;
                    let v = top * (1.0 - ay) + bot * ay;
                    row[x * ch + c] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        });
        let (ow, oh) = (self.pinhole.width as usize, self.pinhole.height as usize);
        if ow == sw && oh == sh {
            return Ok(warped);
        }
        // Design decision: downscaling uses an exact area average.
        let mut out = ImageBuffer::new(ow, oh, ch);
        let planes: Vec<Vec<f32>> = (0..ch)
            .into_par_iter()
            .map(|c| {
                let g = GrayImage { width: sw, height: sh, data: (0..sw * sh).map(|i| warped.data[i * ch + c] as f32).collect() };
                Integral::new(&g).resample(ow, oh).data
            })
            .collect();
        for (c, plane) in planes.iter().enumerate() {
            for (i, v) in plane.iter().enumerate() {
                out.data[i * ch + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(out)
    }
}

/// Pluggable image resampling (for example on the GPU). Must produce the same geometry as
/// [`CameraUndistortion::undistort_image`]: an image of `u.pinhole` size with the input's channel count.
pub trait UndistortBackend: Send + Sync {
    /// Undistorts (and downscales) one image of the source camera size.
    fn undistort_image(&self, u: &CameraUndistortion, img: &ImageBuffer) -> Result<ImageBuffer>;
}

fn camera_key(cam: &Camera, opts: &UndistortOptions) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    cam.model.id().hash(&mut h);
    cam.width.hash(&mut h);
    cam.height.hash(&mut h);
    for p in &cam.params {
        p.to_bits().hash(&mut h);
    }
    opts.hash_into(&mut h);
    h.finish()
}

/// Camera parameter hash → undistortion result (with LUT) cache. Shared between zones.
#[derive(Default)]
pub struct UndistortCache {
    map: Mutex<HashMap<u64, Arc<CameraUndistortion>>>,
}

impl UndistortCache {
    /// Empty cache.
    pub fn new() -> Self {
        Self::default()
    }
    /// Looks up in the cache or computes anew.
    pub fn get(&self, cam: &Camera, opts: &UndistortOptions) -> Result<Arc<CameraUndistortion>> {
        let key = camera_key(cam, opts);
        if let Some(v) = self.map.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(v.clone());
        }
        let v = Arc::new(CameraUndistortion::new(cam, opts)?);
        self.map.lock().unwrap_or_else(|e| e.into_inner()).insert(key, v.clone());
        Ok(v)
    }
    /// Number of cached cameras.
    pub fn len(&self) -> usize {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Sparse model conversion: cameras become PINHOLE, 2D observations are mapped to the new camera coordinates (NaN on failure).
/// Poses, 3D points, tracks, names and ids are unchanged.
pub fn undistort_reconstruction(rec: &Reconstruction, opts: &UndistortOptions, cache: &UndistortCache) -> Result<Reconstruction> {
    let mut new_cams: BTreeMap<CameraId, Camera> = BTreeMap::new();
    for (id, cam) in rec.cameras() {
        let nc = if has_no_distortion(cam) && opts.max_image_size <= 0 {
            cam.clone()
        } else {
            // The cache is shared by parameter hash (different cameras with equal values), so restore the id to this camera's.
            let mut c = cache.get(cam, opts)?.pinhole.clone();
            c.camera_id = *id;
            c
        };
        new_cams.insert(*id, nc);
    }
    let mut out = Reconstruction::new();
    for c in new_cams.values() {
        out.add_camera(c.clone())?;
    }
    for r in rec.rigs().values() {
        out.add_rig(r.clone())?;
    }
    for f in rec.frames().values() {
        out.add_frame(f.clone())?;
    }
    let images: Vec<&Image> = rec.images().collect();
    let converted: Vec<Image> = images
        .par_iter()
        .map(|im| {
            let src = &rec.cameras()[&im.camera_id];
            let dst = &new_cams[&im.camera_id];
            let same = src == dst;
            let pts = im.points2d().iter().map(|p| {
                if same {
                    return p.xy;
                }
                match src.img_to_normalized(&p.xy) {
                    Some(uv) => dst.normalized_to_img(&uv),
                    None => Vec2::new(f64::NAN, f64::NAN),
                }
            });
            let mut ni = Image::new(im.image_id, im.name.clone(), im.camera_id, pts);
            ni.frame_id = im.frame_id;
            ni
        })
        .collect();
    for im in converted {
        out.add_image(im)?;
    }
    for (id, p) in rec.points3d() {
        let np = Point3D { xyz: p.xyz, color: p.color, error: p.error, track: p.track.clone() };
        out.add_point3d_with_id(id, np)?;
    }
    for fid in rec.registered_frames() {
        out.register_frame(*fid)?;
    }
    Ok(out)
}

/// In-memory undistortion result.
pub struct UndistortResult {
    /// Converted sparse model (PINHOLE).
    pub reconstruction: Reconstruction,
    /// Undistorted image per registered image. Images that failed to read are absent.
    pub images: BTreeMap<ImageId, Arc<ImageBuffer>>,
    /// Ids of images whose processing failed (e.g. read failure).
    pub failed: Vec<ImageId>,
}

/// Undistorts all registered images. `load` returns the input image (treated as a failure if absent).
pub fn undistort<F>(rec: &Reconstruction, opts: &UndistortOptions, cache: &UndistortCache, load: F) -> Result<UndistortResult>
where
    F: Fn(&Image) -> Option<Arc<ImageBuffer>> + Sync,
{
    undistort_with(rec, opts, cache, load, None)
}

/// Like [`undistort`], with an optional image resampling backend (`None` = the CPU path).
pub fn undistort_with<F>(
    rec: &Reconstruction,
    opts: &UndistortOptions,
    cache: &UndistortCache,
    load: F,
    backend: Option<&dyn UndistortBackend>,
) -> Result<UndistortResult>
where
    F: Fn(&Image) -> Option<Arc<ImageBuffer>> + Sync,
{
    let reconstruction = undistort_reconstruction(rec, opts, cache)?;
    let ids = rec.registered_images();
    // Fill the camera LUTs first to avoid duplicate computation during parallel image processing.
    for im in ids.iter().filter_map(|i| rec.image(*i)) {
        cache.get(&rec.cameras()[&im.camera_id], opts)?;
    }
    let results: Vec<(ImageId, Option<Arc<ImageBuffer>>)> = ids
        .par_iter()
        .map(|&iid| {
            let Some(im) = rec.image(iid) else { return (iid, None) };
            let cam = &rec.cameras()[&im.camera_id];
            let Some(src) = load(im) else { return (iid, None) };
            let r = cache
                .get(cam, opts)
                .and_then(|u| match backend {
                    Some(b) => b.undistort_image(&u, &src),
                    None => u.undistort_image(&src),
                })
                .ok()
                .map(Arc::new);
            (iid, r)
        })
        .collect();
    let mut images = BTreeMap::new();
    let mut failed = Vec::new();
    for (iid, r) in results {
        match r {
            Some(b) => {
                images.insert(iid, b);
            }
            None => failed.push(iid),
        }
    }
    Ok(UndistortResult { reconstruction, images, failed })
}

/// Reads from the input image folder and undistorts.
pub fn undistort_from_dir(
    rec: &Reconstruction,
    image_dir: impl AsRef<Path>,
    opts: &UndistortOptions,
    cache: &UndistortCache,
) -> Result<UndistortResult> {
    let dir = image_dir.as_ref().to_path_buf();
    undistort(rec, opts, cache, |im| ImageBuffer::load(dir.join(&im.name)).ok().map(Arc::new))
}

/// Writes the undistortion result as a dense reconstruction working folder: images/ (including subfolders), sparse/ (binary model),
/// empty stereo/{depth_maps,normal_maps,consistency_graphs}/ folders and config files.
/// Folder layout and file formats are defined by [`cumulus3d_core::interop`].
pub fn write_undistorted_workspace(result: &UndistortResult, out_dir: impl AsRef<Path>, opts: &UndistortOptions) -> Result<()> {
    let out = out_dir.as_ref();
    let rec = &result.reconstruction;
    std::fs::create_dir_all(out.join("images"))?;
    std::fs::create_dir_all(out.join("sparse"))?;
    interop::create_stereo_dirs(out, Path::new(""))?;
    let q = if opts.jpeg_quality < 0 { 100 } else { opts.jpeg_quality.clamp(1, 100) as u8 };
    let names: Vec<(ImageId, String)> =
        rec.registered_images().into_iter().filter_map(|i| rec.image(i).map(|im| (i, im.name.clone()))).collect();
    names.par_iter().filter_map(|(iid, name)| result.images.get(iid).map(|b| (name, b))).try_for_each(|(name, buf)| -> Result<()> {
        let p = out.join("images").join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(parent) = Path::new(name).parent() {
            interop::create_stereo_dirs(out, parent)?;
        }
        buf.save(p, q)
    })?;
    interop::write_model_binary(rec, out.join("sparse"), interop::ImageOrder::Registration)?;
    let written = names.iter().filter(|(iid, _)| result.images.contains_key(iid)).map(|(_, n)| n.as_str());
    interop::write_stereo_configs(out, written, opts.num_patch_match_src_images)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Marker;
    impl UndistortBackend for Marker {
        fn undistort_image(&self, u: &CameraUndistortion, _img: &ImageBuffer) -> Result<ImageBuffer> {
            let mut o = ImageBuffer::new(u.pinhole.width as usize, u.pinhole.height as usize, 1);
            o.data.fill(77);
            Ok(o)
        }
    }

    #[test]
    fn backend_trait_dispatch() {
        let cam = Camera::new(1, CameraModelKind::OpenCv, 64, 48, vec![60.0, 60.0, 32.0, 24.0, 0.05, 0.0, 0.0, 0.0]).unwrap();
        let u = CameraUndistortion::new(&cam, &UndistortOptions::default()).unwrap();
        let b: &dyn UndistortBackend = &Marker;
        let out = b.undistort_image(&u, &ImageBuffer::new(64, 48, 1)).unwrap();
        assert_eq!((out.width as u64, out.height as u64), (u.pinhole.width, u.pinhole.height));
        assert!(out.data.iter().all(|&v| v == 77));
    }

    fn front_cam() -> Camera {
        Camera::new(1, CameraModelKind::OpenCv, 2048, 1152, vec![1609.22, 1608.21, 1024.0, 576.0, 0.0824, -0.0849, 0.0022, 0.0007]).unwrap()
    }

    #[test]
    fn spec_example_camera() {
        let cam = front_cam();
        let mut o = UndistortOptions::pipeline();
        o.max_image_size = -1;
        let mid = undistorted_camera(&cam, &o).unwrap();
        assert_eq!((mid.width, mid.height), (2002, 1123));
        assert!((mid.principal_point_x() - 1001.0).abs() < 1e-9);
        assert!((mid.principal_point_y() - 561.5).abs() < 1e-9);
        let c = undistorted_camera(&cam, &UndistortOptions::pipeline()).unwrap();
        assert_eq!(c.model, CameraModelKind::Pinhole);
        assert_eq!((c.width, c.height), (960, 539));
        let fx = 1609.22 * 960.0 / 2002.0;
        let fy = 1608.21 * 539.0 / 1123.0;
        assert!((c.params[0] - fx).abs() / fx < 1e-12);
        assert!((c.params[1] - fy).abs() / fy < 1e-12);
        assert!((c.params[0] - 771.654).abs() / 771.654 < 1e-6, "fx {}", c.params[0]);
        assert!((c.params[1] - 771.884).abs() / 771.884 < 1e-6, "fy {}", c.params[1]);
        assert!((c.params[2] - 480.0).abs() < 1e-9 && (c.params[3] - 269.5).abs() < 1e-9);
    }

    #[test]
    fn lut_maps_pinhole_to_source() {
        // Reprojecting the undistorted image positions of a small camera back to the input camera must match the LUT.
        let cam = Camera::new(1, CameraModelKind::OpenCv, 200, 120, vec![160.0, 160.0, 100.0, 60.0, 0.05, -0.02, 0.001, 0.0005]).unwrap();
        let u = CameraUndistortion::new(&cam, &UndistortOptions::default()).unwrap();
        // Checkerboard of input size → check the geometric alignment of the undistorted image: pixel values near the center match.
        let mut img = ImageBuffer::new(200, 120, 1);
        for y in 0..120 {
            for x in 0..200 {
                img.data[y * 200 + x] = (((x as f64 * 0.3).sin() * 60.0) + 128.0) as u8;
            }
        }
        let out = u.undistort_image(&img).unwrap();
        assert_eq!((out.width as u64, out.height as u64), (u.pinhole.width, u.pinhole.height));
        // Center pixel of the undistorted image → input position
        let (x, y) = (u.pinhole.width as f64 / 2.0, u.pinhole.height as f64 / 2.0);
        let uv = u.pinhole.img_to_normalized(&Vec2::new(x, y)).unwrap();
        let p = cam.normalized_to_img(&uv);
        let expect = img.get((p.x - 0.5).round() as usize, (p.y - 0.5).round() as usize, 0) as i32;
        let got = out.get(x as usize, y as usize, 0) as i32;
        assert!((expect - got).abs() <= 25, "{expect} vs {got}");
    }
}
