/*
 * extract.rs
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

//! Image list -> camera grouping -> SIFT -> incremental `FeatureStore` writes.

use crate::camera_init::init_camera;
use crate::exif_info::ExifInfo;
use crate::gray::{self, GrayImage};
use crate::sift::{CpuSift, SiftEngine, SiftOptions, SiftOutput};
use cumulus3d_core::store::PosePrior;
use cumulus3d_core::{CameraId, CameraModelKind, Error, FeatureStore, ImageId, Keypoint, Result, Vec3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Camera grouping mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CameraMode {
    /// A new camera per image.
    PerImage,
    /// All images in this call share one camera.
    Single,
    /// One camera per folder (parent path of the name). Sets position 0.
    #[default]
    PerFolder,
    /// Attach to an existing camera id (no parameter update). Sets position p >= 1.
    Existing(CameraId),
}

/// Image reading and camera options.
#[derive(Clone, Debug, PartialEq)]
pub struct ReaderOptions {
    /// Model type of new cameras.
    pub camera_model: CameraModelKind,
    /// Camera grouping mode.
    pub camera_mode: CameraMode,
    /// If given, used as-is for the new camera parameters (pre-focal flag set).
    pub camera_params: Option<Vec<f64>>,
    /// Focal length when EXIF is missing = factor x max(W, H).
    pub default_focal_length_factor: f64,
    /// Maximum image size (downscale criterion). Effective SIFT value 3200.
    pub max_image_size: usize,
    /// In `Existing` mode, fail if the image size differs from the camera (if disabled, convert without error).
    pub strict_existing_camera_size: bool,
    /// Record EXIF GPS as the prior position.
    pub read_pose_priors: bool,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            camera_model: CameraModelKind::OpenCv,
            camera_mode: CameraMode::PerFolder,
            camera_params: None,
            default_focal_length_factor: 1.2,
            max_image_size: 3200,
            strict_existing_camera_size: false,
            read_pose_priors: true,
        }
    }
}

/// Extraction option bundle.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ExtractionOptions {
    /// Image reading and camera options.
    pub reader: ReaderOptions,
    /// SIFT options.
    pub sift: SiftOptions,
    /// Process several images concurrently (memory ~ image count x pyramid).
    pub sequential_images: bool,
}

/// In-memory input image (already decoded).
#[derive(Clone, Debug)]
pub struct ImageSource {
    /// Image name in the store (relative path separated by '/').
    pub name: String,
    /// Grayscale image at input resolution.
    pub gray: GrayImage,
    /// EXIF summary of the image.
    pub exif: ExifInfo,
}

/// Per-image processing result.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageStatus {
    /// Features extracted and written to the store.
    Extracted {
        /// Issued image id.
        image_id: ImageId,
        /// Assigned camera id.
        camera_id: CameraId,
        /// Number of features recorded.
        num_features: usize,
    },
    /// Same name already exists and has features -> skipped.
    AlreadyExists {
        /// Existing image id.
        image_id: ImageId,
    },
    /// Reading/extraction failed (processing of other images continues).
    Failed {
        /// Failure reason.
        error: String,
    },
}

/// Processing report for one image.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageReport {
    /// Image name in the store.
    pub name: String,
    /// Processing result.
    pub status: ImageStatus,
}

/// Image list file: trim whitespace on each line, skip blank lines.
pub fn read_image_list(path: &Path) -> Result<Vec<String>> {
    let s = std::fs::read_to_string(path)?;
    Ok(s.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect())
}

/// Parent folder of the name ("camF/x.jpg" -> "camF").
pub fn image_folder(name: &str) -> &str {
    match name.rfind('/') {
        Some(i) => &name[..i],
        None => "",
    }
}

/// Input grayscale image -> (downscale, EXIF orientation rotation) -> SIFT -> keypoints in camera-size (cam_w, cam_h) coordinates.
pub fn extract_for_camera(
    backend: &dyn SiftEngine,
    gray: &GrayImage,
    orientation: Option<u32>,
    cam_w: u64,
    cam_h: u64,
    max_image_size: usize,
    sift: &SiftOptions,
) -> Result<(Vec<Keypoint>, SiftOutput)> {
    let (rw, rh) = gray::limited_size(gray.width, gray.height, max_image_size);
    let resized;
    let img = if (rw, rh) != (gray.width, gray.height) {
        resized = gray.resized(rw, rh);
        &resized
    } else {
        gray
    };
    let k = gray::orientation_to_rotation(orientation);
    let rotated;
    let input = if k != 0 {
        rotated = img.rotate_ccw(k);
        &rotated
    } else {
        img
    };
    let out = backend.extract(input, sift)?;
    let (iw, ih) = (input.width as f32, input.height as f32);
    let sx = cam_w as f32 / rw as f32;
    let sy = cam_h as f32 / rh as f32;
    let kps = out
        .features
        .iter()
        .map(|f| {
            let mut kp = f.keypoint();
            if k != 0 {
                kp = gray::rotate_keypoint_ccw(&kp, 4 - k, iw, ih);
            }
            if (rw as u64, rh as u64) != (cam_w, cam_h) {
                kp.rescale(sx, sy);
            }
            kp
        })
        .collect();
    Ok((kps, out))
}

/// Feature extractor: backend + store writes.
#[derive(Clone)]
pub struct FeatureExtractor {
    backend: Arc<dyn SiftEngine>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// One job after camera assignment.
struct Job {
    idx: usize,
    image_id: ImageId,
    camera_id: CameraId,
}

impl FeatureExtractor {
    /// CPU backend.
    pub fn new() -> Self {
        Self { backend: Arc::new(CpuSift::new()) }
    }

    /// Creates one with the given SIFT backend (e.g. GPU).
    pub fn with_backend(backend: Arc<dyn SiftEngine>) -> Self {
        Self { backend }
    }

    /// SIFT backend in use.
    pub fn backend(&self) -> &dyn SiftEngine {
        self.backend.as_ref()
    }

    /// Reads a file list (names relative to `root`) and adds them to the store. Names are processed in lexicographic order.
    /// With `Existing(N)`, fails immediately if camera N does not exist. Per-image failures are recorded in the report.
    pub fn extract_files(&self, store: &FeatureStore, root: &Path, names: &[String], opts: &ExtractionOptions) -> Result<Vec<ImageReport>> {
        let mut names: Vec<String> = names.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        names.sort();
        names.dedup();
        check_existing_camera(store, &opts.reader)?;
        let mut reports: Vec<Option<ImageReport>> = vec![None; names.len()];
        let mut todo = Vec::new();
        for (i, n) in names.iter().enumerate() {
            match already_done(store, n) {
                Some(id) => reports[i] = Some(ImageReport { name: n.clone(), status: ImageStatus::AlreadyExists { image_id: id } }),
                None => todo.push(i),
            }
        }
        let decode = |i: &usize| -> std::result::Result<ImageSource, String> {
            let n = &names[*i];
            let path = root.join(n);
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let exif = ExifInfo::from_bytes(&bytes);
            let img = image::load_from_memory(&bytes).map_err(|e| format!("{}: 디코딩 실패: {e}", path.display()))?;
            let gray = gray::to_gray(&img).map_err(|e| e.to_string())?;
            Ok(ImageSource { name: n.clone(), gray, exif })
        };
        let decoded: Vec<std::result::Result<ImageSource, String>> =
            if opts.sequential_images { todo.iter().map(decode).collect() } else { todo.par_iter().map(decode).collect() };
        let mut inputs = Vec::new();
        let mut input_slot = Vec::new();
        for (i, d) in todo.iter().zip(decoded) {
            match d {
                Ok(inp) => {
                    inputs.push(inp);
                    input_slot.push(*i);
                }
                Err(e) => reports[*i] = Some(ImageReport { name: names[*i].clone(), status: ImageStatus::Failed { error: e } }),
            }
        }
        let sub = self.extract_inputs(store, inputs, opts)?;
        // extract_inputs returns results in name-sorted order. The inputs are sorted too, so the order matches.
        for (slot, r) in input_slot.into_iter().zip(sub) {
            reports[slot] = Some(r);
        }
        Ok(reports.into_iter().flatten().collect())
    }

    /// Adds decoded images to the store (for per-position calls). Issues ids in lexicographic name order.
    pub fn extract_inputs(&self, store: &FeatureStore, mut inputs: Vec<ImageSource>, opts: &ExtractionOptions) -> Result<Vec<ImageReport>> {
        inputs.sort_by(|a, b| a.name.cmp(&b.name));
        let ro = &opts.reader;
        check_existing_camera(store, ro)?;
        let mut reports: Vec<ImageReport> =
            inputs.iter().map(|i| ImageReport { name: i.name.clone(), status: ImageStatus::Failed { error: String::new() } }).collect();

        // 1) Assign camera and image rows (sequential, deterministic ids).
        let mut jobs = Vec::new();
        let mut folder_cam: HashMap<String, CameraId> = HashMap::new();
        let mut single_cam: Option<CameraId> = None;
        for (idx, inp) in inputs.iter().enumerate() {
            if let Some(id) = already_done(store, &inp.name) {
                reports[idx].status = ImageStatus::AlreadyExists { image_id: id };
                continue;
            }
            let (w, h) = (inp.gray.width as u64, inp.gray.height as u64);
            let assigned: std::result::Result<(ImageId, CameraId), String> = (|| {
                // If only the row exists without features, just refill the features.
                if let Some(img) = store.image_by_name(&inp.name) {
                    return Ok((img.image_id, img.camera_id));
                }
                let new_cam = |store: &FeatureStore| -> std::result::Result<CameraId, String> {
                    let cam = init_camera(ro.camera_model, w, h, &inp.exif, ro.camera_params.as_deref(), ro.default_focal_length_factor)
                        .map_err(|e| e.to_string())?;
                    store.add_camera(cam).map_err(|e| e.to_string())
                };
                let check_size = |cid: CameraId| -> std::result::Result<(), String> {
                    let c = store.camera(cid).ok_or_else(|| format!("카메라 {cid} 없음"))?;
                    if (c.width, c.height) != (w, h) {
                        return Err(format!("크기 불일치: 영상 {w}x{h}, 카메라 {cid} {}x{}", c.width, c.height));
                    }
                    Ok(())
                };
                let cam = match ro.camera_mode {
                    CameraMode::PerImage => new_cam(store)?,
                    CameraMode::Single => match single_cam {
                        Some(c) => {
                            check_size(c)?;
                            c
                        }
                        None => {
                            let c = new_cam(store)?;
                            single_cam = Some(c);
                            c
                        }
                    },
                    // Design decision: group via a folder->camera map. On a sorted list this equals the
                    // "previous camera + first-seen folder" rule.
                    CameraMode::PerFolder => {
                        let folder = image_folder(&inp.name).to_string();
                        match folder_cam.get(&folder) {
                            Some(&c) => {
                                check_size(c)?;
                                c
                            }
                            None => {
                                let c = new_cam(store)?;
                                folder_cam.insert(folder, c);
                                c
                            }
                        }
                    }
                    CameraMode::Existing(n) => {
                        if ro.strict_existing_camera_size {
                            check_size(n)?;
                        }
                        n
                    }
                };
                let id = store.add_image(&inp.name, cam).map_err(|e| e.to_string())?;
                if ro.read_pose_priors {
                    if let Some((lat, lon, alt)) = inp.exif.gps {
                        let gravity = gray::orientation_gravity(inp.exif.orientation).map(|(gx, gy)| Vec3::new(gx, gy, 0.0));
                        store.set_pose_prior(id, PosePrior { position: Vec3::new(lat, lon, alt), coordinate_system: 0, gravity });
                    }
                }
                Ok((id, cam))
            })();
            match assigned {
                Ok((image_id, camera_id)) => jobs.push(Job { idx, image_id, camera_id }),
                Err(e) => reports[idx].status = ImageStatus::Failed { error: e },
            }
        }

        // 2) SIFT (images in parallel) -> store.
        let run = |job: &Job| -> (usize, ImageStatus) {
            let inp = &inputs[job.idx];
            let Some(cam) = store.camera(job.camera_id) else {
                return (job.idx, ImageStatus::Failed { error: format!("카메라 {} 없음", job.camera_id) });
            };
            match extract_for_camera(
                self.backend.as_ref(),
                &inp.gray,
                inp.exif.orientation,
                cam.width,
                cam.height,
                ro.max_image_size,
                &opts.sift,
            ) {
                Ok((kps, out)) => {
                    let n = kps.len();
                    store.set_keypoints(job.image_id, kps);
                    store.set_descriptors(job.image_id, out.descriptors);
                    (job.idx, ImageStatus::Extracted { image_id: job.image_id, camera_id: job.camera_id, num_features: n })
                }
                Err(e) => (job.idx, ImageStatus::Failed { error: e.to_string() }),
            }
        };
        let results: Vec<(usize, ImageStatus)> =
            if opts.sequential_images { jobs.iter().map(run).collect() } else { jobs.par_iter().map(run).collect() };
        for (idx, st) in results {
            reports[idx].status = st;
        }
        Ok(reports)
    }
}

/// The id if it has a name and both keypoints and descriptors.
fn already_done(store: &FeatureStore, name: &str) -> Option<ImageId> {
    let id = store.image_id_by_name(name)?;
    (store.exists_keypoints(id) && store.exists_descriptors(id)).then_some(id)
}

fn check_existing_camera(store: &FeatureStore, ro: &ReaderOptions) -> Result<Option<CameraId>> {
    match ro.camera_mode {
        CameraMode::Existing(n) => {
            if store.camera(n).is_none() {
                return Err(Error::NotFound(format!("existing_camera_id {n}")));
            }
            Ok(Some(n))
        }
        _ => Ok(None),
    }
}
