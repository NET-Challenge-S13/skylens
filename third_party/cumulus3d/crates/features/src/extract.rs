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

//! 영상 목록 → 카메라 묶기 → SIFT → `FeatureStore` 증분 기록.

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

/// 카메라 묶기 방식.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CameraMode {
    /// 영상마다 새 카메라.
    PerImage,
    /// 이번 호출의 모든 영상이 카메라 하나.
    Single,
    /// 폴더(이름의 부모 경로)마다 카메라 하나. 위치 0 설정.
    #[default]
    PerFolder,
    /// 기존 카메라 id 에 붙임(파라미터 갱신 없음). 위치 p ≥ 1 설정.
    Existing(CameraId),
}

/// 영상 읽기·카메라 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct ReaderOptions {
    /// 새 카메라의 모델 종류.
    pub camera_model: CameraModelKind,
    /// 카메라 묶기 방식.
    pub camera_mode: CameraMode,
    /// 주어지면 새 카메라 파라미터로 그대로 사용(사전 초점 플래그 참).
    pub camera_params: Option<Vec<f64>>,
    /// EXIF 가 없을 때 초점거리 = 계수 × max(W, H).
    pub default_focal_length_factor: f64,
    /// 최대 영상 크기(축소 기준). SIFT 실효값 3200.
    pub max_image_size: usize,
    /// `Existing` 모드에서 영상 크기가 카메라와 다르면 실패 처리(끄면 오류 없이 환산).
    pub strict_existing_camera_size: bool,
    /// EXIF GPS 를 사전 위치로 기록.
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

/// 추출 옵션 묶음.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ExtractionOptions {
    /// 영상 읽기·카메라 옵션.
    pub reader: ReaderOptions,
    /// SIFT 옵션.
    pub sift: SiftOptions,
    /// 여러 영상을 동시에 처리(메모리 ≈ 영상 수 × 피라미드).
    pub sequential_images: bool,
}

/// 메모리 입력 영상(디코딩 완료).
#[derive(Clone, Debug)]
pub struct ImageSource {
    /// 저장소 영상 이름('/' 구분 상대 경로).
    pub name: String,
    /// 입력 해상도 회색 영상.
    pub gray: GrayImage,
    /// 영상의 EXIF 요약.
    pub exif: ExifInfo,
}

/// 영상별 처리 결과.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageStatus {
    /// 특징을 추출해 저장소에 기록함.
    Extracted {
        /// 발급된 영상 id.
        image_id: ImageId,
        /// 배정된 카메라 id.
        camera_id: CameraId,
        /// 기록한 특징 수.
        num_features: usize,
    },
    /// 같은 이름이 이미 있고 특징도 있음 → 건너뜀.
    AlreadyExists {
        /// 기존 영상 id.
        image_id: ImageId,
    },
    /// 읽기·추출 실패(다른 영상 처리는 계속).
    Failed {
        /// 실패 사유.
        error: String,
    },
}

/// 영상 하나의 처리 보고.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageReport {
    /// 저장소 영상 이름.
    pub name: String,
    /// 처리 결과.
    pub status: ImageStatus,
}

/// 이미지 목록 파일: 줄마다 앞뒤 공백 제거, 빈 줄 건너뜀.
pub fn read_image_list(path: &Path) -> Result<Vec<String>> {
    let s = std::fs::read_to_string(path)?;
    Ok(s.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect())
}

/// 이름의 부모 폴더("camF/x.jpg" → "camF").
pub fn image_folder(name: &str) -> &str {
    match name.rfind('/') {
        Some(i) => &name[..i],
        None => "",
    }
}

/// 입력 회색 영상 → (축소, EXIF 방향 회전) → SIFT → 키포인트를 카메라 크기 (cam_w, cam_h) 좌표로.
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

/// 특징 추출기: 백엔드 + 저장소 기록.
#[derive(Clone)]
pub struct FeatureExtractor {
    backend: Arc<dyn SiftEngine>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// 카메라 배정을 마친 작업 하나.
struct Job {
    idx: usize,
    image_id: ImageId,
    camera_id: CameraId,
}

impl FeatureExtractor {
    /// CPU 백엔드.
    pub fn new() -> Self {
        Self { backend: Arc::new(CpuSift::new()) }
    }

    /// 지정한 SIFT 백엔드(예: GPU)로 만든다.
    pub fn with_backend(backend: Arc<dyn SiftEngine>) -> Self {
        Self { backend }
    }

    /// 사용 중인 SIFT 백엔드.
    pub fn backend(&self) -> &dyn SiftEngine {
        self.backend.as_ref()
    }

    /// 파일 목록(`root` 기준 상대 이름)을 읽어 저장소에 추가. 이름은 사전순으로 처리.
    /// `Existing(N)` 에서 카메라 N 이 없으면 즉시 오류. 영상별 실패는 보고서에 담는다.
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
        // extract_inputs 는 이름 정렬 순서로 돌려준다. 입력도 정렬돼 있으므로 순서가 같다.
        for (slot, r) in input_slot.into_iter().zip(sub) {
            reports[slot] = Some(r);
        }
        Ok(reports.into_iter().flatten().collect())
    }

    /// 디코딩된 영상들을 저장소에 추가(위치 하나 단위 호출용). 이름 사전순으로 id 를 발급한다.
    pub fn extract_inputs(&self, store: &FeatureStore, mut inputs: Vec<ImageSource>, opts: &ExtractionOptions) -> Result<Vec<ImageReport>> {
        inputs.sort_by(|a, b| a.name.cmp(&b.name));
        let ro = &opts.reader;
        check_existing_camera(store, ro)?;
        let mut reports: Vec<ImageReport> =
            inputs.iter().map(|i| ImageReport { name: i.name.clone(), status: ImageStatus::Failed { error: String::new() } }).collect();

        // 1) 카메라·영상 행 배정(순차, 결정적 id).
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
                // 행만 있고 특징이 없으면 특징만 다시 채운다.
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
                    // 설계 결정: 폴더→카메라 사전으로 묶는다. 정렬된 목록에서는 "직전 카메라 + 처음 본 폴더" 규칙과 같다.
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

        // 2) SIFT (영상 병렬) → 저장.
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

/// 이름이 있고 키포인트·기술자가 모두 있으면 그 id.
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
