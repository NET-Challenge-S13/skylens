/*
 * store.rs
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

//! 메모리 특징 저장소(영상·카메라·키포인트·기술자·매칭·두 뷰 기하). 이진 파일로 저장·적재 가능.
//!
//! 스레드 안전(RwLock). 큰 자료(키포인트·기술자·매칭·기하)는 `Arc` 로 공유해 조회 시 복사가 없다.
//! 짝 자료는 항상 "작은 id → 큰 id" 방향으로 저장하고, 반대 방향 조회 시 역변환해 돌려준다.

use crate::camera::{Camera, CameraModelKind};
use crate::error::{Error, Result};
use crate::features::{Descriptors, FeatureMatch, Keypoint, TwoViewGeometry, TwoViewGeometryConfig};
use crate::geometry::{Mat3, Rigid3, Vec3};
use crate::ids::*;
use crate::io::binary::{check_count, ReadLe, WriteLe};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// 저장소의 영상 행.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreImage {
    /// 영상 id.
    pub image_id: ImageId,
    /// 영상 이름(고유).
    pub name: String,
    /// 카메라 id.
    pub camera_id: CameraId,
}

/// 사전 위치(WGS84 위도·경도(도), 고도(m)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosePrior {
    /// (위도, 경도, 고도) 또는 좌표계에 따른 위치.
    pub position: Vec3,
    /// 0 = WGS84.
    pub coordinate_system: i32,
    /// 중력 방향(선택).
    pub gravity: Option<Vec3>,
}

#[derive(Default)]
struct Inner {
    cameras: BTreeMap<CameraId, Camera>,
    images: BTreeMap<ImageId, StoreImage>,
    name_to_id: HashMap<String, ImageId>,
    keypoints: HashMap<ImageId, Arc<Vec<Keypoint>>>,
    descriptors: HashMap<ImageId, Arc<Descriptors>>,
    pose_priors: HashMap<ImageId, PosePrior>,
    matches: HashMap<PairId, Arc<Vec<FeatureMatch>>>,
    two_view: HashMap<PairId, Arc<TwoViewGeometry>>,
    /// 두 뷰 기하 기록 로그(증분 소비용). 같은 짝이 다시 기록되면 중복될 수 있다.
    tvg_log: Vec<PairId>,
}

/// 메모리 특징 저장소.
#[derive(Default)]
pub struct FeatureStore {
    inner: RwLock<Inner>,
}

fn swap_matches(m: &[FeatureMatch]) -> Vec<FeatureMatch> {
    m.iter().map(|x| x.swapped()).collect()
}

impl FeatureStore {
    /// 빈 저장소.
    pub fn new() -> Self {
        Self::default()
    }

    fn r(&self) -> RwLockReadGuard<'_, Inner> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }
    fn w(&self) -> RwLockWriteGuard<'_, Inner> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }

    // ---------- 카메라 ----------

    /// 카메라 추가. `camera_id` 가 무효값이면 "최대 id + 1"(빈 저장소에서 1)을 발급.
    pub fn add_camera(&self, mut camera: Camera) -> Result<CameraId> {
        let mut g = self.w();
        if camera.camera_id == INVALID_CAMERA_ID {
            camera.camera_id = g.cameras.keys().next_back().map_or(1, |k| k + 1);
        } else if g.cameras.contains_key(&camera.camera_id) {
            return Err(Error::AlreadyExists(format!("카메라 {}", camera.camera_id)));
        }
        let id = camera.camera_id;
        g.cameras.insert(id, camera);
        Ok(id)
    }
    /// 기존 카메라 덮어쓰기(없으면 오류).
    pub fn update_camera(&self, camera: Camera) -> Result<()> {
        let mut g = self.w();
        match g.cameras.get_mut(&camera.camera_id) {
            Some(c) => {
                *c = camera;
                Ok(())
            }
            None => Err(Error::NotFound(format!("카메라 {}", camera.camera_id))),
        }
    }
    /// 카메라 조회(복사).
    pub fn camera(&self, id: CameraId) -> Option<Camera> {
        self.r().cameras.get(&id).cloned()
    }
    /// 카메라 전체(id 순).
    pub fn cameras(&self) -> Vec<Camera> {
        self.r().cameras.values().cloned().collect()
    }
    /// 카메라 수.
    pub fn num_cameras(&self) -> usize {
        self.r().cameras.len()
    }

    // ---------- 영상 ----------

    /// 영상 추가(이름 고유, 카메라 존재 필요). id 는 "최대 id + 1"(빈 저장소에서 1).
    pub fn add_image(&self, name: &str, camera_id: CameraId) -> Result<ImageId> {
        let mut g = self.w();
        let id = g.images.keys().next_back().map_or(1, |k| k + 1);
        Self::add_image_locked(&mut g, id, name, camera_id)
    }
    /// 지정 id 로 영상 추가.
    pub fn add_image_with_id(&self, image_id: ImageId, name: &str, camera_id: CameraId) -> Result<ImageId> {
        Self::add_image_locked(&mut self.w(), image_id, name, camera_id)
    }
    fn add_image_locked(g: &mut Inner, image_id: ImageId, name: &str, camera_id: CameraId) -> Result<ImageId> {
        if image_id as u64 >= MAX_NUM_IMAGES {
            return Err(Error::InvalidArgument(format!("영상 id {image_id} 가 상한 이상")));
        }
        if !g.cameras.contains_key(&camera_id) {
            return Err(Error::NotFound(format!("카메라 {camera_id}")));
        }
        if g.name_to_id.contains_key(name) {
            return Err(Error::AlreadyExists(format!("영상 이름 {name}")));
        }
        if g.images.contains_key(&image_id) {
            return Err(Error::AlreadyExists(format!("영상 {image_id}")));
        }
        g.name_to_id.insert(name.to_string(), image_id);
        g.images.insert(image_id, StoreImage { image_id, name: name.to_string(), camera_id });
        Ok(image_id)
    }
    /// 영상 조회.
    pub fn image(&self, id: ImageId) -> Option<StoreImage> {
        self.r().images.get(&id).cloned()
    }
    /// 이름 → 영상 id.
    pub fn image_id_by_name(&self, name: &str) -> Option<ImageId> {
        self.r().name_to_id.get(name).copied()
    }
    /// 이름으로 영상 조회.
    pub fn image_by_name(&self, name: &str) -> Option<StoreImage> {
        let g = self.r();
        g.name_to_id.get(name).and_then(|id| g.images.get(id)).cloned()
    }
    /// 모든 영상(id 오름차순).
    pub fn images(&self) -> Vec<StoreImage> {
        self.r().images.values().cloned().collect()
    }
    /// 영상 id 목록(오름차순).
    pub fn image_ids(&self) -> Vec<ImageId> {
        self.r().images.keys().copied().collect()
    }
    /// 영상 수.
    pub fn num_images(&self) -> usize {
        self.r().images.len()
    }
    /// 영상이 있는지.
    pub fn exists_image(&self, id: ImageId) -> bool {
        self.r().images.contains_key(&id)
    }

    /// 영상 사전 위치 설정.
    pub fn set_pose_prior(&self, image_id: ImageId, prior: PosePrior) {
        self.w().pose_priors.insert(image_id, prior);
    }
    /// 영상 사전 위치 조회.
    pub fn pose_prior(&self, image_id: ImageId) -> Option<PosePrior> {
        self.r().pose_priors.get(&image_id).copied()
    }

    // ---------- 특징 ----------

    /// 키포인트 기록(덮어씀).
    pub fn set_keypoints(&self, image_id: ImageId, kps: Vec<Keypoint>) {
        self.w().keypoints.insert(image_id, Arc::new(kps));
    }
    /// 기술자 기록(덮어씀).
    pub fn set_descriptors(&self, image_id: ImageId, d: Descriptors) {
        self.w().descriptors.insert(image_id, Arc::new(d));
    }
    /// 키포인트 조회(공유).
    pub fn keypoints(&self, image_id: ImageId) -> Option<Arc<Vec<Keypoint>>> {
        self.r().keypoints.get(&image_id).cloned()
    }
    /// 기술자 조회(공유).
    pub fn descriptors(&self, image_id: ImageId) -> Option<Arc<Descriptors>> {
        self.r().descriptors.get(&image_id).cloned()
    }
    /// 키포인트가 있는지.
    pub fn exists_keypoints(&self, image_id: ImageId) -> bool {
        self.r().keypoints.contains_key(&image_id)
    }
    /// 기술자가 있는지.
    pub fn exists_descriptors(&self, image_id: ImageId) -> bool {
        self.r().descriptors.contains_key(&image_id)
    }
    /// 영상의 키포인트 수(없으면 0).
    pub fn num_keypoints(&self, image_id: ImageId) -> usize {
        self.r().keypoints.get(&image_id).map_or(0, |k| k.len())
    }
    /// 영상별 키포인트 수의 최댓값.
    pub fn largest_keypoint_count(&self) -> usize {
        self.r().keypoints.values().map(|k| k.len()).max().unwrap_or(0)
    }

    // ---------- 매칭 ----------

    /// 원시 매칭 기록(덮어씀). (id1, id2) 순서의 인덱스 쌍을 받는다.
    pub fn write_matches(&self, id1: ImageId, id2: ImageId, matches: &[FeatureMatch]) -> Result<()> {
        let pid = pair_id_of(id1, id2)?;
        let v = if swap_image_pair(id1, id2) { swap_matches(matches) } else { matches.to_vec() };
        self.w().matches.insert(pid, Arc::new(v));
        Ok(())
    }
    /// (id1, id2) 방향으로 원시 매칭 조회.
    pub fn read_matches(&self, id1: ImageId, id2: ImageId) -> Option<Arc<Vec<FeatureMatch>>> {
        let pid = pair_id_of(id1, id2).ok()?;
        let m = self.r().matches.get(&pid).cloned()?;
        if swap_image_pair(id1, id2) {
            Some(Arc::new(swap_matches(&m)))
        } else {
            Some(m)
        }
    }
    /// 원시 매칭이 있는지.
    pub fn exists_matches(&self, id1: ImageId, id2: ImageId) -> bool {
        pair_id_of(id1, id2).is_ok_and(|p| self.r().matches.contains_key(&p))
    }
    /// 원시 매칭 삭제.
    pub fn delete_matches(&self, id1: ImageId, id2: ImageId) {
        if let Ok(p) = pair_id_of(id1, id2) {
            self.w().matches.remove(&p);
        }
    }

    /// 두 뷰 기하 기록(덮어씀). (id1 → id2) 방향 기하를 받는다.
    pub fn put_two_view(&self, id1: ImageId, id2: ImageId, tvg: &TwoViewGeometry) -> Result<()> {
        let pid = pair_id_of(id1, id2)?;
        let v = if swap_image_pair(id1, id2) { tvg.inverted() } else { tvg.clone() };
        let mut g = self.w();
        g.two_view.insert(pid, Arc::new(v));
        g.tvg_log.push(pid);
        Ok(())
    }
    /// (id1 → id2) 방향 두 뷰 기하.
    pub fn get_two_view(&self, id1: ImageId, id2: ImageId) -> Option<Arc<TwoViewGeometry>> {
        let pid = pair_id_of(id1, id2).ok()?;
        let t = self.r().two_view.get(&pid).cloned()?;
        if swap_image_pair(id1, id2) {
            Some(Arc::new(t.inverted()))
        } else {
            Some(t)
        }
    }
    /// 두 뷰 기하가 있는지.
    pub fn contains_two_view(&self, id1: ImageId, id2: ImageId) -> bool {
        pair_id_of(id1, id2).is_ok_and(|p| self.r().two_view.contains_key(&p))
    }
    /// 두 뷰 기하 삭제.
    pub fn remove_two_view(&self, id1: ImageId, id2: ImageId) {
        if let Ok(p) = pair_id_of(id1, id2) {
            self.w().two_view.remove(&p);
        }
    }
    /// 저장된 모든 두 뷰 기하(짝 id 오름차순, 저장 방향 = 작은 id → 큰 id).
    pub fn two_view_geometries(&self) -> Vec<(PairId, Arc<TwoViewGeometry>)> {
        let g = self.r();
        let mut v: Vec<_> = g.two_view.iter().map(|(k, t)| (*k, t.clone())).collect();
        v.sort_by_key(|x| x.0);
        v
    }
    /// 로그 위치 `since` 이후 기록된 짝들과 새 로그 위치. 증분 대응 그래프 갱신용.
    pub fn two_view_geometries_since(&self, since: usize) -> (Vec<(PairId, Arc<TwoViewGeometry>)>, usize) {
        let g = self.r();
        let end = g.tvg_log.len();
        let mut out = Vec::new();
        for &pid in &g.tvg_log[since.min(end)..] {
            if let Some(t) = g.two_view.get(&pid) {
                out.push((pid, t.clone()));
            }
        }
        (out, end)
    }
    /// 저장된 두 뷰 기하 수.
    pub fn num_two_view_geometries(&self) -> usize {
        self.r().two_view.len()
    }
    /// 원시 매칭이 저장된 짝 수.
    pub fn num_matched_pairs(&self) -> usize {
        self.r().matches.len()
    }

    /// `keep` 이 참인 영상만 남긴 새 저장소. 카메라는 모두, 영상 id·짝 자료·기하 기록 순서는 그대로 유지한다.
    /// 큰 자료(키포인트·기술자·매칭·기하)는 `Arc` 로 공유하므로 복사가 싸다. 점진 처리의 되감기용.
    pub fn retain_copy(&self, keep: impl Fn(&StoreImage) -> bool) -> FeatureStore {
        let g = self.r();
        let images: BTreeMap<ImageId, StoreImage> = g.images.iter().filter(|(_, im)| keep(im)).map(|(k, v)| (*k, v.clone())).collect();
        let pair_ok = |pid: &PairId| {
            let (a, b) = images_of_pair(*pid);
            images.contains_key(&a) && images.contains_key(&b)
        };
        let inner = Inner {
            cameras: g.cameras.clone(),
            name_to_id: images.values().map(|im| (im.name.clone(), im.image_id)).collect(),
            keypoints: g.keypoints.iter().filter(|(k, _)| images.contains_key(k)).map(|(k, v)| (*k, v.clone())).collect(),
            descriptors: g.descriptors.iter().filter(|(k, _)| images.contains_key(k)).map(|(k, v)| (*k, v.clone())).collect(),
            pose_priors: g.pose_priors.iter().filter(|(k, _)| images.contains_key(k)).map(|(k, v)| (*k, *v)).collect(),
            matches: g.matches.iter().filter(|(k, _)| pair_ok(k)).map(|(k, v)| (*k, v.clone())).collect(),
            two_view: g.two_view.iter().filter(|(k, _)| pair_ok(k)).map(|(k, v)| (*k, v.clone())).collect(),
            tvg_log: g.tvg_log.iter().copied().filter(|k| pair_ok(k)).collect(),
            images,
        };
        FeatureStore { inner: RwLock::new(inner) }
    }

    // ---------- 저장/로드 ----------

    const MAGIC: &'static [u8; 8] = b"C3DFS\0v1";
    /// 0.3.0 까지 쓰던 표지. 읽을 때만 받아들인다(형식은 같음).
    const LEGACY_MAGIC: &'static [u8; 8] = &[b'S', b'K', b'Y', b'F', b'S', 0, b'v', b'1'];

    /// 자체 이진 형식으로 저장(리틀 엔디언).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let g = self.r();
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        w.write_all(Self::MAGIC)?;
        w.put_u64(g.cameras.len() as u64)?;
        for c in g.cameras.values() {
            w.put_u32(c.camera_id)?;
            w.put_i32(c.model.id())?;
            w.put_u64(c.width)?;
            w.put_u64(c.height)?;
            w.put_u8(c.focal_from_prior as u8)?;
            w.put_u64(c.params.len() as u64)?;
            w.put_f64s(&c.params)?;
        }
        w.put_u64(g.images.len() as u64)?;
        for im in g.images.values() {
            w.put_u32(im.image_id)?;
            w.put_u32(im.camera_id)?;
            w.put_u64(im.name.len() as u64)?;
            w.write_all(im.name.as_bytes())?;
        }
        let mut ids: Vec<_> = g.keypoints.keys().copied().collect();
        ids.sort();
        w.put_u64(ids.len() as u64)?;
        for id in ids {
            let k = &g.keypoints[&id];
            w.put_u32(id)?;
            w.put_u64(k.len() as u64)?;
            for kp in k.iter() {
                for x in kp.to_row() {
                    w.put_f32(x)?;
                }
            }
        }
        let mut ids: Vec<_> = g.descriptors.keys().copied().collect();
        ids.sort();
        w.put_u64(ids.len() as u64)?;
        for id in ids {
            let d = &g.descriptors[&id];
            w.put_u32(id)?;
            w.put_u64(d.len() as u64)?;
            w.write_all(d.as_slice())?;
        }
        let mut ids: Vec<_> = g.pose_priors.keys().copied().collect();
        ids.sort();
        w.put_u64(ids.len() as u64)?;
        for id in ids {
            let p = &g.pose_priors[&id];
            w.put_u32(id)?;
            w.put_f64s(p.position.as_slice())?;
            w.put_i32(p.coordinate_system)?;
            match p.gravity {
                Some(gv) => {
                    w.put_u8(1)?;
                    w.put_f64s(gv.as_slice())?;
                }
                None => w.put_u8(0)?,
            }
        }
        let mut pids: Vec<_> = g.matches.keys().copied().collect();
        pids.sort();
        w.put_u64(pids.len() as u64)?;
        for pid in pids {
            let m = &g.matches[&pid];
            w.put_u64(pid)?;
            write_match_list(&mut w, m)?;
        }
        let mut pids: Vec<_> = g.two_view.keys().copied().collect();
        pids.sort();
        w.put_u64(pids.len() as u64)?;
        for pid in pids {
            let t = &g.two_view[&pid];
            w.put_u64(pid)?;
            w.put_i32(t.config.as_i32())?;
            for m in [&t.e, &t.f, &t.h] {
                match m {
                    Some(m) => {
                        w.put_u8(1)?;
                        for r in 0..3 {
                            for c in 0..3 {
                                w.put_f64(m[(r, c)])?;
                            }
                        }
                    }
                    None => w.put_u8(0)?,
                }
            }
            match &t.cam1_to_cam2 {
                Some(p) => {
                    w.put_u8(1)?;
                    w.put_f64s(&p.to_params())?;
                }
                None => w.put_u8(0)?,
            }
            write_match_list(&mut w, &t.inlier_matches)?;
        }
        w.flush()?;
        Ok(())
    }

    /// `save` 형식 읽기.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut r = BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != Self::MAGIC && &magic != Self::LEGACY_MAGIC {
            return Err(Error::Format("특징 저장소 파일 표지가 다름".into()));
        }
        let mut g = Inner::default();
        for _ in 0..check_count(r.get_u64()?, "카메라")? {
            let id = r.get_u32()?;
            let model = CameraModelKind::from_id(r.get_i32()?)?;
            let width = r.get_u64()?;
            let height = r.get_u64()?;
            let prior = r.get_u8()? != 0;
            let n = check_count(r.get_u64()?, "파라미터")?;
            let mut params = Vec::with_capacity(n);
            for _ in 0..n {
                params.push(r.get_f64()?);
            }
            let mut c = Camera::new(id, model, width, height, params)?;
            c.focal_from_prior = prior;
            g.cameras.insert(id, c);
        }
        for _ in 0..check_count(r.get_u64()?, "영상")? {
            let image_id = r.get_u32()?;
            let camera_id = r.get_u32()?;
            let n = check_count(r.get_u64()?, "이름")?;
            let name = String::from_utf8(r.get_vec_u8(n)?).map_err(|_| Error::Format("이름 UTF-8 아님".into()))?;
            g.name_to_id.insert(name.clone(), image_id);
            g.images.insert(image_id, StoreImage { image_id, name, camera_id });
        }
        for _ in 0..check_count(r.get_u64()?, "키포인트 묶음")? {
            let id = r.get_u32()?;
            let n = check_count(r.get_u64()?, "키포인트")?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let mut row = [0f32; 6];
                for x in row.iter_mut() {
                    *x = r.get_f32()?;
                }
                v.push(Keypoint::from_row(&row)?);
            }
            g.keypoints.insert(id, Arc::new(v));
        }
        for _ in 0..check_count(r.get_u64()?, "기술자 묶음")? {
            let id = r.get_u32()?;
            let n = check_count(r.get_u64()?, "기술자")?;
            let d = Descriptors::from_vec(r.get_vec_u8(n * crate::features::DESCRIPTOR_DIM)?)?;
            g.descriptors.insert(id, Arc::new(d));
        }
        for _ in 0..check_count(r.get_u64()?, "사전 위치")? {
            let id = r.get_u32()?;
            let p = r.get_f64_array::<3>()?;
            let cs = r.get_i32()?;
            let gravity = if r.get_u8()? != 0 { Some(Vec3::from(r.get_f64_array::<3>()?)) } else { None };
            g.pose_priors.insert(id, PosePrior { position: Vec3::from(p), coordinate_system: cs, gravity });
        }
        for _ in 0..check_count(r.get_u64()?, "매칭 짝")? {
            let pid = r.get_u64()?;
            g.matches.insert(pid, Arc::new(read_match_list(&mut r)?));
        }
        for _ in 0..check_count(r.get_u64()?, "두 뷰 기하")? {
            let pid = r.get_u64()?;
            let config =
                TwoViewGeometryConfig::from_i32(r.get_i32()?).ok_or_else(|| Error::Format("두 뷰 기하 config 값이 잘못됨".into()))?;
            let mut mats = [None, None, None];
            for m in mats.iter_mut() {
                if r.get_u8()? != 0 {
                    let a = r.get_f64_array::<9>()?;
                    *m = Some(Mat3::from_row_slice(&a));
                }
            }
            let cam1_to_cam2 = if r.get_u8()? != 0 { Some(Rigid3::from_params(&r.get_f64_array::<7>()?)) } else { None };
            let inlier_matches = read_match_list(&mut r)?;
            let [e, f, h] = mats;
            g.two_view.insert(pid, Arc::new(TwoViewGeometry { config, e, f, h, cam1_to_cam2, inlier_matches, tri_angle: None }));
            g.tvg_log.push(pid);
        }
        Ok(Self { inner: RwLock::new(g) })
    }
}

fn write_match_list(w: &mut impl Write, m: &[FeatureMatch]) -> Result<()> {
    w.put_u64(m.len() as u64)?;
    for x in m {
        w.put_u32(x.idx1)?;
        w.put_u32(x.idx2)?;
    }
    Ok(())
}

fn read_match_list(r: &mut impl Read) -> Result<Vec<FeatureMatch>> {
    let n = check_count(r.get_u64()?, "매칭")?;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let a = r.get_u32()?;
        let b = r.get_u32()?;
        v.push(FeatureMatch::new(a, b));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_basic_and_roundtrip() {
        let s = FeatureStore::new();
        let mut cam = Camera::from_focal(CameraModelKind::OpenCv, 1000.0, 2000, 1000);
        cam.focal_from_prior = true;
        let c1 = s.add_camera(cam.clone()).unwrap();
        assert_eq!(c1, 1);
        let c2 = s.add_camera(cam).unwrap();
        assert_eq!(c2, 2);
        let i1 = s.add_image("camF/0.jpg", c1).unwrap();
        let i2 = s.add_image("camR/0.jpg", c2).unwrap();
        assert!(s.add_image("camF/0.jpg", c1).is_err());
        assert_eq!(s.image_id_by_name("camR/0.jpg"), Some(i2));
        s.set_keypoints(i1, vec![Keypoint::new(1.5, 2.5), Keypoint::from_scale_orientation(3.0, 4.0, 2.0, 0.5)]);
        s.set_keypoints(i2, vec![Keypoint::new(0.5, 0.5)]);
        let mut d = Descriptors::new();
        d.push(&[7u8; 128]);
        s.set_descriptors(i1, d);
        s.set_pose_prior(i1, PosePrior { position: Vec3::new(37.5, 127.0, 50.0), coordinate_system: 0, gravity: None });
        // reversed write
        s.write_matches(i2, i1, &[FeatureMatch::new(0, 1)]).unwrap();
        assert_eq!(s.read_matches(i1, i2).unwrap()[0], FeatureMatch::new(1, 0));
        assert_eq!(s.read_matches(i2, i1).unwrap()[0], FeatureMatch::new(0, 1));
        let h = Mat3::new(2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let tvg = TwoViewGeometry {
            config: TwoViewGeometryConfig::PlanarOrRotation,
            h: Some(h),
            cam1_to_cam2: Some(Rigid3::new(crate::geometry::Quat::from_axis_angle(&Vec3::y(), 0.3), Vec3::new(1.0, 0.0, 0.0))),
            inlier_matches: vec![FeatureMatch::new(0, 1)],
            ..Default::default()
        };
        s.put_two_view(i2, i1, &tvg).unwrap();
        let back = s.get_two_view(i2, i1).unwrap();
        assert!((back.h.unwrap() - h).norm() < 1e-12);
        let fwd = s.get_two_view(i1, i2).unwrap();
        assert_eq!(fwd.inlier_matches[0], FeatureMatch::new(1, 0));
        let p = fwd.cam1_to_cam2.unwrap() * tvg.cam1_to_cam2.unwrap();
        assert!(p.rotation.angular_distance(&crate::geometry::Quat::IDENTITY) < 1e-7);
        let (since, pos) = s.two_view_geometries_since(0);
        assert_eq!(since.len(), 1);
        assert_eq!(s.two_view_geometries_since(pos).0.len(), 0);

        let dir = std::env::temp_dir().join(format!("cumulus3d_store_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("store.bin");
        s.save(&path).unwrap();
        let l = FeatureStore::load(&path).unwrap();
        assert_eq!(l.cameras(), s.cameras());
        assert_eq!(l.images(), s.images());
        assert_eq!(l.keypoints(i1), s.keypoints(i1));
        assert_eq!(l.descriptors(i1), s.descriptors(i1));
        assert_eq!(l.pose_prior(i1), s.pose_prior(i1));
        assert_eq!(l.read_matches(i1, i2), s.read_matches(i1, i2));
        assert_eq!(l.get_two_view(i1, i2), s.get_two_view(i1, i2));
        // 옛 표지로 쓴 파일도 읽힌다.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[..8].copy_from_slice(FeatureStore::LEGACY_MAGIC);
        let legacy = dir.join("legacy.bin");
        std::fs::write(&legacy, &bytes).unwrap();
        assert_eq!(FeatureStore::load(&legacy).unwrap().images(), s.images());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn store_concurrent_adds() {
        let s = Arc::new(FeatureStore::new());
        s.add_camera(Camera::from_focal(CameraModelKind::OpenCv, 1.0, 10, 10)).unwrap();
        std::thread::scope(|sc| {
            for t in 0..8 {
                let s = s.clone();
                sc.spawn(move || {
                    for k in 0..50 {
                        let id = s.add_image(&format!("t{t}/{k}"), 1).unwrap();
                        s.set_keypoints(id, vec![Keypoint::new(0.5, 0.5); k]);
                    }
                });
            }
        });
        assert_eq!(s.num_images(), 400);
        assert_eq!(s.largest_keypoint_count(), 49);
    }

    #[test]
    fn retain_copy_drops_images_and_pairs() {
        let s = FeatureStore::new();
        let c = s.add_camera(Camera::from_focal(CameraModelKind::OpenCv, 100.0, 64, 48)).unwrap();
        let ids: Vec<ImageId> = (0..3).map(|i| s.add_image(&format!("cam/{i}.jpg"), c).unwrap()).collect();
        for &i in &ids {
            s.set_keypoints(i, vec![Keypoint::new(1.0, 1.0)]);
        }
        let tvg = TwoViewGeometry { inlier_matches: vec![FeatureMatch::new(0, 0)], ..Default::default() };
        s.put_two_view(ids[0], ids[1], &tvg).unwrap();
        s.put_two_view(ids[1], ids[2], &tvg).unwrap();
        s.write_matches(ids[0], ids[2], &[FeatureMatch::new(0, 0)]).unwrap();
        let r = s.retain_copy(|im| im.image_id != ids[2]);
        assert_eq!(r.num_images(), 2);
        assert_eq!(r.num_cameras(), 1);
        assert!(r.image_id_by_name("cam/2.jpg").is_none());
        assert!(r.contains_two_view(ids[0], ids[1]));
        assert!(!r.contains_two_view(ids[1], ids[2]));
        assert!(!r.exists_matches(ids[0], ids[2]));
        assert_eq!(r.two_view_geometries_since(0).1, 1);
        // 새 영상 id 는 남은 최대 id + 1.
        assert_eq!(r.add_image("cam/9.jpg", c).unwrap(), ids[2]);
        // 원래 저장소는 그대로.
        assert_eq!(s.num_images(), 3);
    }
}
