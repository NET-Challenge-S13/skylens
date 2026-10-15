/*
 * reconstruction.rs
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

//! 희소 재구성 자료 구조.
//!
//! 사본 비용: 영상과 3D 점은 `Arc` 로 보관하고 수정 시 `Arc::make_mut`(copy-on-write)로 복제한다.
//! 따라서 `Reconstruction::clone()` 은 참조 수 증가만 하고, 이후 실제로 바뀐 영상/점만 복사된다.
//! 불변식(`check_invariants`)은 모든 공개 수정 연산이 유지한다.

use crate::camera::Camera;
use crate::error::{Error, Result};
use crate::geometry::{max_triangulation_angle, Rigid3, Sim3, Vec2, Vec3};
use crate::ids::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// 트랙 원소 (영상 id, 2D 점 인덱스).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TrackEntry {
    /// 영상 id.
    pub image_id: ImageId,
    /// 영상 내 2D 점 인덱스.
    pub point2d_idx: Point2DIdx,
}

impl TrackEntry {
    /// (영상, 2D 점)으로 생성.
    pub fn new(image_id: ImageId, point2d_idx: Point2DIdx) -> Self {
        Self { image_id, point2d_idx }
    }
}

/// 영상의 2D 점: 픽셀 좌표 + 연결된 3D 점 id(없으면 `INVALID_POINT3D_ID`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2D {
    /// 픽셀 좌표.
    pub xy: Vec2,
    /// 연결된 3D 점 id.
    pub point3d_id: Point3DId,
}

impl Point2D {
    /// 연결 없는 2D 점 생성.
    pub fn new(xy: Vec2) -> Self {
        Self { xy, point3d_id: INVALID_POINT3D_ID }
    }
    /// 3D 점과 연결됐는지.
    pub fn has_point3d(&self) -> bool {
        self.point3d_id != INVALID_POINT3D_ID
    }
}

/// 3D 점. error 는 픽셀 평균 재투영 오차, −1 = 없음.
#[derive(Clone, Debug, PartialEq)]
pub struct Point3D {
    /// 세계 좌표.
    pub xyz: Vec3,
    /// RGB 색.
    pub color: [u8; 3],
    /// 평균 재투영 오차(픽셀, −1 = 없음).
    pub error: f64,
    /// 관측 트랙.
    pub track: Vec<TrackEntry>,
}

impl Point3D {
    /// 빈 트랙·검정색·오차 없음으로 생성.
    pub fn new(xyz: Vec3) -> Self {
        Self { xyz, color: [0, 0, 0], error: -1.0, track: Vec::new() }
    }
    /// 오차가 기록돼 있는지.
    pub fn has_error(&self) -> bool {
        self.error != -1.0
    }
    /// 트랙 길이.
    pub fn track_len(&self) -> usize {
        self.track.len()
    }
}

/// Rig: 기준 센서(자세 = 항등) + 비기준 센서별 선택적 rig_to_sensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Rig {
    /// rig id.
    pub rig_id: RigId,
    /// 기준 센서.
    pub ref_sensor_id: Option<SensorKey>,
    /// 비기준 센서(정렬 보관). None = 미보정.
    pub sensors: BTreeMap<SensorKey, Option<Rigid3>>,
}

impl Rig {
    /// 빈 rig.
    pub fn new(rig_id: RigId) -> Self {
        Self { rig_id, ref_sensor_id: None, sensors: BTreeMap::new() }
    }
    /// 단일 카메라 rig(rig id = 카메라 id).
    pub fn trivial(camera_id: CameraId) -> Self {
        Self { rig_id: camera_id, ref_sensor_id: Some(SensorKey::camera(camera_id)), sensors: BTreeMap::new() }
    }
    /// 센서 수(기준 포함).
    pub fn num_sensors(&self) -> usize {
        self.sensors.len() + self.ref_sensor_id.is_some() as usize
    }
    /// 기준 센서인지.
    pub fn is_reference_sensor(&self, s: SensorKey) -> bool {
        self.ref_sensor_id == Some(s)
    }
    /// 센서가 rig 에 있는지.
    pub fn has_sensor(&self, s: SensorKey) -> bool {
        self.is_reference_sensor(s) || self.sensors.contains_key(&s)
    }
    /// rig_to_sensor. 기준 센서면 항등, 미보정/없음이면 None.
    pub fn rig_to_sensor(&self, s: SensorKey) -> Option<Rigid3> {
        if self.is_reference_sensor(s) {
            Some(Rigid3::identity())
        } else {
            self.sensors.get(&s).copied().flatten()
        }
    }
}

/// 프레임: 같은 시각 rig 의 데이터 묶음 + 선택적 world_to_rig.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// 프레임 id.
    pub frame_id: FrameId,
    /// 소속 rig id.
    pub rig_id: RigId,
    data_ids: Vec<SensorDataKey>,
    /// 세계 → rig 자세(없으면 미정).
    pub world_to_rig: Option<Rigid3>,
}

impl Frame {
    /// 데이터·자세 없는 프레임.
    pub fn new(frame_id: FrameId, rig_id: RigId) -> Self {
        Self { frame_id, rig_id, data_ids: Vec::new(), world_to_rig: None }
    }
    /// 정렬 상태를 유지하며 추가(중복 무시).
    pub fn attach_data(&mut self, d: SensorDataKey) {
        if let Err(pos) = self.data_ids.binary_search(&d) {
            self.data_ids.insert(pos, d);
        }
    }
    /// 데이터 키 목록(정렬).
    pub fn data_ids(&self) -> &[SensorDataKey] {
        &self.data_ids
    }
    /// 자세가 있는지.
    pub fn has_pose(&self) -> bool {
        self.world_to_rig.is_some()
    }
    /// 카메라 데이터의 영상 id(데이터 순서).
    pub fn image_ids(&self) -> impl Iterator<Item = ImageId> + '_ {
        self.data_ids.iter().filter(|d| d.sensor_id.sensor_type == SensorKind::Camera).map(|d| d.id as ImageId)
    }
}

/// 영상. 자세는 저장하지 않고 프레임에서 유도한다.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    /// 영상 id.
    pub image_id: ImageId,
    /// 공백 불가(텍스트 형식 구분자).
    pub name: String,
    /// 카메라 id.
    pub camera_id: CameraId,
    /// 소속 프레임 id.
    pub frame_id: FrameId,
    points2d: Vec<Point2D>,
    num_points3d: usize,
}

impl Image {
    /// 2D 점 좌표로 생성(연결 없음, 프레임 미지정).
    pub fn new(image_id: ImageId, name: impl Into<String>, camera_id: CameraId, points: impl IntoIterator<Item = Vec2>) -> Self {
        Self {
            image_id,
            name: name.into(),
            camera_id,
            frame_id: INVALID_FRAME_ID,
            points2d: points.into_iter().map(Point2D::new).collect(),
            num_points3d: 0,
        }
    }
    /// 2D 점 전체.
    pub fn points2d(&self) -> &[Point2D] {
        &self.points2d
    }
    /// idx 번째 2D 점.
    pub fn point2d(&self, idx: Point2DIdx) -> &Point2D {
        &self.points2d[idx as usize]
    }
    /// 2D 점 수.
    pub fn num_points2d(&self) -> usize {
        self.points2d.len()
    }
    /// 3D 점이 연결된 2D 점 수(캐시).
    pub fn num_points3d(&self) -> usize {
        self.num_points3d
    }
}

/// 관측 필터 후 점 error 갱신 방식.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterErrorUpdate {
    /// 갱신 안 함.
    None,
    /// (인라이어 오차 합) / (삭제 전 트랙 길이)
    SumOverOriginalLength,
    /// 남은 관측 오차 평균
    MeanOfRemaining,
}

/// 정규화 옵션.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizeOptions {
    /// 참이면 스케일 1(평행이동만).
    pub fixed_scale: bool,
    /// 정규화 후 범위 크기.
    pub extent: f64,
    /// 범위 계산용 하위 분위수.
    pub p0: f64,
    /// 범위 계산용 상위 분위수.
    pub p1: f64,
    /// 참이면 등록 영상 투영 중심, 거짓이면 3D 점 기준.
    pub use_images: bool,
}

impl Default for NormalizeOptions {
    fn default() -> Self {
        Self { fixed_scale: false, extent: 10.0, p0: 0.1, p1: 0.9, use_images: true }
    }
}

/// 희소 재구성.
#[derive(Clone, Debug, Default)]
pub struct Reconstruction {
    cameras: BTreeMap<CameraId, Camera>,
    rigs: BTreeMap<RigId, Rig>,
    frames: BTreeMap<FrameId, Frame>,
    images: BTreeMap<ImageId, Arc<Image>>,
    points3d: BTreeMap<Point3DId, Arc<Point3D>>,
    /// 등록 순서.
    registered_frames: Vec<FrameId>,
    registered_image_count: usize,
    max_point3d_id: Point3DId,
}

fn nf<T>(what: impl std::fmt::Display) -> Result<T> {
    Err(Error::NotFound(what.to_string()))
}

impl Reconstruction {
    /// 빈 재구성.
    pub fn new() -> Self {
        Self::default()
    }

    // ================= 조회 =================

    /// 카메라 전체(id 순).
    pub fn cameras(&self) -> &BTreeMap<CameraId, Camera> {
        &self.cameras
    }
    /// 카메라 조회.
    pub fn camera(&self, id: CameraId) -> Option<&Camera> {
        self.cameras.get(&id)
    }
    /// 카메라 내부 파라미터 수정(불변식과 무관).
    pub fn camera_mut(&mut self, id: CameraId) -> Option<&mut Camera> {
        self.cameras.get_mut(&id)
    }
    /// rig 전체(id 순).
    pub fn rigs(&self) -> &BTreeMap<RigId, Rig> {
        &self.rigs
    }
    /// rig 조회.
    pub fn rig(&self, id: RigId) -> Option<&Rig> {
        self.rigs.get(&id)
    }
    /// rig 가변 조회.
    pub fn rig_mut(&mut self, id: RigId) -> Option<&mut Rig> {
        self.rigs.get_mut(&id)
    }
    /// 프레임 전체(id 순).
    pub fn frames(&self) -> &BTreeMap<FrameId, Frame> {
        &self.frames
    }
    /// 프레임 조회.
    pub fn frame(&self, id: FrameId) -> Option<&Frame> {
        self.frames.get(&id)
    }
    /// 영상(id 오름차순).
    pub fn images(&self) -> impl Iterator<Item = &Image> + '_ {
        self.images.values().map(|a| &**a)
    }
    /// 영상 조회.
    pub fn image(&self, id: ImageId) -> Option<&Image> {
        self.images.get(&id).map(|a| &**a)
    }
    /// 영상 id(오름차순).
    pub fn image_ids(&self) -> impl Iterator<Item = ImageId> + '_ {
        self.images.keys().copied()
    }
    /// 영상이 있는지.
    pub fn exists_image(&self, id: ImageId) -> bool {
        self.images.contains_key(&id)
    }
    /// 이름으로 영상 조회.
    pub fn image_by_name(&self, name: &str) -> Option<&Image> {
        self.images().find(|im| im.name == name)
    }
    /// 3D 점(id 오름차순).
    pub fn points3d(&self) -> impl Iterator<Item = (Point3DId, &Point3D)> + '_ {
        self.points3d.iter().map(|(k, v)| (*k, &**v))
    }
    /// 3D 점 조회.
    pub fn point3d(&self, id: Point3DId) -> Option<&Point3D> {
        self.points3d.get(&id).map(|a| &**a)
    }
    /// 3D 점 id 목록(오름차순).
    pub fn point3d_ids(&self) -> Vec<Point3DId> {
        self.points3d.keys().copied().collect()
    }
    /// 3D 점이 있는지.
    pub fn exists_point3d(&self, id: Point3DId) -> bool {
        self.points3d.contains_key(&id)
    }
    /// 카메라 수.
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }
    /// rig 수.
    pub fn num_rigs(&self) -> usize {
        self.rigs.len()
    }
    /// 프레임 수.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
    /// 영상 수.
    pub fn num_images(&self) -> usize {
        self.images.len()
    }
    /// 3D 점 수.
    pub fn num_points3d(&self) -> usize {
        self.points3d.len()
    }
    /// 등록 프레임 수.
    pub fn registered_frame_count(&self) -> usize {
        self.registered_frames.len()
    }
    /// 등록 영상 수.
    pub fn registered_image_count(&self) -> usize {
        self.registered_image_count
    }
    /// 지금까지 발급/관측한 최대 3D 점 id.
    pub fn max_point3d_id(&self) -> Point3DId {
        self.max_point3d_id
    }
    /// 등록 프레임(등록 순서).
    pub fn registered_frames(&self) -> &[FrameId] {
        &self.registered_frames
    }
    /// 등록 영상: 등록 프레임 순서 → 프레임 내 데이터 순서.
    pub fn registered_images(&self) -> Vec<ImageId> {
        let mut out = Vec::with_capacity(self.registered_image_count);
        for fid in &self.registered_frames {
            if let Some(f) = self.frames.get(fid) {
                out.extend(f.image_ids().filter(|i| self.images.contains_key(i)));
            }
        }
        out
    }
    /// 프레임이 등록됐는지.
    pub fn is_frame_registered(&self, frame_id: FrameId) -> bool {
        self.registered_frames.contains(&frame_id)
    }
    /// 영상(의 프레임)이 등록됐는지.
    pub fn is_image_registered(&self, image_id: ImageId) -> bool {
        self.image(image_id).is_some_and(|im| self.is_frame_registered(im.frame_id))
    }

    /// 영상의 world_to_cam. 자세 없으면 None.
    pub fn world_to_cam(&self, image_id: ImageId) -> Option<Rigid3> {
        let im = self.image(image_id)?;
        let frame = self.frames.get(&im.frame_id)?;
        let world_to_rig = frame.world_to_rig?;
        let rig = self.rigs.get(&frame.rig_id)?;
        let s = SensorKey::camera(im.camera_id);
        if rig.is_reference_sensor(s) {
            Some(world_to_rig)
        } else {
            Some(rig.rig_to_sensor(s)?.compose(&world_to_rig))
        }
    }
    /// 영상 자세가 있는지.
    pub fn has_pose(&self, image_id: ImageId) -> bool {
        self.image(image_id).and_then(|im| self.frames.get(&im.frame_id)).is_some_and(|f| f.has_pose())
    }
    /// 투영 중심 C = −Rᵀ t.
    pub fn projection_center(&self, image_id: ImageId) -> Option<Vec3> {
        self.world_to_cam(image_id).map(|p| p.center())
    }

    // ================= 추가 =================

    /// 카메라 추가(id 중복이면 오류).
    pub fn add_camera(&mut self, camera: Camera) -> Result<()> {
        if self.cameras.contains_key(&camera.camera_id) {
            return Err(Error::AlreadyExists(format!("카메라 {}", camera.camera_id)));
        }
        if !camera.verify_params() {
            return Err(Error::InvalidArgument(format!("카메라 {} 파라미터 개수", camera.camera_id)));
        }
        self.cameras.insert(camera.camera_id, camera);
        Ok(())
    }
    /// 카메라와 단일 카메라 rig(rig id = 카메라 id)를 함께 추가. rig 가 이미 있으면 생략.
    pub fn add_camera_own_rig(&mut self, camera: Camera) -> Result<()> {
        let id = camera.camera_id;
        self.add_camera(camera)?;
        self.rigs.entry(id).or_insert_with(|| Rig::trivial(id));
        Ok(())
    }
    /// rig 추가(id 중복이면 오류).
    pub fn add_rig(&mut self, rig: Rig) -> Result<()> {
        if self.rigs.contains_key(&rig.rig_id) {
            return Err(Error::AlreadyExists(format!("rig {}", rig.rig_id)));
        }
        self.rigs.insert(rig.rig_id, rig);
        Ok(())
    }
    /// 프레임 추가(등록 안 됨).
    pub fn add_frame(&mut self, frame: Frame) -> Result<()> {
        if self.frames.contains_key(&frame.frame_id) {
            return Err(Error::AlreadyExists(format!("프레임 {}", frame.frame_id)));
        }
        if !self.rigs.contains_key(&frame.rig_id) {
            return nf(format!("rig {}", frame.rig_id));
        }
        self.frames.insert(frame.frame_id, frame);
        Ok(())
    }
    /// 영상 추가. 카메라가 있어야 하고, frame_id 가 유효하면 그 프레임이 있어야 한다.
    /// 2D 점의 기존 3D 연결은 지워진다(연결은 3D 점 추가로만 만든다).
    pub fn add_image(&mut self, mut image: Image) -> Result<()> {
        if self.images.contains_key(&image.image_id) {
            return Err(Error::AlreadyExists(format!("영상 {}", image.image_id)));
        }
        if !self.cameras.contains_key(&image.camera_id) {
            return nf(format!("카메라 {}", image.camera_id));
        }
        if image.frame_id != INVALID_FRAME_ID && !self.frames.contains_key(&image.frame_id) {
            return nf(format!("프레임 {}", image.frame_id));
        }
        for p in image.points2d.iter_mut() {
            p.point3d_id = INVALID_POINT3D_ID;
        }
        image.num_points3d = 0;
        self.images.insert(image.image_id, Arc::new(image));
        Ok(())
    }
    /// 영상 + 자명 프레임(frame id = 영상 id, rig id = 카메라 id) 추가. rig 가 없으면 만든다.
    pub fn add_image_own_frame(&mut self, mut image: Image, world_to_cam: Option<Rigid3>) -> Result<()> {
        if !self.cameras.contains_key(&image.camera_id) {
            return nf(format!("카메라 {}", image.camera_id));
        }
        if self.images.contains_key(&image.image_id) {
            return Err(Error::AlreadyExists(format!("영상 {}", image.image_id)));
        }
        let cam = image.camera_id;
        self.rigs.entry(cam).or_insert_with(|| Rig::trivial(cam));
        let mut frame = Frame::new(image.image_id, cam);
        frame.attach_data(SensorDataKey::image(cam, image.image_id));
        frame.world_to_rig = world_to_cam;
        self.add_frame(frame)?;
        image.frame_id = image.image_id;
        self.add_image(image)
    }

    // ================= 자세와 등록 =================

    /// 프레임 자세 설정/제거. 등록된 프레임의 자세는 제거할 수 없다.
    pub fn set_frame_pose(&mut self, frame_id: FrameId, world_to_rig: Option<Rigid3>) -> Result<()> {
        if world_to_rig.is_none() && self.is_frame_registered(frame_id) {
            return Err(Error::InvalidArgument(format!("등록된 프레임 {frame_id} 의 자세 제거")));
        }
        match self.frames.get_mut(&frame_id) {
            Some(f) => {
                f.world_to_rig = world_to_rig;
                Ok(())
            }
            None => nf(format!("프레임 {frame_id}")),
        }
    }
    /// 영상의 world_to_cam 로 프레임 자세를 설정(비기준 센서면 world_to_rig 를 역산).
    pub fn set_world_to_cam(&mut self, image_id: ImageId, world_to_cam: Rigid3) -> Result<()> {
        let im = self.image(image_id).ok_or_else(|| Error::NotFound(format!("영상 {image_id}")))?;
        let (fid, s) = (im.frame_id, SensorKey::camera(im.camera_id));
        let frame = self.frames.get(&fid).ok_or_else(|| Error::NotFound(format!("프레임 {fid}")))?;
        let rig = self.rigs.get(&frame.rig_id).ok_or_else(|| Error::NotFound(format!("rig {}", frame.rig_id)))?;
        let world_to_rig = if rig.is_reference_sensor(s) {
            world_to_cam
        } else {
            let rig_to_cam =
                rig.rig_to_sensor(s).ok_or_else(|| Error::InvalidArgument(format!("영상 {image_id} 센서가 rig 에서 미보정")))?;
            rig_to_cam.inverse().compose(&world_to_cam)
        };
        self.frames.get_mut(&fid).expect("checked").world_to_rig = Some(world_to_rig);
        Ok(())
    }
    /// 프레임 등록(자세 필요). 이미 등록됐으면 Ok(false).
    pub fn register_frame(&mut self, frame_id: FrameId) -> Result<bool> {
        let f = self.frames.get(&frame_id).ok_or_else(|| Error::NotFound(format!("프레임 {frame_id}")))?;
        if !f.has_pose() {
            return Err(Error::InvalidArgument(format!("자세 없는 프레임 {frame_id} 등록")));
        }
        if self.is_frame_registered(frame_id) {
            return Ok(false);
        }
        let n = f.image_ids().filter(|i| self.images.contains_key(i)).count();
        self.registered_frames.push(frame_id);
        self.registered_image_count += n;
        Ok(true)
    }
    /// 영상이 속한 프레임 등록(`register_frame`).
    pub fn register_image(&mut self, image_id: ImageId) -> Result<bool> {
        let fid = self.image(image_id).ok_or_else(|| Error::NotFound(format!("영상 {image_id}")))?.frame_id;
        self.register_frame(fid)
    }
    /// 프레임 등록 해제: 각 영상의 관측을 2D 점 인덱스 오름차순으로 삭제 →
    /// 등록 영상 수 감소 → 자세 제거 → 등록 목록에서 제거. 등록되지 않았으면 Ok(false).
    pub fn deregister_frame(&mut self, frame_id: FrameId) -> Result<bool> {
        let Some(pos) = self.registered_frames.iter().position(|f| *f == frame_id) else {
            return Ok(false);
        };
        let image_ids: Vec<ImageId> = self.frames[&frame_id].image_ids().filter(|i| self.images.contains_key(i)).collect();
        for &iid in &image_ids {
            let n = self.images[&iid].points2d.len();
            for idx in 0..n {
                if self.images[&iid].points2d[idx].has_point3d() {
                    self.delete_observation(iid, idx as Point2DIdx)?;
                }
            }
        }
        self.registered_image_count -= image_ids.len();
        if let Some(f) = self.frames.get_mut(&frame_id) {
            f.world_to_rig = None;
        }
        self.registered_frames.remove(pos);
        Ok(true)
    }
    /// 영상이 속한 프레임 등록 해제(`deregister_frame`).
    pub fn deregister_image(&mut self, image_id: ImageId) -> Result<bool> {
        let fid = self.image(image_id).ok_or_else(|| Error::NotFound(format!("영상 {image_id}")))?.frame_id;
        self.deregister_frame(fid)
    }

    // ================= 3D 점·관측 =================

    fn link(&mut self, e: TrackEntry, pid: Point3DId) {
        let im = Arc::make_mut(self.images.get_mut(&e.image_id).expect("검증됨"));
        let p = &mut im.points2d[e.point2d_idx as usize];
        if !p.has_point3d() {
            im.num_points3d += 1;
        }
        p.point3d_id = pid;
    }
    fn unlink(&mut self, e: TrackEntry, pid: Point3DId) {
        if let Some(a) = self.images.get_mut(&e.image_id) {
            if a.points2d.get(e.point2d_idx as usize).is_some_and(|p| p.point3d_id == pid) {
                let im = Arc::make_mut(a);
                im.points2d[e.point2d_idx as usize].point3d_id = INVALID_POINT3D_ID;
                im.num_points3d -= 1;
            }
        }
    }
    fn check_element(&self, e: &TrackEntry, pid: Point3DId) -> Result<()> {
        let im = self.image(e.image_id).ok_or_else(|| Error::NotFound(format!("영상 {}", e.image_id)))?;
        let p = im
            .points2d
            .get(e.point2d_idx as usize)
            .ok_or_else(|| Error::InvalidArgument(format!("영상 {} 의 2D 점 {} 범위 밖", e.image_id, e.point2d_idx)))?;
        if p.has_point3d() && p.point3d_id != pid {
            return Err(Error::Invariant(format!("영상 {} 2D 점 {} 은 이미 3D 점 {} 에 연결됨", e.image_id, e.point2d_idx, p.point3d_id)));
        }
        Ok(())
    }

    /// 새 3D 점 추가(id = 최대 id + 1). 트랙 원소의 2D 점이 다른 점에 연결돼 있으면 오류.
    pub fn add_point3d(&mut self, xyz: Vec3, track: Vec<TrackEntry>, color: [u8; 3]) -> Result<Point3DId> {
        // 설계 결정: 빈 재구성의 첫 id. "최대 id + 1" 규칙에서 최대값 초기값 0 → 첫 id 1.
        let id = self.max_point3d_id + 1;
        self.add_point3d_with_id(id, Point3D { xyz, color, error: -1.0, track })?;
        Ok(id)
    }
    /// 지정 id 로 3D 점 추가(파일 읽기·전역 매퍼용). 최대 id 를 갱신한다.
    pub fn add_point3d_with_id(&mut self, id: Point3DId, point: Point3D) -> Result<()> {
        if id == INVALID_POINT3D_ID {
            return Err(Error::InvalidArgument("무효 3D 점 id".into()));
        }
        if self.points3d.contains_key(&id) {
            return Err(Error::AlreadyExists(format!("3D 점 {id}")));
        }
        for e in &point.track {
            self.check_element(e, id)?;
        }
        for e in point.track.clone() {
            self.link(e, id);
        }
        self.points3d.insert(id, Arc::new(point));
        self.max_point3d_id = self.max_point3d_id.max(id);
        Ok(())
    }
    /// 기존 3D 점에 관측 추가.
    pub fn add_observation(&mut self, point3d_id: Point3DId, e: TrackEntry) -> Result<()> {
        if !self.points3d.contains_key(&point3d_id) {
            return nf(format!("3D 점 {point3d_id}"));
        }
        self.check_element(&e, point3d_id)?;
        if self.image(e.image_id).expect("검증됨").points2d[e.point2d_idx as usize].has_point3d() {
            return Err(Error::Invariant(format!("관측 ({}, {}) 중복", e.image_id, e.point2d_idx)));
        }
        self.link(e, point3d_id);
        Arc::make_mut(self.points3d.get_mut(&point3d_id).expect("검증됨")).track.push(e);
        Ok(())
    }
    /// 관측 삭제: 트랙 길이 ≤ 2 면 3D 점 전체 삭제, 아니면 그 원소만 제거.
    /// 반환: 점 전체가 삭제됐으면 true.
    pub fn delete_observation(&mut self, image_id: ImageId, point2d_idx: Point2DIdx) -> Result<bool> {
        let im = self.image(image_id).ok_or_else(|| Error::NotFound(format!("영상 {image_id}")))?;
        let pid =
            im.points2d.get(point2d_idx as usize).ok_or_else(|| Error::InvalidArgument(format!("2D 점 {point2d_idx} 범위 밖")))?.point3d_id;
        if pid == INVALID_POINT3D_ID {
            return Err(Error::InvalidArgument(format!("영상 {image_id} 2D 점 {point2d_idx} 에 3D 점 없음")));
        }
        let e = TrackEntry::new(image_id, point2d_idx);
        if self.points3d[&pid].track.len() <= 2 {
            self.delete_point3d(pid)?;
            return Ok(true);
        }
        let p = Arc::make_mut(self.points3d.get_mut(&pid).expect("연결 불변식"));
        if let Some(pos) = p.track.iter().position(|t| *t == e) {
            p.track.remove(pos);
        }
        self.unlink(e, pid);
        Ok(false)
    }
    /// 3D 점 삭제(모든 관측 연결 해제).
    pub fn delete_point3d(&mut self, id: Point3DId) -> Result<()> {
        let p = self.points3d.remove(&id).ok_or_else(|| Error::NotFound(format!("3D 점 {id}")))?;
        for e in &p.track {
            self.unlink(*e, id);
        }
        Ok(())
    }
    /// 두 3D 점 병합: 트랙 길이 가중 평균 위치·색(색은 절삭), 트랙 연결(a 다음 b), 새 id.
    pub fn merge_points3d(&mut self, a: Point3DId, b: Point3DId) -> Result<Point3DId> {
        if a == b {
            return Err(Error::InvalidArgument("같은 점 병합".into()));
        }
        let pa = self.points3d.get(&a).ok_or_else(|| Error::NotFound(format!("3D 점 {a}")))?.clone();
        let pb = self.points3d.get(&b).ok_or_else(|| Error::NotFound(format!("3D 점 {b}")))?.clone();
        let (wa, wb) = (pa.track.len() as f64, pb.track.len() as f64);
        let w = (wa + wb).max(1.0);
        let xyz = (pa.xyz * wa + pb.xyz * wb) / w;
        let mut color = [0u8; 3];
        for (k, c) in color.iter_mut().enumerate() {
            *c = ((pa.color[k] as f64 * wa + pb.color[k] as f64 * wb) / w) as u8;
        }
        let mut track = pa.track.clone();
        track.extend_from_slice(&pb.track);
        self.delete_point3d(a)?;
        self.delete_point3d(b)?;
        // 설계 결정: 병합 점의 error 는 −1(없음)로 둔다. 이후 단계가 재계산한다.
        self.add_point3d(xyz, track, color)
    }
    /// 3D 점 위치 설정.
    pub fn set_point3d_xyz(&mut self, id: Point3DId, xyz: Vec3) -> Result<()> {
        let p = self.points3d.get_mut(&id).ok_or_else(|| Error::NotFound(format!("3D 점 {id}")))?;
        Arc::make_mut(p).xyz = xyz;
        Ok(())
    }
    /// 3D 점 오차 설정.
    pub fn set_point3d_error(&mut self, id: Point3DId, error: f64) -> Result<()> {
        let p = self.points3d.get_mut(&id).ok_or_else(|| Error::NotFound(format!("3D 점 {id}")))?;
        Arc::make_mut(p).error = error;
        Ok(())
    }
    /// 3D 점 색 설정.
    pub fn set_point3d_color(&mut self, id: Point3DId, color: [u8; 3]) -> Result<()> {
        let p = self.points3d.get_mut(&id).ok_or_else(|| Error::NotFound(format!("3D 점 {id}")))?;
        Arc::make_mut(p).color = color;
        Ok(())
    }
    /// 모든 3D 점 삭제(관측 연결 해제).
    pub fn delete_all_points3d(&mut self) {
        self.points3d.clear();
        for a in self.images.values_mut() {
            if a.num_points3d > 0 {
                let im = Arc::make_mut(a);
                for p in im.points2d.iter_mut() {
                    p.point3d_id = INVALID_POINT3D_ID;
                }
                im.num_points3d = 0;
            }
        }
    }

    // ================= 통계 =================

    /// 등록 영상들의 "3D 점이 연결된 2D 점 수" 합.
    pub fn total_observations(&self) -> usize {
        self.registered_images().iter().map(|i| self.images[i].num_points3d).sum()
    }
    /// 3D 점 평균 트랙 길이.
    pub fn mean_track_len(&self) -> f64 {
        if self.points3d.is_empty() {
            0.0
        } else {
            self.total_observations() as f64 / self.points3d.len() as f64
        }
    }
    /// 등록 영상당 평균 관측 수.
    pub fn mean_obs_per_registered_image(&self) -> f64 {
        if self.registered_image_count == 0 {
            0.0
        } else {
            self.total_observations() as f64 / self.registered_image_count as f64
        }
    }
    /// 저장된 점 error 중 −1 이 아닌 것의 평균(재계산 안 함). 없으면 0.
    pub fn mean_reproj_error(&self) -> f64 {
        let (mut s, mut n) = (0.0, 0usize);
        for p in self.points3d.values() {
            if p.has_error() {
                s += p.error;
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            s / n as f64
        }
    }

    // ================= 재투영 오차 =================

    /// 영상별 (world_to_cam, 카메라) 캐시.
    fn pose_cache(&self) -> HashMap<ImageId, (Rigid3, &Camera)> {
        let mut m = HashMap::with_capacity(self.images.len());
        for im in self.images() {
            if let (Some(p), Some(c)) = (self.world_to_cam(im.image_id), self.cameras.get(&im.camera_id)) {
                m.insert(im.image_id, (p, c));
            }
        }
        m
    }

    /// 픽셀 재투영 제곱 오차. 투영 실패(깊이 < ε)면 f64::MAX.
    pub fn squared_reprojection_error(pose: &Rigid3, camera: &Camera, xy: &Vec2, xyz: &Vec3) -> f64 {
        match camera.cam_to_img(&pose.transform_point(xyz)) {
            Some(p) => (p - xy).norm_squared(),
            None => f64::MAX,
        }
    }

    fn compute_point_error(&self, cache: &HashMap<ImageId, (Rigid3, &Camera)>, p: &Point3D) -> f64 {
        if p.track.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        for e in &p.track {
            let sq = match cache.get(&e.image_id) {
                Some((pose, cam)) => {
                    let xy = self.images[&e.image_id].points2d[e.point2d_idx as usize].xy;
                    Self::squared_reprojection_error(pose, cam, &xy, &p.xyz)
                }
                None => f64::MAX,
            };
            s += sq.sqrt();
        }
        s / p.track.len() as f64
    }

    /// 모든 3D 점의 error 를 픽셀 평균 재투영 오차로 재계산.
    pub fn update_point3d_errors(&mut self) {
        let cache = self.pose_cache();
        let errs: Vec<(Point3DId, f64)> = self.points3d.par_iter().map(|(id, p)| (*id, self.compute_point_error(&cache, p))).collect();
        drop(cache);
        for (id, e) in errs {
            let p = self.points3d.get_mut(&id).expect("존재");
            if p.error != e {
                Arc::make_mut(p).error = e;
            }
        }
    }
    /// 점 하나의 평균 픽셀 재투영 오차(저장하지 않음).
    pub fn point3d_reprojection_error(&self, id: Point3DId) -> Option<f64> {
        let p = self.point3d(id)?;
        let mut cache = HashMap::new();
        for e in &p.track {
            if let (Some(pose), Some(im)) = (self.world_to_cam(e.image_id), self.image(e.image_id)) {
                cache.insert(e.image_id, (pose, &self.cameras[&im.camera_id]));
            }
        }
        Some(self.compute_point_error(&cache, p))
    }

    // ================= 필터 =================

    /// 범용 관측 필터: 관측마다 `error_fn(카메라, world_to_cam, 2D 좌표, 3D 점)` 를 구해
    /// max_error 초과(NaN 포함)인 것을 삭제 대상으로 한다. 트랙 길이 < 2 인 점은 즉시 삭제,
    /// 삭제 대상 수가 (길이 − 1) 이상이면 점 전체 삭제, 아니면 해당 관측만 삭제.
    /// 자세 없는 영상의 관측 오차는 +∞. 반환: 삭제된 관측 수.
    pub fn filter_observations<F>(
        &mut self,
        point_ids: Option<&[Point3DId]>,
        max_error: f64,
        error_fn: F,
        update: FilterErrorUpdate,
    ) -> usize
    where
        F: Fn(&Camera, &Rigid3, &Vec2, &Vec3) -> f64 + Sync,
    {
        let ids: Vec<Point3DId> = match point_ids {
            Some(v) => v.iter().copied().filter(|i| self.points3d.contains_key(i)).collect(),
            None => self.point3d_ids(),
        };
        let cache = self.pose_cache();
        // (점 id, 관측별 오차)
        let evals: Vec<(Point3DId, Vec<f64>)> = ids
            .par_iter()
            .map(|id| {
                let p = &self.points3d[id];
                let errs = p
                    .track
                    .iter()
                    .map(|e| match cache.get(&e.image_id) {
                        Some((pose, cam)) => {
                            let xy = self.images[&e.image_id].points2d[e.point2d_idx as usize].xy;
                            error_fn(cam, pose, &xy, &p.xyz)
                        }
                        None => f64::INFINITY,
                    })
                    .collect();
                (*id, errs)
            })
            .collect();
        drop(cache);
        let mut removed = 0usize;
        for (id, errs) in evals {
            let track = self.points3d[&id].track.clone();
            let len = track.len();
            if len < 2 {
                removed += len;
                self.delete_point3d(id).expect("존재");
                continue;
            }
            let bad: Vec<usize> = (0..len).filter(|&k| errs[k].is_nan() || errs[k] > max_error).collect();
            if bad.len() >= len - 1 {
                removed += len;
                self.delete_point3d(id).expect("존재");
                continue;
            }
            let mut point_deleted = false;
            for &k in &bad {
                removed += 1;
                if self.delete_observation(track[k].image_id, track[k].point2d_idx).expect("연결 불변식") {
                    point_deleted = true;
                    break;
                }
            }
            if point_deleted {
                continue;
            }
            let inl: Vec<f64> = (0..len).filter(|k| !bad.contains(k)).map(|k| errs[k]).collect();
            let new_err = match update {
                FilterErrorUpdate::None => None,
                FilterErrorUpdate::SumOverOriginalLength => Some(inl.iter().sum::<f64>() / len as f64),
                FilterErrorUpdate::MeanOfRemaining => Some(inl.iter().sum::<f64>() / inl.len() as f64),
            };
            if let Some(e) = new_err {
                self.set_point3d_error(id, e).expect("존재");
            }
        }
        removed
    }

    /// 픽셀 재투영 오차 필터. 반환: 삭제된 관측 수.
    pub fn filter_observations_with_large_reprojection_error(
        &mut self,
        max_reproj_error_px: f64,
        point_ids: Option<&[Point3DId]>,
    ) -> usize {
        self.filter_observations(
            point_ids,
            max_reproj_error_px,
            |cam, pose, xy, xyz| Self::squared_reprojection_error(pose, cam, xy, xyz).sqrt(),
            FilterErrorUpdate::SumOverOriginalLength,
        )
    }

    /// 삼각측량 각 필터: 트랙 영상 쌍 중 하나라도 각 ≥ min_tri_angle_deg 이면 유지, 아니면 점 삭제.
    /// 반환: 삭제된 관측 수.
    pub fn filter_points3d_with_small_triangulation_angle(&mut self, min_tri_angle_deg: f64, point_ids: Option<&[Point3DId]>) -> usize {
        let min_rad = min_tri_angle_deg.to_radians();
        let ids: Vec<Point3DId> = match point_ids {
            Some(v) => v.iter().copied().filter(|i| self.points3d.contains_key(i)).collect(),
            None => self.point3d_ids(),
        };
        let centers: HashMap<ImageId, Vec3> = self.pose_cache().into_iter().map(|(k, (p, _))| (k, p.center())).collect();
        let to_delete: Vec<(Point3DId, usize)> = ids
            .par_iter()
            .filter_map(|id| {
                let p = &self.points3d[id];
                let cs: Vec<Vec3> = p.track.iter().filter_map(|e| centers.get(&e.image_id).copied()).collect();
                if max_triangulation_angle(&cs, &p.xyz) >= min_rad {
                    None
                } else {
                    Some((*id, p.track.len()))
                }
            })
            .collect();
        let mut removed = 0;
        for (id, n) in to_delete {
            removed += n;
            self.delete_point3d(id).expect("존재");
        }
        removed
    }

    // ================= 변환·정규화 =================

    /// 상사 변환 적용: 점, 프레임 자세, rig 센서 평행이동(s 배).
    pub fn transform(&mut self, t: &Sim3) {
        for p in self.points3d.values_mut() {
            let m = Arc::make_mut(p);
            m.xyz = t.transform_point(&m.xyz);
        }
        for f in self.frames.values_mut() {
            if let Some(p) = f.world_to_rig {
                f.world_to_rig = Some(t.transform_pose(&p));
            }
        }
        for r in self.rigs.values_mut() {
            for v in r.sensors.values_mut().flatten() {
                v.translation *= t.scale;
            }
        }
    }

    /// 정규화. 적용한 Sim3 반환, 등록 프레임 < 2 면 None.
    pub fn normalize(&mut self, opts: &NormalizeOptions) -> Option<Sim3> {
        if self.registered_frames.len() < 2 {
            return None;
        }
        let coords: Vec<Vec3> = if opts.use_images {
            self.registered_images().iter().filter_map(|i| self.projection_center(*i)).collect()
        } else {
            self.points3d.values().map(|p| p.xyz).collect()
        };
        if coords.is_empty() {
            return None;
        }
        let n = coords.len();
        let e = n - 1;
        let lo = e.min((opts.p0 * e as f64).floor() as usize);
        let hi = e.min((opts.p1 * e as f64).ceil() as usize);
        let mut bmin = Vec3::zeros();
        let mut bmax = Vec3::zeros();
        let mut center = Vec3::zeros();
        for ax in 0..3 {
            let mut v: Vec<f64> = coords.iter().map(|c| c[ax]).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            bmin[ax] = v[lo];
            bmax[ax] = v[hi];
            let slice = &v[lo..=hi];
            center[ax] = slice.iter().sum::<f64>() / slice.len() as f64;
        }
        let diag = (bmax - bmin).norm();
        let s = if opts.fixed_scale || diag < f64::EPSILON { 1.0 } else { opts.extent / diag };
        let t = Sim3::new(s, crate::geometry::Quat::IDENTITY, -s * center);
        self.transform(&t);
        Some(t)
    }

    // ================= 영상 삭제(image_deleter) =================

    /// id 목록의 영상이 속한 프레임을 등록 해제. 반환: 경고 문구 목록.
    pub fn deregister_images_by_id(&mut self, ids: &[ImageId]) -> Vec<String> {
        let mut warnings = Vec::new();
        for &id in ids {
            match self.image(id).map(|im| im.frame_id) {
                None => warnings.push(format!("Image with ID {id} does not exist")),
                Some(fid) => match self.deregister_frame(fid) {
                    Ok(true) => {}
                    _ => warnings.push(format!("Image with ID {id} is not registered")),
                },
            }
        }
        warnings
    }
    /// 이름 목록의 영상을 등록 해제. 반환: 경고 문구 목록.
    pub fn deregister_images_by_name(&mut self, names: &[&str]) -> Vec<String> {
        let mut warnings = Vec::new();
        for &name in names {
            match self.image_by_name(name).map(|im| im.image_id) {
                None => warnings.push(format!("Image with name {name} does not exist")),
                Some(id) => warnings.extend(self.deregister_images_by_id(&[id])),
            }
        }
        warnings
    }

    /// 등록되지 않은 프레임·영상을 삭제하고, 쓰이지 않는 카메라·rig 도 삭제.
    pub fn remove_unregistered(&mut self) {
        let unreg: Vec<ImageId> = self.images.keys().copied().filter(|i| !self.is_image_registered(*i)).collect();
        for &iid in &unreg {
            let n = self.images[&iid].points2d.len();
            for idx in 0..n {
                if self.images[&iid].points2d[idx].has_point3d() {
                    let _ = self.delete_observation(iid, idx as Point2DIdx);
                }
            }
            self.images.remove(&iid);
        }
        let reg: std::collections::HashSet<FrameId> = self.registered_frames.iter().copied().collect();
        self.frames.retain(|k, _| reg.contains(k));
        let used_cams: std::collections::HashSet<CameraId> = self.images.values().map(|im| im.camera_id).collect();
        let used_rigs: std::collections::HashSet<RigId> = self.frames.values().map(|f| f.rig_id).collect();
        self.cameras.retain(|k, _| used_cams.contains(k));
        self.rigs.retain(|k, _| used_rigs.contains(k));
    }

    // ================= 색 =================

    /// 색 추출: 등록 영상마다 `load(영상)` 으로 표본기를 얻어
    /// 각 관측의 (x − 0.5, y − 0.5) 를 넘기고, 점별 평균을 반올림. 표본이 없는 점은 검정.
    pub fn extract_colors<S, F>(&mut self, mut load: F)
    where
        F: FnMut(&Image) -> Option<S>,
        S: Fn(f64, f64) -> Option<[f64; 3]>,
    {
        let mut acc: HashMap<Point3DId, ([f64; 3], usize)> = HashMap::new();
        for iid in self.registered_images() {
            let im = &self.images[&iid];
            let Some(sampler) = load(im) else { continue };
            for p in im.points2d.iter().filter(|p| p.has_point3d()) {
                if let Some(c) = sampler(p.xy.x - 0.5, p.xy.y - 0.5) {
                    let a = acc.entry(p.point3d_id).or_insert(([0.0; 3], 0));
                    for (dst, v) in a.0.iter_mut().zip(c) {
                        *dst += v;
                    }
                    a.1 += 1;
                }
            }
        }
        for (id, p) in self.points3d.iter_mut() {
            let color = match acc.get(id) {
                Some((s, n)) => s.map(|v| (v / *n as f64).round().clamp(0.0, 255.0) as u8),
                None => [0, 0, 0],
            };
            if p.color != color {
                Arc::make_mut(p).color = color;
            }
        }
    }

    // ================= 검사 =================

    /// 불변식 검사: 양방향 연결, 영상별 연결 수 캐시, 등록 영상 수.
    pub fn check_invariants(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Invariant(m));
        for (pid, p) in &self.points3d {
            for e in &p.track {
                let Some(im) = self.image(e.image_id) else { return bad(format!("점 {pid}: 영상 {} 없음", e.image_id)) };
                match im.points2d.get(e.point2d_idx as usize) {
                    Some(q) if q.point3d_id == *pid => {}
                    _ => return bad(format!("점 {pid}: ({}, {}) 역연결 불일치", e.image_id, e.point2d_idx)),
                }
            }
        }
        for im in self.images() {
            let mut n = 0;
            for (idx, q) in im.points2d.iter().enumerate() {
                if q.has_point3d() {
                    n += 1;
                    let Some(p) = self.point3d(q.point3d_id) else {
                        return bad(format!("영상 {} 2D {idx}: 3D 점 {} 없음", im.image_id, q.point3d_id));
                    };
                    if !p.track.contains(&TrackEntry::new(im.image_id, idx as u32)) {
                        return bad(format!("영상 {} 2D {idx}: 트랙에 없음", im.image_id));
                    }
                }
            }
            if n != im.num_points3d {
                return bad(format!("영상 {} 연결 수 캐시 {} != {}", im.image_id, im.num_points3d, n));
            }
        }
        if self.registered_images().len() != self.registered_image_count {
            return bad("등록 영상 수 불일치".into());
        }
        if let Some(m) = self.points3d.keys().next_back() {
            if *m > self.max_point3d_id {
                return bad("최대 3D 점 id 불일치".into());
            }
        }
        Ok(())
    }
}

/// u8 RGB(행 우선, 3채널) 영상에서 양선형 보간. (x, y) 는 화소 중심이 정수인 좌표.
/// 범위 밖이면 None.
pub fn bilinear_rgb(data: &[u8], width: usize, height: usize, x: f64, y: f64) -> Option<[f64; 3]> {
    if width == 0 || height == 0 || !(x >= 0.0 && y >= 0.0 && x <= (width - 1) as f64 && y <= (height - 1) as f64) {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;
    let px = |xx: usize, yy: usize, c: usize| data[(yy * width + xx) * 3 + c] as f64;
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        *o = (1.0 - dy) * ((1.0 - dx) * px(x0, y0, c) + dx * px(x1, y0, c)) + dy * ((1.0 - dx) * px(x0, y1, c) + dx * px(x1, y1, c));
    }
    Some(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::camera::CameraModelKind;
    use crate::geometry::Quat;

    /// 카메라 1개, 영상 n 장(x 축으로 늘어선 카메라), 각 영상 2D 점 m 개(모두 같은 좌표 집합).
    pub(crate) fn line_scene(n: u32, m: usize) -> Reconstruction {
        let mut r = Reconstruction::new();
        let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 100.0, 200, 200);
        cam.camera_id = 1;
        r.add_camera_own_rig(cam).unwrap();
        for i in 1..=n {
            let pts = (0..m).map(|k| Vec2::new(10.0 + k as f64, 20.0));
            let pose = Rigid3::new(Quat::IDENTITY, Vec3::new(-(i as f64), 0.0, 0.0));
            r.add_image_own_frame(Image::new(i, format!("im{i}.jpg"), 1, pts), Some(pose)).unwrap();
            r.register_image(i).unwrap();
        }
        r
    }

    #[test]
    fn add_delete_observations_and_invariants() {
        let mut r = line_scene(4, 5);
        let t = |i, k| TrackEntry::new(i, k);
        let p2 = r.add_point3d(Vec3::new(0.0, 0.0, 5.0), vec![t(1, 0), t(2, 0)], [1, 2, 3]).unwrap();
        let p3 = r.add_point3d(Vec3::new(0.0, 0.0, 5.0), vec![t(1, 1), t(2, 1), t(3, 1)], [0; 3]).unwrap();
        let p4 = r.add_point3d(Vec3::new(0.0, 0.0, 5.0), vec![t(1, 2), t(2, 2), t(3, 2), t(4, 2)], [0; 3]).unwrap();
        assert_eq!((p2, p3, p4), (1, 2, 3));
        r.check_invariants().unwrap();
        // conflicting link
        assert!(r.add_point3d(Vec3::zeros(), vec![t(1, 0), t(3, 0)], [0; 3]).is_err());
        r.check_invariants().unwrap();
        assert_eq!(r.image(1).unwrap().num_points3d(), 3);
        r.add_observation(p2, t(3, 0)).unwrap();
        assert!(r.add_observation(p2, t(3, 0)).is_err());
        assert_eq!(r.point3d(p2).unwrap().track.len(), 3);
        assert!(!r.delete_observation(3, 0).unwrap());
        assert!(r.delete_observation(1, 0).unwrap()); // len 2 → point deleted
        assert!(!r.exists_point3d(p2));
        assert_eq!(r.image(2).unwrap().num_points3d(), 2);
        r.check_invariants().unwrap();
        // merge
        let m = r.merge_points3d(p3, p4).unwrap();
        assert_eq!(m, 4);
        assert_eq!(r.point3d(m).unwrap().track.len(), 7);
        r.check_invariants().unwrap();
        // cheap clone + copy on write
        let snapshot = r.clone();
        r.delete_point3d(m).unwrap();
        assert!(snapshot.exists_point3d(m));
        assert_eq!(snapshot.image(1).unwrap().num_points3d(), 2);
        assert_eq!(r.image(1).unwrap().num_points3d(), 0);
        snapshot.check_invariants().unwrap();
        r.check_invariants().unwrap();
    }

    #[test]
    fn image_deleter_rule() {
        let mut r = line_scene(4, 5);
        let t = |i, k| TrackEntry::new(i, k);
        let a = r.add_point3d(Vec3::new(1.0, 0.0, 5.0), vec![t(1, 0), t(2, 0)], [0; 3]).unwrap();
        let b = r.add_point3d(Vec3::new(1.0, 0.0, 5.0), vec![t(1, 1), t(2, 1), t(3, 1)], [0; 3]).unwrap();
        let c = r.add_point3d(Vec3::new(1.0, 0.0, 5.0), vec![t(1, 2), t(2, 2), t(3, 2), t(4, 2)], [0; 3]).unwrap();
        let d = r.add_point3d(Vec3::new(1.0, 0.0, 5.0), vec![t(2, 3), t(3, 3)], [0; 3]).unwrap();
        for (id, e) in [(a, 1.0), (b, 2.0), (c, 3.0), (d, 4.0)] {
            r.set_point3d_error(id, e).unwrap();
        }
        let w = r.deregister_images_by_id(&[1, 99]);
        assert_eq!(w.len(), 1);
        assert!(!r.exists_point3d(a));
        assert_eq!(r.point3d(b).unwrap().track.len(), 2);
        assert_eq!(r.point3d(c).unwrap().track.len(), 3);
        assert_eq!(r.point3d(d).unwrap().track.len(), 2);
        assert_eq!(r.point3d(b).unwrap().error, 2.0);
        assert_eq!(r.point3d(c).unwrap().error, 3.0);
        assert!(!r.is_image_registered(1));
        assert!(!r.registered_images().contains(&1));
        assert_eq!(r.registered_image_count(), 3);
        assert_eq!(r.num_images(), 4);
        // second delete: warning only
        assert_eq!(r.deregister_images_by_name(&["im1.jpg"]).len(), 1);
        r.check_invariants().unwrap();
        r.remove_unregistered();
        assert_eq!(r.num_images(), 3);
        r.check_invariants().unwrap();
    }

    #[test]
    fn filters() {
        let r = line_scene(3, 4);
        let t = |i, k| TrackEntry::new(i, k);
        // project a point at (0.5,0,10) in camera of image 1: x=0.5+... set 2D obs to exact projection for images
        let x = Vec3::new(2.0, 0.0, 10.0);
        let mut obs = Vec::new();
        for i in 1..=3u32 {
            let pose = r.world_to_cam(i).unwrap();
            let px = r.camera(1).unwrap().cam_to_img(&(pose * x)).unwrap();
            obs.push(px);
        }
        // overwrite image 2D coords via re-creating scene
        let mut r2 = Reconstruction::new();
        r2.add_camera_own_rig(r.camera(1).unwrap().clone()).unwrap();
        for i in 1..=3u32 {
            let mut pts = vec![obs[(i - 1) as usize]; 4];
            pts[1].x += 4.01; // obs idx 1 is 4.01 px off
            pts[2].x += 3.99;
            r2.add_image_own_frame(Image::new(i, format!("{i}"), 1, pts), r.world_to_cam(i)).unwrap();
            r2.register_image(i).unwrap();
        }
        let r = &mut r2;
        let good = r.add_point3d(x, vec![t(1, 0), t(2, 0), t(3, 0)], [0; 3]).unwrap();
        let one_bad = r.add_point3d(x, vec![t(1, 1), t(2, 2), t(3, 3)], [0; 3]);
        // t(2,2) is 3.99 off → inlier; t(1,1) 4.01 → bad
        let one_bad = one_bad.unwrap();
        let two_bad = r.add_point3d(x, vec![t(2, 1), t(3, 1), t(1, 3)], [0; 3]).unwrap();
        let removed = r.filter_observations_with_large_reprojection_error(4.0, None);
        assert_eq!(removed, 1 + 3);
        assert!(r.exists_point3d(good));
        assert_eq!(r.point3d(one_bad).unwrap().track.len(), 2);
        assert!((r.point3d(one_bad).unwrap().error - 3.99 / 3.0).abs() < 1e-9);
        assert!(!r.exists_point3d(two_bad));
        r.check_invariants().unwrap();
        // triangulation angle: centers at x=1,2,3, point at (2,0,10): max angle between c1,c3
        let ang = crate::geometry::triangulation_angle(&Vec3::new(1.0, 0.0, 0.0), &Vec3::new(3.0, 0.0, 0.0), &x).to_degrees();
        // one_bad 는 영상 2,3 만 남아 최대 각이 작다 → 먼저 삭제(관측 2개).
        assert_eq!(r.filter_points3d_with_small_triangulation_angle(ang - 1e-9, None), 2);
        assert!(r.exists_point3d(good));
        assert_eq!(r.filter_points3d_with_small_triangulation_angle(ang + 1e-9, None), 3);
        assert_eq!(r.num_points3d(), 0);
        r.check_invariants().unwrap();
    }

    #[test]
    fn errors_and_normalize() {
        let mut r = line_scene(11, 1);
        r.update_point3d_errors();
        let opts = NormalizeOptions::default();
        let t = r.normalize(&opts).unwrap();
        let centers: Vec<Vec3> = r.registered_images().iter().map(|i| r.projection_center(*i).unwrap()).collect();
        // n=11 → lo=1, hi=9 → x from 2..10 originally, diag 8 → s=10/8
        assert!((t.scale - 10.0 / 8.0).abs() < 1e-12);
        let mut xs: Vec<f64> = centers.iter().map(|c| c.x).collect();
        xs.sort_by(|a, b| a.total_cmp(b));
        assert!(((xs[9] - xs[1]) - 10.0).abs() < 1e-9);
        let mean: f64 = xs[1..=9].iter().sum::<f64>() / 9.0;
        assert!(mean.abs() < 1e-9);
    }

    #[test]
    fn rig_pose_derivation() {
        let mut r = Reconstruction::new();
        for c in [1u32, 2] {
            let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 100.0, 200, 200);
            cam.camera_id = c;
            r.add_camera(cam).unwrap();
        }
        let mut rig = Rig::new(7);
        rig.ref_sensor_id = Some(SensorKey::camera(1));
        let rig_to_c2 = Rigid3::new(Quat::from_axis_angle(&Vec3::y(), 0.2), Vec3::new(0.5, 0.0, 0.0));
        rig.sensors.insert(SensorKey::camera(2), Some(rig_to_c2));
        r.add_rig(rig).unwrap();
        let mut f = Frame::new(3, 7);
        f.attach_data(SensorDataKey::image(2, 11));
        f.attach_data(SensorDataKey::image(1, 10));
        r.add_frame(f).unwrap();
        let mut im = Image::new(10, "a", 1, []);
        im.frame_id = 3;
        r.add_image(im).unwrap();
        let mut im = Image::new(11, "b", 2, []);
        im.frame_id = 3;
        r.add_image(im).unwrap();
        let want = Rigid3::new(Quat::from_axis_angle(&Vec3::x(), 0.1), Vec3::new(1.0, 2.0, 3.0));
        r.set_world_to_cam(11, want).unwrap();
        let got = r.world_to_cam(11).unwrap();
        assert!((got.translation - want.translation).norm() < 1e-12);
        r.register_frame(3).unwrap();
        assert_eq!(r.registered_image_count(), 2);
        assert_eq!(r.registered_images(), vec![10, 11]);
        let s = Sim3::new(2.0, Quat::from_axis_angle(&Vec3::z(), 0.4), Vec3::new(1.0, 1.0, 1.0));
        let c_before = r.projection_center(11).unwrap();
        r.transform(&s);
        assert!((r.projection_center(11).unwrap() - s.transform_point(&c_before)).norm() < 1e-10);
    }

    #[test]
    fn bilinear() {
        let data = vec![0u8, 0, 0, 100, 100, 100, 200, 200, 200, 255, 255, 255];
        let c = bilinear_rgb(&data, 2, 2, 0.5, 0.5).unwrap();
        assert!((c[0] - 138.75).abs() < 1e-9);
        assert!(bilinear_rgb(&data, 2, 2, 1.5, 0.0).is_none());
    }
}
