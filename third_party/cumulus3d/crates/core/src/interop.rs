/*
 * interop.rs
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

//! 외부 도구와 주고받는 파일 형식(상호운용).
//!
//! 외부 SfM 도구(COLMAP)의 모델 형식(`cameras`/`images`/`points3D`, 선택적으로 `rigs`/`frames`,
//! 각각 `.bin`·`.txt`)과 조밀 복원용 작업 폴더 구성(`stereo/patch-match.cfg`, `stereo/fusion.cfg`)을
//! 이 모듈 한 곳에서만 다룬다. 파일 이름·헤더 문자열·필드 순서는 그 형식을 그대로 따른다.

use crate::camera::{Camera, CameraModelKind};
use crate::error::{Error, Result};
use crate::geometry::{Rigid3, Vec2, Vec3};
use crate::ids::*;
use crate::io::binary::{check_count, ReadLe, WriteLe};
use crate::io::fmt::g17;
use crate::reconstruction::{Frame, Image, Point3D, Reconstruction, Rig, TrackEntry};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// images 파일의 영상 순서.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ImageOrder {
    /// 기본 동작: 등록 프레임 순서 → 프레임 내 데이터 순서.
    #[default]
    Registration,
    /// 영상 id 오름차순(결정적, 비교 테스트용).
    ById,
}

struct RawImage {
    image_id: ImageId,
    world_to_cam: Rigid3,
    camera_id: CameraId,
    name: String,
    points: Vec<Vec2>,
}

struct RawModel {
    cameras: Vec<Camera>,
    rigs: Option<Vec<Rig>>,
    frames: Option<Vec<Frame>>,
    images: Vec<RawImage>,
    points: Vec<(Point3DId, Point3D)>,
}

fn assemble(raw: RawModel) -> Result<Reconstruction> {
    let mut rec = Reconstruction::new();
    for c in raw.cameras {
        rec.add_camera(c)?;
    }
    let mut reg_order: Vec<FrameId> = Vec::new();
    match (raw.rigs, raw.frames) {
        (None, None) => {
            // 구버전 모델: 카메라마다 rig, 영상마다 프레임.
            let cam_ids: Vec<CameraId> = rec.cameras().keys().copied().collect();
            for c in cam_ids {
                rec.add_rig(Rig::trivial(c))?;
            }
            for im in raw.images {
                let image = Image::new(im.image_id, im.name, im.camera_id, im.points);
                rec.add_image_own_frame(image, Some(im.world_to_cam))?;
                reg_order.push(im.image_id);
            }
        }
        (Some(rigs), Some(frames)) => {
            for r in rigs {
                rec.add_rig(r)?;
            }
            let mut data_to_frame: HashMap<SensorDataKey, FrameId> = HashMap::new();
            let mut frame_order = Vec::new();
            for f in frames {
                for d in f.data_ids() {
                    data_to_frame.insert(*d, f.frame_id);
                }
                frame_order.push(f.frame_id);
                rec.add_frame(f)?;
            }
            // 설계 결정: 읽은 모델의 등록 순서. images 파일에 처음 나타나는 프레임 순서를 쓰면
            // 다시 쓸 때 images 레코드 순서가 보존된다(바이트 왕복). 나머지 프레임은 frames 파일 순서.
            let mut seen = std::collections::HashSet::new();
            for im in &raw.images {
                if let Some(f) = data_to_frame.get(&SensorDataKey::image(im.camera_id, im.image_id)) {
                    if seen.insert(*f) {
                        reg_order.push(*f);
                    }
                }
            }
            reg_order.extend(frame_order.into_iter().filter(|f| !seen.contains(f)));
            for im in raw.images {
                let fid = *data_to_frame
                    .get(&SensorDataKey::image(im.camera_id, im.image_id))
                    .ok_or_else(|| Error::Format(format!("영상 {} 이 어떤 프레임에도 없음", im.image_id)))?;
                let mut image = Image::new(im.image_id, im.name, im.camera_id, im.points);
                image.frame_id = fid;
                rec.add_image(image)?;
            }
        }
        _ => {
            // 설계 결정: rigs 와 frames 중 하나만 있는 모델은 불완전한 입력 → 오류.
            return Err(Error::Format("rigs 와 frames 는 함께 있어야 함".into()));
        }
    }
    for fid in reg_order {
        if rec.frame(fid).is_some_and(|f| f.has_pose()) {
            rec.register_frame(fid)?;
        }
    }
    for (id, p) in raw.points {
        // 트랙이 연결을 정의한다(images 의 3D 점 id 는 트랙과 일치한다고 가정).
        rec.add_point3d_with_id(id, p)?;
    }
    Ok(rec)
}

/// 모델 디렉터리 읽기: 이진(cameras/images/points3D.bin) 우선, 아니면 텍스트.
pub fn read_model(dir: impl AsRef<Path>) -> Result<Reconstruction> {
    let d = dir.as_ref();
    let all = |ext: &str| ["cameras", "images", "points3D"].iter().all(|n| d.join(format!("{n}.{ext}")).is_file());
    if all("bin") {
        read_model_binary(d)
    } else if all("txt") {
        read_model_text(d)
    } else {
        Err(Error::NotFound(format!("{} 에 모델 파일이 없음", d.display())))
    }
}

fn open(p: &Path) -> Result<BufReader<File>> {
    Ok(BufReader::new(File::open(p).map_err(|e| Error::Io(std::io::Error::new(e.kind(), format!("{}: {e}", p.display()))))?))
}

// ======================= 이진 =======================

/// 이진 모델 파일(`*.bin`) 읽기.
pub fn read_model_binary(dir: impl AsRef<Path>) -> Result<Reconstruction> {
    let d = dir.as_ref();
    let cameras = {
        let mut r = open(&d.join("cameras.bin"))?;
        let n = check_count(r.get_u64()?, "카메라")?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let id = r.get_u32()?;
            let model = CameraModelKind::from_id(r.get_i32()?)?;
            let w = r.get_u64()?;
            let h = r.get_u64()?;
            let mut params = Vec::with_capacity(model.num_params());
            for _ in 0..model.num_params() {
                params.push(r.get_f64()?);
            }
            v.push(Camera::new(id, model, w, h, params)?);
        }
        v
    };
    let rigs = if d.join("rigs.bin").is_file() {
        let mut r = open(&d.join("rigs.bin"))?;
        let n = check_count(r.get_u64()?, "rig")?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let mut rig = Rig::new(r.get_u32()?);
            let ns = r.get_u32()?;
            if ns > 0 {
                let t = sensor_type(r.get_i32()?)?;
                rig.ref_sensor_id = Some(SensorKey::new(t, r.get_u32()?));
                for _ in 1..ns {
                    let t = sensor_type(r.get_i32()?)?;
                    let sid = SensorKey::new(t, r.get_u32()?);
                    let pose = if r.get_u8()? != 0 { Some(Rigid3::from_params(&r.get_f64_array::<7>()?)) } else { None };
                    rig.sensors.insert(sid, pose);
                }
            }
            v.push(rig);
        }
        Some(v)
    } else {
        None
    };
    let frames = if d.join("frames.bin").is_file() {
        let mut r = open(&d.join("frames.bin"))?;
        let n = check_count(r.get_u64()?, "프레임")?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let fid = r.get_u32()?;
            let rid = r.get_u32()?;
            let mut f = Frame::new(fid, rid);
            f.world_to_rig = Some(Rigid3::from_params(&r.get_f64_array::<7>()?));
            let nd = r.get_u32()?;
            for _ in 0..nd {
                let t = sensor_type(r.get_i32()?)?;
                let sid = r.get_u32()?;
                let did = r.get_u64()?;
                f.attach_data(SensorDataKey::new(SensorKey::new(t, sid), did));
            }
            v.push(f);
        }
        Some(v)
    } else {
        None
    };
    let images = {
        let mut r = open(&d.join("images.bin"))?;
        let n = check_count(r.get_u64()?, "영상")?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let image_id = r.get_u32()?;
            let pose = Rigid3::from_params(&r.get_f64_array::<7>()?);
            let camera_id = r.get_u32()?;
            let name = r.get_cstring()?;
            let np = check_count(r.get_u64()?, "2D 점")?;
            let mut points = Vec::with_capacity(np);
            for _ in 0..np {
                let x = r.get_f64()?;
                let y = r.get_f64()?;
                let _point3d_id = r.get_u64()?; // 연결은 points3D 트랙이 정의
                points.push(Vec2::new(x, y));
            }
            v.push(RawImage { image_id, world_to_cam: pose, camera_id, name, points });
        }
        v
    };
    let points = {
        let mut r = open(&d.join("points3D.bin"))?;
        let n = check_count(r.get_u64()?, "3D 점")?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let id = r.get_u64()?;
            let xyz = Vec3::from(r.get_f64_array::<3>()?);
            let color = [r.get_u8()?, r.get_u8()?, r.get_u8()?];
            let error = r.get_f64()?;
            let tl = check_count(r.get_u64()?, "트랙")?;
            let mut track = Vec::with_capacity(tl);
            for _ in 0..tl {
                let i = r.get_u32()?;
                let k = r.get_u32()?;
                track.push(TrackEntry::new(i, k));
            }
            v.push((id, Point3D { xyz, color, error, track }));
        }
        v
    };
    assemble(RawModel { cameras, rigs, frames, images, points })
}

fn sensor_type(v: i32) -> Result<SensorKind> {
    SensorKind::from_i32(v).ok_or_else(|| Error::Format(format!("잘못된 센서 종류 {v}")))
}

fn create(p: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(File::create(p).map_err(|e| Error::Io(std::io::Error::new(e.kind(), format!("{}: {e}", p.display()))))?))
}

/// 쓰기 대상 영상 id(등록 영상만).
fn images_to_write(rec: &Reconstruction, order: ImageOrder) -> Vec<ImageId> {
    let mut ids = rec.registered_images();
    if order == ImageOrder::ById {
        ids.sort();
    }
    ids
}

fn posed_frames(rec: &Reconstruction) -> Vec<&Frame> {
    rec.frames().values().filter(|f| f.has_pose()).collect()
}

/// 이진 다섯 파일 쓰기(디렉터리는 존재해야 함).
pub fn write_model_binary(rec: &Reconstruction, dir: impl AsRef<Path>, order: ImageOrder) -> Result<()> {
    let d = dir.as_ref();
    check_dir(d)?;
    {
        let mut w = create(&d.join("cameras.bin"))?;
        w.put_u64(rec.num_cameras() as u64)?;
        for c in rec.cameras().values() {
            w.put_u32(c.camera_id)?;
            w.put_i32(c.model.id())?;
            w.put_u64(c.width)?;
            w.put_u64(c.height)?;
            w.put_f64s(&c.params)?;
        }
        w.flush()?;
    }
    {
        let mut w = create(&d.join("rigs.bin"))?;
        w.put_u64(rec.num_rigs() as u64)?;
        for r in rec.rigs().values() {
            w.put_u32(r.rig_id)?;
            w.put_u32(r.num_sensors() as u32)?;
            if let Some(rs) = r.ref_sensor_id {
                w.put_i32(rs.sensor_type.as_i32())?;
                w.put_u32(rs.id)?;
                for (s, pose) in &r.sensors {
                    w.put_i32(s.sensor_type.as_i32())?;
                    w.put_u32(s.id)?;
                    match pose {
                        Some(p) => {
                            w.put_u8(1)?;
                            w.put_f64s(&p.to_params())?;
                        }
                        None => w.put_u8(0)?,
                    }
                }
            }
        }
        w.flush()?;
    }
    {
        let frames = posed_frames(rec);
        let mut w = create(&d.join("frames.bin"))?;
        w.put_u64(frames.len() as u64)?;
        for f in frames {
            w.put_u32(f.frame_id)?;
            w.put_u32(f.rig_id)?;
            w.put_f64s(&f.world_to_rig.expect("자세 있음").to_params())?;
            w.put_u32(f.data_ids().len() as u32)?;
            for dd in f.data_ids() {
                w.put_i32(dd.sensor_id.sensor_type.as_i32())?;
                w.put_u32(dd.sensor_id.id)?;
                w.put_u64(dd.id)?;
            }
        }
        w.flush()?;
    }
    {
        let ids = images_to_write(rec, order);
        let mut w = create(&d.join("images.bin"))?;
        w.put_u64(ids.len() as u64)?;
        for id in ids {
            let im = rec.image(id).expect("등록 영상");
            let pose = rec.world_to_cam(id).expect("등록 영상 자세");
            w.put_u32(id)?;
            w.put_f64s(&pose.to_params())?;
            w.put_u32(im.camera_id)?;
            w.write_all(im.name.as_bytes())?;
            w.put_u8(0)?;
            w.put_u64(im.num_points2d() as u64)?;
            for p in im.points2d() {
                w.put_f64(p.xy.x)?;
                w.put_f64(p.xy.y)?;
                w.put_u64(p.point3d_id)?;
            }
        }
        w.flush()?;
    }
    {
        let mut w = create(&d.join("points3D.bin"))?;
        w.put_u64(rec.num_points3d() as u64)?;
        for (id, p) in rec.points3d() {
            w.put_u64(id)?;
            w.put_f64s(p.xyz.as_slice())?;
            w.write_all(&p.color)?;
            w.put_f64(p.error)?;
            w.put_u64(p.track.len() as u64)?;
            for e in &p.track {
                w.put_u32(e.image_id)?;
                w.put_u32(e.point2d_idx)?;
            }
        }
        w.flush()?;
    }
    Ok(())
}

fn check_dir(d: &Path) -> Result<()> {
    if !d.is_dir() {
        return Err(Error::NotFound(format!("출력 디렉터리 {} 가 없음", d.display())));
    }
    Ok(())
}

// ======================= 텍스트 =======================

/// 주석·빈 줄을 건너뛰는 줄 읽기기. images.txt 둘째 줄은 `next_raw` 로 읽는다.
struct Lines {
    path: PathBuf,
    lines: std::io::Lines<BufReader<File>>,
    line_no: usize,
}

impl Lines {
    fn open(path: PathBuf) -> Result<Self> {
        let r = open(&path)?;
        Ok(Self { path, lines: r.lines(), line_no: 0 })
    }
    fn next_raw(&mut self) -> Result<Option<String>> {
        match self.lines.next() {
            Some(l) => {
                self.line_no += 1;
                Ok(Some(l?.trim().to_string()))
            }
            None => Ok(None),
        }
    }
    fn next_data(&mut self) -> Result<Option<String>> {
        while let Some(l) = self.next_raw()? {
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            return Ok(Some(l));
        }
        Ok(None)
    }
    fn err(&self, msg: impl Into<String>) -> Error {
        Error::parse(&self.path, self.line_no, msg)
    }
}

struct Toks<'a> {
    it: std::str::SplitWhitespace<'a>,
}

impl<'a> Toks<'a> {
    fn new(s: &'a str) -> Self {
        Self { it: s.split_whitespace() }
    }
    fn s(&mut self, l: &Lines) -> Result<&'a str> {
        self.it.next().ok_or_else(|| l.err("필드 부족"))
    }
    fn p<T: std::str::FromStr>(&mut self, l: &Lines) -> Result<T> {
        let t = self.s(l)?;
        t.parse::<T>().map_err(|_| l.err(format!("값 해석 실패: {t}")))
    }
    fn pose(&mut self, l: &Lines) -> Result<Rigid3> {
        let mut a = [0.0; 7];
        for x in a.iter_mut() {
            *x = self.p(l)?;
        }
        Ok(Rigid3::from_params(&a))
    }
    fn sensor_type(&mut self, l: &Lines) -> Result<SensorKind> {
        let t = self.s(l)?;
        SensorKind::from_name(t).ok_or_else(|| l.err(format!("센서 종류 {t}")))
    }
    fn rest(self) -> Vec<&'a str> {
        self.it.collect()
    }
}

/// 텍스트 모델 파일(`*.txt`) 읽기.
pub fn read_model_text(dir: impl AsRef<Path>) -> Result<Reconstruction> {
    let d = dir.as_ref();
    let mut cameras = Vec::new();
    {
        let mut l = Lines::open(d.join("cameras.txt"))?;
        while let Some(line) = l.next_data()? {
            let mut t = Toks::new(&line);
            let id: u32 = t.p(&l)?;
            let model = CameraModelKind::from_name(t.s(&l)?)?;
            let w: u64 = t.p(&l)?;
            let h: u64 = t.p(&l)?;
            let mut params = Vec::new();
            for s in t.rest() {
                params.push(s.parse::<f64>().map_err(|_| l.err(format!("파라미터 {s}")))?);
            }
            cameras.push(Camera::new(id, model, w, h, params)?);
        }
    }
    let rigs = if d.join("rigs.txt").is_file() {
        let mut l = Lines::open(d.join("rigs.txt"))?;
        let mut v = Vec::new();
        while let Some(line) = l.next_data()? {
            let mut t = Toks::new(&line);
            let mut rig = Rig::new(t.p(&l)?);
            let ns: u32 = t.p(&l)?;
            if ns > 0 {
                let st = t.sensor_type(&l)?;
                rig.ref_sensor_id = Some(SensorKey::new(st, t.p(&l)?));
                for _ in 1..ns {
                    let st = t.sensor_type(&l)?;
                    let sid = SensorKey::new(st, t.p(&l)?);
                    let has: u8 = t.p(&l)?;
                    let pose = if has == 1 { Some(t.pose(&l)?) } else { None };
                    rig.sensors.insert(sid, pose);
                }
            }
            v.push(rig);
        }
        Some(v)
    } else {
        None
    };
    let frames = if d.join("frames.txt").is_file() {
        let mut l = Lines::open(d.join("frames.txt"))?;
        let mut v = Vec::new();
        while let Some(line) = l.next_data()? {
            let mut t = Toks::new(&line);
            let fid: u32 = t.p(&l)?;
            let rid: u32 = t.p(&l)?;
            let mut f = Frame::new(fid, rid);
            f.world_to_rig = Some(t.pose(&l)?);
            let nd: u32 = t.p(&l)?;
            for _ in 0..nd {
                let st = t.sensor_type(&l)?;
                let sid: u32 = t.p(&l)?;
                let did: u64 = t.p(&l)?;
                f.attach_data(SensorDataKey::new(SensorKey::new(st, sid), did));
            }
            v.push(f);
        }
        Some(v)
    } else {
        None
    };
    let mut images = Vec::new();
    {
        let mut l = Lines::open(d.join("images.txt"))?;
        while let Some(line) = l.next_data()? {
            let mut t = Toks::new(&line);
            let image_id: u32 = t.p(&l)?;
            let pose = t.pose(&l)?;
            let camera_id: u32 = t.p(&l)?;
            let name = t.s(&l)?.to_string();
            let pts_line = l.next_raw()?.unwrap_or_default();
            let toks: Vec<&str> = pts_line.split_whitespace().collect();
            if !toks.len().is_multiple_of(3) {
                return Err(l.err("2D 점 줄의 필드 수가 3의 배수가 아님"));
            }
            let mut points = Vec::with_capacity(toks.len() / 3);
            for c in toks.chunks(3) {
                let x: f64 = c[0].parse().map_err(|_| l.err("2D x"))?;
                let y: f64 = c[1].parse().map_err(|_| l.err("2D y"))?;
                let _id: i64 = c[2].parse().map_err(|_| l.err("3D 점 id"))?;
                points.push(Vec2::new(x, y));
            }
            images.push(RawImage { image_id, world_to_cam: pose, camera_id, name, points });
        }
    }
    let mut points = Vec::new();
    {
        let mut l = Lines::open(d.join("points3D.txt"))?;
        while let Some(line) = l.next_data()? {
            let mut t = Toks::new(&line);
            let id: u64 = t.p(&l)?;
            let xyz = Vec3::new(t.p(&l)?, t.p(&l)?, t.p(&l)?);
            let color = [t.p(&l)?, t.p(&l)?, t.p(&l)?];
            let error: f64 = t.p(&l)?;
            let rest = t.rest();
            if !rest.len().is_multiple_of(2) {
                return Err(l.err("트랙 필드 수가 짝수가 아님"));
            }
            let mut track = Vec::with_capacity(rest.len() / 2);
            for c in rest.chunks(2) {
                let i: u32 = c[0].parse().map_err(|_| l.err("트랙 영상 id"))?;
                let k: u32 = c[1].parse().map_err(|_| l.err("트랙 2D 인덱스"))?;
                track.push(TrackEntry::new(i, k));
            }
            points.push((id, Point3D { xyz, color, error, track }));
        }
    }
    assemble(RawModel { cameras, rigs, frames, images, points })
}

fn pose_str(p: &Rigid3) -> String {
    p.to_params().iter().map(|v| g17(*v)).collect::<Vec<_>>().join(" ")
}

/// 텍스트 다섯 파일 쓰기(디렉터리는 존재해야 함). 숫자는 %.17g.
pub fn write_model_text(rec: &Reconstruction, dir: impl AsRef<Path>, order: ImageOrder) -> Result<()> {
    let d = dir.as_ref();
    check_dir(d)?;
    {
        let mut w = create(&d.join("rigs.txt"))?;
        writeln!(w, "# Rig calib list with one line of data per calib:")?;
        writeln!(w, "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])")?;
        writeln!(w, "# Number of rigs: {}", rec.num_rigs())?;
        for r in rec.rigs().values() {
            let mut s = format!("{} {}", r.rig_id, r.num_sensors());
            if let Some(rs) = r.ref_sensor_id {
                s += &format!(" {} {}", rs.sensor_type.name(), rs.id);
                for (sid, pose) in &r.sensors {
                    s += &format!(" {} {}", sid.sensor_type.name(), sid.id);
                    match pose {
                        Some(p) => s += &format!(" 1 {}", pose_str(p)),
                        None => s += " 0",
                    }
                }
            }
            writeln!(w, "{s}")?;
        }
        w.flush()?;
    }
    {
        let mut w = create(&d.join("cameras.txt"))?;
        writeln!(w, "# Camera list with one line of data per camera:")?;
        writeln!(w, "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]")?;
        writeln!(w, "# Number of cameras: {}", rec.num_cameras())?;
        for c in rec.cameras().values() {
            let mut s = format!("{} {} {} {}", c.camera_id, c.model.name(), c.width, c.height);
            for p in &c.params {
                s.push(' ');
                s += &g17(*p);
            }
            writeln!(w, "{s}")?;
        }
        w.flush()?;
    }
    {
        let frames = posed_frames(rec);
        let mut w = create(&d.join("frames.txt"))?;
        writeln!(w, "# Frame list with one line of data per frame:")?;
        writeln!(w, "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)")?;
        writeln!(w, "# Number of frames: {}", frames.len())?;
        for f in frames {
            let mut s = format!("{} {} {} {}", f.frame_id, f.rig_id, pose_str(&f.world_to_rig.expect("자세 있음")), f.data_ids().len());
            for dd in f.data_ids() {
                s += &format!(" {} {} {}", dd.sensor_id.sensor_type.name(), dd.sensor_id.id, dd.id);
            }
            writeln!(w, "{s}")?;
        }
        w.flush()?;
    }
    {
        let ids = images_to_write(rec, order);
        let nobs: usize = ids.iter().map(|i| rec.image(*i).expect("영상").num_points3d()).sum();
        let mean = if ids.is_empty() { 0.0 } else { nobs as f64 / ids.len() as f64 };
        let mut w = create(&d.join("images.txt"))?;
        writeln!(w, "# Image list with two lines of data per image:")?;
        writeln!(w, "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME")?;
        writeln!(w, "#   POINTS2D[] as (X, Y, POINT3D_ID)")?;
        writeln!(w, "# Number of images: {}, mean observations per image: {}", ids.len(), g17(mean))?;
        for id in ids {
            let im = rec.image(id).expect("영상");
            let pose = rec.world_to_cam(id).expect("등록 영상 자세");
            writeln!(w, "{} {} {} {}", id, pose_str(&pose), im.camera_id, im.name)?;
            let mut s = String::with_capacity(im.num_points2d() * 40);
            for (k, p) in im.points2d().iter().enumerate() {
                if k > 0 {
                    s.push(' ');
                }
                s += &g17(p.xy.x);
                s.push(' ');
                s += &g17(p.xy.y);
                s.push(' ');
                if p.has_point3d() {
                    s += &p.point3d_id.to_string();
                } else {
                    s += "-1";
                }
            }
            writeln!(w, "{s}")?;
        }
        w.flush()?;
    }
    {
        let total: usize = rec.points3d().map(|(_, p)| p.track.len()).sum();
        let mean = if rec.num_points3d() == 0 { 0.0 } else { total as f64 / rec.num_points3d() as f64 };
        let mut w = create(&d.join("points3D.txt"))?;
        writeln!(w, "# 3D point list with one line of data per point:")?;
        writeln!(w, "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)")?;
        writeln!(w, "# Number of points: {}, mean track length: {}", rec.num_points3d(), g17(mean))?;
        for (id, p) in rec.points3d() {
            let mut s = format!(
                "{} {} {} {} {} {} {} {}",
                id,
                g17(p.xyz.x),
                g17(p.xyz.y),
                g17(p.xyz.z),
                p.color[0],
                p.color[1],
                p.color[2],
                g17(p.error)
            );
            for e in &p.track {
                s += &format!(" {} {}", e.image_id, e.point2d_idx);
            }
            writeln!(w, "{s}")?;
        }
        w.flush()?;
    }
    Ok(())
}

/// 조밀 복원 작업 폴더의 `stereo/` 하위 폴더 이름.
pub const STEREO_SUBDIRS: [&str; 3] = ["depth_maps", "normal_maps", "consistency_graphs"];

/// 조밀 복원 작업 폴더의 `stereo/` 아래 하위 폴더를 만든다. `rel` 은 영상 이름의 부모 경로(없으면 빈 경로).
pub fn create_stereo_dirs(out_dir: &Path, rel: &Path) -> Result<()> {
    for sub in STEREO_SUBDIRS {
        std::fs::create_dir_all(out_dir.join("stereo").join(sub).join(rel))?;
    }
    Ok(())
}

/// 조밀 복원 작업 폴더 설정 파일(`stereo/patch-match.cfg`, `stereo/fusion.cfg`)을 쓴다.
/// 영상마다 원천 영상 자동 선택(`__auto__, n`) 한 줄을 붙인다.
pub fn write_stereo_configs<'a>(out_dir: &Path, names: impl IntoIterator<Item = &'a str>, num_src_images: usize) -> Result<()> {
    let mut pm = String::new();
    let mut fu = String::new();
    for name in names {
        pm += &format!("{name}\n__auto__, {num_src_images}\n");
        fu += &format!("{name}\n");
    }
    std::fs::write(out_dir.join("stereo/patch-match.cfg"), pm)?;
    std::fs::write(out_dir.join("stereo/fusion.cfg"), fu)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Quat;
    use rand::{RngExt, SeedableRng};
    use rand_pcg::Pcg64;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cumulus3d_interop_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// 카메라 3개(OPENCV), 영상 39장, 점 n_pts 개, 일부 error = −1, 일부 2D 점 무연결.
    pub(crate) fn random_model(n_pts: usize, seed: u64) -> Reconstruction {
        let mut rng = Pcg64::seed_from_u64(seed);
        let mut r = Reconstruction::new();
        for c in 1..=3u32 {
            let mut cam = Camera::from_focal(CameraModelKind::OpenCv, 1609.22 + c as f64 * 0.1, 2048, 1152);
            cam.camera_id = c;
            cam.params[4] = rng.random_range(-0.1..0.1);
            cam.params[6] = rng.random_range(-0.01..0.01);
            r.add_camera_own_rig(cam).unwrap();
        }
        let npts2d = 2000usize;
        for i in 1..=39u32 {
            let pts: Vec<Vec2> = (0..npts2d).map(|_| Vec2::new(rng.random_range(0.0..2048.0), rng.random_range(0.0..1152.0))).collect();
            let q = Quat::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), 0.5).normalized();
            let pose = Rigid3::new(q, Vec3::new(rng.random_range(-5.0..5.0), 0.1 / 3.0, i as f64 * 1e-7));
            let cam = (i - 1) % 3 + 1;
            let pose = if i == 39 { None } else { Some(pose) };
            r.add_image_own_frame(Image::new(i, format!("cam{cam}/{i:04}.jpg"), cam, pts), pose).unwrap();
        }
        // 등록 순서를 섞는다(쓰기 순서 = 등록 순서 확인용). 영상 39 는 미등록.
        let mut order: Vec<u32> = (1..=38).collect();
        for k in (1..order.len()).rev() {
            let j = rng.random_range(0..=k);
            order.swap(k, j);
        }
        for i in order {
            r.register_image(i).unwrap();
        }
        let mut used = std::collections::HashSet::new();
        let mut k = 0;
        while r.num_points3d() < n_pts {
            let tl = rng.random_range(2..6usize);
            let mut track = Vec::new();
            let mut imgs = std::collections::HashSet::new();
            while track.len() < tl {
                let i = rng.random_range(1..=38u32);
                let idx = rng.random_range(0..npts2d as u32);
                if imgs.insert(i) && used.insert((i, idx)) {
                    track.push(TrackEntry::new(i, idx));
                }
            }
            let xyz = Vec3::new(rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0), rng.random::<f64>() * 1e-3);
            let id = r.add_point3d(xyz, track, [rng.random(), rng.random(), rng.random()]).unwrap();
            let err = if k % 7 == 0 { -1.0 } else { rng.random_range(0.0..4.0) };
            r.set_point3d_error(id, err).unwrap();
            k += 1;
        }
        r
    }

    fn assert_models_equal(a: &Reconstruction, b: &Reconstruction) {
        assert_eq!(a.cameras(), b.cameras());
        assert_eq!(a.rigs(), b.rigs());
        let fa: Vec<_> = a.frames().values().filter(|f| f.has_pose()).collect();
        let fb: Vec<_> = b.frames().values().filter(|f| f.has_pose()).collect();
        assert_eq!(fa, fb);
        assert_eq!(a.registered_images(), b.registered_images());
        for id in a.registered_images() {
            assert_eq!(a.image(id), b.image(id));
            assert_eq!(a.world_to_cam(id), b.world_to_cam(id));
        }
        let pa: Vec<_> = a.points3d().collect();
        let pb: Vec<_> = b.points3d().collect();
        assert_eq!(pa.len(), pb.len());
        for ((ia, x), (ib, y)) in pa.iter().zip(pb.iter()) {
            assert_eq!(ia, ib);
            assert_eq!(x.xyz.x.to_bits(), y.xyz.x.to_bits());
            assert_eq!(x.xyz.z.to_bits(), y.xyz.z.to_bits());
            assert_eq!(x.error.to_bits(), y.error.to_bits());
            assert_eq!(x.color, y.color);
            assert_eq!(x.track, y.track);
        }
    }

    #[test]
    fn binary_roundtrip() {
        let mut m = random_model(10000, 1);
        // 미등록 영상의 점은 없어야 저장 후 일치(영상 39 에는 점이 없음).
        m.check_invariants().unwrap();
        let d = tmpdir("bin");
        write_model_binary(&m, &d, ImageOrder::Registration).unwrap();
        let r = read_model(&d).unwrap();
        r.check_invariants().unwrap();
        // 미등록 영상은 사라짐
        assert_eq!(r.num_images(), 38);
        m.remove_unregistered();
        assert_models_equal(&m, &r);
        // 다시 쓰면 바이트 동일
        let d2 = tmpdir("bin2");
        write_model_binary(&r, &d2, ImageOrder::Registration).unwrap();
        for f in ["cameras.bin", "rigs.bin", "frames.bin", "images.bin", "points3D.bin"] {
            assert_eq!(std::fs::read(d.join(f)).unwrap(), std::fs::read(d2.join(f)).unwrap(), "{f}");
        }
        std::fs::remove_dir_all(&d).ok();
        std::fs::remove_dir_all(&d2).ok();
    }

    #[test]
    fn text_roundtrip_and_headers() {
        let mut m = random_model(2000, 2);
        // 2D 점 0개 영상 하나 추가·등록
        m.add_image_own_frame(Image::new(100, "cam1/empty.jpg", 1, []), Some(Rigid3::identity())).unwrap();
        m.register_image(100).unwrap();
        let d = tmpdir("txt");
        write_model_text(&m, &d, ImageOrder::Registration).unwrap();
        let r = read_model(&d).unwrap();
        m.remove_unregistered();
        assert_models_equal(&m, &r);
        let images = std::fs::read_to_string(d.join("images.txt")).unwrap();
        let lines: Vec<&str> = images.split('\n').collect();
        assert_eq!(lines[0], "# Image list with two lines of data per image:");
        assert_eq!(lines[1], "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME");
        assert_eq!(lines[2], "#   POINTS2D[] as (X, Y, POINT3D_ID)");
        assert!(lines[3].starts_with("# Number of images: 39, mean observations per image: "));
        assert!(images.contains("cam1/empty.jpg\n\n"));
        let cams = std::fs::read_to_string(d.join("cameras.txt")).unwrap();
        assert!(cams.starts_with(
            "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: 3\n1 OPENCV 2048 1152 "
        ));
        let pts = std::fs::read_to_string(d.join("points3D.txt")).unwrap();
        assert!(pts.starts_with("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n# Number of points: 2000, mean track length: "));
        let rigs = std::fs::read_to_string(d.join("rigs.txt")).unwrap();
        assert!(rigs.contains("# Number of rigs: 3\n1 1 CAMERA 1\n"));
        let frames = std::fs::read_to_string(d.join("frames.txt")).unwrap();
        assert!(frames.contains("# Number of frames: 39\n"));
        assert!(!images.contains(" \n"));
        // text → bin → equal
        let d2 = tmpdir("txt2");
        write_model_binary(&r, &d2, ImageOrder::ById).unwrap();
        let r2 = read_model_binary(&d2).unwrap();
        assert_eq!(r2.registered_images().len(), r.registered_images().len());
        std::fs::remove_dir_all(&d).ok();
        std::fs::remove_dir_all(&d2).ok();
    }

    #[test]
    fn legacy_model_without_rigs_frames() {
        let m = random_model(100, 3);
        let d = tmpdir("legacy");
        write_model_binary(&m, &d, ImageOrder::ById).unwrap();
        std::fs::remove_file(d.join("rigs.bin")).unwrap();
        std::fs::remove_file(d.join("frames.bin")).unwrap();
        let r = read_model(&d).unwrap();
        for c in r.cameras().keys() {
            assert_eq!(r.rig(*c).unwrap().ref_sensor_id, Some(SensorKey::camera(*c)));
        }
        for im in r.images() {
            assert_eq!(im.frame_id, im.image_id);
            assert_eq!(r.world_to_cam(im.image_id), m.world_to_cam(im.image_id));
        }
        write_model_binary(&r, &d, ImageOrder::ById).unwrap();
        assert!(d.join("rigs.bin").is_file() && d.join("frames.bin").is_file());
        // rigs 만 있으면 오류
        std::fs::remove_file(d.join("frames.bin")).unwrap();
        assert!(read_model(&d).is_err());
        std::fs::remove_dir_all(&d).ok();
    }
}
