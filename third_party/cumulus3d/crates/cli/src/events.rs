/*
 * events.rs
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
 * SPDX-License-Identifier: Apache-2.0
 */

//! 파이프라인 이벤트. 새 프레임 묶음이 처리될 때마다 단계별로 분해된 결과를 내보낸다.
//! 이 파일은 세 모듈(session·pipeline·sinks)의 공용 계약이다. 필드 추가는 가능, 기존 필드 변경·삭제는 금지.

use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::{ImageId, Reconstruction, Rigid3, Sim3};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// 모든 이벤트에 붙는 머리말.
#[derive(Clone, Debug)]
pub struct Meta {
    /// 세션 버전(이벤트를 낸 상태의 단조 증가 번호).
    pub version: u64,
    /// 발생 시각.
    pub at: SystemTime,
}

/// 구역 번호와 위치 범위 [lo, hi).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoneRange {
    /// 구역 번호.
    pub zone: usize,
    /// 첫 위치(포함).
    pub lo: usize,
    /// 끝 위치(제외).
    pub hi: usize,
}

/// 디코딩된 입력 프레임(카메라 한 장). 처리(특징·매칭·등록) 전에 [`Event::FrameDecoded`] 로 먼저 전달된다.
#[derive(Clone, Debug)]
pub struct Frame {
    /// 위치 번호.
    pub position: usize,
    /// 카메라 이름(예: `camF`).
    pub camera: String,
    /// 영상 이름(`image_root` 기준 상대 경로).
    pub name: String,
    /// 영상 파일 경로.
    pub path: std::path::PathBuf,
    /// 가로 화소 수.
    pub width: u32,
    /// 세로 화소 수.
    pub height: u32,
    /// RGB8 화소(행 우선, 길이 `width * height * 3`). 훅 사이에서 복사 없이 공유된다.
    pub rgb: Arc<[u8]>,
    /// 이 영상의 GPS 기록(있으면).
    pub gps: Option<cumulus3d_core::io::GpsRecord>,
}

/// 파이프라인 이벤트(단계별 결과). 모든 변형에 [`Meta`] 가 붙는다.
#[derive(Clone, Debug)]
pub enum Event {
    /// 위치 하나의 프레임 묶음이 들어옴.
    FrameIngested {
        /// 공통 머리말.
        meta: Meta,
        /// 위치 번호.
        position: usize,
        /// 이 위치의 영상 이름들.
        images: Vec<String>,
    },
    /// 입력 프레임 한 장 디코딩 완료. 위치 처리 전에 먼저 나온다(`SessionConfig::decode_frames` 가 켜졌을 때만).
    FrameDecoded {
        /// 공통 머리말.
        meta: Meta,
        /// 디코딩된 프레임.
        frame: Arc<Frame>,
    },
    /// 세션 시작(첫 입력 때 한 번). `positions` = 알려진 전체 위치 수. (session 추가)
    Started {
        /// 공통 머리말.
        meta: Meta,
        /// 알려진 전체 위치 수(모르면 None).
        positions: Option<usize>,
    },
    /// run.log 한 줄. `stamped` 이면 `[HH:MM:SS] ` 머리(meta.at 현지 시각)를 붙여 쓴다. (session 추가)
    Log {
        /// 공통 머리말.
        meta: Meta,
        /// 기록할 줄.
        line: String,
        /// `[HH:MM:SS] ` 머리를 붙일지.
        stamped: bool,
    },
    /// 영상 하나의 특징 추출 완료.
    FeaturesExtracted {
        /// 공통 머리말.
        meta: Meta,
        /// 영상 이름.
        image: String,
        /// 추출한 특징 수.
        count: usize,
        /// 위치 번호.
        position: usize,
        /// 영상 id.
        image_id: ImageId,
    },
    /// 새 짝 매칭 결과.
    /// `pairs` = 이번 위치에서 매칭한 짝 수, `verified` = 유효 기하 짝 수, `matched` = 기술자 매칭을 수행한 짝 수.
    PairsMatched {
        /// 공통 머리말.
        meta: Meta,
        /// 이번 위치에서 매칭한 짝 수.
        pairs: usize,
        /// 유효 기하 짝 수.
        verified: usize,
        /// 위치 번호.
        position: usize,
        /// 기술자 매칭을 수행한 짝 수.
        matched: usize,
    },
    /// 첫 모델이 만들어짐.
    ModelInitialized {
        /// 공통 머리말.
        meta: Meta,
        /// 등록된 영상 수.
        registered: usize,
        /// 위치 번호.
        position: usize,
        /// 관련 모델 사본.
        model: Arc<Reconstruction>,
    },
    /// 영상이 모델에 등록됨.
    FrameRegistered {
        /// 공통 머리말.
        meta: Meta,
        /// 위치 번호.
        position: usize,
        /// 영상 이름.
        image: String,
        /// 영상 id.
        image_id: ImageId,
        /// 등록된 자세(world → 카메라).
        cam_from_world: Rigid3,
    },
    /// 위치 하나의 등록·삼각측량이 끝남.
    PositionDone {
        /// 공통 머리말.
        meta: Meta,
        /// 위치 번호.
        position: usize,
        /// 등록된 영상 수.
        registered: usize,
        /// 이 위치까지 기대한 영상 수.
        expected: usize,
    },
    /// 구역의 마지막 위치가 도착함.
    /// `model` = 그 시점 체인 모델 사본(정렬 전).
    ZoneArrived {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 범위.
        range: ZoneRange,
        /// 관련 모델 사본.
        model: Arc<Reconstruction>,
    },
    /// 구역 초벌 점군(BA 없음).
    /// `model` = 초벌 모델(구역 0 은 GPS 정렬본), `dense` = 거짓이면 조밀화를 하지 않았거나 실패해 `cloud` 가 빈 자리표시.
    /// `frame` = 점군 좌표계("gps" | "refined_K" | "unaligned").
    ZonePreview {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 범위.
        range: ZoneRange,
        /// 점군(조밀화가 없으면 빈 자리표시).
        cloud: Arc<PointCloud>,
        /// 관련 모델 사본.
        model: Arc<Reconstruction>,
        /// 조밀화를 실제로 했는지.
        dense: bool,
        /// 점군 좌표계("gps" | "refined_K" | "unaligned").
        frame: String,
    },
    /// 정밀 작업 시작(배경). (session 추가)
    RefineStarted {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 번호.
        zone: usize,
    },
    /// 정밀 작업의 BA 직후 모델(GPS 정렬 전). (session 추가)
    ZoneAdjusted {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 번호.
        zone: usize,
        /// 관련 모델 사본.
        model: Arc<Reconstruction>,
    },
    /// 구역 정밀 자세 확정(BA + GPS 정렬).
    /// `model` = BA + GPS 정렬된 정밀 모델.
    ZoneRefinedPose {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 범위.
        range: ZoneRange,
        /// 등록된 영상 수.
        registered: usize,
        /// 평균 재투영 오차(픽셀).
        mean_reproj_px: f64,
        /// 관련 모델 사본.
        model: Arc<Reconstruction>,
    },
    /// 구역 정밀 점군. 같은 구역 초벌을 대체한다.
    /// `dense` = 거짓이면 `cloud` 는 빈 자리표시.
    ZoneRefined {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 범위.
        range: ZoneRange,
        /// 점군(조밀화가 없으면 빈 자리표시).
        cloud: Arc<PointCloud>,
        /// 관련 모델 사본.
        model: Arc<Reconstruction>,
        /// 조밀화를 실제로 했는지.
        dense: bool,
    },
    /// 정밀 모델이 다음 등록의 기준으로 채택됨.
    BaseAdopted {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 번호.
        zone: usize,
    },
    /// 이미 보낸 구역의 좌표계 변환(점군은 다시 보내지 않음): new_from_old.
    Reanchored {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 번호.
        zone: usize,
        /// new_from_old 변환.
        transform: Sim3,
    },
    /// 화면용 스냅샷(사건 순서 번호와 합성 점군).
    Snapshot {
        /// 공통 머리말.
        meta: Meta,
        /// 사건 순서 번호.
        index: usize,
        /// 점군(조밀화가 없으면 빈 자리표시).
        cloud: Arc<PointCloud>,
    },
    /// 모든 위치 처리 완료.
    AllPositionsDone {
        /// 공통 머리말.
        meta: Meta,
    },
    /// 모든 정밀 작업 완료.
    /// `chain` = 최종 체인 모델.
    AllRefinedDone {
        /// 공통 머리말.
        meta: Meta,
        /// 최종 체인 모델.
        chain: Option<Arc<Reconstruction>>,
    },
    /// 세션 종료 요약(모든 이벤트의 마지막). `stage_times` = (단계, 누적 시간, 횟수) 단계 이름순,
    /// `pipeline_secs` = 시작 ~ 모든 정밀 완료, `chain_stats` = 최종 체인 모델 통계 줄. (session 추가)
    Finished {
        /// 공통 머리말.
        meta: Meta,
        /// 시작 ~ 모든 정밀 완료(초).
        pipeline_secs: f64,
        /// (단계, 누적 시간, 횟수) 단계 이름순.
        stage_times: Vec<(String, Duration, usize)>,
        /// 최종 체인 모델 통계 줄.
        chain_stats: Vec<String>,
    },
    /// `Command::ResetFrom` 처리됨. (session 추가)
    Reset {
        /// 공통 머리말.
        meta: Meta,
        /// 위치 번호.
        position: usize,
    },
    /// `Command::InvalidateZone` 처리됨. (session 추가)
    ZoneInvalidated {
        /// 공통 머리말.
        meta: Meta,
        /// 구역 번호.
        zone: usize,
    },
    /// 경고.
    Warning {
        /// 공통 머리말.
        meta: Meta,
        /// 메시지.
        message: String,
    },
    /// 오류(훅 패닉 포함).
    Error {
        /// 공통 머리말.
        meta: Meta,
        /// 메시지.
        message: String,
    },
}

/// 이벤트 종류(훅 등록·필터링용).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// [`Event::FrameIngested`].
    FrameIngested,
    /// [`Event::FrameDecoded`].
    FrameDecoded,
    /// [`Event::Started`].
    Started,
    /// [`Event::Log`].
    Log,
    /// [`Event::RefineStarted`].
    RefineStarted,
    /// [`Event::ZoneAdjusted`].
    ZoneAdjusted,
    /// [`Event::Finished`].
    Finished,
    /// [`Event::Reset`].
    Reset,
    /// [`Event::ZoneInvalidated`].
    ZoneInvalidated,
    /// [`Event::FeaturesExtracted`].
    FeaturesExtracted,
    /// [`Event::PairsMatched`].
    PairsMatched,
    /// [`Event::ModelInitialized`].
    ModelInitialized,
    /// [`Event::FrameRegistered`].
    FrameRegistered,
    /// [`Event::PositionDone`].
    PositionDone,
    /// [`Event::ZoneArrived`].
    ZoneArrived,
    /// [`Event::ZonePreview`].
    ZonePreview,
    /// [`Event::ZoneRefinedPose`].
    ZoneRefinedPose,
    /// [`Event::ZoneRefined`].
    ZoneRefined,
    /// [`Event::BaseAdopted`].
    BaseAdopted,
    /// [`Event::Reanchored`].
    Reanchored,
    /// [`Event::Snapshot`].
    Snapshot,
    /// [`Event::AllPositionsDone`].
    AllPositionsDone,
    /// [`Event::AllRefinedDone`].
    AllRefinedDone,
    /// [`Event::Warning`].
    Warning,
    /// [`Event::Error`].
    Error,
}

impl Event {
    /// 이벤트 종류.
    pub fn kind(&self) -> EventKind {
        use Event::*;
        match self {
            FrameIngested { .. } => EventKind::FrameIngested,
            FrameDecoded { .. } => EventKind::FrameDecoded,
            Started { .. } => EventKind::Started,
            Log { .. } => EventKind::Log,
            RefineStarted { .. } => EventKind::RefineStarted,
            ZoneAdjusted { .. } => EventKind::ZoneAdjusted,
            Finished { .. } => EventKind::Finished,
            Reset { .. } => EventKind::Reset,
            ZoneInvalidated { .. } => EventKind::ZoneInvalidated,
            FeaturesExtracted { .. } => EventKind::FeaturesExtracted,
            PairsMatched { .. } => EventKind::PairsMatched,
            ModelInitialized { .. } => EventKind::ModelInitialized,
            FrameRegistered { .. } => EventKind::FrameRegistered,
            PositionDone { .. } => EventKind::PositionDone,
            ZoneArrived { .. } => EventKind::ZoneArrived,
            ZonePreview { .. } => EventKind::ZonePreview,
            ZoneRefinedPose { .. } => EventKind::ZoneRefinedPose,
            ZoneRefined { .. } => EventKind::ZoneRefined,
            BaseAdopted { .. } => EventKind::BaseAdopted,
            Reanchored { .. } => EventKind::Reanchored,
            Snapshot { .. } => EventKind::Snapshot,
            AllPositionsDone { .. } => EventKind::AllPositionsDone,
            AllRefinedDone { .. } => EventKind::AllRefinedDone,
            Warning { .. } => EventKind::Warning,
            Error { .. } => EventKind::Error,
        }
    }
    /// 공통 머리말.
    pub fn meta(&self) -> &Meta {
        use Event::*;
        match self {
            FrameIngested { meta, .. }
            | FeaturesExtracted { meta, .. }
            | PairsMatched { meta, .. }
            | ModelInitialized { meta, .. }
            | FrameRegistered { meta, .. }
            | PositionDone { meta, .. }
            | ZoneArrived { meta, .. }
            | ZonePreview { meta, .. }
            | ZoneRefinedPose { meta, .. }
            | ZoneRefined { meta, .. }
            | BaseAdopted { meta, .. }
            | Reanchored { meta, .. }
            | Snapshot { meta, .. }
            | AllPositionsDone { meta }
            | AllRefinedDone { meta, .. }
            | FrameDecoded { meta, .. }
            | Started { meta, .. }
            | Log { meta, .. }
            | RefineStarted { meta, .. }
            | ZoneAdjusted { meta, .. }
            | Finished { meta, .. }
            | Reset { meta, .. }
            | ZoneInvalidated { meta, .. }
            | Warning { meta, .. }
            | Error { meta, .. } => meta,
        }
    }
}

impl Event {
    /// timeline.txt 의 사건 문구(없으면 None). 문구·형식은 기존 기록과 같다.
    pub fn timeline_text(&self) -> Option<String> {
        use Event::*;
        Some(match self {
            Started { positions, .. } => match positions {
                Some(n) => format!("start NPOS={n}"),
                None => "start".to_string(),
            },
            ModelInitialized { registered, .. } => format!("init_model Registered images: {registered}"),
            BaseAdopted { zone, .. } => format!("adopt refined {zone} as base"),
            PositionDone { position, registered, expected, .. } => format!("pos {position} registered {registered}/{expected}"),
            ZoneArrived { range, .. } => format!("region {} arrived pos[{},{})", range.zone, range.lo, range.hi),
            ZonePreview { range, .. } => format!("preview {} ready", range.zone),
            RefineStarted { zone, .. } => format!("refined {zone} ba_start"),
            ZoneRefinedPose { range, registered, mean_reproj_px, .. } => {
                format!("refined {} pose_ready Registered images: {registered} Mean reprojection error: {mean_reproj_px:.6}px ", range.zone)
            }
            ZoneRefined { range, .. } => format!("refined {} ready", range.zone),
            AllPositionsDone { .. } => "all positions done".to_string(),
            AllRefinedDone { .. } => "all refined done".to_string(),
            _ => return None,
        })
    }
}

/// 리듀서에 넣는 명령.
#[derive(Clone, Debug)]
pub enum Command {
    /// 이 위치부터 다시 처리(그 이후 상태 폐기).
    ResetFrom(usize),
    /// 구역 결과를 무효화하고 다시 만들게 함.
    InvalidateZone(usize),
}
