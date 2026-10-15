/*
 * declare.rs
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

//! 선언형 빌더 층: 계획을 먼저 기록하고 나중에 한 번에 실행한다.
//!
//! [`Recon::declare`] 가 돌려주는 [`ReconBuilder`] 의 메서드는 "이렇게 하겠다"는 **계획만 기록**하고 아무것도
//! 실행하지 않는다(파일·폴더를 만들거나 지우지 않고, 장치를 열지 않는다). [`ReconBuilder::build`] 가 계획을 검사해
//! 실행 가능한 재구성([`Recon`])을 만들고, [`Recon::run`] 이 세션 리듀서 + 파이프라인 + 훅 + 기본 출력 훅을 조립해
//! 위치 반복·배경 작업 대기·종료까지 한 번에 실행한다(지연 실행).
//!
//! 계획 자체는 값([`Plan`])이라 비교·복제·TOML 저장이 된다([`Plan::to_toml`] / [`Plan::from_toml`]).
//! 훅(클로저)은 계획이 아니라 빌더에 보관한다.
//!
//! # 사용 예
//!
//! ```no_run
//! use cumulus3d_cli::declare::{Dense, Recon, Sinks};
//! use cumulus3d_cli::events::{Event, EventKind};
//!
//! fn main() -> Result<(), String> {
//!     let recon = Recon::declare()
//!         .input("data")                      // data/images/<cam>/, data/gps_ref.txt
//!         .preset("aerial-formation")         // camF/camR/camL, 편대 짝 규칙, stride 3
//!         .stride(3)
//!         .zones(12, 2)
//!         .dense(Dense::profile("fast").fusion_min_views(3))
//!         .sinks(Sinks::default_files("out").models(true))
//!         .seed(7)
//!         .on(EventKind::PositionDone, |e: &Event| {
//!             if let Some(t) = e.timeline_text() {
//!                 println!("{t}");
//!             }
//!         })
//!         .on_zone_preview(|range, cloud| println!("초벌 {}: 점 {}", range.zone, cloud.len()))
//!         .build()                            // 검사만(오류는 모아서 한 번에)
//!         .map_err(|e| e.to_string())?;
//!     println!("{}", recon.plan().to_toml());
//!     let summary = recon.run()?;              // 여기서 실제 실행
//!     println!("이벤트 {} 개", summary.events);
//!     Ok(())
//! }
//! ```
//!
//! TOML 계획 파일로 실행: `cumulus3d run plan.toml`. 기본 계획은 `cumulus3d plan --print-default`.
//! 계획 파일의 상대 경로는 실행 위치(현재 폴더) 기준이다.

use crate::densewrap::{make_match_backend, make_pm_backend, make_sift_backend, parse_profile, DenseConfig};
use crate::events::{EventKind, Frame, ZoneRange};
use crate::pipeline::{Pipeline, QueuePolicy, Summary};
use crate::session::{zones_upto, Input, Session, SessionConfig};
use crate::sinks::{DefaultSinks, SinkSet};
use crate::stream::{frame_set_of, Layout, StreamConfig, CAMS};
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::io::{read_gps_file, GpsRecord};
use cumulus3d_dense::{DensifyOptions, DepthMapCache, FusionMode, FusionResidual};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// 기본 프리셋 이름.
pub const DEFAULT_PRESET: &str = "aerial-formation";

/// 알려진 프리셋 이름.
pub const PRESETS: [&str; 1] = [DEFAULT_PRESET];

// ============================================================================================
// 계획 값
// ============================================================================================

/// 재구성 계획(값). 실행하지 않고 "무엇을 어떻게 할지"만 담는다.
///
/// 기본값([`Plan::default`])은 `aerial-formation` 프리셋 + `cumulus3d stream` 기본 설정이며,
/// 경로는 자리 표시(`data/images`, `data/gps_ref.txt`, `out`)다.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Plan {
    /// 두 뷰 기하·GPS 정렬 RANSAC 시드(없으면 실행마다 다름).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// rayon 스레드 수(0 = 코어 수).
    pub threads: usize,
    /// GPS 기록 파일(`이름 위도 경도 고도` 줄). 없으면 정렬할 수 없다.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gps: Option<PathBuf>,
    /// 짝 전략.
    pub pairing: Pairing,
    /// 입력 영상.
    pub input: Source,
    /// 구역 나누기.
    pub zones: Zones,
    /// 좌표 정렬.
    pub align: Align,
    /// 특징 추출.
    pub features: Features,
    /// 기술자 매칭.
    pub matching: Matching,
    /// 희소 재구성.
    pub sparse: Sparse,
    /// 조밀화.
    pub dense: Dense,
    /// 출력 훅.
    pub sinks: Sinks,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            seed: None,
            threads: 0,
            gps: Some(PathBuf::from("data/gps_ref.txt")),
            pairing: Pairing::Formation,
            input: Source::default(),
            zones: Zones::default(),
            align: Align::default(),
            features: Features::default(),
            matching: Matching::default(),
            sparse: Sparse::default(),
            dense: Dense::default(),
            sinks: Sinks::default_files("out"),
        }
    }
}

/// 입력 영상: `<images>/<camera>/<camera>_NNNN.jpg`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Source {
    /// 영상 폴더(카메라 폴더들의 부모).
    pub images: PathBuf,
    /// 카메라 폴더 목록(위치 하나의 프레임 묶음 순서). 위치 목록은 첫 카메라 폴더에서 찾는다.
    pub cameras: Vec<String>,
    /// 파일 번호 간격: 번호 % stride == 0 인 프레임만 쓴다. 위치 = 선택된 프레임 목록의 순번.
    pub stride: usize,
    /// 앞 N 위치만(없으면 전부).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub positions: Option<usize>,
}

impl Default for Source {
    fn default() -> Self {
        Self { images: PathBuf::from("data/images"), cameras: CAMS.iter().map(|c| c.to_string()).collect(), stride: 3, positions: None }
    }
}

/// 짝 전략.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pairing {
    /// 편대 규칙: 같은 카메라 위치 간격 1..5, 8, 16 / 다른 카메라 위치 차 0..4.
    #[default]
    Formation,
}

/// 구역(위치 단위).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Zones {
    /// 구역 크기.
    pub span: usize,
    /// 구역 겹침(< span).
    pub overlap: usize,
}

impl Default for Zones {
    fn default() -> Self {
        Self { span: 12, overlap: 2 }
    }
}

/// 정렬 좌표계.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlignFrame {
    /// GPS 기록으로 ENU(동·북·위) 좌표계에 맞춘다(GPS 파일 필요).
    #[default]
    Enu,
    /// 정렬하지 않음(GPS 파일을 쓰지 않는다).
    None,
}

/// 좌표 정렬.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Align {
    /// 좌표계.
    pub frame: AlignFrame,
    /// 첫 GPS 줄로 세션 ENU 원점 고정(끄면 정렬마다 첫 공통 기록).
    pub fixed_origin: bool,
}

/// 특징 추출.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Features {
    /// SIFT 백엔드(cpu|cuda). 결과는 cpu 와 같다.
    pub backend: String,
    /// 최대 특징 수.
    pub max_num_features: usize,
    /// 입력 최대 영상 크기.
    pub max_image_size: usize,
}

impl Default for Features {
    fn default() -> Self {
        Self { backend: "cpu".into(), max_num_features: 8192, max_image_size: 3200 }
    }
}

/// 기술자 매칭.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Matching {
    /// 매칭 백엔드(cpu|cuda). 결과는 cpu 와 같다.
    pub backend: String,
}

impl Default for Matching {
    fn default() -> Self {
        Self { backend: "cpu".into() }
    }
}

/// 희소 재구성.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sparse {
    /// 새로 등록된 영상만 삼각측량(결과가 달라질 수 있는 개선).
    pub incremental_triangulation: bool,
}

/// 조밀화 계획. [`Dense::profile`] 로 시작해 메서드로 덧붙인다. 비운 융합·필터 값은 프로파일 기본값.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Dense {
    /// 조밀화 여부(끄면 후처리 점군 없음).
    pub enabled: bool,
    /// PatchMatch 백엔드(cuda).
    pub backend: String,
    /// 프로파일(fast|quality).
    pub profile: String,
    /// 왜곡 보정 최대 영상 크기.
    pub max_image_size: i64,
    /// 원천 뷰 수(1..=32).
    pub views: usize,
    /// 초벌·정밀 조밀화 백엔드 호출 직렬화.
    pub serialize: bool,
    /// 겹침 영상 깊이맵 재사용.
    pub depth_cache: bool,
    /// 융합 덮어쓰기.
    pub fusion: Fusion,
    /// 깊이맵 필터 덮어쓰기.
    pub filter: Filter,
}

impl Default for Dense {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: "cuda".into(),
            profile: "fast".into(),
            max_image_size: 960,
            views: 10,
            serialize: false,
            depth_cache: false,
            fusion: Fusion::default(),
            filter: Filter::default(),
        }
    }
}

/// 융합 덮어쓰기(`densify` 하위 명령의 `--fusion-*` 와 같은 뜻). 비우면 프로파일 기본값.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fusion {
    /// 융합 방식(consistency|traversal).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// 일치 융합: 기준 외 일치 뷰 최소 수(≥ 1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_views: Option<usize>,
    /// 상대 깊이 허용치(> 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_error: Option<f64>,
    /// 법선 허용 각(도, (0, 180]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal_error: Option<f64>,
    /// 재투영 허용치(픽셀, > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reproj_error: Option<f64>,
    /// 남은 픽셀 처리(none|release|second-pass).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual: Option<String>,
}

/// 깊이맵 필터 덮어쓰기(`densify` 하위 명령의 `--filter-*`, `--median-filter` 와 같은 뜻). 비우면 프로파일 기본값.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Filter {
    /// 일치 뷰 최소 수.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_views: Option<u32>,
    /// 순·역 재투영 허용치(픽셀, > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geom_error: Option<f32>,
    /// 최소 NCC([-1, 1]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_ncc: Option<f32>,
    /// 판독 전 5×5 중앙값 평면 필터.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median: Option<bool>,
}

impl Dense {
    /// 프로파일(fast|quality)과 기본값으로 시작(켜짐, 백엔드 cuda).
    pub fn profile(name: &str) -> Self {
        Self { profile: name.to_string(), ..Self::default() }
    }
    /// 꺼진 조밀화.
    pub fn off() -> Self {
        Self { enabled: false, ..Self::default() }
    }
    /// PatchMatch 백엔드 이름.
    pub fn backend(mut self, name: &str) -> Self {
        self.backend = name.to_string();
        self
    }
    /// 왜곡 보정 최대 영상 크기.
    pub fn max_image_size(mut self, n: i64) -> Self {
        self.max_image_size = n;
        self
    }
    /// 원천 뷰 수.
    pub fn views(mut self, n: usize) -> Self {
        self.views = n;
        self
    }
    /// 초벌·정밀 백엔드 호출 직렬화.
    pub fn serialize(mut self, on: bool) -> Self {
        self.serialize = on;
        self
    }
    /// 겹침 영상 깊이맵 재사용.
    pub fn depth_cache(mut self, on: bool) -> Self {
        self.depth_cache = on;
        self
    }
    /// 융합 방식(consistency|traversal).
    pub fn fusion_mode(mut self, mode: &str) -> Self {
        self.fusion.mode = Some(mode.to_string());
        self
    }
    /// 일치 융합: 기준 외 일치 뷰 최소 수.
    pub fn fusion_min_views(mut self, n: usize) -> Self {
        self.fusion.min_views = Some(n);
        self
    }
    /// 융합 상대 깊이 허용치.
    pub fn fusion_depth_error(mut self, v: f64) -> Self {
        self.fusion.depth_error = Some(v);
        self
    }
    /// 융합 법선 허용 각(도).
    pub fn fusion_normal_error(mut self, deg: f64) -> Self {
        self.fusion.normal_error = Some(deg);
        self
    }
    /// 융합 재투영 허용치(픽셀).
    pub fn fusion_reproj_error(mut self, px: f64) -> Self {
        self.fusion.reproj_error = Some(px);
        self
    }
    /// 일치 융합 남은 픽셀 처리(none|release|second-pass).
    pub fn fusion_residual(mut self, mode: &str) -> Self {
        self.fusion.residual = Some(mode.to_string());
        self
    }
    /// 깊이맵 필터: 일치 뷰 최소 수.
    pub fn filter_min_views(mut self, n: u32) -> Self {
        self.filter.min_views = Some(n);
        self
    }
    /// 깊이맵 필터: 순·역 재투영 허용치(픽셀).
    pub fn filter_geom_error(mut self, px: f32) -> Self {
        self.filter.geom_error = Some(px);
        self
    }
    /// 깊이맵 필터: 최소 NCC.
    pub fn filter_min_ncc(mut self, v: f32) -> Self {
        self.filter.min_ncc = Some(v);
        self
    }
    /// 판독 전 중앙값 평면 필터.
    pub fn median_filter(mut self, on: bool) -> Self {
        self.filter.median = Some(on);
        self
    }

    /// 덮어쓰기를 조밀화 옵션에 반영(이름이 틀리면 오류).
    pub fn apply(&self, o: &mut DensifyOptions) -> Result<(), String> {
        let f = &self.fusion;
        if let Some(m) = &f.mode {
            o.fusion.mode = parse_fusion_mode(m)?;
        }
        if let Some(v) = f.min_views {
            o.fusion.min_consistent_views = v;
        }
        if let Some(v) = f.depth_error {
            o.fusion.max_depth_error = v;
        }
        if let Some(v) = f.normal_error {
            o.fusion.max_normal_error_deg = v;
        }
        if let Some(v) = f.reproj_error {
            o.fusion.max_reproj_error = v;
        }
        if let Some(m) = &f.residual {
            o.fusion.residual =
                FusionResidual::parse(m).ok_or_else(|| format!("알 수 없는 융합 남은 픽셀 처리 {m} (none|release|second-pass)"))?;
        }
        let g = &self.filter;
        if let Some(v) = g.min_views {
            o.filter.min_num_consistent = v;
        }
        if let Some(v) = g.geom_error {
            o.filter.geom_max_cost = v;
        }
        if let Some(v) = g.min_ncc {
            o.filter.min_ncc = v;
        }
        if let Some(v) = g.median {
            o.filter.median_filter = v;
        }
        Ok(())
    }

    /// 실행용 조밀화 설정. 백엔드는 여기서 만든다(장치가 없으면 그 오류를 조밀화 단계에서 보고).
    pub fn config(&self) -> Result<DenseConfig, String> {
        let mut d = DenseConfig::new(make_pm_backend(&self.backend), parse_profile(&self.profile)?);
        d.undistort.max_image_size = self.max_image_size;
        d.densify.neighbors.num_views = self.views;
        self.apply(&mut d.densify)?;
        if self.serialize {
            d.lock = Some(Arc::new(Mutex::new(())));
        }
        if self.depth_cache {
            d.depth_cache = Some(Arc::new(DepthMapCache::new()));
        }
        Ok(d)
    }
}

fn parse_fusion_mode(m: &str) -> Result<FusionMode, String> {
    FusionMode::parse(m).ok_or_else(|| format!("알 수 없는 융합 방식 {m} (consistency|traversal)"))
}

/// 출력 훅 계획. `out` 이 없으면 파일을 쓰지 않는다(켜 둔 파일 항목도 무시, 훅만).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sinks {
    /// 출력 폴더.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out: Option<PathBuf>,
    /// 실행 시작 때 출력 폴더를 지우고 새로 만든다.
    pub clean: bool,
    /// run.log 내용을 표준 출력에도.
    pub echo: bool,
    /// `timeline.txt`.
    pub timeline: bool,
    /// `run.log`, `DONE`.
    pub run_log: bool,
    /// 구역 PLY(`full/preview`, `full/refined`).
    pub zone_ply: bool,
    /// 정렬·스냅샷·manifest·final_frame.
    pub snapshots: bool,
    /// 구역별 희소 모델(`work/models/`).
    pub models: bool,
}

impl Default for Sinks {
    /// 출력 폴더 없는 [`Sinks::default_files`](TOML 에서 `[sinks]` 일부만 적었을 때의 나머지 값).
    fn default() -> Self {
        Self { out: None, ..Self::default_files("") }
    }
}

impl Sinks {
    /// `cumulus3d stream` 과 같은 파일 출력(희소 모델 제외, 표준 출력에도 기록, 시작 때 폴더 새로 만듦).
    pub fn default_files(out: impl Into<PathBuf>) -> Self {
        Self {
            out: Some(out.into()),
            clean: true,
            echo: true,
            timeline: true,
            run_log: true,
            zone_ply: true,
            snapshots: true,
            models: false,
        }
    }
    /// 파일 출력 없음(출력 폴더 없음, 표준 출력 기록 없음).
    pub fn none() -> Self {
        Self { out: None, clean: false, echo: false, timeline: false, run_log: false, zone_ply: false, snapshots: false, models: false }
    }
    /// 시작 때 출력 폴더 지움 여부.
    pub fn clean(mut self, on: bool) -> Self {
        self.clean = on;
        self
    }
    /// 표준 출력 기록 여부.
    pub fn echo(mut self, on: bool) -> Self {
        self.echo = on;
        self
    }
    /// `timeline.txt` 여부.
    pub fn timeline(mut self, on: bool) -> Self {
        self.timeline = on;
        self
    }
    /// `run.log` 여부.
    pub fn run_log(mut self, on: bool) -> Self {
        self.run_log = on;
        self
    }
    /// 구역 PLY 여부.
    pub fn zone_ply(mut self, on: bool) -> Self {
        self.zone_ply = on;
        self
    }
    /// 스냅샷·정렬 출력 여부.
    pub fn snapshots(mut self, on: bool) -> Self {
        self.snapshots = on;
        self
    }
    /// 구역별 희소 모델 여부.
    pub fn models(mut self, on: bool) -> Self {
        self.models = on;
        self
    }
    fn set(&self) -> SinkSet {
        SinkSet { timeline: self.timeline, run_log: self.run_log, zone_ply: self.zone_ply, models: self.models, snapshots: self.snapshots }
    }
}

impl Plan {
    /// TOML 문자열로.
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("계획은 항상 TOML 로 바꿀 수 있다")
    }

    /// TOML 문자열에서(없는 항목은 기본값, 모르는 항목은 오류).
    pub fn from_toml(s: &str) -> Result<Self, String> {
        toml::from_str(s).map_err(|e| format!("계획 TOML: {e}"))
    }

    /// TOML 파일에서.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let p = path.as_ref();
        let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        Self::from_toml(&s).map_err(|e| format!("{}: {e}", p.display()))
    }

    /// `cumulus3d stream` 설정과 같은 계획(`<src>/images`, `<src>/gps_ref.txt`, aerial-formation 카메라).
    pub fn from_stream(c: &StreamConfig) -> Self {
        Self {
            seed: c.seed,
            threads: c.threads,
            gps: Some(c.src.join("gps_ref.txt")),
            pairing: Pairing::Formation,
            input: Source {
                images: c.src.join("images"),
                cameras: CAMS.iter().map(|s| s.to_string()).collect(),
                stride: c.stride,
                positions: c.max_positions,
            },
            zones: Zones { span: c.span, overlap: c.overlap },
            align: Align { frame: AlignFrame::Enu, fixed_origin: c.fixed_enu_origin },
            features: Features {
                backend: c.sift_backend.clone(),
                max_num_features: c.max_num_features,
                max_image_size: c.sift_max_image_size,
            },
            matching: Matching { backend: c.match_backend.clone() },
            sparse: Sparse { incremental_triangulation: c.incremental_triangulation },
            dense: Dense {
                enabled: !c.no_dense,
                backend: c.pm_backend.clone(),
                profile: c.mvs_profile.clone(),
                max_image_size: c.dense_max_image_size,
                views: c.number_views,
                serialize: c.serialize_dense,
                depth_cache: c.depth_cache,
                fusion: Fusion::default(),
                filter: Filter::default(),
            },
            sinks: Sinks { models: c.save_models, echo: c.echo, ..Sinks::default_files(&c.out) },
        }
    }
}

// ============================================================================================
// 검사 오류
// ============================================================================================

/// 계획 검사 문제 하나.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// 계획 항목(TOML 경로, 예: `zones.overlap`).
    pub field: String,
    /// 설명.
    pub message: String,
}

/// [`ReconBuilder::build`] 의 검사 오류. 찾은 문제를 모두 담는다.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanError {
    /// 문제 목록(찾은 순서).
    pub problems: Vec<Problem>,
}

impl PlanError {
    /// 해당 항목의 문제가 있는지.
    pub fn has(&self, field: &str) -> bool {
        self.problems.iter().any(|p| p.field == field)
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "계획 검사 실패({} 건)", self.problems.len())?;
        for p in &self.problems {
            write!(f, "\n  - {}: {}", p.field, p.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for PlanError {}

#[derive(Default)]
struct Problems(Vec<Problem>);

impl Problems {
    fn add(&mut self, field: &str, message: impl Into<String>) {
        self.0.push(Problem { field: field.to_string(), message: message.into() });
    }
    fn check(&mut self, ok: bool, field: &str, message: impl FnOnce() -> String) {
        if !ok {
            self.add(field, message());
        }
    }
}

// ============================================================================================
// 빌더
// ============================================================================================

type Attach = Box<dyn FnOnce(Pipeline<Session>) -> Pipeline<Session>>;

/// 계획 기록기. 메서드는 기록만 하고 실행하지 않는다. [`ReconBuilder::build`] 로 검사한다.
pub struct ReconBuilder {
    plan: Plan,
    attach: Vec<Attach>,
    pending: Vec<Problem>,
    frames: bool,
}

impl fmt::Debug for ReconBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReconBuilder").field("plan", &self.plan).field("hooks", &self.attach.len()).field("pending", &self.pending).finish()
    }
}

impl ReconBuilder {
    /// 지금까지 기록한 계획.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// 입력 폴더: 영상 `<src>/images/<camera>/`, GPS `<src>/gps_ref.txt`.
    pub fn input(mut self, src: impl AsRef<Path>) -> Self {
        let src = src.as_ref();
        self.plan.input.images = src.join("images");
        self.plan.gps = Some(src.join("gps_ref.txt"));
        self
    }

    /// 영상 폴더(카메라 폴더들의 부모)만 지정.
    pub fn images(mut self, dir: impl Into<PathBuf>) -> Self {
        self.plan.input.images = dir.into();
        self
    }

    /// 카메라 폴더 목록(프레임 묶음 순서).
    pub fn cameras<S: Into<String>>(mut self, cams: impl IntoIterator<Item = S>) -> Self {
        self.plan.input.cameras = cams.into_iter().map(Into::into).collect();
        self
    }

    /// 프리셋 적용(카메라·짝 전략·stride). 모르는 이름은 `build` 에서 오류.
    /// `aerial-formation`: camF/camR/camL, 편대 짝 규칙, stride 3.
    pub fn preset(mut self, name: &str) -> Self {
        match name {
            DEFAULT_PRESET => {
                self.plan.input.cameras = CAMS.iter().map(|c| c.to_string()).collect();
                self.plan.input.stride = 3;
                self.plan.pairing = Pairing::Formation;
            }
            other => self
                .pending
                .push(Problem { field: "preset".into(), message: format!("알 수 없는 프리셋 {other} ({})", PRESETS.join("|")) }),
        }
        self
    }

    /// 파일 번호 간격.
    pub fn stride(mut self, n: usize) -> Self {
        self.plan.input.stride = n;
        self
    }

    /// 앞 N 위치만.
    pub fn positions(mut self, n: usize) -> Self {
        self.plan.input.positions = Some(n);
        self
    }

    /// 짝 전략.
    pub fn pairing(mut self, p: Pairing) -> Self {
        self.plan.pairing = p;
        self
    }

    /// 구역 크기·겹침(위치 수).
    pub fn zones(mut self, span: usize, overlap: usize) -> Self {
        self.plan.zones = Zones { span, overlap };
        self
    }

    /// GPS 기록 파일.
    pub fn gps(mut self, path: impl Into<PathBuf>) -> Self {
        self.plan.gps = Some(path.into());
        self
    }

    /// GPS 파일 없음(정렬도 함께 끄려면 [`ReconBuilder::align`] 에 [`AlignFrame::None`]).
    pub fn no_gps(mut self) -> Self {
        self.plan.gps = None;
        self
    }

    /// 정렬 좌표계.
    pub fn align(mut self, frame: AlignFrame) -> Self {
        self.plan.align.frame = frame;
        self
    }

    /// 첫 GPS 줄로 세션 ENU 원점 고정.
    pub fn fixed_enu_origin(mut self, on: bool) -> Self {
        self.plan.align.fixed_origin = on;
        self
    }

    /// 특징 추출 설정.
    pub fn features(mut self, f: Features) -> Self {
        self.plan.features = f;
        self
    }

    /// 매칭 백엔드(cpu|cuda).
    pub fn match_backend(mut self, name: &str) -> Self {
        self.plan.matching.backend = name.to_string();
        self
    }

    /// SIFT·매칭·PatchMatch 백엔드를 모두 cuda 로.
    pub fn gpu(mut self) -> Self {
        self.plan.features.backend = "cuda".into();
        self.plan.matching.backend = "cuda".into();
        self.plan.dense.backend = "cuda".into();
        self
    }

    /// 새로 등록된 영상만 삼각측량.
    pub fn incremental_triangulation(mut self, on: bool) -> Self {
        self.plan.sparse.incremental_triangulation = on;
        self
    }

    /// 조밀화 계획(켜짐).
    pub fn dense(mut self, d: Dense) -> Self {
        self.plan.dense = Dense { enabled: true, ..d };
        self
    }

    /// 조밀화 끔.
    pub fn no_dense(mut self) -> Self {
        self.plan.dense.enabled = false;
        self
    }

    /// 출력 훅 계획.
    pub fn sinks(mut self, s: Sinks) -> Self {
        self.plan.sinks = s;
        self
    }

    /// RANSAC 시드.
    pub fn seed(mut self, seed: u64) -> Self {
        self.plan.seed = Some(seed);
        self
    }

    /// rayon 스레드 수(0 = 코어 수).
    pub fn threads(mut self, n: usize) -> Self {
        self.plan.threads = n;
        self
    }

    fn hook(mut self, f: impl FnOnce(Pipeline<Session>) -> Pipeline<Session> + 'static) -> Self {
        self.attach.push(Box::new(f));
        self
    }

    /// 특정 종류 훅([`Pipeline::on`]).
    /// `EventKind::FrameDecoded` 를 고르면 프레임 디코딩이 켜진다.
    pub fn on<F>(mut self, kind: EventKind, f: F) -> Self
    where
        F: FnMut(&crate::events::Event) + Send + 'static,
    {
        self.frames |= kind == EventKind::FrameDecoded;
        self.hook(move |p| p.on(kind, f))
    }

    /// 프레임 훅([`Pipeline::on_frame`]): 디코딩된 입력 프레임(RGB)을 카메라 한 장씩, 위치 처리 전에 받는다.
    /// 등록하면 프레임 디코딩이 켜진다.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: FnMut(&Arc<Frame>) + Send + 'static,
    {
        self.frames = true;
        self.hook(move |p| p.on_frame(f))
    }

    /// 모든 이벤트 훅([`Pipeline::on_any`]).
    pub fn on_any<F>(self, f: F) -> Self
    where
        F: FnMut(&crate::events::Event) + Send + 'static,
    {
        self.hook(move |p| p.on_any(f))
    }

    /// 구역 초벌 점군 훅([`Pipeline::on_zone_preview`]).
    pub fn on_zone_preview<F>(self, f: F) -> Self
    where
        F: FnMut(ZoneRange, &Arc<PointCloud>) + Send + 'static,
    {
        self.hook(move |p| p.on_zone_preview(f))
    }

    /// 구역 정밀 점군 훅([`Pipeline::on_zone_refined`]).
    pub fn on_zone_refined<F>(self, f: F) -> Self
    where
        F: FnMut(ZoneRange, &Arc<PointCloud>) + Send + 'static,
    {
        self.hook(move |p| p.on_zone_refined(f))
    }

    /// 스냅샷 훅([`Pipeline::on_snapshot`]).
    pub fn on_snapshot<F>(self, f: F) -> Self
    where
        F: FnMut(usize, &Arc<PointCloud>) + Send + 'static,
    {
        self.hook(move |p| p.on_snapshot(f))
    }

    /// 위치 처리 완료 훅([`Pipeline::on_position_done`]).
    pub fn on_position_done<F>(self, f: F) -> Self
    where
        F: FnMut(usize, usize, usize) + Send + 'static,
    {
        self.hook(move |p| p.on_position_done(f))
    }

    /// 경고·오류 메시지 훅([`Pipeline::on_message`]).
    pub fn on_message<F>(self, f: F) -> Self
    where
        F: FnMut(EventKind, &str) + Send + 'static,
    {
        self.hook(move |p| p.on_message(f))
    }

    /// 종류별 큐 정책([`Pipeline::policy`]).
    pub fn policy(self, kind: EventKind, p: QueuePolicy) -> Self {
        self.hook(move |pl| pl.policy(kind, p))
    }

    /// 계획 검사 → 실행 가능한 [`Recon`]. 파일·폴더를 만들거나 지우지 않는다(입력 폴더 목록과 GPS 파일만 읽는다).
    ///
    /// 검사: 입력·카메라 폴더 존재, 선택된 위치 수, 정렬과 GPS 파일, 백엔드 이름과 장치(CUDA 요청 시),
    /// 출력 폴더 쓰기 가능, 옵션 값 범위. 문제는 모두 모아 [`PlanError`] 로 한 번에 돌려준다.
    pub fn build(self) -> Result<Recon, PlanError> {
        let ReconBuilder { plan, attach, pending, frames } = self;
        let mut pr = Problems(pending);
        let (layout, gps) = check(&plan, &mut pr);
        if !pr.0.is_empty() {
            return Err(PlanError { problems: pr.0 });
        }
        let (Some(layout), Some(gps)) = (layout, gps) else {
            unreachable!("문제가 없으면 입력과 GPS 를 읽었다");
        };
        Ok(Recon { plan, attach, layout, gps, frames })
    }
}

fn backend_ok(pr: &mut Problems, field: &str, name: &str, allowed: &[&str]) {
    if !allowed.contains(&name) {
        pr.add(field, format!("알 수 없는 백엔드 {name} ({})", allowed.join("|")));
    } else if name == "cuda" && !cumulus3d_cuda::is_available() {
        pr.add(field, "CUDA 장치를 찾지 못함 — cuda 백엔드를 쓸 수 없다");
    }
}

/// 가장 가까운 기존 상위 폴더가 쓰기 가능한지(만들지 않고 검사).
fn writable_target(p: &Path) -> Result<(), String> {
    if p.exists() {
        let m = std::fs::metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
        if !m.is_dir() {
            return Err(format!("{} 는 폴더가 아님", p.display()));
        }
        if m.permissions().readonly() {
            return Err(format!("{} 에 쓸 수 없음", p.display()));
        }
        return Ok(());
    }
    let mut a = p.parent();
    while let Some(d) = a {
        let d = if d.as_os_str().is_empty() { Path::new(".") } else { d };
        if d.exists() {
            let m = std::fs::metadata(d).map_err(|e| format!("{}: {e}", d.display()))?;
            if !m.is_dir() {
                return Err(format!("{} 는 폴더가 아님", d.display()));
            }
            if m.permissions().readonly() {
                return Err(format!("{} 에 쓸 수 없음", d.display()));
            }
            return Ok(());
        }
        a = d.parent();
    }
    Ok(())
}

fn check(plan: &Plan, pr: &mut Problems) -> (Option<Layout>, Option<Vec<GpsRecord>>) {
    // 값 범위.
    let inp = &plan.input;
    pr.check(inp.stride >= 1, "input.stride", || "1 이상이어야 함".into());
    pr.check(inp.positions != Some(0), "input.positions", || "1 이상이어야 함".into());
    pr.check(!inp.cameras.is_empty(), "input.cameras", || "카메라 폴더 목록이 비어 있음".into());
    let mut seen = std::collections::BTreeSet::new();
    for c in &inp.cameras {
        if c.is_empty() || c.contains(['/', '\\']) {
            pr.add("input.cameras", format!("잘못된 카메라 폴더 이름 {c:?}"));
        } else if !seen.insert(c.as_str()) {
            pr.add("input.cameras", format!("카메라 {c} 중복"));
        }
    }
    let z = plan.zones;
    pr.check(z.span >= 1, "zones.span", || "1 이상이어야 함".into());
    pr.check(z.overlap < z.span.max(1), "zones.overlap", || format!("구역 크기({})보다 작아야 함 (지금 {})", z.span, z.overlap));
    let f = &plan.features;
    backend_ok(pr, "features.backend", &f.backend, &["cpu", "cuda"]);
    pr.check(f.max_num_features >= 1, "features.max_num_features", || "1 이상이어야 함".into());
    pr.check(f.max_image_size >= 16, "features.max_image_size", || "16 이상이어야 함".into());
    backend_ok(pr, "matching.backend", &plan.matching.backend, &["cpu", "cuda"]);

    let d = &plan.dense;
    if d.enabled {
        if d.backend != "cuda" {
            pr.add("dense.backend", format!("알 수 없는 백엔드 {} (cuda)", d.backend));
        } else if !cumulus3d_cuda::is_available() {
            pr.add("dense.backend", "조밀화에는 CUDA 장치가 필요한데 장치를 찾지 못함 — `.no_dense()` 또는 [dense] enabled = false");
        }
        if let Err(e) = parse_profile(&d.profile) {
            pr.add("dense.profile", e);
        }
        pr.check(d.max_image_size >= 16, "dense.max_image_size", || "16 이상이어야 함".into());
        pr.check((1..=32).contains(&d.views), "dense.views", || format!("1..=32 이어야 함 (지금 {})", d.views));
        let fu = &d.fusion;
        if let Some(m) = &fu.mode {
            if let Err(e) = parse_fusion_mode(m) {
                pr.add("dense.fusion.mode", e);
            }
        }
        if let Some(m) = &fu.residual {
            pr.check(FusionResidual::parse(m).is_some(), "dense.fusion.residual", || {
                format!("알 수 없는 처리 {m} (none|release|second-pass)")
            });
        }
        if let Some(v) = fu.min_views {
            pr.check(v >= 1, "dense.fusion.min_views", || "1 이상이어야 함".into());
        }
        if let Some(v) = fu.depth_error {
            pr.check(v > 0.0 && v.is_finite(), "dense.fusion.depth_error", || format!("양수여야 함 (지금 {v})"));
        }
        if let Some(v) = fu.normal_error {
            pr.check(v > 0.0 && v <= 180.0, "dense.fusion.normal_error", || format!("(0, 180] 이어야 함 (지금 {v})"));
        }
        if let Some(v) = fu.reproj_error {
            pr.check(v > 0.0 && v.is_finite(), "dense.fusion.reproj_error", || format!("양수여야 함 (지금 {v})"));
        }
        let fi = &d.filter;
        if let Some(v) = fi.geom_error {
            pr.check(v > 0.0 && v.is_finite(), "dense.filter.geom_error", || format!("양수여야 함 (지금 {v})"));
        }
        if let Some(v) = fi.min_ncc {
            pr.check((-1.0..=1.0).contains(&v), "dense.filter.min_ncc", || format!("[-1, 1] 이어야 함 (지금 {v})"));
        }
    }

    // 출력.
    let s = &plan.sinks;
    if let Some(out) = &s.out {
        if let Err(e) = writable_target(out) {
            pr.add("sinks.out", e);
        }
        if s.clean && (inp.images.starts_with(out) || plan.gps.as_ref().is_some_and(|g| g.starts_with(out))) {
            pr.add("sinks.out", format!("시작 때 지울 출력 폴더 {} 안에 입력이 있음", out.display()));
        }
    }

    // 입력 폴더.
    let mut layout = None;
    if !inp.images.is_dir() {
        pr.add("input.images", format!("영상 폴더 없음: {}", inp.images.display()));
    } else {
        let missing: Vec<&String> = inp.cameras.iter().filter(|c| !inp.images.join(c).is_dir()).collect();
        for c in &missing {
            pr.add("input.cameras", format!("카메라 폴더 없음: {}", inp.images.join(c).display()));
        }
        if missing.is_empty() && !inp.cameras.is_empty() && inp.stride >= 1 {
            match Layout::discover_in(&inp.images, &inp.cameras, inp.stride) {
                Ok(mut l) => {
                    if let Some(n) = inp.positions {
                        l.truncate(n);
                    }
                    if l.frames.is_empty() {
                        pr.add(
                            "input",
                            format!(
                                "선택된 영상 없음: {}/{c}/{c}_NNNN.jpg 중 번호 % {} == 0 인 파일",
                                inp.images.display(),
                                inp.stride,
                                c = inp.cameras[0]
                            ),
                        );
                    } else {
                        layout = Some(l);
                    }
                }
                Err(e) => pr.add("input.images", e),
            }
        }
    }

    // GPS·정렬.
    let mut gps = None;
    match (plan.align.frame, &plan.gps) {
        (AlignFrame::None, _) => gps = Some(Vec::new()),
        (AlignFrame::Enu, None) => pr.add("gps", "ENU 정렬에는 GPS 파일이 필요함 — `.gps(경로)` 또는 정렬 끔([align] frame = \"none\")"),
        (AlignFrame::Enu, Some(p)) => match read_gps_file(p) {
            Err(e) => pr.add("gps", format!("{}: {e}", p.display())),
            Ok(all) => {
                if let Some(l) = &layout {
                    let sel: Vec<GpsRecord> = all.into_iter().filter(|g| l.pos_of.contains_key(&g.name)).collect();
                    if sel.is_empty() {
                        pr.add("gps", format!("{}: 선택된 영상의 GPS 기록이 하나도 없음", p.display()));
                    } else {
                        gps = Some(sel);
                    }
                } else {
                    gps = Some(Vec::new());
                }
            }
        },
    }
    (layout, gps)
}

// ============================================================================================
// 실행
// ============================================================================================

/// 검사를 마친 실행 가능한 재구성. [`Recon::run`] 으로 실행한다.
pub struct Recon {
    plan: Plan,
    attach: Vec<Attach>,
    layout: Layout,
    gps: Vec<GpsRecord>,
    frames: bool,
}

impl fmt::Debug for Recon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Recon")
            .field("plan", &self.plan)
            .field("hooks", &self.attach.len())
            .field("positions", &self.layout.frames.len())
            .finish()
    }
}

impl Recon {
    /// 기본 계획([`Plan::default`])으로 기록을 시작한다.
    pub fn declare() -> ReconBuilder {
        Self::from_plan(Plan::default())
    }

    /// 주어진 계획으로 기록을 시작한다(TOML 에서 읽은 계획 등).
    pub fn from_plan(plan: Plan) -> ReconBuilder {
        ReconBuilder { plan, attach: Vec::new(), pending: Vec::new(), frames: false }
    }

    /// 계획.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// 선택된 위치 수.
    pub fn positions(&self) -> usize {
        self.layout.frames.len()
    }

    /// 위치 ↔ 영상 이름 배치.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// 실행: 출력 폴더 준비 → 세션 + 파이프라인 + 기본 출력 훅 + 사용자 훅 → 위치 반복 → 배경 작업 대기·종료.
    /// 치명 오류(세션 실패)면 후처리 없이 그 오류를 돌려준다.
    pub fn run(self) -> Result<Summary, String> {
        let Recon { plan, attach, layout, gps, frames } = self;
        if plan.threads > 0 {
            // 이미 만들어졌으면(테스트 등) 무시.
            let _ = rayon::ThreadPoolBuilder::new().num_threads(plan.threads).build_global();
        }
        // 백엔드가 없으면 조밀화 단계에서 오류로 기록한다.
        let dense = if plan.dense.enabled { Some(plan.dense.config()?) } else { None };
        let s = &plan.sinks;
        let sinks = match &s.out {
            Some(out) => {
                if s.clean && out.exists() {
                    std::fs::remove_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
                }
                Some(DefaultSinks::with_set(out, s.echo, s.set()).map_err(|e| e.to_string())?)
            }
            None => None,
        };
        let npos = layout.frames.len();
        let mut sc = SessionConfig::new(plan.input.images.clone());
        sc.gps = gps;
        sc.span = plan.zones.span;
        sc.overlap = plan.zones.overlap;
        sc.total_positions = Some(npos);
        sc.incremental_triangulation = plan.sparse.incremental_triangulation;
        sc.fixed_enu_origin = plan.align.fixed_origin;
        sc.seed = plan.seed;
        sc.extraction.sift.max_num_features = plan.features.max_num_features;
        sc.extraction.reader.max_image_size = plan.features.max_image_size;
        sc.sift = make_sift_backend(&plan.features.backend)?;
        sc.matcher = Arc::from(make_match_backend(&plan.matching.backend)?);
        sc.dense = dense;
        // 되감지 않는다(메모리 절약).
        sc.history = 0;
        sc.decode_frames = frames;

        let mut pl = Pipeline::new(Session::new(sc));
        if let Some(sinks) = sinks {
            // 출력 훅은 하나(on_any)라 비동기여도 이벤트 순서대로 처리된다.
            pl = pl.on_any(sinks.into_hook());
        }
        for a in attach {
            pl = a(pl);
        }

        let (span, ovl) = (plan.zones.span, plan.zones.overlap);
        let mut reg_end: BTreeMap<usize, Vec<ZoneRange>> = BTreeMap::new();
        for z in zones_upto(npos, span, ovl) {
            reg_end.entry(z.hi - 1).or_default().push(z);
        }
        pl.push(Input::Log(format!(
            "regions: {}",
            reg_end
                .iter()
                .map(|(e, v)| format!("{e} ->{}", v.iter().map(|z| format!(" {}:{}:{}", z.zone, z.lo, z.hi)).collect::<String>()))
                .collect::<Vec<_>>()
                .join(", ")
        )));
        let d = &plan.dense;
        pl.push(Input::Log(format!(
            "설정: span {span} overlap {ovl} stride {} 증분삼각측량 {} 깊이캐시 {} 고정원점 {} pm {}({})/sift {}/match {} 직렬조밀화 {} 왜곡보정최대 {} 뷰 {} 스레드 {}",
            plan.input.stride,
            plan.sparse.incremental_triangulation,
            d.depth_cache,
            plan.align.fixed_origin,
            d.backend,
            d.profile,
            plan.features.backend,
            plan.matching.backend,
            d.serialize,
            d.max_image_size,
            d.views,
            rayon::current_num_threads()
        )));
        for p in 0..npos {
            pl.push(frame_set_of(&layout, &plan.input.cameras, p).into());
            if let Some(e) = pl.reducer().failed() {
                // 치명 오류: 후처리 없이 멈춘다(훅 큐는 파이프라인을 버릴 때 비워진다).
                return Err(e.to_string());
            }
        }
        let (_session, summary) = pl.finish();
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_plan_toml_roundtrip() {
        let p = Plan::default();
        let s = p.to_toml();
        assert_eq!(Plan::from_toml(&s).unwrap(), p, "{s}");
        // 빈 문서 = 기본 계획.
        assert_eq!(Plan::from_toml("").unwrap(), p);
    }

    #[test]
    fn full_plan_toml_roundtrip() {
        let b = Recon::declare()
            .input("some/where")
            .cameras(["a", "b"])
            .stride(2)
            .positions(5)
            .zones(6, 1)
            .fixed_enu_origin(true)
            .incremental_triangulation(true)
            .dense(
                Dense::profile("quality")
                    .fusion_mode("traversal")
                    .fusion_min_views(3)
                    .fusion_depth_error(0.02)
                    .fusion_normal_error(12.5)
                    .fusion_reproj_error(1.5)
                    .fusion_residual("second-pass")
                    .filter_min_views(2)
                    .filter_geom_error(1.25)
                    .filter_min_ncc(0.1)
                    .median_filter(true)
                    .depth_cache(true)
                    .serialize(true),
            )
            .sinks(Sinks::default_files("o").models(true).echo(false))
            .seed(42)
            .threads(3);
        let p = b.plan().clone();
        let s = p.to_toml();
        assert_eq!(Plan::from_toml(&s).unwrap(), p, "{s}");
        assert!(s.contains("[dense.fusion]") && s.contains("min_views = 3"), "{s}");
    }

    #[test]
    fn example_plan_parses() {
        let s = include_str!("../../../examples/plans/aerial-formation.toml");
        let p = Plan::from_toml(s).unwrap();
        assert_eq!(p.input.cameras, ["camF", "camR", "camL"]);
        assert_eq!(p.pairing, Pairing::Formation);
        assert_eq!(Plan::from_toml(&p.to_toml()).unwrap(), p);
    }

    #[test]
    fn unknown_field_rejected() {
        assert!(Plan::from_toml("[zones]\nspam = 3\n").is_err());
        assert!(Plan::from_toml("pairing = \"grid\"\n").is_err());
    }

    #[test]
    fn stream_plan_matches_builder() {
        let mut c = StreamConfig::new("src".into(), "dst".into());
        (c.stride, c.span, c.overlap, c.seed, c.save_models, c.echo) = (1, 4, 1, Some(7), true, false);
        let b = Recon::declare()
            .input("src")
            .preset(DEFAULT_PRESET)
            .stride(1)
            .zones(4, 1)
            .seed(7)
            .sinks(Sinks::default_files("dst").models(true).echo(false));
        assert_eq!(Plan::from_stream(&c), *b.plan());
    }

    #[test]
    fn build_collects_all_problems() {
        let tmp = std::env::temp_dir().join(format!("cumulus3d_declare_missing_{}", std::process::id()));
        let e = Recon::declare()
            .input(tmp.join("in"))
            .preset("grid")
            .stride(0)
            .zones(4, 4)
            .dense(Dense::profile("slow").views(40).fusion_mode("magic").filter_min_ncc(2.0))
            .features(Features { backend: "fpga".into(), ..Features::default() })
            .sinks(Sinks::default_files(tmp.join("in")).clean(true))
            .build()
            .unwrap_err();
        for f in [
            "preset",
            "input.stride",
            "zones.overlap",
            "features.backend",
            "dense.profile",
            "dense.views",
            "dense.fusion.mode",
            "dense.filter.min_ncc",
            "sinks.out",
            "input.images",
            "gps",
        ] {
            assert!(e.has(f), "{f} 없음:\n{e}");
        }
        assert_eq!(e.has("dense.backend"), !cumulus3d_cuda::is_available());
        assert!(!tmp.exists());
        assert!(e.to_string().starts_with("계획 검사 실패("));
    }

    #[test]
    fn enu_without_gps_is_error() {
        let e = Recon::declare().no_gps().no_dense().build().unwrap_err();
        assert!(e.has("gps"), "{e}");
        let e = Recon::declare().no_gps().align(AlignFrame::None).no_dense().build().unwrap_err();
        assert!(!e.has("gps"), "{e}");
    }
}
