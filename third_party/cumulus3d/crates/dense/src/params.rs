/*
 * params.rs
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

//! 조밀화 매개변수: 프로파일, PatchMatch 상수, 필터·융합 허용치.

/// 조밀화 프로파일.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MvsProfile {
    /// 짧은 반복 일정(기본). 세부 복원 단계는 상향 표본 가설에서 시작한다.
    #[default]
    Fast,
    /// 긴 반복 일정. 세부 복원 단계는 무작위 초기화에서 시작한다.
    Quality,
}

impl MvsProfile {
    /// 이름("fast" / "quality", 대소문자 무시)으로 찾는다.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "quality" => Some(Self::Quality),
            _ => None,
        }
    }
    /// 프로파일 이름.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Quality => "quality",
        }
    }
}

/// 한 스케일의 반복 일정.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LevelSchedule {
    /// 광도 실행의 적·흑 반복 수(최저 스케일: 무작위 초기화, 그 위: 세부 복원 실행).
    pub photometric_iters: u32,
    /// 세부 복원 실행을 무작위 초기화로 시작할지(거짓이면 상향 표본 가설로 시작).
    pub restorer_random_init: bool,
    /// 기하 일관성 실행 회차 수(모든 기준 뷰 동기).
    pub geometric_rounds: u32,
    /// 기하 실행 한 회차의 반복 수.
    pub geometric_iters: u32,
}

/// PatchMatch 상수(비용·뷰 선택·정제·기하 항).
#[derive(Clone, Debug, PartialEq)]
pub struct PmParams {
    /// 창 반경(픽셀).
    pub window_radius: i32,
    /// 창 안 표본 간격(1 = 전수, 2 = 행·열 하나 걸러).
    pub window_step: i32,
    /// 양방향 가중치 공간 σ(픽셀). ≤ 0 이면 창 반경.
    pub sigma_spatial: f32,
    /// 양방향 가중치 밝기 σ(\[0,1\] 밝기).
    pub sigma_color: f32,
    /// 좋은 비용 경계 τ(t) = tau0·exp(−t²/tau_alpha).
    pub tau0: f32,
    /// 좋은 비용 경계의 감쇠 상수 α.
    pub tau_alpha: f32,
    /// 나쁜 비용 경계.
    pub tau1: f32,
    /// 비용 신뢰도 exp(−m²/(2β²)).
    pub beta: f32,
    /// 뷰 채택: 좋은 비용 수 > n1 이고 나쁜 비용 수 < n2.
    pub n1: u32,
    /// 뷰 채택의 나쁜 비용 수 상한 n2.
    pub n2: u32,
    /// 채택되지 않은 직전 최중요 뷰의 가중치.
    pub prev_view_weight: f32,
    /// 삼각측량 사전 최소 각(도).
    pub min_triangulation_angle_deg: f32,
    /// 입사각 사전 σ.
    pub incident_angle_sigma: f32,
    /// 기하 항 가중치 λ 와 상한 δ(픽셀).
    pub geom_lambda: f32,
    /// 기하 항 상한 δ(픽셀).
    pub geom_max_cost: f32,
    /// 가중치 합이 0 일 때 쓰는 상위 K 평균의 K.
    pub top_k: u32,
    /// 정제 섭동 시작 크기(역깊이 상대, 법선 각 도): 광도/기하.
    pub eps0_photometric: f32,
    /// 정제 섭동 시작 크기(역깊이 상대): 기하 실행.
    pub eps0_geometric: f32,
    /// 정제 법선 섭동 시작 각(도): 광도 실행.
    pub phi0_photometric_deg: f32,
    /// 정제 법선 섭동 시작 각(도): 기하 실행.
    pub phi0_geometric_deg: f32,
    /// 역깊이 섭동 하한.
    pub eps_min: f32,
    /// 법선 섭동 각 하한(도).
    pub phi_min_deg: f32,
    /// 거친 스케일 가설을 후보로 더할 기준 패치 분산 문턱(\[0,1\] 밝기²). 0 이면 끔.
    pub weak_texture_var: f32,
}

impl Default for PmParams {
    fn default() -> Self {
        Self {
            window_radius: 5,
            window_step: 2,
            sigma_spatial: -1.0,
            sigma_color: 0.2,
            tau0: 0.8,
            tau_alpha: 90.0,
            tau1: 1.2,
            beta: 0.3,
            n1: 2,
            n2: 3,
            prev_view_weight: 0.2,
            min_triangulation_angle_deg: 1.0,
            incident_angle_sigma: 0.9,
            geom_lambda: 0.2,
            geom_max_cost: 3.0,
            top_k: 3,
            eps0_photometric: 0.2,
            eps0_geometric: 0.05,
            phi0_photometric_deg: 30.0,
            phi0_geometric_deg: 10.0,
            eps_min: 0.005,
            phi_min_deg: 1.0,
            weak_texture_var: 0.0,
        }
    }
}

impl PmParams {
    /// 실효 공간 σ.
    pub fn sigma_s(&self) -> f32 {
        if self.sigma_spatial > 0.0 {
            self.sigma_spatial
        } else {
            self.window_radius as f32
        }
    }
    /// 창 표본 오프셋(−R..R, 간격 step).
    pub fn window_offsets(&self) -> Vec<i32> {
        let r = self.window_radius;
        let s = self.window_step.max(1);
        let mut v = Vec::new();
        let mut o = -r;
        while o <= r {
            v.push(o);
            o += s;
        }
        v
    }
}

/// 깊이맵 필터 허용치.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterParams {
    /// 광도 조건: 가시 확률 ≥ E(1 − min_ncc).
    pub min_ncc: f32,
    /// 이 각(도)보다 작은 삼각측량각의 뷰는 세지 않는다.
    pub min_triangulation_angle_deg: f32,
    /// 일치 뷰 최소 수(실효값은 min(원천 수, 이 값)).
    pub min_num_consistent: u32,
    /// 순·역 재투영 오차 허용(픽셀).
    pub geom_max_cost: f32,
    /// 확률 방출 σ.
    pub ncc_sigma: f32,
    /// 판독 전 5×5 중앙값 평면 필터.
    pub median_filter: bool,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            min_ncc: 0.1,
            min_triangulation_angle_deg: 3.0,
            min_num_consistent: 2,
            geom_max_cost: 1.0,
            ncc_sigma: 0.6,
            median_filter: true,
        }
    }
}

/// 융합 방식.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FusionMode {
    /// 이웃의 이웃으로 깊이 우선 확장해 모은 픽셀의 성분별 중앙값 점(표면당 점 하나에 가깝다).
    Traversal,
    /// 기준 픽셀마다 이웃 뷰에서 깊이·법선·재투영이 일치하는 뷰 수를 세어, 충분하면 일치 점들의 평균 점.
    #[default]
    Consistency,
}

impl FusionMode {
    /// 이름("traversal"/"median", "consistency"/"average")으로 찾는다.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "traversal" | "median" => Some(Self::Traversal),
            "consistency" | "average" => Some(Self::Consistency),
            _ => None,
        }
    }
}

/// 일치 융합에서 기준 미달 픽셀(남은 픽셀) 처리.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FusionResidual {
    /// 처리 없음(기본).
    #[default]
    None,
    /// 버려진 후보가 일시 점유했던 픽셀(기준·이웃)을 되돌려 뒤 기준 영상이 다시 쓸 수 있게 한다.
    Release,
    /// 1차 융합 뒤, 점이 되지 못한 유효 깊이 픽셀만으로 더 엄격한 허용치의 2차 융합.
    SecondPass,
}

impl FusionResidual {
    /// 이름("none", "release", "second-pass")으로 찾는다.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "none" => Some(Self::None),
            "release" => Some(Self::Release),
            "second-pass" | "secondpass" => Some(Self::SecondPass),
            _ => None,
        }
    }
    /// 처리 방식 이름.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Release => "release",
            Self::SecondPass => "second-pass",
        }
    }
}

/// 2차 융합 허용치.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualParams {
    /// 기준 외 일치 뷰 최소 수.
    pub min_views: usize,
    /// 상대 깊이 허용치(None = 1차의 0.5배).
    pub depth_error: Option<f64>,
    /// 법선 허용 각(도, None = 1차의 0.67배).
    pub normal_error_deg: Option<f64>,
    /// 1차 점과 이 거리(GSD 배수) 이내인 2차 점은 버린다.
    pub min_dist_gsd: f64,
}

impl Default for ResidualParams {
    fn default() -> Self {
        Self { min_views: 2, depth_error: None, normal_error_deg: None, min_dist_gsd: 0.5 }
    }
}

/// 융합 허용치.
#[derive(Clone, Debug, PartialEq)]
pub struct FusionParams {
    /// 융합 방식.
    pub mode: FusionMode,
    /// 일치 융합: 남은 픽셀 처리.
    pub residual: FusionResidual,
    /// 2차 융합 허용치.
    pub residual_params: ResidualParams,
    /// 일치 융합: 기준 외 일치 뷰 최소 수.
    pub min_consistent_views: usize,
    /// 일치 융합: 이미 다른 기준 픽셀의 점에 쓰인 픽셀은 기준으로 다시 쓰지 않음.
    pub mark_used: bool,
    /// 일치 융합: 표시 반경(픽셀, 0 = 투영 픽셀만).
    pub mark_radius: usize,
    /// 일치 융합: 깊이 불확실성 역분산 가중 평균(끄면 단순 평균).
    pub inverse_variance: bool,
    /// 정합 불확실성 σ_px = sigma_px0 + sigma_px_slope·비용 (픽셀).
    pub sigma_px0: f64,
    /// σ_px 의 비용 기울기.
    pub sigma_px_slope: f64,
    /// 일치 융합: 검사할 겹침 뷰 수(공유 점 순).
    pub consistency_num_images: usize,
    /// 확장 융합: 점 하나에 필요한 최소 픽셀 수.
    pub min_num_pixels: usize,
    /// 확장 융합: 점 하나에 모을 최대 픽셀 수.
    pub max_num_pixels: usize,
    /// 확장 융합: 최대 확장 깊이.
    pub max_traversal_depth: usize,
    /// 재투영 오차 허용치(픽셀).
    pub max_reproj_error: f64,
    /// 상대 깊이 허용치.
    pub max_depth_error: f64,
    /// 법선 허용 각(도).
    pub max_normal_error_deg: f64,
    /// 확장 융합: 검사할 겹침 뷰 수.
    pub check_num_images: usize,
}

impl Default for FusionParams {
    fn default() -> Self {
        Self {
            mode: FusionMode::Consistency,
            residual: FusionResidual::None,
            residual_params: ResidualParams::default(),
            min_consistent_views: 5,
            mark_used: true,
            mark_radius: 0,
            inverse_variance: true,
            sigma_px0: 0.25,
            sigma_px_slope: 1.0,
            consistency_num_images: 12,
            min_num_pixels: 5,
            max_num_pixels: 10000,
            max_traversal_depth: 100,
            max_reproj_error: 2.0,
            max_depth_error: 0.01,
            max_normal_error_deg: 10.0,
            check_num_images: 50,
        }
    }
}

/// 이웃 뷰 선택.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborParams {
    /// 기준 뷰당 원천 뷰 최대 수(≤ 32).
    pub num_views: usize,
    /// 75 분위 삼각측량각 하한(도).
    pub min_triangulation_angle_deg: f64,
    /// 방향 다양성: 기준선 방위각 구간 수와 같은 구간 반복 선택 감쇠(1 = 끔).
    pub direction_bins: usize,
    /// 같은 구간 반복 선택 감쇠(1 = 방향 다양성 끔).
    pub diversity_decay: f64,
}

impl Default for NeighborParams {
    fn default() -> Self {
        Self { num_views: 10, min_triangulation_angle_deg: 1.0, direction_bins: 8, diversity_decay: 1.0 }
    }
}

/// `densify` 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct DensifyOptions {
    /// 반복 일정 프로파일.
    pub profile: MvsProfile,
    /// 이웃 뷰 선택.
    pub neighbors: NeighborParams,
    /// PatchMatch 상수.
    pub pm: PmParams,
    /// 깊이맵 필터 허용치.
    pub filter: FilterParams,
    /// 융합 허용치.
    pub fusion: FusionParams,
    /// 필터 뒤 후처리.
    pub post: crate::postproc::PostParams,
    /// 스케일 수 상한(축소율 0.5).
    pub max_levels: usize,
    /// 최저 스케일의 짧은 변 하한(이보다 작아지면 스케일 수를 줄인다).
    pub min_level_size: usize,
    /// 결합 양방향 상향 표본 매개변수: 공간 σ(저해상 픽셀), 밝기 σ.
    pub jbu_sigma_spatial: f32,
    /// 결합 양방향 상향 표본 밝기 σ.
    pub jbu_sigma_color: f32,
    /// 세부 복원기 교체 문턱 ξ.
    pub restorer_threshold: f32,
    /// 난수 시드.
    pub seed: u64,
    /// 일정 덮어쓰기(측정용): 기하 실행 반복 수, 회차 수.
    pub geometric_iters_override: Option<u32>,
    /// 일정 덮어쓰기: 기하 실행 회차 수.
    pub geometric_rounds_override: Option<u32>,
}

impl Default for DensifyOptions {
    fn default() -> Self {
        Self {
            profile: MvsProfile::Fast,
            neighbors: NeighborParams::default(),
            pm: PmParams::default(),
            filter: FilterParams::default(),
            fusion: FusionParams::default(),
            post: crate::postproc::PostParams::default(),
            max_levels: 3,
            min_level_size: 64,
            jbu_sigma_spatial: 1.0,
            jbu_sigma_color: 0.04,
            restorer_threshold: 0.1,
            seed: 0x005e_ed0f_de75,
            geometric_iters_override: None,
            geometric_rounds_override: None,
        }
    }
}

impl DensifyOptions {
    /// 프로파일 기본값: fast 는 창 반경 5·간격 2(36 표본), quality 는 반경 5·간격 1(121 표본).
    pub fn with_profile(profile: MvsProfile) -> Self {
        let mut o = Self { profile, ..Self::default() };
        o.pm.window_step = match profile {
            MvsProfile::Fast => 2,
            MvsProfile::Quality => 1,
        };
        o
    }

    /// 스케일 `level`(0 = 최저)의 일정. `top` 은 최고 스케일 번호.
    pub fn schedule(&self, level: usize, top: usize) -> LevelSchedule {
        let mut s = self.base_schedule(level, top);
        if let Some(v) = self.geometric_iters_override {
            s.geometric_iters = v;
        }
        if let Some(v) = self.geometric_rounds_override {
            s.geometric_rounds = v;
        }
        s
    }

    fn base_schedule(&self, level: usize, top: usize) -> LevelSchedule {
        match self.profile {
            MvsProfile::Quality => LevelSchedule {
                photometric_iters: if level == 0 { 7 } else { 6 },
                restorer_random_init: true,
                geometric_rounds: 2,
                geometric_iters: 6,
            },
            MvsProfile::Fast => {
                let _ = top;
                LevelSchedule {
                    photometric_iters: if level == 0 { 6 } else { 3 },
                    restorer_random_init: false,
                    geometric_rounds: 2,
                    geometric_iters: 2,
                }
            }
        }
    }

    /// 설정 해시(깊이맵 캐시 열쇠용).
    pub fn fingerprint(&self) -> u64 {
        let s = format!(
            "{:?}",
            (
                self.profile,
                &self.neighbors,
                &self.pm,
                &self.filter,
                &self.post,
                self.max_levels,
                self.min_level_size,
                self.jbu_sigma_spatial,
                self.jbu_sigma_color,
                (self.restorer_threshold, self.seed, self.geometric_iters_override, self.geometric_rounds_override)
            )
        );
        crate::math::hash_bytes(s.as_bytes())
    }
}
