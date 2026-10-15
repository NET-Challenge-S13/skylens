/*
 * kernel.rs
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

//! PatchMatch 커널 경계. 한 스케일의 전파·뷰 선택·정제(적·흑 반복 실행), 단일 가설 비용 평가, 최종 판독을
//! 백엔드가 맡고, 다중 스케일 진행·상향 표본·세부 복원 판정·필터 문턱·융합은 이 크레이트가 맡는다.
//!
//! # 커널이 지켜야 할 규칙
//! 좌표는 정수 픽셀 = 픽셀 중심. 광선 `ray(x, y) = ((x−cx)/fx, (y−cy)/fy, 1)`, 깊이는 카메라 z,
//! 법선은 기준 카메라 좌표의 단위 벡터이고 `n·ray < 0`.
//!
//! - **비용** `m = 1 − ρ`: 기준 영상 색만 쓰는 양방향 가중 NCC. 창은 `PmParams::window_offsets()` 의 정사각 격자,
//!   가중치 `exp(−(dx²+dy²)/(2σs²) − (I_c − I_k)²/(2σc²))`, 영상 밖 밝기는 0. 원천 밝기는 평면 유도 호모그래피
//!   `H = K_j (R + t nᵀ/δ) K⁻¹`, `δ = n·(d·ray)` 로 옮긴 위치의 쌍선형 값(밖은 0). 분산 < 1e−5 이면 2,
//!   아니면 `clamp(1 − cov/√(var_r var_s), 0, 2)`. 중심 투영이 카메라 뒤이거나 원천 영상 밖이면 2.
//! - **기하 오차** `Δe`: 기준 점을 원천에 투영해 원천 깊이맵의 최근접 값으로 되돌려 기준에 재투영한 픽셀 거리.
//!   투영 실패·영상 밖·깊이 0 이면 무한. 비용에서는 `min(Δe, geom_max_cost)`.
//! - **사전확률**(반쪽 단계 시작 시점의 현재 가설로 뷰마다 1회): 삼각측량·입사각·해상도 사전의 곱([`crate::math`]).
//! - **반복**: 적·흑 반쪽 단계. 색 `(x+y) mod 2`. 활성 픽셀은 8개 영역(위·오른쪽·아래·왼쪽 방향마다
//!   V자 7개 `(0,−1),(±1,−2),(±2,−3),(±3,−4)` 와 띠 11개 `(0,−3),(0,−5),…,(0,−23)` 를 90° 씩 돌린 것)에서
//!   저장 비용이 가장 작은 이웃(동점이면 먼저 나온 것)의 평면을 자기 광선에 옮긴 가설 8개와 현재 가설을 평가한다.
//!   영역에 유효 이웃이 없거나 옮기기가 실패하면 현재 가설로 대신한다.
//! - **뷰 선택**: 전파 가설 8개의 비용으로 투표(좋음 `m < τ(t) = τ0·exp(−t²/α)`, 나쁨 `m > τ1`; 좋음 > n1 이고
//!   나쁨 < n2 이면 채택). 채택 뷰 가중 `w = 좋은 비용들의 exp(−m²/(2β²)) 평균`, 직전 반복의 최중요 뷰면 2배.
//!   채택되지 않았어도 직전 최중요 뷰면 `prev_view_weight`. 여기에 사전확률을 곱한 `w''` 로 가설별 가중 평균
//!   (기하 실행이면 `m + λ·min(Δe, δ)`). 가중 합이 0 이면 같은 순회에서 모은 사전확률 가중 평균으로 대신한다.
//!   사전확률이 0 인 뷰는 비용을 계산하지 않는다. 최소 비용 가설 채택(동점은 앞 번호).
//!   최중요 뷰 = `w''` 최대(동점은 앞 뷰).
//! - **정제**(같은 반쪽 단계): 무작위 깊이(역깊이 균등 `[1/dmax, 1/dmin]`), 무작위 법선(구면 균등, 카메라 쪽으로 뒤집음),
//!   섭동 깊이 `1/d' = (1/d)(1 + u·ε_t)`, 섭동 법선(무작위 수직 축으로 `u·φ_t` 회전, 실패 시 각 절반으로 최대 3회 재시도),
//!   `ε_t = max(ε_min, ε0·2^−t)`, `φ_t = max(φ_min, φ0·2^−t)` 로 조합 6개 `(d_r,n)(d,n_r)(d_r,n_r)(d_p,n)(d,n_p)(d_p,n_p)`.
//!   같은 가중치(양수인 뷰만)로 집계해 더 작으면 채택.
//! - **난수**: 픽셀마다 `splitmix64` 수열. 시작 상태 `mix64(seed ^ mix64(view_key ^ mix64(level·2⁴⁸ + run·2³² + t·2¹⁶ + half)) ^ pixel)`
//!   (초기화는 t = 0xFFFF). 균등 (0,1] 값은 `((x >> 40) + 1)·2⁻²⁴`. 스레드 배치와 무관하게 결정적이다.
//! - **거친 스케일 후보**: `use_prior` 이고 기준 패치 분산이 문턱보다 작으면 상태의 `prior` 평면을 정제 뒤 후보 하나로 더 평가한다.
//! - **실행 시작**: 무작위 초기화(요청 시) 후 현재 가설의 상위 K 평균 비용으로 저장 비용을 채우고, 최중요 뷰는 비운다.
//! - **수치**: 커널은 근사 나눗셈·지수(속도 우선)를 쓸 수 있다. 결과는 같은 장치·같은 입력에서 비트 단위로 재현된다.

use crate::params::{FilterParams, PmParams};
use cumulus3d_core::Result;
use std::sync::Arc;

/// 한 스케일의 뷰 영상.
#[derive(Clone, Debug)]
pub struct LevelImage {
    /// 너비(픽셀).
    pub width: usize,
    /// 높이(픽셀).
    pub height: usize,
    /// 회색 8비트(행 우선).
    pub gray: Arc<Vec<u8>>,
    /// fx, fy, cx, cy (정수 중심 규약).
    pub k: [f32; 4],
}

/// 기준 → 원천 상대 기하(원천 카메라 좌표 = r·X + t, 원천 중심의 기준 좌표 `center`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PairGeometry {
    /// 기준 → 원천 회전(행 우선 3×3).
    pub r: [f32; 9],
    /// 기준 → 원천 이동.
    pub t: [f32; 3],
    /// 원천 카메라 중심(기준 카메라 좌표).
    pub center: [f32; 3],
}

/// 커널 입력 뷰.
#[derive(Clone, Debug)]
pub struct KernelView {
    /// 난수 열쇠(영상 이름 해시).
    pub key: u64,
    /// 스케일별 영상(0 = 최저 해상도).
    pub levels: Vec<LevelImage>,
    /// 원천 뷰 색인(≤ 32).
    pub sources: Vec<usize>,
    /// 원천별 상대 기하.
    pub pairs: Vec<PairGeometry>,
    /// 무작위 초기화 깊이 하한.
    pub depth_min: f32,
    /// 무작위 초기화 깊이 상한.
    pub depth_max: f32,
}

/// 커널 입력 전체(한 조밀화 호출 동안 고정).
#[derive(Clone, Debug)]
pub struct KernelInput {
    /// 커널 입력 뷰 전체.
    pub views: Vec<KernelView>,
    /// 스케일 수.
    pub num_levels: usize,
    /// PatchMatch 상수.
    pub pm: PmParams,
    /// 판독·필터 허용치.
    pub filter: FilterParams,
    /// 난수 시드.
    pub seed: u64,
}

/// 기준 뷰 하나의 픽셀 상태(한 스케일).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewState {
    /// 너비(픽셀).
    pub width: usize,
    /// 높이(픽셀).
    pub height: usize,
    /// 픽셀별 깊이(카메라 z).
    pub depth: Vec<f32>,
    /// 픽셀별 법선(기준 카메라 좌표).
    pub normal: Vec<[f32; 3]>,
    /// 픽셀별 저장 비용.
    pub cost: Vec<f32>,
    /// 거친 스케일 가설(nx, ny, nz, d). 비어 있지 않으면 무늬 약한 픽셀에서 후보로 쓴다.
    pub prior: Vec<[f32; 4]>,
}

impl ViewState {
    /// 깊이·법선 0, 비용 2(최대)로 채운 상태.
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self { width, height, depth: vec![0.0; n], normal: vec![[0.0; 3]; n], cost: vec![2.0; n], prior: Vec::new() }
    }
}

/// 기하 실행이 읽는 깊이 스냅숏(뷰 색인 → 그 스케일 깊이맵). `id` 가 같으면 내용도 같다.
#[derive(Clone, Debug)]
pub struct DepthSnapshot {
    /// 스냅숏 식별자.
    pub id: u64,
    /// 깊이맵의 스케일.
    pub level: usize,
    /// 뷰 색인별 깊이맵(없으면 None).
    pub maps: Vec<Option<Arc<Vec<f32>>>>,
}

/// 실행 하나의 매개변수.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunParams {
    /// 스케일 번호(0 = 최저).
    pub level: usize,
    /// 기하 일관성 실행인지.
    pub geometric: bool,
    /// 시작 전에 무작위 초기화할지.
    pub random_init: bool,
    /// 적·흑 반복 수.
    pub iterations: u32,
    /// 난수 열쇠용 실행 번호.
    pub run_id: u32,
    /// 상태의 `prior` 가설을 무늬 약한 픽셀(기준 패치 분산 < `PmParams::weak_texture_var`)의 추가 후보로 쓴다.
    pub use_prior: bool,
}

/// 조밀화 호출 하나 동안의 백엔드 상태(장치 상주 자료 등).
pub trait PatchMatchSession {
    /// `views[i]` 의 상태 `states[i]` 에서 적·흑 반복 실행. 상태를 제자리 갱신한다.
    fn run(&mut self, views: &[usize], states: &mut [ViewState], params: &RunParams, snapshot: Option<&DepthSnapshot>) -> Result<()>;
    /// 현재 가설의 상위 K 평균 비용(기하면 `m + λ·min(Δe, δ)`).
    fn evaluate(
        &mut self,
        level: usize,
        views: &[usize],
        states: &[ViewState],
        geometric: bool,
        snapshot: Option<&DepthSnapshot>,
    ) -> Result<Vec<Vec<f32>>>;
    /// 최종 스케일 판독: 픽셀마다 광도·기하 조건을 모두 만족한 원천 뷰 수.
    /// 조건: 삼각측량각 ≥ 필터 최소각, 입사 cos > 0, 방출 가시 확률 ≥ E(1 − min_ncc), 비절단 Δe ≤ 허용치.
    fn filter(&mut self, views: &[usize], states: &[ViewState], snapshot: &DepthSnapshot) -> Result<Vec<Vec<u8>>>;
    /// 스케일 `level − 1` 상태를 `level` 로 결합 양방향 상향 표본([`crate::upsample::joint_bilateral_upsample`] 과 같은 규칙).
    /// 기본 구현은 호스트 계산.
    fn upsample(
        &mut self,
        input: &KernelInput,
        level: usize,
        views: &[usize],
        low: &[ViewState],
        sigma_s: f32,
        sigma_c: f32,
    ) -> Result<Vec<ViewState>> {
        Ok(views
            .iter()
            .zip(low)
            .map(|(&v, s)| {
                crate::upsample::joint_bilateral_upsample(
                    s,
                    &input.views[v].levels[level - 1],
                    &input.views[v].levels[level],
                    sigma_s,
                    sigma_c,
                )
            })
            .collect())
    }
    /// 5×5 중앙값 평면 필터([`crate::upsample::median_plane_filter`] 와 같은 규칙). 기본 구현은 호스트 계산.
    fn median_filter(&mut self, input: &KernelInput, level: usize, views: &[usize], states: &[ViewState]) -> Result<Vec<ViewState>> {
        Ok(views.iter().zip(states).map(|(&v, s)| crate::upsample::median_plane_filter(s, &input.views[v].levels[level])).collect())
    }
}

/// PatchMatch 백엔드.
pub trait PatchMatchBackend: Send + Sync {
    /// 백엔드 이름(로그용).
    fn name(&self) -> String;
    /// 입력을 받아 세션을 연다(영상 업로드 등).
    fn begin<'a>(&'a self, input: &'a KernelInput) -> Result<Box<dyn PatchMatchSession + 'a>>;
}

/// 픽셀 난수 시작 상태(커널과 같은 정의).
pub fn rng_state(seed: u64, view_key: u64, level: u32, run: u32, t: u32, half: u32, pixel: u64) -> u64 {
    use crate::math::mix64;
    let tag = ((level as u64) << 48) ^ ((run as u64) << 32) ^ ((t as u64) << 16) ^ half as u64;
    mix64(seed ^ mix64(view_key ^ mix64(tag)) ^ pixel)
}

/// splitmix64 한 걸음과 (0,1] 균등 값.
pub fn rng_next(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let z = crate::math::mix64(*state);
    ((z >> 40) as f32 + 1.0) * (1.0 / 16_777_216.0)
}
