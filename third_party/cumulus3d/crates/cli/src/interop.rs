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

//! 외부 SfM 도구(COLMAP) 단계별 명령 호환 하위 명령(얇은 래퍼). 하위 명령 이름과 옵션 문자열은 그 도구의 명령줄을 따른다.
//! 단계 사이 상태는 `--database_path` (FeatureStore 이진 파일)와 모델 폴더([`cumulus3d_core::interop`] 형식)로 잇는다.
//! GPU 관련 옵션(`--FeatureExtraction.use_gpu` 등)은 받기만 하고 무시한다.

use crate::densewrap::{dense_model, make_pm_backend, parse_profile, DenseConfig};
use clap::{Args, Subcommand};
use cumulus3d_align::{align_to_gps, ModelAlignerOptions};
use cumulus3d_ba::{bundle_adjust, BaConfig};
use cumulus3d_core::analyzer::analyzer_lines;
use cumulus3d_core::interop::{read_model, write_model_binary, write_model_text, ImageOrder};
use cumulus3d_core::io::{read_gps_file, write_ply, PlyLayout, PointCloud};
use cumulus3d_core::reconstruction::bilinear_rgb;
use cumulus3d_core::{CameraModelKind, FeatureStore, ImageId, MatchGraph, MatchGraphOptions, Reconstruction};
use cumulus3d_dense::densify::densify_with;
use cumulus3d_dense::fusion_score::ScoreFusionOptions;
use cumulus3d_dense::{write_undistorted_workspace, DenseScene, UndistortCache, UndistortOptions};
use cumulus3d_features::{CameraMode, ExtractionOptions, FeatureExtractor, ImageStatus};
use cumulus3d_matching::{match_pair_list_file, CpuMatcher, PairMatchingOptions};
use cumulus3d_sfm::{global_mapper, register_images, triangulate_points, GlobalSfmOptions, PointTriangulatorOptions, RegistrationOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;

type R = Result<(), String>;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

/// `1/0/true/false` 를 받는다(명령줄 관례).
pub fn parse_flag(s: &str) -> Result<bool, String> {
    match s.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("불린 값이 아님: {s}")),
    }
}

/// 단계별 하위 명령(명령 이름은 `feature_extractor` 등 밑줄 형식).
#[derive(Subcommand, Debug)]
pub enum InteropCmd {
    /// 영상 폴더 → 특징 저장소(SIFT 추출, 카메라 초기화).
    #[command(name = "feature_extractor")]
    FeatureExtractor(FeatureExtractorArgs),
    /// 짝 목록 파일 → 기술자 매칭 + 두 뷰 기하 검증.
    #[command(name = "matches_importer")]
    MatchesImporter(MatchesImporterArgs),
    /// 특징 저장소 → 전역 SfM 모델(`output_path/0`).
    #[command(name = "global_mapper")]
    GlobalMapper(GlobalMapperArgs),
    /// 기존 모델에 아직 등록되지 않은 영상을 등록.
    #[command(name = "image_registrator")]
    ImageRegistrator(ImageRegistratorArgs),
    /// 등록된 자세로 3D 점 삼각측량.
    #[command(name = "point_triangulator")]
    PointTriangulator(PointTriangulatorArgs),
    /// 모델 번들 조정.
    #[command(name = "bundle_adjuster")]
    BundleAdjuster(BundleAdjusterArgs),
    /// 모델을 GPS ENU 좌표계로 정렬.
    #[command(name = "model_aligner")]
    ModelAligner(ModelAlignerArgs),
    /// 모델 통계 출력.
    #[command(name = "model_analyzer")]
    ModelAnalyzer(ModelAnalyzerArgs),
    /// 모델 형식 변환(BIN/TXT/PLY).
    #[command(name = "model_converter")]
    ModelConverter(ModelConverterArgs),
    /// 모델에서 영상 삭제.
    #[command(name = "image_deleter")]
    ImageDeleter(ImageDeleterArgs),
    /// 왜곡 보정 작업 폴더 생성.
    #[command(name = "image_undistorter")]
    ImageUndistorter(ImageUndistorterArgs),
    /// 조밀화: image_undistorter 출력 폴더 → dense.ply.
    #[command(name = "densify")]
    Densify(Box<DensifyArgs>),
}

/// `feature_extractor` 인자.
#[derive(Args, Debug)]
pub struct FeatureExtractorArgs {
    /// 특징 저장소 파일 경로(없으면 새로 만든다).
    #[arg(long = "database_path")]
    pub database_path: PathBuf,
    /// 입력 영상 폴더.
    #[arg(long = "image_path")]
    pub image_path: PathBuf,
    /// 처리할 영상 이름 목록 파일(없으면 폴더 전체).
    #[arg(long = "image_list_path")]
    pub image_list_path: Option<PathBuf>,
    /// 카메라 모델 이름.
    #[arg(long = "ImageReader.camera_model", default_value = "SIMPLE_RADIAL")]
    pub camera_model: String,
    /// 모든 영상이 카메라 하나를 공유.
    #[arg(long = "ImageReader.single_camera", default_value = "0", value_parser = parse_flag, action = clap::ArgAction::Set)]
    pub single_camera: bool,
    /// 폴더마다 카메라 하나.
    #[arg(long = "ImageReader.single_camera_per_folder", default_value = "0", value_parser = parse_flag, action = clap::ArgAction::Set)]
    pub single_camera_per_folder: bool,
    /// 기존 카메라 id 사용(≥ 0 일 때).
    #[arg(long = "ImageReader.existing_camera_id")]
    pub existing_camera_id: Option<i64>,
    /// 카메라 파라미터 직접 지정(쉼표 구분).
    #[arg(long = "ImageReader.camera_params")]
    pub camera_params: Option<String>,
    /// EXIF 초점이 없을 때 초점 = 계수 × max(w, h).
    #[arg(long = "ImageReader.default_focal_length_factor", default_value_t = 1.2)]
    pub default_focal_length_factor: f64,
    /// 영상당 최대 특징 수.
    #[arg(long = "SiftExtraction.max_num_features", default_value_t = 8192)]
    pub max_num_features: usize,
    /// SIFT 입력 최대 크기(픽셀).
    #[arg(long = "SiftExtraction.max_image_size", default_value_t = 3200)]
    pub max_image_size: usize,
    /// GPU 사용 여부(받기만 하고 무시).
    #[arg(long = "FeatureExtraction.use_gpu", value_parser = parse_flag)]
    pub use_gpu: Option<bool>,
    /// GPU 사용 여부(받기만 하고 무시).
    #[arg(long = "SiftExtraction.use_gpu", value_parser = parse_flag)]
    pub sift_use_gpu: Option<bool>,
}

/// `matches_importer` 인자.
#[derive(Args, Debug)]
pub struct MatchesImporterArgs {
    /// 특징 저장소 파일 경로.
    #[arg(long = "database_path")]
    pub database_path: PathBuf,
    /// 짝 목록 파일(줄마다 `이름1 이름2`).
    #[arg(long = "match_list_path")]
    pub match_list_path: PathBuf,
    /// 매칭 종류(`pairs` 만 지원).
    #[arg(long = "match_type", default_value = "pairs")]
    pub match_type: String,
    /// GPU 사용 여부(받기만 하고 무시).
    #[arg(long = "FeatureMatching.use_gpu", value_parser = parse_flag)]
    pub use_gpu: Option<bool>,
    /// GPU 사용 여부(받기만 하고 무시).
    #[arg(long = "SiftMatching.use_gpu", value_parser = parse_flag)]
    pub sift_use_gpu: Option<bool>,
    /// 짝당 최대 매칭 수.
    #[arg(long = "SiftMatching.max_num_matches", default_value_t = 32768)]
    pub max_num_matches: usize,
}

/// `global_mapper` 인자.
#[derive(Args, Debug)]
pub struct GlobalMapperArgs {
    /// 특징 저장소 파일 경로.
    #[arg(long = "database_path")]
    pub database_path: PathBuf,
    /// 영상 폴더(주면 점 색 추출).
    #[arg(long = "image_path")]
    pub image_path: Option<PathBuf>,
    /// 출력 폴더(모델은 `0/` 아래).
    #[arg(long = "output_path")]
    pub output_path: PathBuf,
    /// 전역 BA 반복 횟수.
    #[arg(long = "GlobalMapper.ba_num_iterations")]
    pub ba_num_iterations: Option<usize>,
    /// 재삼각측량 생략.
    #[arg(long = "GlobalMapper.skip_retriangulation", value_parser = parse_flag)]
    pub skip_retriangulation: Option<bool>,
    /// 유지할 최대 트랙 수.
    #[arg(long = "GlobalMapper.keep_max_num_tracks")]
    pub keep_max_num_tracks: Option<usize>,
}

/// `image_registrator` 인자.
#[derive(Args, Debug)]
pub struct ImageRegistratorArgs {
    /// 특징 저장소 파일 경로.
    #[arg(long = "database_path")]
    pub database_path: PathBuf,
    /// 입력 모델 폴더.
    #[arg(long = "input_path")]
    pub input_path: PathBuf,
    /// 출력 모델 폴더.
    #[arg(long = "output_path")]
    pub output_path: PathBuf,
}

/// `point_triangulator` 인자.
#[derive(Args, Debug)]
pub struct PointTriangulatorArgs {
    /// 특징 저장소 파일 경로.
    #[arg(long = "database_path")]
    pub database_path: PathBuf,
    /// 영상 폴더(주면 점 색 추출).
    #[arg(long = "image_path")]
    pub image_path: Option<PathBuf>,
    /// 입력 모델 폴더.
    #[arg(long = "input_path")]
    pub input_path: PathBuf,
    /// 출력 모델 폴더.
    #[arg(long = "output_path")]
    pub output_path: PathBuf,
    /// 기존 3D 점을 지우고 다시 삼각측량.
    #[arg(long = "clear_points", default_value = "1", value_parser = parse_flag, action = clap::ArgAction::Set)]
    pub clear_points: bool,
}

/// `bundle_adjuster` 인자.
#[derive(Args, Debug)]
pub struct BundleAdjusterArgs {
    /// 입력 모델 폴더.
    #[arg(long = "input_path")]
    pub input_path: PathBuf,
    /// 출력 모델 폴더.
    #[arg(long = "output_path")]
    pub output_path: PathBuf,
    /// 최대 반복 수.
    #[arg(long = "BundleAdjustment.max_num_iterations", default_value_t = 100)]
    pub max_num_iterations: usize,
    /// 초점 거리 정제.
    #[arg(long = "BundleAdjustment.refine_focal_length", default_value = "1", value_parser = parse_flag, action = clap::ArgAction::Set)]
    pub refine_focal_length: bool,
    /// 주점 정제.
    #[arg(long = "BundleAdjustment.refine_principal_point", default_value = "0", value_parser = parse_flag, action = clap::ArgAction::Set)]
    pub refine_principal_point: bool,
    /// 왜곡 등 추가 파라미터 정제.
    #[arg(long = "BundleAdjustment.refine_extra_params", default_value = "1", value_parser = parse_flag, action = clap::ArgAction::Set)]
    pub refine_extra_params: bool,
}

/// `model_aligner` 인자.
#[derive(Args, Debug)]
pub struct ModelAlignerArgs {
    /// 입력 모델 폴더.
    #[arg(long = "input_path")]
    pub input_path: PathBuf,
    /// 출력 모델 폴더.
    #[arg(long = "output_path")]
    pub output_path: PathBuf,
    /// 기준 GPS 파일(줄: `이름 위도 경도 고도`).
    #[arg(long = "ref_images_path")]
    pub ref_images_path: PathBuf,
    /// 기준이 GPS 인지(1 만 지원).
    #[arg(long = "ref_is_gps", default_value = "1", value_parser = parse_flag, action = clap::ArgAction::Set)]
    pub ref_is_gps: bool,
    /// 정렬 좌표계(`enu` 만 지원).
    #[arg(long = "alignment_type", default_value = "enu")]
    pub alignment_type: String,
    /// 견고 추정 최대 오차(미터, 0 = 견고 추정 없음).
    #[arg(long = "alignment_max_error", default_value_t = 0.0)]
    pub alignment_max_error: f64,
    /// 정렬에 필요한 최소 공통 영상 수.
    #[arg(long = "min_common_images", default_value_t = 3)]
    pub min_common_images: usize,
}

/// `model_analyzer` 인자.
#[derive(Args, Debug)]
pub struct ModelAnalyzerArgs {
    /// 모델 폴더.
    #[arg(long = "path")]
    pub path: PathBuf,
    /// 자세한 통계 출력.
    #[arg(long = "verbose", default_value = "0", value_parser = parse_flag, action = clap::ArgAction::Set)]
    pub verbose: bool,
}

/// `model_converter` 인자.
#[derive(Args, Debug)]
pub struct ModelConverterArgs {
    /// 입력 모델 폴더.
    #[arg(long = "input_path")]
    pub input_path: PathBuf,
    /// 출력 폴더 또는 파일.
    #[arg(long = "output_path")]
    pub output_path: PathBuf,
    /// 출력 형식(BIN|TXT|PLY).
    #[arg(long = "output_type")]
    pub output_type: String,
}

/// `image_deleter` 인자.
#[derive(Args, Debug)]
pub struct ImageDeleterArgs {
    /// 입력 모델 폴더.
    #[arg(long = "input_path")]
    pub input_path: PathBuf,
    /// 출력 모델 폴더.
    #[arg(long = "output_path")]
    pub output_path: PathBuf,
    /// 지울 영상 id 목록 파일.
    #[arg(long = "image_ids_path")]
    pub image_ids_path: Option<PathBuf>,
    /// 지울 영상 이름 목록 파일.
    #[arg(long = "image_names_path")]
    pub image_names_path: Option<PathBuf>,
}

/// `image_undistorter` 인자.
#[derive(Args, Debug)]
pub struct ImageUndistorterArgs {
    /// 입력 영상 폴더.
    #[arg(long = "image_path")]
    pub image_path: PathBuf,
    /// 입력 모델 폴더.
    #[arg(long = "input_path")]
    pub input_path: PathBuf,
    /// 출력 작업 폴더.
    #[arg(long = "output_path")]
    pub output_path: PathBuf,
    /// 출력 형식(이 값만 지원).
    #[arg(long = "output_type", default_value = "COLMAP")]
    pub output_type: String,
    /// 왜곡 보정 영상 최대 크기(-1 = 제한 없음).
    #[arg(long = "max_image_size", default_value_t = -1)]
    pub max_image_size: i64,
}

/// `densify` 인자.
#[derive(Args, Debug)]
pub struct DensifyArgs {
    /// image_undistorter 출력 폴더(images/, sparse/). `--image_path` 를 주면 대신 왜곡 있는 모델(`--input_path`)을 메모리에서 보정.
    #[arg(long = "input_path", short = 'i')]
    pub input_path: PathBuf,
    /// 출력 PLY.
    #[arg(long = "output_path", short = 'o')]
    pub output_path: PathBuf,
    /// 주면 `input_path` 를 왜곡 있는 입력 모델로 보고 이 폴더의 영상을 메모리에서 보정한다.
    #[arg(long = "image_path")]
    pub image_path: Option<PathBuf>,
    /// 왜곡 보정 최대 크기(메모리 보정 경로).
    #[arg(long = "max_image_size", default_value_t = 960)]
    pub max_image_size: i64,
    /// 기준 뷰당 원천 뷰 수(≤ 32).
    #[arg(long = "number-views", default_value_t = 10)]
    pub number_views: usize,
    /// PatchMatch 백엔드(cuda).
    #[arg(long = "pm-backend", default_value = "cuda")]
    pub pm_backend: String,
    /// 프로파일(fast|quality).
    #[arg(long = "mvs-profile", default_value = "fast")]
    pub mvs_profile: String,
    /// 비용 창 반경(기본: 프로파일 값 5).
    #[arg(long = "window-radius")]
    pub window_radius: Option<i32>,
    /// 비용 창 표본 간격 1|2(기본: fast 2, quality 1).
    #[arg(long = "window-step")]
    pub window_step: Option<i32>,
    /// 스케일 수 상한.
    #[arg(long = "max-levels")]
    pub max_levels: Option<usize>,
    /// 등록 순서로 앞 N 장만(`--image_path` 사용 시).
    #[arg(long = "max-images")]
    pub max_images: Option<usize>,
    /// 융합 방식(consistency|traversal|score).
    #[arg(long = "fusion-mode")]
    pub fusion_mode: Option<String>,
    /// 일치 융합: 기준 외 일치 뷰 최소 수.
    #[arg(long = "fusion-min-views")]
    pub fusion_min_views: Option<usize>,
    /// 융합 상대 깊이 허용치.
    #[arg(long = "fusion-depth-error")]
    pub fusion_depth_error: Option<f64>,
    /// 융합 법선 허용 각(도).
    #[arg(long = "fusion-normal-error")]
    pub fusion_normal_error: Option<f64>,
    /// 융합 재투영 허용치(픽셀).
    #[arg(long = "fusion-reproj-error")]
    pub fusion_reproj_error: Option<f64>,
    /// 일치 융합 남은 픽셀 처리(none|release|second-pass).
    #[arg(long = "fusion-residual")]
    pub fusion_residual: Option<String>,
    /// 2차 융합: 기준 외 일치 뷰 최소 수(기본 2).
    #[arg(long = "residual-min-views")]
    pub residual_min_views: Option<usize>,
    /// 2차 융합: 상대 깊이 허용치(기본 1차의 0.5배).
    #[arg(long = "residual-depth-error")]
    pub residual_depth_error: Option<f64>,
    /// 2차 융합: 법선 허용 각(도, 기본 1차의 0.67배).
    #[arg(long = "residual-normal-error")]
    pub residual_normal_error: Option<f64>,
    /// 2차 융합: 1차 점과 이 거리(GSD 배수, 기본 0.5) 이내인 2차 점은 버림.
    #[arg(long = "residual-min-dist")]
    pub residual_min_dist: Option<f64>,
    /// 일치 융합: 쓰인 픽셀 표시(1|0).
    #[arg(long = "fusion-mark-used", value_parser = parse_flag)]
    pub fusion_mark_used: Option<bool>,
    /// 일치 융합: 표시 반경(픽셀).
    #[arg(long = "fusion-mark-radius")]
    pub fusion_mark_radius: Option<usize>,
    /// 일치 융합: 검사할 겹침 뷰 수.
    #[arg(long = "fusion-num-images")]
    pub fusion_num_images: Option<usize>,
    /// 모든 스케일의 기하 실행 반복 수 덮어쓰기.
    #[arg(long = "geo-iters")]
    pub geo_iters: Option<u32>,
    /// 기하 실행 회차 수 덮어쓰기.
    #[arg(long = "geo-rounds")]
    pub geo_rounds: Option<u32>,
    /// 일치 융합: 역분산 가중(1|0).
    #[arg(long = "fusion-inverse-variance", value_parser = parse_flag)]
    pub fusion_inverse_variance: Option<bool>,
    /// 깊이맵 필터: 일치 뷰 최소 수.
    #[arg(long = "filter-min-views")]
    pub filter_min_views: Option<u32>,
    /// 깊이맵 필터: 순·역 재투영 허용치(픽셀).
    #[arg(long = "filter-geom-error")]
    pub filter_geom_error: Option<f32>,
    /// 깊이맵 필터: 최소 NCC.
    #[arg(long = "filter-min-ncc")]
    pub filter_min_ncc: Option<f32>,
    /// 작은 조각 제거(1|0).
    #[arg(long = "remove-speckles", value_parser = parse_flag)]
    pub remove_speckles: Option<bool>,
    /// 경계 인식 틈 메우기(1|0).
    #[arg(long = "fill-holes", value_parser = parse_flag)]
    pub fill_holes: Option<bool>,
    /// 이웃 선택 방향 다양성 감쇠(1 = 끔).
    #[arg(long = "diversity-decay")]
    pub diversity_decay: Option<f64>,
    /// 거친 스케일 후보를 더할 기준 패치 분산 문턱(0 = 끔).
    #[arg(long = "weak-texture-var")]
    pub weak_texture_var: Option<f32>,
    /// 판독 전 5×5 중앙값 평면 필터(1|0).
    #[arg(long = "median-filter", value_parser = parse_flag)]
    pub median_filter: Option<bool>,
    /// 결과 점군 통계(간격·이상점·중복) 출력.
    #[arg(long = "stats")]
    pub stats: bool,
    /// 점수 융합: 채택 문턱 τ(기본 2.0).
    #[arg(long = "score-tau")]
    pub score_tau: Option<f64>,
    /// 점수 융합: 재투영 오차 가중 σ_e(픽셀, 기본 1.0).
    #[arg(long = "score-sigma-e")]
    pub score_sigma_e: Option<f64>,
    /// 점수 융합: 시선 사잇각 가중 σ_θ(도, 기본 2.0).
    #[arg(long = "score-sigma-theta")]
    pub score_sigma_theta: Option<f64>,
    /// 점수 융합: 자유공간 위반 벌점 계수 λ(기본 1.0).
    #[arg(long = "score-lambda")]
    pub score_lambda: Option<f64>,
    /// 2차 융합 점만 따로 쓸 PLY(`--fusion-residual second-pass` 일 때).
    #[arg(long = "residual-out")]
    pub residual_out: Option<PathBuf>,
    /// 융합 변형 파일(줄마다 `이름|융합 플래그들`). 깊이맵은 한 번만 만들고 변형마다 `<output_path 폴더>/<이름>.ply` 를 쓴다.
    #[arg(long = "fusion-variants")]
    pub fusion_variants: Option<PathBuf>,
}

/// 조밀화 하위 명령 옵션을 설정에 반영.
pub(crate) fn apply_densify_args(a: &DensifyArgs, o: &mut cumulus3d_dense::DensifyOptions) -> R {
    if let Some(r) = a.window_radius {
        o.pm.window_radius = r;
    }
    if let Some(st) = a.window_step {
        o.pm.window_step = st;
    }
    if let Some(l) = a.max_levels {
        o.max_levels = l;
    }
    if let Some(m) = a.fusion_mode.as_deref().filter(|m| !is_score_mode(m)) {
        o.fusion.mode = cumulus3d_dense::FusionMode::parse(m).ok_or_else(|| format!("알 수 없는 --fusion-mode {m}"))?;
    }
    let f = &mut o.fusion;
    if let Some(v) = a.fusion_min_views {
        f.min_consistent_views = v;
    }
    if let Some(v) = a.fusion_depth_error {
        f.max_depth_error = v;
    }
    if let Some(v) = a.fusion_normal_error {
        f.max_normal_error_deg = v;
    }
    if let Some(v) = a.fusion_reproj_error {
        f.max_reproj_error = v;
    }
    if let Some(m) = &a.fusion_residual {
        f.residual = cumulus3d_dense::FusionResidual::parse(m).ok_or_else(|| format!("알 수 없는 --fusion-residual {m}"))?;
    }
    if let Some(v) = a.residual_min_views {
        f.residual_params.min_views = v;
    }
    if let Some(v) = a.residual_depth_error {
        f.residual_params.depth_error = Some(v);
    }
    if let Some(v) = a.residual_normal_error {
        f.residual_params.normal_error_deg = Some(v);
    }
    if let Some(v) = a.residual_min_dist {
        f.residual_params.min_dist_gsd = v;
    }
    if let Some(v) = a.fusion_mark_used {
        f.mark_used = v;
    }
    if let Some(v) = a.fusion_mark_radius {
        f.mark_radius = v;
    }
    if let Some(v) = a.fusion_num_images {
        f.consistency_num_images = v;
    }
    if let Some(v) = a.geo_iters {
        o.geometric_iters_override = Some(v);
    }
    if let Some(v) = a.geo_rounds {
        o.geometric_rounds_override = Some(v);
    }
    if let Some(v) = a.fusion_inverse_variance {
        f.inverse_variance = v;
    }
    if let Some(v) = a.filter_min_views {
        o.filter.min_num_consistent = v;
    }
    if let Some(v) = a.filter_geom_error {
        o.filter.geom_max_cost = v;
    }
    if let Some(v) = a.filter_min_ncc {
        o.filter.min_ncc = v;
    }
    if let Some(v) = a.remove_speckles {
        o.post.remove_speckles = v;
    }
    if let Some(v) = a.fill_holes {
        o.post.fill_holes = v;
    }
    if let Some(v) = a.diversity_decay {
        o.neighbors.diversity_decay = v;
    }
    if let Some(v) = a.weak_texture_var {
        o.pm.weak_texture_var = v;
    }
    if let Some(v) = a.median_filter {
        o.filter.median_filter = v;
    }
    Ok(())
}

fn is_score_mode(m: &str) -> bool {
    m.eq_ignore_ascii_case("score")
}

fn apply_score_args(a: &DensifyArgs, s: &mut ScoreFusionOptions) {
    if let Some(v) = a.score_tau {
        s.tau = v;
    }
    if let Some(v) = a.score_sigma_e {
        s.sigma_e_px = v;
    }
    if let Some(v) = a.score_sigma_theta {
        s.sigma_theta_deg = v;
    }
    if let Some(v) = a.score_lambda {
        s.lambda = v;
    }
}

fn has_score_args(a: &DensifyArgs) -> bool {
    a.score_tau.is_some() || a.score_sigma_e.is_some() || a.score_sigma_theta.is_some() || a.score_lambda.is_some()
}

/// 점수 융합 설정. `--fusion-mode score` 면 `o.fusion`(다른 융합 플래그 반영 후)의 허용치를 이어받고 `--score-*` 를 덮어쓴다.
/// `base` 는 융합 변형에서 densify 명령 자체의 인자: 변형 줄에 `--fusion-mode` 가 없으면 그 방식을 따르고, 그 `--score-*` 를 먼저 반영한다.
pub(crate) fn score_options(
    a: &DensifyArgs,
    o: &cumulus3d_dense::DensifyOptions,
    base: Option<&DensifyArgs>,
) -> Result<Option<ScoreFusionOptions>, String> {
    let mode = a.fusion_mode.as_deref().or_else(|| base.and_then(|b| b.fusion_mode.as_deref()));
    if !mode.is_some_and(is_score_mode) {
        if has_score_args(a) || base.is_some_and(|b| a.fusion_mode.is_none() && has_score_args(b)) {
            return Err("--score-* 는 --fusion-mode score 와 함께".into());
        }
        return Ok(None);
    }
    let mut s = ScoreFusionOptions::from_fusion(&o.fusion);
    if let Some(b) = base {
        apply_score_args(b, &mut s);
    }
    apply_score_args(a, &mut s);
    Ok(Some(s))
}

fn load_store(p: &Path) -> Result<FeatureStore, String> {
    if p.exists() {
        FeatureStore::load(p).map_err(|x| format!("{}: {x}", p.display()))
    } else {
        Ok(FeatureStore::new())
    }
}

fn load_model(p: &Path) -> Result<Reconstruction, String> {
    read_model(p).map_err(|x| format!("{}: {x}", p.display()))
}

fn save_model(rec: &Reconstruction, p: &Path) -> R {
    std::fs::create_dir_all(p).map_err(e)?;
    write_model_binary(rec, p, ImageOrder::Registration).map_err(e)
}

fn graph_of(store: &FeatureStore) -> MatchGraph {
    MatchGraph::from_store(store, &MatchGraphOptions::default())
}

/// 영상 폴더에서 3D 점 색 추출(모델을 쓰는 명령들의 공통 마무리 단계).
pub fn extract_colors(rec: &mut Reconstruction, image_path: &Path) {
    rec.extract_colors(|im| {
        let img = image::open(image_path.join(&im.name)).ok()?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let data = img.into_raw();
        Some(move |x: f64, y: f64| bilinear_rgb(&data, w, h, x, y))
    });
}

fn list_images(root: &Path) -> Vec<String> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else if let Some(ext) = p.extension().and_then(|x| x.to_str()) {
                if matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png") {
                    if let Ok(rel) = p.strip_prefix(root) {
                        out.push(rel.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/"));
                    }
                }
            }
        }
    }
    let mut v = Vec::new();
    walk(root, root, &mut v);
    v.sort();
    v
}

/// 하위 명령 하나를 실행한다.
pub fn run(cmd: InteropCmd) -> R {
    let t = Instant::now();
    match cmd {
        InteropCmd::FeatureExtractor(a) => {
            let store = load_store(&a.database_path)?;
            let names = match &a.image_list_path {
                Some(p) => cumulus3d_features::read_image_list(p).map_err(e)?,
                None => list_images(&a.image_path),
            };
            let mut o = ExtractionOptions::default();
            o.reader.camera_model = CameraModelKind::from_name(&a.camera_model).map_err(e)?;
            o.reader.camera_mode = if let Some(id) = a.existing_camera_id.filter(|x| *x >= 0) {
                CameraMode::Existing(id as u32)
            } else if a.single_camera {
                CameraMode::Single
            } else if a.single_camera_per_folder {
                CameraMode::PerFolder
            } else {
                CameraMode::PerImage
            };
            if let Some(p) = &a.camera_params {
                o.reader.camera_params = Some(p.split(',').map(|x| x.trim().parse::<f64>().map_err(e)).collect::<Result<_, _>>()?);
            }
            o.reader.default_focal_length_factor = a.default_focal_length_factor;
            o.reader.max_image_size = a.max_image_size;
            o.sift.max_num_features = a.max_num_features;
            let rep = FeatureExtractor::new().extract_files(&store, &a.image_path, &names, &o).map_err(e)?;
            for r in &rep {
                match &r.status {
                    ImageStatus::Extracted { image_id, camera_id, num_features } => {
                        println!("{} image_id={image_id} camera_id={camera_id} features={num_features}", r.name)
                    }
                    ImageStatus::AlreadyExists { image_id } => println!("{} image_id={image_id} (이미 있음)", r.name),
                    ImageStatus::Failed { error } => println!("{} 실패: {error}", r.name),
                }
            }
            store.save(&a.database_path).map_err(e)?;
        }
        InteropCmd::MatchesImporter(a) => {
            if a.match_type != "pairs" {
                return Err(format!("--match_type {} 미지원(pairs 만)", a.match_type));
            }
            let store = load_store(&a.database_path)?;
            let opts = PairMatchingOptions { max_num_matches: a.max_num_matches, ..Default::default() };
            let (list, st) = match_pair_list_file(&store, &a.match_list_path, &opts, &CpuMatcher::default()).map_err(e)?;
            for n in &list.missing_names {
                println!("경고: 이름 없음 {n}");
            }
            println!(
                "짝 {} 건너뜀 {} 매칭 {} 검증만 {} 유효 기하 {}",
                list.pairs.len(),
                st.num_skipped_existing,
                st.num_matched,
                st.num_verified_only,
                st.num_valid_geometries
            );
            store.save(&a.database_path).map_err(e)?;
        }
        InteropCmd::GlobalMapper(a) => {
            let store = load_store(&a.database_path)?;
            let graph = graph_of(&store);
            let mut o = GlobalSfmOptions::default();
            if let Some(n) = a.ba_num_iterations {
                o.ba_num_iterations = n;
            }
            if let Some(s) = a.skip_retriangulation {
                o.skip_retriangulation = s;
            }
            if let Some(n) = a.keep_max_num_tracks {
                o.tracks.max_num_tracks = n;
            }
            let out = global_mapper(&store, &graph, &o).map_err(e)?;
            if let Some(f) = &out.failure {
                println!("경고: {f}");
            }
            let mut rec = out.reconstruction;
            if let Some(ip) = &a.image_path {
                extract_colors(&mut rec, ip);
            }
            if rec.registered_image_count() > 0 {
                save_model(&rec, &a.output_path.join("0"))?;
            }
            println!("등록 {} 점 {}", rec.registered_image_count(), rec.num_points3d());
        }
        InteropCmd::ImageRegistrator(a) => {
            let store = load_store(&a.database_path)?;
            let graph = graph_of(&store);
            let mut rec = load_model(&a.input_path)?;
            let rep = register_images(&mut rec, &store, &graph, &RegistrationOptions::default()).map_err(e)?;
            println!("시도 {} 등록 {}", rep.attempts.len(), rep.registered().len());
            save_model(&rec, &a.output_path)?;
        }
        InteropCmd::PointTriangulator(a) => {
            let store = load_store(&a.database_path)?;
            let graph = graph_of(&store);
            let mut rec = load_model(&a.input_path)?;
            let o = PointTriangulatorOptions { clear_points: a.clear_points, ..Default::default() };
            let rep = triangulate_points(&mut rec, &graph, &o).map_err(e)?;
            if let Some(ip) = &a.image_path {
                extract_colors(&mut rec, ip);
            }
            println!("{rep:?}");
            save_model(&rec, &a.output_path)?;
        }
        InteropCmd::BundleAdjuster(a) => {
            let mut rec = load_model(&a.input_path)?;
            let cfg = BaConfig {
                max_num_iterations: a.max_num_iterations,
                refine_focal_length: a.refine_focal_length,
                refine_principal_point: a.refine_principal_point,
                refine_extra_params: a.refine_extra_params,
                ..Default::default()
            };
            let s = bundle_adjust(&mut rec, &cfg).map_err(e)?;
            println!("반복 {} 비용 {:.6e} → {:.6e} {:?}", s.num_iterations, s.initial_cost, s.final_cost, s.termination);
            save_model(&rec, &a.output_path)?;
        }
        InteropCmd::ModelAligner(a) => {
            if !a.ref_is_gps || a.alignment_type != "enu" {
                return Err("model_aligner: --ref_is_gps 1 --alignment_type enu 만 지원".into());
            }
            let mut rec = load_model(&a.input_path)?;
            let gps = read_gps_file(&a.ref_images_path).map_err(e)?;
            let o = ModelAlignerOptions { max_error: a.alignment_max_error, min_common_images: a.min_common_images, ..Default::default() };
            let al = align_to_gps(&mut rec, &gps, &o).map_err(e)?;
            for (n, er) in &al.errors {
                println!("{n}: {er:.4}");
            }
            println!("평균 오차 {:.4} 중앙 {:.4} 인라이어 {}/{}", al.mean_error, al.median_error, al.num_inliers, al.common.len());
            save_model(&rec, &a.output_path)?;
        }
        InteropCmd::ModelAnalyzer(a) => {
            let rec = load_model(&a.path)?;
            for l in analyzer_lines(&rec, a.verbose) {
                println!("{l}");
            }
            return Ok(());
        }
        InteropCmd::ModelConverter(a) => {
            let rec = load_model(&a.input_path)?;
            match a.output_type.to_ascii_uppercase().as_str() {
                "BIN" => save_model(&rec, &a.output_path)?,
                "TXT" => {
                    std::fs::create_dir_all(&a.output_path).map_err(e)?;
                    write_model_text(&rec, &a.output_path, ImageOrder::Registration).map_err(e)?
                }
                "PLY" => {
                    let mut c = PointCloud::default();
                    for (_, p) in rec.points3d() {
                        c.positions.push([p.xyz.x as f32, p.xyz.y as f32, p.xyz.z as f32]);
                        c.normals.push([0.0; 3]);
                        c.colors.push(p.color);
                    }
                    write_ply(&a.output_path, &c, PlyLayout::XyzNormalRgb).map_err(e)?
                }
                o => return Err(format!("--output_type {o} 미지원(BIN|TXT|PLY)")),
            }
        }
        InteropCmd::ImageDeleter(a) => {
            let mut rec = load_model(&a.input_path)?;
            let mut warn = Vec::new();
            if let Some(p) = &a.image_ids_path {
                let ids: Vec<ImageId> = std::fs::read_to_string(p).map_err(e)?.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                warn.extend(rec.deregister_images_by_id(&ids));
            }
            if let Some(p) = &a.image_names_path {
                let text = std::fs::read_to_string(p).map_err(e)?;
                let names: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
                warn.extend(rec.deregister_images_by_name(&names));
            }
            for w in warn {
                println!("{w}");
            }
            save_model(&rec, &a.output_path)?;
        }
        InteropCmd::ImageUndistorter(a) => {
            if a.output_type != "COLMAP" {
                return Err("image_undistorter: --output_type COLMAP 만 지원".into());
            }
            let rec = load_model(&a.input_path)?;
            let o = UndistortOptions { max_image_size: a.max_image_size, ..Default::default() };
            let r = cumulus3d_dense::undistort_from_dir(&rec, &a.image_path, &o, &UndistortCache::new()).map_err(e)?;
            write_undistorted_workspace(&r, &a.output_path, &o).map_err(e)?;
            println!("보정 {} 실패 {}", r.images.len(), r.failed.len());
        }
        InteropCmd::Densify(a) => {
            let profile = parse_profile(&a.mvs_profile)?;
            let mut cfg = DenseConfig::new(make_pm_backend(&a.pm_backend), profile);
            cfg.undistort.max_image_size = a.max_image_size;
            cfg.densify.neighbors.num_views = a.number_views;
            apply_densify_args(&a, &mut cfg.densify)?;
            cfg.score = score_options(&a, &cfg.densify, None)?;
            // 변형 파일은 깊이 추정 전에 검사한다.
            let variants =
                a.fusion_variants.as_deref().map(|vp| crate::fusion_variants::read_variants(vp, &a, &cfg.densify)).transpose()?;
            let backend = cfg.backend.clone()?;
            let tl = Instant::now();
            let (out, frames, scene) = match &a.image_path {
                Some(ip) => {
                    let model = load_model(&a.input_path)?;
                    let keep: std::collections::HashSet<String> = model
                        .registered_images()
                        .into_iter()
                        .take(a.max_images.unwrap_or(usize::MAX))
                        .filter_map(|i| model.image(i).map(|im| im.name.clone()))
                        .collect();
                    let r = dense_model(&model, |n| keep.contains(n), ip, &cfg)?;
                    eprintln!("[time] 왜곡 보정+장면 {:.2}s", r.undistort_time.as_secs_f64());
                    (r.output, r.frames, r.scene)
                }
                None => {
                    let scene = DenseScene::from_workspace_dir(&a.input_path, &cfg.scene).map_err(e)?;
                    eprintln!("[time] 장면 읽기 {:.2}s", tl.elapsed().as_secs_f64());
                    let n = scene.views.len();
                    (densify_with(&scene, &cfg.densify, cfg.score.as_ref(), backend.as_ref(), None).map_err(e)?, n, scene)
                }
            };
            let tm = &out.timings;
            let lv: Vec<String> = tm.levels.iter().map(|d| format!("{:.2}", d.as_secs_f64())).collect();
            eprintln!(
                "[time] 뷰 {frames}, 깊이맵 {}: 이웃 {:.2}s 준비 {:.2}s 깊이 {:.2}s (스케일별 [{}], 그중 상향표본·판정 {:.2}s) 필터 {:.2}s 융합 {:.2}s 합 {:.2}s",
                out.depth_views,
                tm.neighbors.as_secs_f64(),
                tm.prepare.as_secs_f64(),
                tm.depth().as_secs_f64(),
                lv.join(", "),
                tm.upsample.as_secs_f64(),
                tm.filter.as_secs_f64(),
                tm.fusion.as_secs_f64(),
                tm.total().as_secs_f64()
            );
            let tw = Instant::now();
            out.write_ply(&a.output_path).map_err(e)?;
            if let Some(rp) = &a.residual_out {
                if !out.write_residual_ply(rp).map_err(e)? {
                    eprintln!("경고: 2차 융합 점 없음 → {} 쓰지 않음", rp.display());
                }
            }
            eprintln!("[time] PLY 쓰기 {:.2}s", tw.elapsed().as_secs_f64());
            println!(
                "점 {} (2차 {}) 융합 {:.2}s (1차 {:.2}s 2차 {:.2}s)",
                out.cloud.len(),
                out.num_residual(),
                tm.fusion.as_secs_f64(),
                tm.fusion_pass1.as_secs_f64(),
                tm.fusion_pass2.as_secs_f64()
            );
            if a.stats {
                let ts = Instant::now();
                let st = cumulus3d_dense::cloud_stats(&scene, &out.depth_maps, &out.cloud, &out.visibility, 200_000, 2.0);
                println!(
                    "통계: 점 {} 표본 {} 이웃 간격 중앙 {:.4} GSD {:.4} 이상점(2px) {:.2}% 평면 이탈(>GSD·고립) {:.2}% 평면 거리 중앙 {:.4} 중복(<GSD/2) {:.2}% ({:.1}s)",
                    st.points,
                    st.sampled,
                    st.nn_spacing_median,
                    st.gsd,
                    100.0 * st.outlier_ratio,
                    100.0 * st.plane_outlier_ratio,
                    st.plane_residual_median,
                    100.0 * st.duplicate_ratio,
                    ts.elapsed().as_secs_f64()
                );
            }
            if let Some(variants) = &variants {
                let dir = a.output_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
                crate::fusion_variants::run_variants(&scene, &out.depth_maps, variants, dir, a.stats)?;
            }
        }
    }
    eprintln!("Elapsed time: {:.3} [minutes]", t.elapsed().as_secs_f64() / 60.0);
    Ok(())
}
