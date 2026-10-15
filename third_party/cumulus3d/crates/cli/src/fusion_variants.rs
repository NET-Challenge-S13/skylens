/*
 * fusion_variants.rs
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

//! `densify --fusion-variants <파일>`: 깊이맵은 한 번만 만들고 융합 설정만 바꿔 여러 번 융합한다.
//!
//! 파일 형식: 줄마다 `이름|융합 플래그들`(densify 와 같은 형식, 공백 구분). 빈 줄과 `#` 로 시작하는 줄은 건너뛴다.
//! 플래그는 densify 하위 명령의 인자 파서로 해석해 기본 설정 위에 덮어쓰므로, 새 융합 플래그도 그대로 쓸 수 있다.
//! 깊이맵을 바꾸는 플래그(필터·PatchMatch·영상 크기 등)는 거부한다.

use crate::interop::{apply_densify_args, score_options, DensifyArgs};
use clap::Parser;
use cumulus3d_dense::densify::fuse_output;
use cumulus3d_dense::fusion_score::ScoreFusionOptions;
use cumulus3d_dense::{DenseScene, DensifyOptions, DepthMapSet};
use std::path::Path;
use std::time::Instant;

/// 융합 변형 하나.
#[derive(Clone, Debug)]
pub struct FusionVariant {
    /// 변형 이름(출력 파일 이름에 쓴다).
    pub name: String,
    /// 원래 줄의 플래그 문자열(보고용).
    pub flags: String,
    /// 이 변형의 조밀화 옵션.
    pub opts: DensifyOptions,
    /// 점수 융합 설정(있으면 점수 융합).
    pub score: Option<ScoreFusionOptions>,
}

#[derive(Parser, Debug)]
#[command(no_binary_name = true)]
struct VariantLine {
    #[command(flatten)]
    a: DensifyArgs,
}

fn parse_line(tokens: &[&str]) -> Result<DensifyArgs, String> {
    let mut argv: Vec<&str> = vec!["--input_path", ".", "--output_path", "."];
    argv.extend_from_slice(tokens);
    VariantLine::try_parse_from(argv).map(|v| v.a).map_err(|e| e.to_string())
}

/// 변형 파일을 읽어 기본 설정(`base`, densify 명령 인자 `base_args` 를 반영한 것) 위에 각 줄의 플래그를 덮어쓴 설정 목록을 만든다.
/// 줄에 `--fusion-mode` 가 없으면 명령의 융합 방식(점수 융합 포함)을 따른다.
pub fn read_variants(path: &Path, base_args: &DensifyArgs, base: &DensifyOptions) -> Result<Vec<FusionVariant>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // 플래그 없는 줄의 해석 결과: 설정 밖 인자(영상 크기 등)가 바뀌었는지 비교하는 기준.
    let empty = parse_line(&[])?;
    let mut out: Vec<FusionVariant> = Vec::new();
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = |m: String| format!("{}:{}: {m}", path.display(), ln + 1);
        let (name, flags) = line.split_once('|').ok_or_else(|| at("`이름|플래그` 형식이 아님".into()))?;
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')) || name.starts_with('.') {
            return Err(at(format!("변형 이름 {name:?} (영숫자·_·-·. 만)")));
        }
        if out.iter().any(|v| v.name == name) {
            return Err(at(format!("변형 이름 중복 {name}")));
        }
        let tokens: Vec<&str> = flags.split_whitespace().collect();
        let a = parse_line(&tokens).map_err(at)?;
        let outside = [
            ("--max_image_size", a.max_image_size != empty.max_image_size),
            ("--number-views", a.number_views != empty.number_views),
            ("--pm-backend", a.pm_backend != empty.pm_backend),
            ("--mvs-profile", a.mvs_profile != empty.mvs_profile),
            ("--max-images", a.max_images != empty.max_images),
            ("--image_path", a.image_path != empty.image_path),
            ("--stats", a.stats != empty.stats),
            ("--fusion-variants", a.fusion_variants != empty.fusion_variants),
            ("--residual-out", a.residual_out != empty.residual_out),
        ];
        if let Some((f, _)) = outside.iter().find(|(_, changed)| *changed) {
            return Err(at(format!("{f} 는 변형마다 바꿀 수 없음(융합 설정만)")));
        }
        let mut opts = base.clone();
        apply_densify_args(&a, &mut opts).map_err(at)?;
        let mut same = opts.clone();
        same.fusion = base.fusion.clone();
        if same != *base {
            return Err(at("깊이맵을 바꾸는 플래그가 있음(융합 설정만 허용)".into()));
        }
        let score = score_options(&a, &opts, Some(base_args)).map_err(at)?;
        out.push(FusionVariant { name: name.to_string(), flags: tokens.join(" "), opts, score });
    }
    if out.is_empty() {
        return Err(format!("{}: 변형 없음", path.display()));
    }
    Ok(out)
}

/// 변형마다 융합해 `out_dir/<이름>.ply` 를 쓴다. 변형별 융합 시간을 출력하고, `stats` 면 점군 통계도 낸다.
/// `out_dir/fusion_variants.tsv` 에 이름·점 수·융합 시간·플래그를 모은다.
pub fn run_variants(scene: &DenseScene, maps: &DepthMapSet, variants: &[FusionVariant], out_dir: &Path, stats: bool) -> Result<(), String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let mut tsv = String::from("name\tpoints\tresidual_points\tfusion_s\tpass1_s\tpass2_s\twrite_s\tflags\n");
    for v in variants {
        let out = fuse_output(scene, maps, &v.opts, v.score.as_ref());
        let fusion_s = out.timings.fusion.as_secs_f64();
        let path = out_dir.join(format!("{}.ply", v.name));
        let tw = Instant::now();
        out.write_ply(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let rpath = out_dir.join(format!("{}.residual.ply", v.name));
        let wrote_residual = out.write_residual_ply(&rpath).map_err(|e| format!("{}: {e}", rpath.display()))?;
        let write_s = tw.elapsed().as_secs_f64();
        let extra = if wrote_residual { format!(" (2차 점 {} → {})", out.num_residual(), rpath.display()) } else { String::new() };
        let (p1, p2) = (out.timings.fusion_pass1.as_secs_f64(), out.timings.fusion_pass2.as_secs_f64());
        println!(
            "변형 {}: 점 {} 융합 {fusion_s:.2}s (1차 {p1:.2}s 2차 {p2:.2}s) 쓰기 {write_s:.2}s → {}{extra}",
            v.name,
            out.cloud.len(),
            path.display()
        );
        if stats {
            let ts = Instant::now();
            let st = cumulus3d_dense::cloud_stats(scene, &out.depth_maps, &out.cloud, &out.visibility, 200_000, 2.0);
            println!(
                "통계[{}]: 점 {} 융합 {fusion_s:.2}s (1차 {p1:.2}s 2차 {p2:.2}s) 이웃 간격 중앙 {:.4} GSD {:.4} 이상점(2px) {:.2}% 평면 이탈(>GSD·고립) {:.2}% 평면 거리 중앙 {:.4} 중복(<GSD/2) {:.2}% ({:.1}s)",
                v.name,
                st.points,
                st.nn_spacing_median,
                st.gsd,
                100.0 * st.outlier_ratio,
                100.0 * st.plane_outlier_ratio,
                st.plane_residual_median,
                100.0 * st.duplicate_ratio,
                ts.elapsed().as_secs_f64()
            );
        }
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{fusion_s:.3}\t{p1:.3}\t{p2:.3}\t{write_s:.3}\t{}\n",
            v.name,
            out.cloud.len(),
            out.num_residual(),
            v.flags
        ));
    }
    let p = out_dir.join("fusion_variants.tsv");
    std::fs::write(&p, tsv).map_err(|e| format!("{}: {e}", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("cumulus3d_fv_{}_{name}", std::process::id()));
        std::fs::write(&p, body).unwrap();
        p
    }

    fn args(t: &[&str]) -> DensifyArgs {
        parse_line(t).unwrap()
    }

    #[test]
    fn parses_fusion_flags_over_base() {
        let base = DensifyOptions::default();
        let p = write_tmp("ok", "# 주석\n\na|--fusion-min-views 3 --fusion-normal-error 30\nb|\nc|--fusion-mode score --score-tau 2.5 --fusion-depth-error 0.02\n");
        let v = read_variants(&p, &args(&[]), &base).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].opts.fusion.min_consistent_views, 3);
        assert_eq!(v[0].opts.fusion.max_normal_error_deg, 30.0);
        assert!(v[0].score.is_none());
        assert_eq!(v[1].opts, base);
        let s = v[2].score.as_ref().unwrap();
        assert_eq!((s.tau, s.depth_error), (2.5, 0.02));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn inherits_score_mode_from_command() {
        let base = DensifyOptions::default();
        let cmd = args(&["--fusion-mode", "score", "--score-lambda", "0.5"]);
        let p = write_tmp("inh", "a|--score-tau 3\nb|--fusion-mode consistency\n");
        let v = read_variants(&p, &cmd, &base).unwrap();
        let s = v[0].score.as_ref().unwrap();
        assert_eq!((s.tau, s.lambda), (3.0, 0.5));
        assert!(v[1].score.is_none());
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn rejects_depth_flags_and_bad_names() {
        let base = DensifyOptions::default();
        for (n, body) in [
            ("f", "a|--filter-min-views 1"),
            ("s", "a|--max_image_size 100"),
            ("n", "a b|--fusion-min-views 2"),
            ("d", "a|\na|"),
            ("u", "a|--nope 1"),
            ("t", "a|--score-tau 2"),
        ] {
            let p = write_tmp(n, body);
            assert!(read_variants(&p, &args(&[]), &base).is_err(), "{body}");
            std::fs::remove_file(p).ok();
        }
    }
}
