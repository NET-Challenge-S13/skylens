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

//! `densify --fusion-variants <file>`: build the depth maps once and fuse several times, changing only the fusion settings.
//!
//! File format: one `name|fusion flags` per line (same format as densify, whitespace separated). Empty lines and lines starting with `#`
//! are skipped.
//! The flags are interpreted by the densify subcommand's argument parser and overlaid on the base settings, so new fusion flags can be
//! used as they are.
//! Flags that change the depth maps (filter, PatchMatch, image size, etc.) are rejected.

use crate::interop::{apply_densify_args, score_options, DensifyArgs};
use clap::Parser;
use cumulus3d_dense::densify::fuse_output;
use cumulus3d_dense::fusion_score::ScoreFusionOptions;
use cumulus3d_dense::{DenseScene, DensifyOptions, DepthMapSet};
use std::path::Path;
use std::time::Instant;

/// One fusion variant.
#[derive(Clone, Debug)]
pub struct FusionVariant {
    /// Variant name (used in the output file name).
    pub name: String,
    /// Flag string of the original line (for reporting).
    pub flags: String,
    /// Densification options of this variant.
    pub opts: DensifyOptions,
    /// Score fusion settings (if present, score fusion).
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

/// Read the variants file and build the settings list in which each line's flags are overlaid on the base settings (`base`, with the
/// densify command arguments `base_args` applied).
/// If a line has no `--fusion-mode`, it follows the command's fusion mode (including score fusion).
pub fn read_variants(path: &Path, base_args: &DensifyArgs, base: &DensifyOptions) -> Result<Vec<FusionVariant>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // Interpretation result of a line without flags: the reference for checking whether arguments outside the settings (image size etc.)
    // changed.
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

/// Fuse per variant and write `out_dir/<name>.ply`. Print the fusion time per variant, and with `stats` also the point-cloud statistics.
/// Collect name, point count, fusion time and flags in `out_dir/fusion_variants.tsv`.
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
        let sm = out.timings.fusion_smooth.as_secs_f64();
        let (vis, vis_n) = (out.timings.fusion_visibility.as_secs_f64(), out.timings.fusion_visibility_removed);
        println!(
            "변형 {}: 점 {} 융합 {fusion_s:.2}s (1차 {p1:.2}s 2차 {p2:.2}s 평활 {sm:.2}s, 가시성 {vis:.2}s, 제거 {vis_n}) 쓰기 {write_s:.2}s → {}{extra}",
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
