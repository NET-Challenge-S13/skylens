/*
 * declare_synthetic.rs
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

//! Test the declarative builder layer (`declare`) with a synthetic scene (3 drones x 8 positions). Densification is off (so it runs
//! without a device).
//! - The builder and build have no side effects before execution (no files created, no hook calls).
//! - The same plan gives the same result (fixed seed).
//! - `cumulus3d run <plan.toml>` produces the same output as `cumulus3d stream` with the same settings.

mod common;
use common::{make_dataset, NPOS};
use cumulus3d_cli::declare::{Plan, Recon, Sinks};
use cumulus3d_cli::events::{Event, EventKind};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

fn base() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("declare_synth")
}

/// Synthetic input folder shared by the tests (created only once).
fn dataset() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = base().join("src");
        let _ = std::fs::remove_dir_all(&d);
        make_dataset(&d);
        d
    })
}

/// The same plan as `stream --stride 1 --span 4 --overlap 1 --no-dense --seed 7 --save-models --quiet`.
fn plan(out: &Path) -> Plan {
    Recon::declare()
        .input(dataset())
        .preset("aerial-formation")
        .stride(1)
        .zones(4, 1)
        .no_dense()
        .seed(7)
        .sinks(Sinks::default_files(out).models(true).echo(false))
        .plan()
        .clone()
}

/// File listing of a folder (relative path -> size).
fn files(dir: &Path) -> BTreeMap<String, u64> {
    fn walk(root: &Path, d: &Path, m: &mut BTreeMap<String, u64>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(root, &p, m);
            } else {
                m.insert(p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), e.metadata().unwrap().len());
            }
        }
    }
    let mut m = BTreeMap::new();
    walk(dir, dir, &mut m);
    m
}

/// Timeline text (excluding time). The order between background refinement and the foreground can differ from run to run, so sort.
fn timeline(dir: &Path) -> Vec<String> {
    let s = std::fs::read_to_string(dir.join("timeline.txt")).unwrap();
    let mut v: Vec<String> = s.lines().map(|l| l.split_once(' ').unwrap().1.to_string()).collect();
    v.sort();
    v
}

/// Whether the sparse model file contents are the same (per-zone snap/preview/ba/refined + chain).
fn assert_same_models(a: &Path, b: &Path) {
    let fa = files(a);
    let models: Vec<&String> = fa.keys().filter(|k| k.starts_with("work/models/") && k.ends_with(".bin")).collect();
    assert!(models.len() >= 20, "모델 파일 {}", models.len());
    for k in models {
        assert_eq!(std::fs::read(a.join(k)).unwrap(), std::fs::read(b.join(k)).unwrap(), "모델 파일 다름: {k}");
    }
}

fn assert_same_outputs(a: &Path, b: &Path) {
    let (fa, fb) = (files(a), files(b));
    assert_eq!(fa.keys().collect::<Vec<_>>(), fb.keys().collect::<Vec<_>>(), "파일 목록 다름");
    assert_eq!(timeline(a), timeline(b));
    assert_same_models(a, b);
}

#[test]
fn builder_and_build_have_no_side_effects() {
    let src = dataset();
    let before = files(src);
    let out = base().join("lazy_out");
    let _ = std::fs::remove_dir_all(&out);
    let calls = Arc::new(AtomicUsize::new(0));
    let (c1, c2) = (calls.clone(), calls.clone());
    let builder = Recon::from_plan(plan(&out))
        .on(EventKind::PositionDone, move |_e: &Event| {
            c1.fetch_add(1, Ordering::SeqCst);
        })
        .on_position_done(move |_, _, _| {
            c2.fetch_add(1, Ordering::SeqCst);
        });
    assert!(!out.exists(), "기록 단계에서 출력 폴더가 생김");
    let recon = builder.build().unwrap_or_else(|e| panic!("{e}"));
    assert!(!out.exists(), "build 단계에서 출력 폴더가 생김");
    assert_eq!(calls.load(Ordering::SeqCst), 0, "실행 전에 훅이 불림");
    assert_eq!(files(src), before, "입력 폴더가 바뀜");
    assert_eq!(recon.positions(), NPOS);

    let summary = recon.run().unwrap();
    assert!(out.join("timeline.txt").exists() && out.join("DONE").exists());
    assert_eq!(calls.load(Ordering::SeqCst), 2 * summary.by_kind[&EventKind::PositionDone]);
    assert!(summary.by_kind[&EventKind::PositionDone] >= 4);
}

#[test]
fn same_plan_same_result() {
    let (a, b) = (base().join("same_a"), base().join("same_b"));
    let p = plan(&a);
    // A plan saved to TOML and read back is the same plan.
    let mut q = Plan::from_toml(&p.to_toml()).unwrap();
    assert_eq!(q, p);
    q.sinks.out = Some(b.clone());
    Recon::from_plan(p).build().unwrap().run().unwrap();
    Recon::from_plan(q).build().unwrap().run().unwrap();
    assert_same_outputs(&a, &b);
}

#[test]
fn cli_run_matches_stream() {
    let src = dataset();
    let (s_out, r_out) = (base().join("cli_stream"), base().join("cli_run"));
    let bin = env!("CARGO_BIN_EXE_cumulus3d");
    let st = Command::new(bin)
        .args(["stream", "--src"])
        .arg(src)
        .arg("--out")
        .arg(&s_out)
        .args(["--stride", "1", "--span", "4", "--overlap", "1", "--no-dense", "--seed", "7", "--save-models", "--quiet"])
        .status()
        .expect("stream 실행");
    assert!(st.success());

    let toml_path = base().join("cli_plan.toml");
    std::fs::write(&toml_path, plan(&r_out).to_toml()).unwrap();
    let st = Command::new(bin).arg("run").arg(&toml_path).status().expect("run 실행");
    assert!(st.success());
    assert_same_outputs(&s_out, &r_out);
    // The settings line of run.log is the same too.
    let conf = |d: &Path| {
        std::fs::read_to_string(d.join("run.log"))
            .unwrap()
            .lines()
            .filter(|l| l.starts_with("설정:") || l.starts_with("regions:"))
            .map(String::from)
            .collect::<Vec<_>>()
    };
    assert_eq!(conf(&s_out), conf(&r_out));

    // The default plan output can be read as TOML.
    let o = Command::new(bin).args(["plan", "--print-default"]).output().unwrap();
    assert!(o.status.success());
    assert_eq!(Plan::from_toml(&String::from_utf8(o.stdout).unwrap()).unwrap(), Plan::default());

    // On validation failure, report all problems without running.
    let bad = base().join("cli_bad.toml");
    let bad_out = base().join("cli_bad_out");
    let mut p = plan(&bad_out);
    (p.zones.span, p.zones.overlap, p.input.stride) = (2, 3, 0);
    std::fs::write(&bad, p.to_toml()).unwrap();
    let o = Command::new(bin).arg("run").arg(&bad).output().unwrap();
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("zones.overlap") && err.contains("input.stride"), "{err}");
    assert!(!bad_out.exists());
}
