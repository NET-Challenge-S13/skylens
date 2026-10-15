/*
 * post.rs
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

//! 후처리: 스크립트 끝의 파이썬 부분(초벌 정렬·사건별 스냅샷·manifest)과 incremental_reanchor.py 를
//! align 크레이트로 수행한다. 모델은 메모리의 Reconstruction 을 그대로 쓴다.

use crate::util::{py_float, Json, Logger};
use cumulus3d_align::shared::NameFilter;
use cumulus3d_align::RobustUmeyamaOptions;
use cumulus3d_align::{
    compose_snapshot, robust_umeyama, shared_point_correspondences, transform_cloud, KdTree, SharedPointOptions, SnapshotOptions,
};
use cumulus3d_core::io::{read_ply, write_ply, PlyLayout, PointCloud};
use cumulus3d_core::{Reconstruction, Sim3, Vec3};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 후처리 입력(PLY 경로·사건 문자열 기반).
pub struct PostInput<'a> {
    /// 출력 폴더.
    pub out: &'a Path,
    /// 구역 크기(위치 수).
    pub span: usize,
    /// 구역 겹침(위치 수).
    pub ovl: usize,
    /// (첫 사건 기준 초, 사건 문자열).
    pub events: Vec<(f64, String)>,
    /// 구역별 초벌 모델.
    pub previews: &'a BTreeMap<usize, Reconstruction>,
    /// 구역별 정밀 모델.
    pub refined: &'a BTreeMap<usize, Reconstruction>,
    /// 구역별 초벌 PLY 경로.
    pub preview_ply: &'a BTreeMap<usize, PathBuf>,
    /// 구역별 정밀 PLY 경로.
    pub refined_ply: &'a BTreeMap<usize, PathBuf>,
    /// 영상 이름 → 위치.
    pub position: &'a (dyn Fn(&str) -> Option<usize> + Sync),
}

/// 파이썬 `tof`: 첫 단어열이 일치하거나 접두가 같은 첫 사건 시각.
fn tof(events: &[(f64, String)], pat: &str) -> Option<f64> {
    events.iter().find(|(_, s)| s.split(" pos[").next() == Some(pat) || s.starts_with(pat)).map(|(t, _)| *t)
}

/// 견고 Umeyama 결과(s, R, t, 잔차 중앙, 인라이어 수).
struct Fit {
    sim3: Sim3,
    median: f64,
    inliers: usize,
}

/// 두 모델의 공유 3D 점(같은 이름 영상·같은 2D 인덱스)으로 a_to_b 를 맞춘다.
fn corr_points(a: &Reconstruction, b: &Reconstruction, filter: Option<NameFilter>) -> (Vec<Vec3>, Vec<Vec3>) {
    let opts = SharedPointOptions { position_range: None, name_filter: filter };
    let pairs = shared_point_correspondences(a, b, &opts);
    let mut pa = Vec::with_capacity(pairs.len());
    let mut pb = Vec::with_capacity(pairs.len());
    for (i, j) in pairs {
        if let (Some(x), Some(y)) = (a.point3d(i), b.point3d(j)) {
            pa.push(x.xyz);
            pb.push(y.xyz);
        }
    }
    (pa, pb)
}

fn fit(pa: &[Vec3], pb: &[Vec3]) -> Option<Fit> {
    let r = robust_umeyama(pa, pb, &RobustUmeyamaOptions::default())?;
    Some(Fit { sim3: r.sim3, median: r.median_residual, inliers: r.num_inliers })
}

fn transformed(c: &PointCloud, t: &Sim3) -> PointCloud {
    let mut c = c.clone();
    transform_cloud(&mut c, t);
    c
}

fn write(path: &Path, c: &PointCloud) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    write_ply(path, c, PlyLayout::XyzRgbNormal).map_err(|e| format!("{}: {e}", path.display()))
}

fn basename(p: &Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
}

/// 앞 구름의 점들에 대한 뒤 구름(1/20 추출) 최근접 거리 중 < 5 m 의 중앙값.
fn overlap_median(prev: &PointCloud, cur: &PointCloud) -> Option<(f64, usize)> {
    if prev.is_empty() {
        return None;
    }
    let tree = KdTree::from_f32(&prev.positions);
    let q: Vec<[f64; 3]> = cur.positions.iter().step_by(20).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
    let mut d: Vec<f64> = tree.nearest_many(&q).into_iter().flatten().map(|(_, d2)| d2.sqrt()).filter(|d| *d < 5.0).collect();
    if d.is_empty() {
        return None;
    }
    let n = d.len();
    d.sort_by(f64::total_cmp);
    let m = if n % 2 == 1 { d[n / 2] } else { 0.5 * (d[n / 2 - 1] + d[n / 2]) };
    Some((m, n))
}

/// 구역별 사건 시각(첫 사건 기준 초). 없으면 None.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZoneTimes {
    /// `region K arrived`.
    pub arrived: Option<f64>,
    /// `preview K ready`.
    pub preview_ready: Option<f64>,
    /// `refined K pose_ready`.
    pub refined_pose: Option<f64>,
    /// `refined K ready`.
    pub refined_ready: Option<f64>,
}

/// 구역 점군 하나: 출력 파일 이름(full/ 과 같은 이름)과 점군.
pub struct ZoneCloud<'a> {
    /// 출력 파일 이름.
    pub file: String,
    /// 점군.
    pub cloud: &'a PointCloud,
}

/// 메모리 입력형 후처리(이벤트로 모은 구역 결과). [`run`] 과 기본 훅이 함께 쓴다.
pub struct PostZones<'a> {
    /// 출력 폴더.
    pub out: &'a Path,
    /// 조밀 점군이 있는 초벌 구역.
    pub preview: BTreeMap<usize, ZoneCloud<'a>>,
    /// 조밀 점군이 있는 정밀 구역.
    pub refined: BTreeMap<usize, ZoneCloud<'a>>,
    /// 초벌 모델(구역 0 은 GPS 정렬본).
    pub preview_models: BTreeMap<usize, &'a Reconstruction>,
    /// 정밀 모델(BA + GPS 정렬).
    pub refined_models: BTreeMap<usize, &'a Reconstruction>,
    /// 구역별 사건 시각.
    pub times: BTreeMap<usize, ZoneTimes>,
    /// 구역 K(≥ 1) 초벌 정렬에 쓰는 위치 구간 [lo, hi) = 앞 구역과의 겹침.
    pub windows: BTreeMap<usize, (usize, usize)>,
    /// 영상 이름 → 위치.
    pub position: &'a (dyn Fn(&str) -> Option<usize> + Sync),
}

/// 후처리 전체(PLY 경로·사건 문자열 입력). 요약은 run.log(+표준 출력)로.
pub fn run(inp: &PostInput, log: &Logger) -> Result<(), String> {
    let ev = &inp.events;
    let mut pv: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (k, p) in inp.preview_ply {
        pv.insert(*k, read_ply(p).map_err(|e| format!("{}: {e}", p.display()))?);
    }
    let mut rf: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (k, p) in inp.refined_ply {
        rf.insert(*k, read_ply(p).map_err(|e| format!("{}: {e}", p.display()))?);
    }
    let mut times: BTreeMap<usize, ZoneTimes> = BTreeMap::new();
    for k in pv.keys() {
        let t = times.entry(*k).or_default();
        t.preview_ready = tof(ev, &format!("preview {k} ready"));
        t.arrived = tof(ev, &format!("region {k} arrived"));
    }
    for k in rf.keys() {
        let t = times.entry(*k).or_default();
        t.refined_ready = tof(ev, &format!("refined {k} ready"));
        t.refined_pose = tof(ev, &format!("refined {k} pose_ready"));
    }
    let windows = pv
        .keys()
        .map(|k| {
            let start = k * inp.span;
            (*k, (start.saturating_sub(inp.ovl), start + inp.ovl))
        })
        .collect();
    let z = PostZones {
        out: inp.out,
        preview: pv.iter().map(|(k, c)| (*k, ZoneCloud { file: basename(&inp.preview_ply[k]), cloud: c })).collect(),
        refined: rf.iter().map(|(k, c)| (*k, ZoneCloud { file: basename(&inp.refined_ply[k]), cloud: c })).collect(),
        preview_models: inp.previews.iter().map(|(k, m)| (*k, m)).collect(),
        refined_models: inp.refined.iter().map(|(k, m)| (*k, m)).collect(),
        times,
        windows,
        position: inp.position,
    };
    run_zones(&z, log)
}

/// 후처리 본체(메모리 입력): 초벌 정렬 → aligned/ → 사건별 스냅샷·manifest → 재고정(final_frame/).
pub fn run_zones(z: &PostZones, log: &Logger) -> Result<(), String> {
    let out = z.out;
    let pv: BTreeMap<usize, &PointCloud> = z.preview.iter().map(|(k, c)| (*k, c.cloud)).collect();
    let rf: BTreeMap<usize, &PointCloud> = z.refined.iter().map(|(k, c)| (*k, c.cloud)).collect();
    let tm = |k: &usize| z.times.get(k).copied().unwrap_or_default();
    let t_p: BTreeMap<usize, Option<f64>> = pv.keys().map(|k| (*k, tm(k).preview_ready)).collect();
    let t_r: BTreeMap<usize, Option<f64>> = rf.keys().map(|k| (*k, tm(k).refined_ready)).collect();
    let t_rp: BTreeMap<usize, Option<f64>> = rf.keys().map(|k| (*k, tm(k).refined_pose)).collect();
    let t_a: BTreeMap<usize, Option<f64>> = pv.keys().map(|k| (*k, tm(k).arrived)).collect();

    // --- 초벌 정렬: 그 시점에 자세가 준비된 최신 정밀 모델(없으면 GPS 정렬된 preview_0)에 공유 점으로.
    let mut align_rows: Vec<Json> = Vec::new();
    let mut pframe: BTreeMap<usize, String> = BTreeMap::new();
    let mut pv_aligned: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (&k, cloud) in &pv {
        let (s, med, nk, reference, t): (f64, f64, usize, String, Sim3);
        if k == 0 {
            (s, med, nk, reference, t) = (1.0, 0.0, 0, "GPS(preview_0)".into(), Sim3::identity());
            pframe.insert(k, "preview_0".into());
        } else {
            let tpk = t_p[&k].unwrap_or(f64::INFINITY);
            let cand = rf.keys().copied().filter(|j| t_rp.get(j).copied().flatten().is_some_and(|x| x <= tpk)).max();
            let (b, refname, frame) = match cand.and_then(|j| z.refined_models.get(&j).copied().map(|m| (m, j))) {
                Some((m, j)) => (Some(m), format!("refined_{j}"), format!("refined_{j}")),
                None => (z.preview_models.get(&0).copied(), "preview_0(GPS)".to_string(), "preview_0".to_string()),
            };
            let a = z.preview_models.get(&k).copied();
            let res = match (a, b) {
                (Some(a), Some(b)) => {
                    let (lo, hi) = z.windows.get(&k).copied().unwrap_or((0, 0));
                    let pos = z.position;
                    // 위치 함수는 'static 이 아니라서 구간 이름 집합을 미리 만든다.
                    let names: BTreeSet<String> =
                        a.images().map(|im| im.name.clone()).filter(|n| pos(n).is_some_and(|p| p >= lo && p < hi)).collect();
                    let filt: NameFilter = Arc::new(move |n: &str| names.contains(n));
                    let (mut pa, mut pb) = corr_points(a, b, Some(filt));
                    if pa.len() < 20 {
                        (pa, pb) = corr_points(a, b, None);
                    }
                    fit(&pa, &pb)
                }
                _ => None,
            };
            match res {
                Some(f) => {
                    (s, med, nk, reference, t) = (f.sim3.scale, f.median, f.inliers, refname, f.sim3);
                    pframe.insert(k, frame);
                }
                None => {
                    // 설계 결정: 대응이 없으면 실패로 끝내지 않고 정렬 없이 두고 표시한다.
                    log.line(&format!("[post] 초벌 {k} 정렬 실패(공유 점 부족) — 원래 좌표 유지"));
                    (s, med, nk, reference, t) = (1.0, 0.0, 0, format!("{refname}(실패)"), Sim3::identity());
                    pframe.insert(k, frame);
                }
            }
        }
        align_rows.push(Json::obj(vec![
            ("region", Json::Int(k as i64)),
            ("reference", Json::Str(reference)),
            ("pairs", Json::Int(nk as i64)),
            ("fit_median_m", Json::f(med, 2)),
            ("scale", Json::f(s, 3)),
        ]));
        let c = transformed(cloud, &t);
        write(&out.join("aligned/preview").join(z.preview[&k].file.clone()), &c)?;
        pv_aligned.insert(k, c);
    }
    for (k, c) in &rf {
        write(&out.join("aligned/refined").join(z.refined[k].file.clone()), c)?;
    }

    // --- 사건 순서별 스냅샷.
    let mut evs: Vec<(f64, &str, usize)> = Vec::new();
    for k in pv.keys() {
        evs.push((t_p[k].unwrap_or(f64::INFINITY), "preview", *k));
    }
    for k in rf.keys() {
        evs.push((t_r[k].unwrap_or(f64::INFINITY), "refined", *k));
    }
    evs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(b.1)).then(a.2.cmp(&b.2)));
    let snap_opts = SnapshotOptions::default();
    let snapdir = out.join("snapshots");
    std::fs::create_dir_all(&snapdir).map_err(|e| e.to_string())?;
    let mut shown_p: BTreeSet<usize> = BTreeSet::new();
    let mut shown_r: BTreeSet<usize> = BTreeSet::new();
    struct Ev {
        i: usize,
        t: f64,
        kind: &'static str,
        k: usize,
        n: usize,
        refined: Vec<usize>,
        preview_only: Vec<usize>,
    }
    let mut man: Vec<Ev> = Vec::new();
    let snap_name = |i: usize, t: f64, kind: &str, k: usize| snapdir.join(format!("event_{i:02}_{t:06.1}s_{kind}_{k}.ply"));
    for (i, &(t, kind, k)) in evs.iter().enumerate() {
        let i = i + 1;
        if kind == "preview" {
            shown_p.insert(k);
        } else {
            shown_r.insert(k);
        }
        let fine: Vec<&PointCloud> = shown_r.iter().map(|j| rf[j]).collect();
        let po: Vec<usize> = shown_p.difference(&shown_r).copied().collect();
        let coarse: Vec<&PointCloud> = po.iter().map(|j| &pv_aligned[j]).collect();
        let a = compose_snapshot(&fine, &coarse, &snap_opts);
        write(&snap_name(i, t, kind, k), &a)?;
        let kind: &'static str = if kind == "preview" { "preview" } else { "refined" };
        man.push(Ev { i, t, kind, k, n: a.len(), refined: shown_r.iter().copied().collect(), preview_only: po });
    }
    let r1 = |x: Option<f64>| Json::of(x, 1);
    let rows: Vec<Json> = pv
        .keys()
        .map(|k| {
            let ta = t_a[k].unwrap_or(0.0);
            let tp = t_p[k].unwrap_or(0.0);
            let trp = t_rp.get(k).copied().flatten();
            let tr = t_r.get(k).copied().flatten();
            Json::obj(vec![
                ("region", Json::Int(*k as i64)),
                ("arrived_s", Json::f(ta, 1)),
                ("preview_ready_s", Json::f(tp, 1)),
                ("preview_latency_s", Json::f(tp - ta, 1)),
                ("refined_pose_s", r1(trp)),
                ("refined_ready_s", r1(tr)),
                ("refined_latency_s", r1(tr.map(|x| x - ta))),
            ])
        })
        .collect();
    let ev_json = |e: &Ev, frame: Option<&str>, n: usize| {
        let mut v = vec![
            ("event", Json::Int(e.i as i64)),
            ("time_s", Json::f(e.t, 1)),
            ("kind", Json::Str(e.kind.into())),
            ("region", Json::Int(e.k as i64)),
            ("points_1of6", Json::Int(n as i64)),
            ("refined", Json::ints(&e.refined)),
            ("preview_only", Json::ints(&e.preview_only)),
        ];
        if let Some(f) = frame {
            v.push(("frame", Json::Str(f.into())));
        }
        Json::obj(v)
    };
    let manifest = |events: Vec<Json>, reanchor: Option<Json>| {
        let mut v = vec![("timeline", Json::Arr(rows.clone())), ("events", Json::Arr(events)), ("align", Json::Arr(align_rows.clone()))];
        if let Some(r) = reanchor {
            v.push(("reanchor", r));
        }
        Json::obj(v)
    };
    let mpath = snapdir.join("manifest.json");
    std::fs::write(&mpath, manifest(man.iter().map(|e| ev_json(e, None, e.n)).collect(), None).dump()).map_err(|e| e.to_string())?;
    log.line("== 구역별 시간(초, 시작 기준)");
    for r in &rows {
        log.line(&r.py_repr());
    }
    log.line("== 정렬");
    for a in &align_rows {
        log.line(&a.py_repr());
    }
    log.line("== 사건 순서");
    for m in &man {
        log.line(&format!("{} {} {} {} {}", m.i, py_float(m.t, 1), m.kind, m.k, m.n));
    }
    let ks: Vec<usize> = rf.keys().copied().collect();
    for w in ks.windows(2) {
        match overlap_median(rf[&w[0]], rf[&w[1]]) {
            Some((m, _)) => log.line(&format!("정밀 {}-{} 겹침 차 중앙 {m:.2}m", w[0], w[1])),
            None => log.line(&format!("정밀 {}-{} 겹침 없음", w[0], w[1])),
        }
    }

    // --- 재기준(incremental_reanchor.py): 새 정밀 모델이 나올 때마다 이미 보낸 구역을 그 좌표계로.
    if rf.is_empty() {
        return Ok(());
    }
    let mut models: BTreeMap<String, &Reconstruction> = BTreeMap::new();
    if let Some(m) = z.preview_models.get(&0).copied() {
        models.insert("preview_0".into(), m);
    }
    for (k, m) in &z.refined_models {
        models.insert(format!("refined_{k}"), m);
    }
    let mut cache: HashMap<(String, String), Sim3> = HashMap::new();
    let mut order: Vec<(String, String)> = Vec::new();
    let mut tf = |a: &str, b: &str| -> Sim3 {
        if a == b {
            return Sim3::identity();
        }
        let key = (a.to_string(), b.to_string());
        if let Some(t) = cache.get(&key) {
            return *t;
        }
        let t = match (models.get(a), models.get(b)) {
            (Some(ma), Some(mb)) => {
                let (pa, pb) = corr_points(ma, mb, None);
                match fit(&pa, &pb) {
                    Some(f) => {
                        log.line(&format!(
                            "{a} -> {b}: 점쌍 {}, 잔차 중앙 {:.3}m, 축척 {:.4}, 이동 {:.2}m",
                            f.inliers,
                            f.median,
                            f.sim3.scale,
                            f.sim3.translation.norm()
                        ));
                        f.sim3
                    }
                    None => {
                        log.line(&format!("{a} -> {b}: 점쌍 부족({}), 항등 사용", pa.len()));
                        Sim3::identity()
                    }
                }
            }
            _ => Sim3::identity(),
        };
        cache.insert(key.clone(), t);
        order.push(key);
        t
    };
    for f in std::fs::read_dir(&snapdir).map_err(|e| e.to_string())?.flatten() {
        let n = f.file_name().to_string_lossy().to_string();
        if n.starts_with("event_") && n.ends_with(".ply") {
            let _ = std::fs::remove_file(f.path());
        }
    }
    let mut cur = "preview_0".to_string();
    let mut ev2: Vec<Json> = Vec::new();
    for e in &man {
        if e.kind == "refined" {
            cur = format!("refined_{}", e.k);
        }
        let rs: Vec<PointCloud> = e.refined.iter().map(|j| transformed(rf[j], &tf(&format!("refined_{j}"), &cur))).collect();
        let ps: Vec<PointCloud> = e
            .preview_only
            .iter()
            .map(|j| transformed(&pv_aligned[j], &tf(pframe.get(j).map(|s| s.as_str()).unwrap_or("preview_0"), &cur)))
            .collect();
        let a = compose_snapshot(&rs.iter().collect::<Vec<_>>(), &ps.iter().collect::<Vec<_>>(), &snap_opts);
        write(&snap_name(e.i, e.t, e.kind, e.k), &a)?;
        log.line(&format!("{} {} {} {} 좌표계 {cur} 점 {}", e.i, py_float(e.t, 1), e.kind, e.k, a.len()));
        ev2.push(ev_json(e, Some(&cur), a.len()));
    }
    let last = format!("refined_{}", rf.keys().max().copied().unwrap_or(0));
    let mut final_rf: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (k, c) in &rf {
        let c2 = transformed(c, &tf(&format!("refined_{k}"), &last));
        write(&out.join("final_frame/refined").join(z.refined[k].file.clone()), &c2)?;
        final_rf.insert(*k, c2);
    }
    for (k, c) in &pv_aligned {
        let fr = pframe.get(k).cloned().unwrap_or_else(|| "preview_0".into());
        write(&out.join("final_frame/preview").join(z.preview[k].file.clone()), &transformed(c, &tf(&fr, &last)))?;
    }
    for w in ks.windows(2) {
        match overlap_median(&final_rf[&w[0]], &final_rf[&w[1]]) {
            Some((m, n)) => log.line(&format!("최종 좌표계 정밀 {}-{} 겹침 차 중앙 {m:.2}m (겹침 점 {n})", w[0], w[1])),
            None => log.line(&format!("최종 좌표계 정밀 {}-{} 겹침 없음", w[0], w[1])),
        }
    }
    let re = Json::Obj(
        order
            .iter()
            .map(|(a, b)| {
                let t = cache[&(a.clone(), b.clone())];
                (format!("{a}->{b}"), Json::obj(vec![("scale", Json::f(t.scale, 4)), ("shift_m", Json::f(t.translation.norm(), 2))]))
            })
            .collect(),
    );
    std::fs::write(&mpath, manifest(ev2, Some(re)).dump()).map_err(|e| e.to_string())?;
    Ok(())
}
