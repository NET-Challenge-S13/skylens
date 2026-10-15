/*
 * positioning.rs
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

//! 전역 위치 추정: BATA 계열 점-카메라 방향 제약, 회전 고정.
//!
//! 잔차 e_k = v_k − d_k (X_p − c_i), Huber(0.1) 손실. 자체 LM: 관측 스케일 d → 점 X 순으로
//! 슈어 소거하고 프레임 중심의 밀집 축약 계통을 촐레스키로 푼다.
//! 병렬 합산은 고정 크기 덩어리 단위로 하고 덩어리 순서대로 더해 스레드 수와 무관하게 결정적이다.

use cumulus3d_core::{Error, ImageId, Mat3, Point3DId, Reconstruction, Result, Rigid3, Vec3};
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// 위치 추정 옵션.
#[derive(Clone, Debug)]
pub struct PositionSolverOptions {
    /// 프레임 중심 위치를 변수로 둘지.
    pub optimize_positions: bool,
    /// 3D 점을 변수로 둘지.
    pub optimize_points: bool,
    /// 관측별 스케일을 변수로 둘지.
    pub optimize_scales: bool,
    /// Huber 척도 a.
    pub loss_scale: f64,
    /// LM 최대 반복 횟수.
    pub max_num_iterations: usize,
    /// 비용 상대 변화 수렴 임계.
    pub function_tolerance: f64,
    /// 기울기 수렴 임계.
    pub gradient_tolerance: f64,
    /// 파라미터 변화 수렴 임계.
    pub parameter_tolerance: f64,
    /// 초기화 난수 시드(기본 0).
    pub random_seed: u64,
    /// 초점 사전값 없는 카메라 관측의 손실 배율.
    pub uncalibrated_weight: f64,
    /// 관측 스케일 하한.
    pub min_scale: f64,
}

impl Default for PositionSolverOptions {
    fn default() -> Self {
        Self {
            optimize_positions: true,
            optimize_points: true,
            optimize_scales: true,
            loss_scale: 0.1,
            max_num_iterations: 100,
            function_tolerance: 1e-5,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-8,
            random_seed: 0,
            uncalibrated_weight: 0.5,
            min_scale: 1e-5,
        }
    }
}

/// 위치 추정 결과 통계.
#[derive(Clone, Debug, Default)]
pub struct PositioningSummary {
    /// 최적화한 프레임 수.
    pub num_frames: usize,
    /// 최적화한 점 수.
    pub num_points: usize,
    /// 방향 관측 수.
    pub num_observations: usize,
    /// LM 반복 횟수.
    pub num_iterations: usize,
    /// 초기 비용.
    pub initial_cost: f64,
    /// 최종 비용.
    pub final_cost: f64,
    /// 수렴 여부.
    pub converged: bool,
}

struct Obs {
    frame: usize,
    /// 세계 좌표 관측 방향(단위).
    v: Vec3,
    factor: f64,
    fixed: bool,
}

struct Problem {
    centers: Vec<Vec3>,
    points: Vec<Vec3>,
    scales: Vec<f64>,
    obs: Vec<Obs>,
    /// 점별 관측 구간.
    ptr: Vec<usize>,
}

/// 점 하나의 축약 결과(역전파용).
struct PointSys {
    inv: Mat3,
    /// (프레임, S_Xc 블록).
    blocks: Vec<(usize, Mat3)>,
    gx: Vec3,
    /// 관측별 (h, g_d, H_dX 행, H_dc 행) — 고정 스케일이면 h = 0.
    od: Vec<(f64, f64, Vec3, Vec3)>,
    /// 프레임별 축약 블록 기여 (프레임, S_cc 대각 기여, g_c 기여).
    cc: Vec<(usize, Mat3, Vec3)>,
}

fn huber(s: f64, a: f64) -> (f64, f64) {
    let a2 = a * a;
    if s <= a2 {
        (s, 1.0)
    } else {
        let r = s.sqrt();
        (2.0 * a * r - a2, a / r)
    }
}

impl Problem {
    fn residual(&self, k: usize, p: usize) -> Vec3 {
        let o = &self.obs[k];
        o.v - self.scales[k] * (self.points[p] - self.centers[o.frame])
    }

    fn cost_of(&self, centers: &[Vec3], points: &[Vec3], scales: &[f64], a: f64) -> f64 {
        let per: Vec<f64> = (0..points.len())
            .into_par_iter()
            .map(|p| {
                let mut c = 0.0;
                for (o, sk) in self.obs[self.ptr[p]..self.ptr[p + 1]].iter().zip(&scales[self.ptr[p]..self.ptr[p + 1]]) {
                    let e = o.v - *sk * (points[p] - centers[o.frame]);
                    c += 0.5 * o.factor * huber(e.norm_squared(), a).0;
                }
                c
            })
            .collect();
        per.iter().sum()
    }

    fn point_system(&self, p: usize, a: f64, mu: f64, opts: &PositionSolverOptions) -> PointSys {
        let x = self.points[p];
        let mut hxx_diag = 0.0;
        let mut sxx = Mat3::zeros();
        let mut gx = Vec3::zeros();
        let mut blocks: Vec<(usize, Mat3)> = Vec::new();
        let mut cc: Vec<(usize, Mat3, Vec3)> = Vec::new();
        let mut od = Vec::with_capacity(self.ptr[p + 1] - self.ptr[p]);
        for k in self.ptr[p]..self.ptr[p + 1] {
            let o = &self.obs[k];
            let f = o.frame;
            let d = self.scales[k];
            let diff = x - self.centers[f];
            let e = self.residual(k, p);
            let w = o.factor * huber(e.norm_squared(), a).1;
            hxx_diag += w * d * d;
            let gd = -diff.dot(&e) * w;
            let mut sxc = Mat3::identity() * (-d * d * w);
            let mut scc = Mat3::identity() * (w * d * d);
            let mut gxk = -d * w * e;
            let mut gck = d * w * e;
            let elim = opts.optimize_scales && !o.fixed;
            let mut entry = (0.0, gd, Vec3::zeros(), Vec3::zeros());
            if elim {
                let hdd = w * diff.norm_squared();
                let h = hdd + mu * hdd.clamp(1e-6, 1e32);
                let hdx = w * d * diff; // H_dX (= H_Xd)
                let hdc = -w * d * diff; // H_dc
                let outer = diff * diff.transpose() * (w * w * d * d / h);
                sxx -= outer;
                sxc += outer;
                scc -= outer;
                gxk -= hdx * (gd / h);
                gck -= hdc * (gd / h);
                entry = (h, gd, hdx, hdc);
            }
            od.push(entry);
            gx += gxk;
            match blocks.iter_mut().find(|(ff, _)| *ff == f) {
                Some(b) => b.1 += sxc,
                None => blocks.push((f, sxc)),
            }
            match cc.iter_mut().find(|(ff, _, _)| *ff == f) {
                Some(b) => {
                    b.1 += scc;
                    b.2 += gck;
                }
                None => cc.push((f, scc, gck)),
            }
        }
        // H_XX = Σ w d² I (+ 감쇠).
        let hxx = hxx_diag + mu * hxx_diag.clamp(1e-6, 1e32);
        sxx += Mat3::identity() * hxx;
        let inv = if opts.optimize_points { sxx.try_inverse().unwrap_or_else(Mat3::zeros) } else { Mat3::zeros() };
        PointSys { inv, blocks, gx, od, cc }
    }
}

/// 등록 영상의 회전을 고정하고 중심·점 위치를 푼다. 결과를 재구성에 기록(t = −R c).
pub fn global_positioning(rec: &mut Reconstruction, opts: &PositionSolverOptions) -> Result<PositioningSummary> {
    let mut frames: Vec<ImageId> = rec.registered_images();
    frames.sort();
    if frames.is_empty() {
        return Err(Error::InvalidArgument("위치 추정: 등록 영상 없음".into()));
    }
    let fidx: BTreeMap<ImageId, usize> = frames.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let rots: Vec<Mat3> =
        frames.iter().map(|id| rec.world_to_cam(*id).map(|p| p.rotation_matrix()).unwrap_or_else(Mat3::identity)).collect();
    // 관측 구성.
    let point_ids: Vec<Point3DId> = rec.point3d_ids();
    let mut obs = Vec::new();
    let mut ptr = vec![0usize];
    let mut used_points: Vec<Point3DId> = Vec::new();
    let mut first = true;
    for &pid in &point_ids {
        let p = rec.point3d(pid).expect("존재");
        let start = obs.len();
        for e in &p.track {
            let Some(&f) = fidx.get(&e.image_id) else { continue };
            let im = rec.image(e.image_id).expect("존재");
            let cam = rec.camera(im.camera_id).expect("존재");
            let Some(r) = cam.img_to_ray(&im.point2d(e.point2d_idx).xy) else { continue };
            let factor = if cam.focal_from_prior { 1.0 } else { opts.uncalibrated_weight };
            obs.push(Obs { frame: f, v: rots[f].transpose() * r, factor, fixed: first });
            first = false;
        }
        if obs.len() > start {
            ptr.push(obs.len());
            used_points.push(pid);
        }
    }
    if used_points.is_empty() {
        return Err(Error::InvalidArgument("위치 추정: 관측 있는 트랙 없음".into()));
    }
    let mut rng = rand_pcg::Pcg64::seed_from_u64(opts.random_seed);
    let mut u = || 100.0 * rng.random_range(-1.0..1.0);
    let centers: Vec<Vec3> = (0..frames.len()).map(|_| Vec3::new(u(), u(), u())).collect();
    let points: Vec<Vec3> = (0..used_points.len()).map(|_| Vec3::new(u(), u(), u())).collect();
    let scales = vec![1.0; obs.len()];
    let mut prob = Problem { centers, points, scales, obs, ptr };
    let summary = solve(&mut prob, opts);
    // 기록.
    for (i, id) in frames.iter().enumerate() {
        let r = rots[i];
        rec.set_world_to_cam(*id, Rigid3::from_rotation_matrix(&r, -(r * prob.centers[i])))?;
    }
    for (k, pid) in used_points.iter().enumerate() {
        rec.set_point3d_xyz(*pid, prob.points[k])?;
    }
    Ok(PositioningSummary { num_frames: frames.len(), num_points: used_points.len(), num_observations: prob.obs.len(), ..summary })
}

fn solve(prob: &mut Problem, opts: &PositionSolverOptions) -> PositioningSummary {
    let a = opts.loss_scale;
    let nf = prob.centers.len();
    let np = prob.points.len();
    let mut cost = prob.cost_of(&prob.centers, &prob.points, &prob.scales, a);
    let mut summary = PositioningSummary { initial_cost: cost, ..Default::default() };
    let mut mu = 1e-4;
    let mut nu = 2.0;
    // 덩어리 수(결정적 병렬 합산).
    let chunk = np.div_ceil(64).max(256);
    for _ in 0..opts.max_num_iterations {
        summary.num_iterations += 1;
        let systems: Vec<PointSys> = (0..np).into_par_iter().map(|p| prob.point_system(p, a, mu, opts)).collect();
        // 축약 계통 S δc = −g.
        let n = 3 * nf;
        let partial: Vec<(DMatrix<f64>, DVector<f64>)> = systems
            .par_chunks(chunk)
            .map(|chunk_sys| {
                let mut s = DMatrix::<f64>::zeros(n, n);
                let mut g = DVector::<f64>::zeros(n);
                for ps in chunk_sys {
                    for (f, scc, gc) in &ps.cc {
                        for r in 0..3 {
                            g[3 * f + r] += gc[r];
                            for c in 0..3 {
                                s[(3 * f + r, 3 * f + c)] += scc[(r, c)];
                            }
                        }
                    }
                    if !opts.optimize_points {
                        continue;
                    }
                    for (fa, ba) in &ps.blocks {
                        let t = ba.transpose() * ps.inv;
                        let gxs = t * ps.gx;
                        for r in 0..3 {
                            g[3 * fa + r] -= gxs[r];
                        }
                        for (fb, bb) in &ps.blocks {
                            let m = t * bb;
                            for r in 0..3 {
                                for c in 0..3 {
                                    s[(3 * fa + r, 3 * fb + c)] -= m[(r, c)];
                                }
                            }
                        }
                    }
                }
                (s, g)
            })
            .collect();
        let mut s = DMatrix::<f64>::zeros(n, n);
        let mut g = DVector::<f64>::zeros(n);
        for (ps, pg) in partial {
            s += ps;
            g += pg;
        }
        // 카메라 대각 감쇠: 원 헤시안 대각 H_cc = Σ w d² 를 다시 계산.
        let mut hcc = vec![0.0; nf];
        for p in 0..np {
            for k in prob.ptr[p]..prob.ptr[p + 1] {
                let o = &prob.obs[k];
                let e = prob.residual(k, p);
                let w = o.factor * huber(e.norm_squared(), a).1;
                hcc[o.frame] += w * prob.scales[k] * prob.scales[k];
            }
        }
        for f in 0..nf {
            for r in 0..3 {
                s[(3 * f + r, 3 * f + r)] += mu * hcc[f].clamp(1e-6, 1e32);
            }
        }
        // 기울기 수렴 검사(축약 기울기 근사).
        let gmax = g.amax().max(systems.iter().map(|ps| ps.gx.amax()).fold(0.0, f64::max));
        if gmax < opts.gradient_tolerance {
            summary.converged = true;
            break;
        }
        let dc: DVector<f64> = if opts.optimize_positions {
            match s.clone().cholesky() {
                Some(ch) => ch.solve(&(-&g)),
                None => {
                    mu *= nu;
                    nu *= 2.0;
                    continue;
                }
            }
        } else {
            DVector::zeros(n)
        };
        // 역대입.
        let steps: Vec<Vec3> = systems
            .par_iter()
            .map(|ps| {
                let mut rhs = -ps.gx;
                for (f, b) in &ps.blocks {
                    rhs -= b * Vec3::new(dc[3 * f], dc[3 * f + 1], dc[3 * f + 2]);
                }
                ps.inv * rhs
            })
            .collect();
        let mut new_centers = prob.centers.clone();
        for f in 0..nf {
            new_centers[f] += Vec3::new(dc[3 * f], dc[3 * f + 1], dc[3 * f + 2]);
        }
        let mut new_points = prob.points.clone();
        let mut new_scales = prob.scales.clone();
        let mut step_norm2 = dc.norm_squared();
        for p in 0..np {
            let dx = steps[p];
            new_points[p] += dx;
            step_norm2 += dx.norm_squared();
            let ps = &systems[p];
            for (j, k) in (prob.ptr[p]..prob.ptr[p + 1]).enumerate() {
                let (h, gd, hdx, hdc) = ps.od[j];
                if h == 0.0 {
                    continue;
                }
                let f = prob.obs[k].frame;
                let dcf = Vec3::new(dc[3 * f], dc[3 * f + 1], dc[3 * f + 2]);
                let dd = (-gd - hdx.dot(&dx) - hdc.dot(&dcf)) / h;
                step_norm2 += dd * dd;
                new_scales[k] = (prob.scales[k] + dd).max(opts.min_scale);
            }
        }
        let new_cost = prob.cost_of(&new_centers, &new_points, &new_scales, a);
        if new_cost.is_finite() && new_cost < cost {
            let x_norm2: f64 = prob.centers.iter().map(|c| c.norm_squared()).sum::<f64>()
                + prob.points.iter().map(|c| c.norm_squared()).sum::<f64>()
                + prob.scales.iter().map(|c| c * c).sum::<f64>();
            let rel = (cost - new_cost) / cost;
            prob.centers = new_centers;
            prob.points = new_points;
            prob.scales = new_scales;
            cost = new_cost;
            mu = (mu / 3.0).max(1e-16);
            nu = 2.0;
            if rel < opts.function_tolerance || step_norm2.sqrt() < opts.parameter_tolerance * (x_norm2.sqrt() + opts.parameter_tolerance) {
                summary.converged = true;
                break;
            }
        } else {
            mu *= nu;
            nu *= 2.0;
            if mu > 1e32 {
                break;
            }
        }
    }
    summary.final_cost = cost;
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::umeyama;
    use cumulus3d_core::{Camera, CameraModelKind, Image, Point3D, TrackEntry, Vec2};

    /// 참 회전 + 참 트랙으로 무작위 초기화 → 상사 정렬 후 중심 오차.
    fn run(noise_px: f64, seed: u64) -> (f64, f64, Reconstruction) {
        use rand::SeedableRng;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
        let mut rec = Reconstruction::new();
        let mut cam = Camera::new(1, CameraModelKind::Pinhole, 1000, 800, vec![800.0, 800.0, 500.0, 400.0]).unwrap();
        cam.focal_from_prior = true;
        rec.add_camera_own_rig(cam.clone()).unwrap();
        let mut poses = Vec::new();
        for i in 0..12u32 {
            let c = Vec3::new(i as f64 * 2.0, (i % 3) as f64 * 3.0, 30.0);
            let r = crate::math::so3_exp(&Vec3::new(std::f64::consts::PI + 0.05 * (i % 3) as f64, 0.02 * i as f64, 0.0));
            poses.push(Rigid3::from_rotation_matrix(&r, -(r * c)));
        }
        let mut pts = Vec::new();
        for _ in 0..400 {
            pts.push(Vec3::new(rng.random_range(-10.0..32.0), rng.random_range(-10.0..16.0), rng.random_range(0.0..5.0)));
        }
        let mut kps: Vec<Vec<Vec2>> = vec![Vec::new(); poses.len()];
        let mut tracks: Vec<Vec<TrackEntry>> = vec![Vec::new(); pts.len()];
        for (i, pose) in poses.iter().enumerate() {
            for (j, x) in pts.iter().enumerate() {
                if let Some(xy) = cam.cam_to_img(&(*pose * *x)) {
                    if xy.x > 0.0 && xy.y > 0.0 && xy.x < 1000.0 && xy.y < 800.0 {
                        let n = Vec2::new(crate::math::gaussian(&mut rng), crate::math::gaussian(&mut rng)) * noise_px;
                        tracks[j].push(TrackEntry::new(i as u32 + 1, kps[i].len() as u32));
                        kps[i].push(xy + n);
                    }
                }
            }
        }
        for (i, pose) in poses.iter().enumerate() {
            let id = i as u32 + 1;
            rec.add_image_own_frame(Image::new(id, format!("{id}"), 1, kps[i].clone()), Some(*pose)).unwrap();
            rec.register_image(id).unwrap();
        }
        let mut truth = BTreeMap::new();
        for (j, t) in tracks.into_iter().enumerate() {
            if t.len() >= 3 {
                let id = j as u64;
                rec.add_point3d_with_id(id, Point3D { xyz: Vec3::zeros(), color: [0; 3], error: -1.0, track: t }).unwrap();
                truth.insert(id, pts[j]);
            }
        }
        let s = global_positioning(&mut rec, &PositionSolverOptions::default()).unwrap();
        assert!(s.final_cost < s.initial_cost);
        let est: Vec<Vec3> = (1..=poses.len() as u32).map(|i| rec.projection_center(i).unwrap()).collect();
        let tru: Vec<Vec3> = poses.iter().map(|p| p.center()).collect();
        let sim = umeyama(&est, &tru).unwrap();
        let err = est.iter().zip(&tru).map(|(a, b)| (sim.transform_point(a) - b).norm()).fold(0.0, f64::max);
        let extent = 25.0;
        (err / extent, s.final_cost, rec)
    }

    #[test]
    fn noise_free_positions() {
        let (rel, _, _) = run(0.0, 1);
        assert!(rel < 1e-4, "rel {rel}");
    }

    #[test]
    fn noisy_positions_and_determinism() {
        let (rel, cost, rec) = run(0.5, 2);
        assert!(rel < 0.01, "rel {rel}");
        let (_, cost2, rec2) = run(0.5, 2);
        assert_eq!(cost.to_bits(), cost2.to_bits());
        for id in rec.registered_images() {
            assert_eq!(rec.world_to_cam(id).unwrap().to_params(), rec2.world_to_cam(id).unwrap().to_params());
        }
    }
}
