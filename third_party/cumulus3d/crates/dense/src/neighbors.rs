/*
 * neighbors.rs
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

//! 이웃(원천) 뷰 선택과 뷰별 깊이 범위. 희소점 트랙만 쓴다.

use crate::math::quantile_sorted;
use crate::scene::DenseScene;
use std::collections::HashMap;

/// 뷰 쌍 통계: 공유 점 수와 쌍마다 기록된 삼각측량각(라디안).
pub struct PairStats {
    n: usize,
    map: HashMap<(u32, u32), (u32, Vec<f32>)>,
}

impl PairStats {
    /// 트랙의 서로 다른 뷰 쌍마다 공유 수 +1, 각 min(a, π − a) 기록. 같은 뷰가 트랙에 두 번 있으면 쌍이 중복 계수된다.
    pub fn new(scene: &DenseScene) -> Self {
        let centers: Vec<[f64; 3]> = scene.views.iter().map(|v| v.center()).collect();
        let mut map: HashMap<(u32, u32), (u32, Vec<f32>)> = HashMap::new();
        for p in &scene.points {
            let x = p.xyz;
            for i in 0..p.views.len() {
                for j in 0..i {
                    let (a, b) = (p.views[i], p.views[j]);
                    if a == b {
                        continue;
                    }
                    let ca = centers[a as usize];
                    let cb = centers[b as usize];
                    let u = [ca[0] - x[0], ca[1] - x[1], ca[2] - x[2]];
                    let v = [cb[0] - x[0], cb[1] - x[1], cb[2] - x[2]];
                    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    let ang = if nu == 0.0 || nv == 0.0 {
                        0.0
                    } else {
                        let c = ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)).clamp(-1.0, 1.0);
                        let a = c.acos();
                        a.min(std::f64::consts::PI - a)
                    };
                    let key = (a.min(b), a.max(b));
                    let e = map.entry(key).or_default();
                    e.0 += 1;
                    e.1.push(ang as f32);
                }
            }
        }
        Self { n: scene.views.len(), map }
    }

    /// 뷰 `r` 의 후보: 75 분위 각 ≥ `min_angle_rad` 인 뷰를 공유 수 내림차순(같으면 색인 오름차순)으로 최대 `max` 개.
    pub fn select(&self, r: usize, max: usize, min_angle_rad: f64) -> Vec<usize> {
        let mut cands: Vec<(u32, usize)> = Vec::new();
        for o in 0..self.n {
            if o == r {
                continue;
            }
            let key = ((r as u32).min(o as u32), (r as u32).max(o as u32));
            let Some((cnt, angs)) = self.map.get(&key) else { continue };
            if min_angle_rad > 0.0 {
                let mut a: Vec<f64> = angs.iter().map(|&x| x as f64).collect();
                a.sort_by(|x, y| x.total_cmp(y));
                if quantile_sorted(&a, 0.75) < min_angle_rad {
                    continue;
                }
            }
            cands.push((*cnt, o));
        }
        cands.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        cands.into_iter().take(max).map(|(_, o)| o).collect()
    }

    /// 방향 다양성 선택: 후보(75 분위 각 ≥ 하한)의 점수 = 공유 수. 기준 카메라 좌표에서 기준선의 영상 평면 방위각을
    /// `bins` 구간으로 나누고(기준선이 광축 방향에 가까우면 별도 구간), 같은 구간에서 이미 k 개 골랐으면 점수에 `decay^k` 를
    /// 곱해 가장 큰 것부터 탐욕적으로 고른다. 같은 위치의 다른 카메라(광축 방향 기준선)와 앞뒤·좌우 위치가 섞인다.
    pub fn select_diverse(&self, scene: &DenseScene, r: usize, max: usize, min_angle_rad: f64, bins: usize, decay: f64) -> Vec<usize> {
        let cands = self.select(r, usize::MAX, min_angle_rad);
        let vr = &scene.views[r];
        let cr = vr.center();
        let bins = bins.max(1);
        let info: Vec<(usize, f64, usize)> = cands
            .iter()
            .map(|&o| {
                let co = scene.views[o].center();
                let b = vr.dir_to_cam(&[co[0] - cr[0], co[1] - cr[1], co[2] - cr[2]]);
                let lat = (b[0] * b[0] + b[1] * b[1]).sqrt();
                let bin = if lat < 0.3 * b[2].abs() {
                    bins
                } else {
                    let a = b[1].atan2(b[0]) + std::f64::consts::PI;
                    ((a / (2.0 * std::f64::consts::PI) * bins as f64) as usize).min(bins - 1)
                };
                let key = ((r as u32).min(o as u32), (r as u32).max(o as u32));
                (o, self.map[&key].0 as f64, bin)
            })
            .collect();
        let mut used = vec![0i32; bins + 1];
        let mut taken = vec![false; info.len()];
        let mut out = Vec::new();
        while out.len() < max {
            let mut best: Option<(f64, usize)> = None;
            for (i, &(_, sc, bin)) in info.iter().enumerate() {
                if taken[i] {
                    continue;
                }
                let s = sc * decay.powi(used[bin]);
                if best.is_none_or(|b| s > b.0) {
                    best = Some((s, i));
                }
            }
            let Some((_, i)) = best else { break };
            taken[i] = true;
            used[info[i].2] += 1;
            out.push(info[i].0);
        }
        out
    }

    /// 모든 뷰의 후보 목록.
    pub fn select_all(&self, max: usize, min_angle_deg: f64) -> Vec<Vec<usize>> {
        let a = min_angle_deg.to_radians();
        (0..self.n).map(|r| self.select(r, max, a)).collect()
    }
}

/// 뷰별 깊이 범위: 관측 희소점 깊이(양수만)의 1%/99% 위치 값(보간 없음)에 0.75 / 1.25 배. 점이 없으면 None.
pub fn depth_ranges(scene: &DenseScene) -> Vec<Option<(f64, f64)>> {
    let mut lists: Vec<Vec<f64>> = vec![Vec::new(); scene.views.len()];
    for p in &scene.points {
        for &v in &p.views {
            let view = &scene.views[v as usize];
            let r = &view.r[2];
            let d = r[0] * p.xyz[0] + r[1] * p.xyz[1] + r[2] * p.xyz[2] + view.t[2];
            if d > 0.0 {
                lists[v as usize].push(d);
            }
        }
    }
    lists.into_iter().map(|mut l| depth_range_of(&mut l)).collect()
}

/// 깊이 목록 하나의 범위(정렬함).
pub fn depth_range_of(l: &mut [f64]) -> Option<(f64, f64)> {
    if l.is_empty() {
        return None;
    }
    l.sort_by(|a, b| a.total_cmp(b));
    let n = l.len();
    let lo = l[((n as f64 * 0.01).floor() as usize).min(n - 1)];
    let hi = l[((n as f64 * 0.99).floor() as usize).min(n - 1)];
    Some((0.75 * lo, 1.25 * hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_range_reference() {
        let mut l: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let (a, b) = depth_range_of(&mut l).unwrap();
        assert!((a - 1.5).abs() < 1e-12 && (b - 125.0).abs() < 1e-12);
    }
}
