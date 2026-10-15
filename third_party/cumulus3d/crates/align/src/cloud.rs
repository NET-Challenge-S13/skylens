/*
 * cloud.rs
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

//! 점군 후처리: Sim3 변환, 정밀 점군 근접 마스킹, 1/N 추출, 스냅샷 합성.

use crate::kdtree::KdTree;
use cumulus3d_core::io::PointCloud;
use cumulus3d_core::{Sim3, Vec3};
use rayon::prelude::*;

/// 점과 법선에 Sim3 적용(점: sRx+t, 법선: Rn). 계산은 f64.
pub fn transform_cloud(cloud: &mut PointCloud, t: &Sim3) {
    let r = t.rotation.to_rotation_matrix();
    let sr = r * t.scale;
    cloud.positions.par_iter_mut().for_each(|p| {
        let v = sr * Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64) + t.translation;
        *p = [v.x as f32, v.y as f32, v.z as f32];
    });
    cloud.normals.par_iter_mut().for_each(|n| {
        let v = r * Vec3::new(n[0] as f64, n[1] as f64, n[2] as f64);
        *n = [v.x as f32, v.y as f32, v.z as f32];
    });
}

/// `mask[i]` 가 true 인 점만 남긴 새 점군.
pub fn select(cloud: &PointCloud, keep: &[bool]) -> PointCloud {
    let pick = |v: &Vec<[f32; 3]>| -> Vec<[f32; 3]> {
        if v.is_empty() {
            Vec::new()
        } else {
            v.iter().zip(keep).filter(|(_, k)| **k).map(|(x, _)| *x).collect()
        }
    };
    PointCloud {
        positions: pick(&cloud.positions),
        normals: pick(&cloud.normals),
        colors: if cloud.colors.is_empty() {
            Vec::new()
        } else {
            cloud.colors.iter().zip(keep).filter(|(_, k)| **k).map(|(c, _)| *c).collect()
        },
    }
}

/// `reference` 와 거리 ≤ radius 인 점을 뺀 점군(정밀 점과 겹치는 초벌 점 제거).
pub fn remove_near(cloud: &PointCloud, reference: &KdTree, radius: f64) -> PointCloud {
    let near = reference.any_within_many_f32(&cloud.positions, radius);
    let keep: Vec<bool> = near.iter().map(|b| !b).collect();
    select(cloud, &keep)
}

/// 1/stride 간격 추출(인덱스 0, stride, 2·stride, …). stride ≤ 1 이면 그대로.
pub fn decimate(cloud: &PointCloud, stride: usize) -> PointCloud {
    if stride <= 1 {
        return cloud.clone();
    }
    let keep: Vec<bool> = (0..cloud.len()).map(|i| i % stride == 0).collect();
    select(cloud, &keep)
}

/// 이어 붙이기. 일부만 법선/색이 있으면 없는 쪽을 (0,0,0) 법선 / 흰색으로 채운다.
pub fn merge_clouds(clouds: &[&PointCloud]) -> PointCloud {
    let total: usize = clouds.iter().map(|c| c.len()).sum();
    let any_n = clouds.iter().any(|c| c.has_normals());
    let any_c = clouds.iter().any(|c| c.has_colors());
    let mut out = PointCloud {
        positions: Vec::with_capacity(total),
        normals: Vec::with_capacity(if any_n { total } else { 0 }),
        colors: Vec::with_capacity(if any_c { total } else { 0 }),
    };
    for c in clouds {
        out.positions.extend_from_slice(&c.positions);
        if any_n {
            if c.has_normals() {
                out.normals.extend_from_slice(&c.normals);
            } else {
                out.normals.extend(std::iter::repeat_n([0.0f32; 3], c.len()));
            }
        }
        if any_c {
            if c.has_colors() {
                out.colors.extend_from_slice(&c.colors);
            } else {
                out.colors.extend(std::iter::repeat_n([255u8; 3], c.len()));
            }
        }
    }
    out
}

/// 스냅샷 합성 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotOptions {
    /// 정밀 점과 이 거리 이내의 초벌 점 제거 (m).
    pub mask_radius: f64,
    /// 최종 1/N 추출.
    pub stride: usize,
}

impl Default for SnapshotOptions {
    fn default() -> Self {
        Self { mask_radius: 1.5, stride: 6 }
    }
}

/// 스냅샷 = 정밀 구역 전부 + (정밀 점과 겹치지 않는) 초벌 구역들, 그 뒤 1/N 추출.
/// 모든 입력은 이미 같은 좌표계에 있어야 한다.
pub fn compose_snapshot(fine: &[&PointCloud], coarse: &[&PointCloud], opts: &SnapshotOptions) -> PointCloud {
    let fine_all = merge_clouds(fine);
    let masked: Vec<PointCloud> = if fine_all.is_empty() {
        coarse.iter().map(|c| (*c).clone()).collect()
    } else {
        let tree = KdTree::from_f32(&fine_all.positions);
        coarse.iter().map(|c| remove_near(c, &tree, opts.mask_radius)).collect()
    };
    let mut parts: Vec<&PointCloud> = vec![&fine_all];
    parts.extend(masked.iter());
    // 설계 결정: 1/N 추출 시점. 마스킹 후 합친 전체에 한 번 적용한다.
    decimate(&merge_clouds(&parts), opts.stride)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_core::Quat;

    fn grid(n: usize, step: f32, z: f32) -> PointCloud {
        let mut c = PointCloud::default();
        for i in 0..n {
            for j in 0..n {
                c.positions.push([i as f32 * step, j as f32 * step, z]);
                c.normals.push([0.0, 0.0, 1.0]);
                c.colors.push([i as u8, j as u8, 0]);
            }
        }
        c
    }

    #[test]
    fn mask_within_1_5m() {
        // 정밀: x ∈ [0, 10] 격자(z=0). 초벌: z 높이별 점 — 1.5 m 이내만 제거.
        let fine = grid(11, 1.0, 0.0);
        let mut coarse = PointCloud::default();
        for z in [0.5f32, 1.4, 1.5, 1.6, 3.0] {
            coarse.positions.push([5.0, 5.0, z]);
        }
        coarse.positions.push([20.0, 5.0, 0.0]); // 수평으로 멀리
        coarse.positions.push([11.4, 5.0, 0.0]); // 가장자리에서 1.4 m
        let tree = KdTree::from_f32(&fine.positions);
        let out = remove_near(&coarse, &tree, 1.5);
        assert_eq!(out.positions, vec![[5.0, 5.0, 1.6], [5.0, 5.0, 3.0], [20.0, 5.0, 0.0]]);
    }

    #[test]
    fn transform_points_and_normals() {
        let mut c = grid(3, 1.0, 2.0);
        let t = Sim3::new(2.0, Quat::from_axis_angle(&Vec3::x(), std::f64::consts::FRAC_PI_2), Vec3::new(1.0, 0.0, 0.0));
        let orig = c.clone();
        transform_cloud(&mut c, &t);
        for (a, b) in orig.positions.iter().zip(&c.positions) {
            let e = t.transform_point(&Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64));
            assert!((e - Vec3::new(b[0] as f64, b[1] as f64, b[2] as f64)).norm() < 1e-5);
        }
        // z 법선 → x 축 90° 회전 → −y.
        let n = c.normals[0];
        assert!((n[1] + 1.0).abs() < 1e-6 && n[2].abs() < 1e-6);
    }

    #[test]
    fn decimate_and_snapshot() {
        let c = grid(4, 1.0, 0.0);
        assert_eq!(decimate(&c, 6).len(), 3); // 16 → 0, 6, 12
        let fine = grid(5, 1.0, 0.0);
        let mut coarse = grid(10, 1.0, 0.0); // 0..9; x,y ≤ 6 인 점은 정밀(0..4)과 1.5 m 이내
        coarse.normals.clear();
        let snap = compose_snapshot(&[&fine], &[&coarse], &SnapshotOptions { mask_radius: 1.5, stride: 1 });
        let kept_coarse = coarse
            .positions
            .iter()
            .filter(|p| {
                let dx = (p[0] - 4.0).max(0.0);
                let dy = (p[1] - 4.0).max(0.0);
                dx * dx + dy * dy > 2.25
            })
            .count();
        assert_eq!(snap.len(), 25 + kept_coarse);
        assert_eq!(snap.normals.len(), snap.len());
        assert_eq!(snap.normals[snap.len() - 1], [0.0; 3]);
    }
}
