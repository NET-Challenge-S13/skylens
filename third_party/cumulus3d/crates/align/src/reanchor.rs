/*
 * reanchor.rs
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

//! 재고정(reanchor): 새 정밀 모델이 나올 때마다 이전 구역들을 최신 좌표계로 다시 잇는 연쇄 Sim3.
//!
//! 구역 k 의 정밀 모델이 도착하면, 직전 최신 좌표계 → 새 좌표계 Sim3 `T_k` 를(공유 3D 점 정렬로) 구하고
//! 모든 이전 구역의 `zone_to_latest[j]` 앞에 `T_k` 를 합성한다. 새 구역 자신은 항등.

use crate::error::Result;
use crate::shared::{align_reconstructions, SharedPointOptions};
use crate::umeyama::{RobustUmeyamaOptions, RobustUmeyamaResult};
use cumulus3d_core::{Reconstruction, Sim3};

/// 구역별 "최신 좌표계 ← 구역 좌표계" 변환 묶음.
#[derive(Clone, Debug, Default)]
pub struct AnchorChain {
    zone_to_latest: Vec<Sim3>,
    /// 각 구역이 직전 구역과 연결됐는지(첫 구역은 false).
    linked: Vec<bool>,
}

impl AnchorChain {
    /// 빈 체인.
    pub fn new() -> Self {
        Self::default()
    }
    /// 등록된 구역 수.
    pub fn len(&self) -> usize {
        self.zone_to_latest.len()
    }
    /// 구역이 없으면 참.
    pub fn is_empty(&self) -> bool {
        self.zone_to_latest.is_empty()
    }

    /// 새 정밀 구역 추가. `new_from_prev` = 직전 최신 좌표계 → 새 좌표계.
    /// None(연결 실패)이면 이전 구역 변환은 그대로 두고 새 구역은 자기 좌표계(항등)로 둔다.
    /// 반환: 새 구역 인덱스.
    pub fn push(&mut self, new_from_prev: Option<Sim3>) -> usize {
        if let Some(t) = new_from_prev {
            for s in &mut self.zone_to_latest {
                *s = t.compose(s);
            }
        }
        self.zone_to_latest.push(Sim3::identity());
        self.linked.push(new_from_prev.is_some() && !self.linked.is_empty());
        self.zone_to_latest.len() - 1
    }

    /// 직전 정밀 모델 `prev` 와 새 정밀 모델 `new` 의 공유 3D 점으로 연결 Sim3 를 구해 추가.
    /// 정렬이 실패해도 구역은 추가되고(연결 없음) 오류를 돌려준다.
    pub fn push_model(
        &mut self,
        prev: Option<&Reconstruction>,
        new: &Reconstruction,
        shared: &SharedPointOptions,
        robust: &RobustUmeyamaOptions,
    ) -> Result<Option<RobustUmeyamaResult>> {
        let Some(prev) = prev else {
            self.push(None);
            return Ok(None);
        };
        match align_reconstructions(prev, new, shared, robust) {
            Ok((r, _)) => {
                self.push(Some(r.sim3));
                Ok(Some(r))
            }
            Err(e) => {
                self.push(None);
                Err(e)
            }
        }
    }

    /// 구역 `zone` 좌표 → 최신 좌표.
    pub fn zone_to_latest(&self, zone: usize) -> Option<Sim3> {
        self.zone_to_latest.get(zone).copied()
    }
    /// 구역 `zone` 이 직전 구역과 연결됐는지.
    pub fn is_linked(&self, zone: usize) -> bool {
        self.linked.get(zone).copied().unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::umeyama::tests::{rand_vec, random_sim3};
    use cumulus3d_core::Vec3;

    #[test]
    fn chain_maps_all_zones_to_latest() {
        let mut rng = cumulus3d_core::ransac::make_rng(Some(31));
        // 구역 k 좌표계 = Z_k ∘ 세계. 진짜 zone_j_to_latest = Z_last ∘ Z_j⁻¹.
        let zs: Vec<Sim3> = (0..4).map(|_| random_sim3(&mut rng)).collect();
        let mut chain = AnchorChain::new();
        chain.push(None);
        for k in 1..zs.len() {
            chain.push(Some(zs[k].compose(&zs[k - 1].inverse())));
        }
        let x = rand_vec(&mut rng, 10.0);
        let last = zs.last().unwrap();
        for (j, z) in zs.iter().enumerate() {
            let in_zone = z.transform_point(&x);
            let got = chain.zone_to_latest(j).unwrap().transform_point(&in_zone);
            let want = last.transform_point(&x);
            assert!((got - want).norm() / want.norm().max(1.0) < 1e-9, "zone {j}");
        }
        assert!(!chain.is_linked(0) && chain.is_linked(3));
        // 연결 실패: 이전 변환 유지, 새 구역 항등.
        let before = chain.zone_to_latest(1).unwrap();
        chain.push(None);
        assert_eq!(chain.zone_to_latest(1).unwrap(), before);
        assert_eq!(chain.zone_to_latest(4).unwrap().transform_point(&Vec3::x()), Vec3::x());
    }
}
