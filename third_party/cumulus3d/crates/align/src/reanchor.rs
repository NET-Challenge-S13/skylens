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

//! Reanchor: a chained Sim3 that reconnects earlier zones to the latest coordinate frame whenever a new refined model appears.
//!
//! When the refined model of zone k arrives, computes the Sim3 `T_k` from the previous latest frame to the new frame
//! (by shared 3D point alignment) and composes `T_k` in front of `zone_to_latest[j]` of every earlier zone.
//! The new zone itself is identity.

use crate::error::Result;
use crate::shared::{align_reconstructions, SharedPointOptions};
use crate::umeyama::{RobustUmeyamaOptions, RobustUmeyamaResult};
use cumulus3d_core::{Reconstruction, Sim3};

/// Bundle of per-zone "latest frame ← zone frame" transforms.
#[derive(Clone, Debug, Default)]
pub struct AnchorChain {
    zone_to_latest: Vec<Sim3>,
    /// Whether each zone is connected to the previous zone (false for the first zone).
    linked: Vec<bool>,
}

impl AnchorChain {
    /// Empty chain.
    pub fn new() -> Self {
        Self::default()
    }
    /// Number of registered zones.
    pub fn len(&self) -> usize {
        self.zone_to_latest.len()
    }
    /// True if there are no zones.
    pub fn is_empty(&self) -> bool {
        self.zone_to_latest.is_empty()
    }

    /// Adds a new refined zone. `new_from_prev` = previous latest frame → new frame.
    /// If None (connection failed), earlier zone transforms are left as they are and the new zone keeps its own frame (identity).
    /// Returns: the new zone index.
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

    /// Finds the connecting Sim3 from the shared 3D points of the previous refined model `prev`
    /// and the new refined model `new`, and adds it.
    /// Even if alignment fails, the zone is added (not connected) and an error is returned.
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

    /// Zone `zone` coordinates to latest coordinates.
    pub fn zone_to_latest(&self, zone: usize) -> Option<Sim3> {
        self.zone_to_latest.get(zone).copied()
    }
    /// Whether zone `zone` is connected to the previous zone.
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
        // Zone k frame = Z_k ∘ world. The true zone_j_to_latest = Z_last ∘ Z_j⁻¹.
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
        // Connection failed: keep earlier transforms, new zone is identity.
        let before = chain.zone_to_latest(1).unwrap();
        chain.push(None);
        assert_eq!(chain.zone_to_latest(1).unwrap(), before);
        assert_eq!(chain.zone_to_latest(4).unwrap().transform_point(&Vec3::x()), Vec3::x());
    }
}
