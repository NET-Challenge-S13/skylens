/*
 * pipeline.rs
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

//! Pair-list-based matching -> two-view geometry verification -> FeatureStore writes.

use crate::batched::{CpuTwoView, TwoViewBackend, VerifyJob};
use crate::descriptor::{DescriptorMatchOptions, MatcherBackend};
use crate::pairs::{read_pair_list, PairList};
use crate::two_view::{derive_seed, estimate_two_view, finalize_geometry, TwoViewOptions};
use cumulus3d_core::{pair_id_of, Error, FeatureMatch, FeatureStore, ImageId, Result, TwoViewGeometry};
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;

/// Full options for pair matching.
#[derive(Clone, Debug, PartialEq)]
pub struct PairMatchingOptions {
    /// Descriptor matching options.
    pub sift: DescriptorMatchOptions,
    /// Two-view geometry verification options.
    pub geometry: TwoViewOptions,
    /// Actual value used = min(this value, maximum per-image keypoint count in the store).
    pub max_num_matches: usize,
    /// Number of pairs processed at a time.
    pub block_size: usize,
    /// If true, record only raw matches (skip geometry verification).
    pub skip_geometric_verification: bool,
}

impl Default for PairMatchingOptions {
    fn default() -> Self {
        Self {
            sift: DescriptorMatchOptions::default(),
            geometry: TwoViewOptions::default(),
            max_num_matches: 32768,
            block_size: 1225,
            skip_geometric_verification: false,
        }
    }
}

/// Processing statistics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MatchingStats {
    /// Pairs skipped because both matches and geometry already exist.
    pub num_skipped_existing: usize,
    /// Pairs for which descriptor matching was performed.
    pub num_matched: usize,
    /// Pairs verified only, using existing raw matches.
    pub num_verified_only: usize,
    /// Pairs recorded with a valid geometry, with inliers >= min_num_inliers.
    pub num_valid_geometries: usize,
}

/// (id1, id2, raw matches, geometry, whether descriptor matching was performed).
type PairResult = (ImageId, ImageId, Vec<FeatureMatch>, Option<TwoViewGeometry>, bool);

struct Task {
    id1: ImageId,
    id2: ImageId,
    pre: Option<Vec<FeatureMatch>>,
}

/// Matches a pair list and records it in the store. Even if the accumulated pair list is passed each time a position arrives,
/// already processed pairs (pairs with both raw matches and geometry) are skipped at no cost.
pub fn match_pairs(
    store: &FeatureStore,
    pairs: &[(ImageId, ImageId)],
    opts: &PairMatchingOptions,
    backend: &dyn MatcherBackend,
) -> Result<MatchingStats> {
    match_pairs_with(store, pairs, opts, backend, &CpuTwoView)
}

/// [`match_pairs`] 와 같되 두 뷰 기하 검증을 `verifier` 로 한다(블록마다 짝 묶음 한 번).
pub fn match_pairs_with(
    store: &FeatureStore,
    pairs: &[(ImageId, ImageId)],
    opts: &PairMatchingOptions,
    backend: &dyn MatcherBackend,
    verifier: &dyn TwoViewBackend,
) -> Result<MatchingStats> {
    let mut stats = MatchingStats::default();
    let max_num_matches = opts.max_num_matches.min(store.largest_keypoint_count());
    let min_inl = opts.geometry.min_num_inliers;
    for block in pairs.chunks(opts.block_size.max(1)) {
        // 1) Pair handling rules within a block.
        let mut seen = HashSet::new();
        let mut tasks = Vec::new();
        for &(id1, id2) in block {
            if id1 == id2 {
                continue;
            }
            let pid = pair_id_of(id1, id2)?;
            if !seen.insert(pid) {
                continue;
            }
            let has_m = store.exists_matches(id1, id2);
            let has_g = store.contains_two_view(id1, id2);
            match (has_m, has_g) {
                (true, true) => {
                    stats.num_skipped_existing += 1;
                }
                (false, true) => {
                    store.remove_two_view(id1, id2);
                    tasks.push(Task { id1, id2, pre: None });
                }
                (true, false) => {
                    let m = store.read_matches(id1, id2).map(|m| m.to_vec()).unwrap_or_default();
                    store.delete_matches(id1, id2);
                    tasks.push(Task { id1, id2, pre: Some(m) });
                }
                (false, false) => tasks.push(Task { id1, id2, pre: None }),
            }
        }
        // Check the required data.
        for t in &tasks {
            for id in [t.id1, t.id2] {
                let img = store.image(id).ok_or_else(|| Error::NotFound(format!("영상 {id}")))?;
                if store.camera(img.camera_id).is_none() {
                    return Err(Error::NotFound(format!("카메라 {}", img.camera_id)));
                }
                if !store.exists_keypoints(id) {
                    return Err(Error::NotFound(format!("영상 {id} 키포인트")));
                }
                if t.pre.is_none() && !store.exists_descriptors(id) {
                    return Err(Error::NotFound(format!("영상 {id} 기술자")));
                }
            }
        }
        // 2) 짝 단위 병렬 매칭 → 블록 단위 검증.
        let raws: Vec<(ImageId, ImageId, Vec<FeatureMatch>, bool)> = tasks
            .into_par_iter()
            .map(|t| {
                let matched = t.pre.is_none();
                let raw = match t.pre {
                    Some(m) => m,
                    None => {
                        let d1 = store.descriptors(t.id1).expect("확인됨");
                        let d2 = store.descriptors(t.id2).expect("확인됨");
                        let m = backend.match_descriptors(&d1, &d2, &opts.sift, max_num_matches);
                        if m.len() < min_inl {
                            Vec::new()
                        } else {
                            m
                        }
                    }
                };
                (t.id1, t.id2, raw, matched)
            })
            .collect();
        let tvgs: Vec<Option<TwoViewGeometry>> = if opts.skip_geometric_verification {
            raws.iter().map(|_| None).collect()
        } else {
            let jobs: Vec<VerifyJob<'_>> = raws.iter().map(|(a, b, m, _)| VerifyJob { id1: *a, id2: *b, matches: m }).collect();
            verifier.verify_pairs(store, &jobs, &opts.geometry).into_iter().map(Some).collect()
        };
        let results: Vec<PairResult> =
            raws.into_iter().zip(tvgs).map(|((id1, id2, raw, matched), tvg)| (id1, id2, raw, tvg, matched)).collect();
        // 3) 블록 단위 기록(입력 순서).
        for (id1, id2, raw, tvg, matched) in results {
            if matched {
                stats.num_matched += 1;
            } else {
                stats.num_verified_only += 1;
            }
            store.write_matches(id1, id2, &raw)?;
            if let Some(g) = tvg {
                if !g.inlier_matches.is_empty() {
                    stats.num_valid_geometries += 1;
                }
                store.put_two_view(id1, id2, &g)?;
            }
        }
    }
    Ok(stats)
}

/// Verifies one pair (reads camera and keypoints from the store). Result after applying the minimum-inlier rule.
pub fn verify_pair(store: &FeatureStore, id1: ImageId, id2: ImageId, matches: &[FeatureMatch], opts: &TwoViewOptions) -> TwoViewGeometry {
    if matches.len() < opts.min_num_inliers {
        return TwoViewGeometry::default();
    }
    let (Some(i1), Some(i2)) = (store.image(id1), store.image(id2)) else { return TwoViewGeometry::default() };
    let (Some(c1), Some(c2)) = (store.camera(i1.camera_id), store.camera(i2.camera_id)) else {
        return TwoViewGeometry::default();
    };
    let (Some(k1), Some(k2)) = (store.keypoints(id1), store.keypoints(id2)) else { return TwoViewGeometry::default() };
    let mut o = opts.clone();
    if let Ok(pid) = pair_id_of(id1, id2) {
        o.ransac.random_seed = derive_seed(opts.ransac.random_seed, pid);
    }
    let g = estimate_two_view(&c1, &k1, &c2, &k2, matches, &o);
    finalize_geometry(g, opts.min_num_inliers)
}

/// Reads a pair list file and matches (corresponds to `matches_importer --match_type pairs`).
pub fn match_pair_list_file(
    store: &FeatureStore,
    path: impl AsRef<Path>,
    opts: &PairMatchingOptions,
    backend: &dyn MatcherBackend,
) -> Result<(PairList, MatchingStats)> {
    let list = read_pair_list(path, |n| store.image_id_by_name(n))?;
    let stats = match_pairs(store, &list.pairs, opts, backend)?;
    Ok((list, stats))
}
