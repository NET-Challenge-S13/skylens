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

//! 짝 목록 기반 매칭 → 두 뷰 기하 검증 → FeatureStore 기록.

use crate::descriptor::{DescriptorMatchOptions, MatcherBackend};
use crate::pairs::{read_pair_list, PairList};
use crate::two_view::{derive_seed, estimate_two_view, finalize_geometry, TwoViewOptions};
use cumulus3d_core::{pair_id_of, Error, FeatureMatch, FeatureStore, ImageId, Result, TwoViewGeometry};
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;

/// 짝 매칭 전체 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct PairMatchingOptions {
    /// 기술자 매칭 옵션.
    pub sift: DescriptorMatchOptions,
    /// 두 뷰 기하 검증 옵션.
    pub geometry: TwoViewOptions,
    /// 실제 사용값 = min(이 값, 저장소 영상별 키포인트 수 최댓값).
    pub max_num_matches: usize,
    /// 한 번에 처리하는 짝 수.
    pub block_size: usize,
    /// 참이면 원시 매칭만 기록(기하 검증 생략).
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

/// 처리 통계.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MatchingStats {
    /// 이미 매칭·기하가 모두 있어 건너뛴 짝.
    pub num_skipped_existing: usize,
    /// 기술자 매칭을 수행한 짝.
    pub num_matched: usize,
    /// 기존 원시 매칭으로 검증만 한 짝.
    pub num_verified_only: usize,
    /// 인라이어 ≥ min_num_inliers 로 유효 기하가 기록된 짝.
    pub num_valid_geometries: usize,
}

/// (id1, id2, 원시 매칭, 기하, 기술자 매칭 수행 여부).
type PairResult = (ImageId, ImageId, Vec<FeatureMatch>, Option<TwoViewGeometry>, bool);

struct Task {
    id1: ImageId,
    id2: ImageId,
    pre: Option<Vec<FeatureMatch>>,
}

/// 짝 목록을 매칭해 저장소에 기록한다. 위치가 도착할 때마다 누적 짝 목록을 넘겨도
/// 이미 처리된 짝(원시 매칭과 기하가 모두 있는 짝)은 비용 없이 건너뛴다.
pub fn match_pairs(
    store: &FeatureStore,
    pairs: &[(ImageId, ImageId)],
    opts: &PairMatchingOptions,
    backend: &dyn MatcherBackend,
) -> Result<MatchingStats> {
    let mut stats = MatchingStats::default();
    let max_num_matches = opts.max_num_matches.min(store.largest_keypoint_count());
    let min_inl = opts.geometry.min_num_inliers;
    for block in pairs.chunks(opts.block_size.max(1)) {
        // 1) 블록 안 짝 처리 규칙.
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
        // 필요한 자료 확인.
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
        // 2) 짝 단위 병렬: 매칭 + 검증.
        let results: Vec<PairResult> = tasks
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
                let tvg = (!opts.skip_geometric_verification).then(|| verify_pair(store, t.id1, t.id2, &raw, &opts.geometry));
                (t.id1, t.id2, raw, tvg, matched)
            })
            .collect();
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

/// 짝 하나 검증(저장소에서 카메라·키포인트를 읽음). 최소 인라이어 규칙 적용 후 결과.
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

/// 짝 목록 파일을 읽어 매칭(`matches_importer --match_type pairs` 대응).
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
