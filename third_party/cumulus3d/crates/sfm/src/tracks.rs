/*
 * tracks.rs
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

//! 트랙 구성·선별: 합집합-찾기 → 영상 내 일관성 검사 → 최소 뷰 수 → 길이순 선별.

use crate::rotation_averaging::ViewGraph;
use cumulus3d_core::graph::Correspondence;
use cumulus3d_core::{ImageId, MatchGraph, Point3D, Reconstruction, Result, TrackEntry, Vec2, Vec3};
use std::collections::{BTreeMap, HashSet};

/// 트랙 옵션.
#[derive(Clone, Debug)]
pub struct TrackOptions {
    /// 같은 영상 두 관측 사이 허용 픽셀 거리.
    pub intra_image_consistency_threshold: f64,
    /// 영상별 필요 트랙 수(기본 2^31−1 = 사실상 무제한).
    pub required_tracks_per_view: usize,
    /// 트랙의 서로 다른 영상 최소 수.
    pub min_num_views_per_track: usize,
    /// 채택할 최대 트랙 수.
    pub max_num_tracks: usize,
}

impl Default for TrackOptions {
    fn default() -> Self {
        Self {
            intra_image_consistency_threshold: 10.0,
            required_tracks_per_view: i32::MAX as usize,
            min_num_views_per_track: 3,
            max_num_tracks: i32::MAX as usize,
        }
    }
}

/// 트랙 구성 통계.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrackSummary {
    /// 후보 트랙 수.
    pub num_candidates: usize,
    /// 영상 내 일관성 검사로 버린 트랙 수.
    pub num_inconsistent: usize,
    /// 뷰 수 부족으로 버린 트랙 수.
    pub num_too_few_views: usize,
    /// 검사를 통과한 트랙 수.
    pub num_valid: usize,
    /// 최종 선별한 트랙 수.
    pub num_selected: usize,
}

/// 후보 트랙 → 일관성·최소 뷰 검사를 통과한 트랙(순서 유지, 인덱스 = 트랙 id).
pub fn filter_candidate_tracks(
    candidates: Vec<Vec<Correspondence>>,
    position: impl Fn(&Correspondence) -> Option<Vec2>,
    opts: &TrackOptions,
    summary: &mut TrackSummary,
) -> Vec<Vec<Correspondence>> {
    let thr2 = opts.intra_image_consistency_threshold.powi(2);
    let mut out = Vec::new();
    summary.num_candidates += candidates.len();
    'track: for t in candidates {
        let mut seen: BTreeMap<ImageId, Vec<Vec2>> = BTreeMap::new();
        for c in &t {
            let Some(xy) = position(c) else {
                summary.num_inconsistent += 1;
                continue 'track;
            };
            let prev = seen.entry(c.image_id).or_default();
            if prev.iter().any(|p| (p - xy).norm_squared() > thr2) {
                summary.num_inconsistent += 1;
                continue 'track;
            }
            prev.push(xy);
        }
        if seen.len() < opts.min_num_views_per_track {
            summary.num_too_few_views += 1;
            continue;
        }
        out.push(t);
    }
    summary.num_valid = out.len();
    out
}

/// 선별: (길이, id) 내림차순으로 상한까지. 반환: 채택된 트랙 id(채택 순).
pub fn select_tracks(tracks: &[Vec<Correspondence>], opts: &TrackOptions) -> Vec<usize> {
    let mut order: Vec<usize> = (0..tracks.len()).collect();
    order.sort_by(|&a, &b| (tracks[b].len(), b).cmp(&(tracks[a].len(), a)));
    let mut count: BTreeMap<ImageId, usize> = BTreeMap::new();
    let all_images: HashSet<ImageId> = tracks.iter().flatten().map(|c| c.image_id).collect();
    let mut satisfied: HashSet<ImageId> = HashSet::new();
    let mut out = Vec::new();
    for id in order {
        if out.len() >= opts.max_num_tracks {
            break;
        }
        let t = &tracks[id];
        let needed = t.iter().any(|c| count.get(&c.image_id).copied().unwrap_or(0) <= opts.required_tracks_per_view);
        if !needed {
            continue;
        }
        for c in t {
            let n = count.entry(c.image_id).or_insert(0);
            *n += 1;
            if *n > opts.required_tracks_per_view {
                satisfied.insert(c.image_id);
            }
        }
        out.push(id);
        if opts.required_tracks_per_view < i32::MAX as usize && satisfied.len() == all_images.len() {
            break;
        }
    }
    out
}

/// 등록 영상의 유효 간선으로 트랙을 만들어 재구성에 3D 점(좌표 0, error −1)으로 추가.
/// 점 id = 통과 트랙의 순번(0부터).
pub fn establish_tracks(rec: &mut Reconstruction, graph: &MatchGraph, view_graph: &ViewGraph, opts: &TrackOptions) -> Result<TrackSummary> {
    let valid: HashSet<(ImageId, ImageId)> = view_graph.edges.iter().filter(|e| e.valid).map(|e| (e.image_id1, e.image_id2)).collect();
    let registered: HashSet<ImageId> = rec.registered_images().into_iter().collect();
    let candidates = graph.union_find_tracks(|a, b| valid.contains(&(a, b)) && registered.contains(&a) && registered.contains(&b));
    let mut summary = TrackSummary::default();
    let tracks = filter_candidate_tracks(
        candidates,
        |c| rec.image(c.image_id).and_then(|im| im.points2d().get(c.point2d_idx as usize)).map(|p| p.xy),
        opts,
        &mut summary,
    );
    let selected = select_tracks(&tracks, opts);
    summary.num_selected = selected.len();
    for id in selected {
        let track: Vec<TrackEntry> = tracks[id].iter().map(|c| TrackEntry::new(c.image_id, c.point2d_idx)).collect();
        rec.add_point3d_with_id(id as u64, Point3D { xyz: Vec3::zeros(), color: [0, 0, 0], error: -1.0, track })?;
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(i: ImageId, p: u32) -> Correspondence {
        Correspondence::new(i, p)
    }

    #[test]
    fn consistency_and_min_views() {
        // 영상 A=1, B=2, C=3. A1–B1, B1–C1, C1–A2 (A 에 두 관측).
        let positions = |dist: f64| {
            move |c: &Correspondence| -> Option<Vec2> {
                Some(match (c.image_id, c.point2d_idx) {
                    (1, 1) => Vec2::new(100.0, 100.0),
                    (1, 2) => Vec2::new(100.0 + dist, 100.0),
                    _ => Vec2::new(50.0, 50.0),
                })
            }
        };
        let cand = vec![vec![c(1, 1), c(1, 2), c(2, 1), c(3, 1)], vec![c(4, 0), c(5, 0)]];
        let mut s = TrackSummary::default();
        let t = filter_candidate_tracks(cand.clone(), positions(5.0), &TrackOptions::default(), &mut s);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].len(), 4);
        assert_eq!(s.num_too_few_views, 1);
        let mut s = TrackSummary::default();
        let t = filter_candidate_tracks(cand, positions(15.0), &TrackOptions::default(), &mut s);
        assert!(t.is_empty());
        assert_eq!(s.num_inconsistent, 1);
    }

    #[test]
    fn keep_max_longest() {
        let tracks = vec![
            vec![c(1, 0), c(2, 0), c(3, 0)],
            vec![c(1, 1), c(2, 1), c(3, 1), c(4, 1), c(5, 1)],
            vec![c(1, 2), c(2, 2), c(3, 2), c(4, 2)],
            vec![c(1, 3), c(2, 3), c(3, 3), c(4, 3)],
        ];
        let o = TrackOptions { max_num_tracks: 2, ..Default::default() };
        // 길이 5 → 길이 4 중 id 큰 것(3) 먼저.
        assert_eq!(select_tracks(&tracks, &o), vec![1, 3]);
        assert_eq!(select_tracks(&tracks, &TrackOptions::default()), vec![1, 3, 2, 0]);
    }
}
