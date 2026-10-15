/*
 * graph.rs
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

//! 대응 그래프.
//!
//! 영상과 짝을 증분으로 추가할 수 있어, 새 영상이 들어올 때 전체를 다시 구성할 필요가 없다.
//! 2D 점별 대응은 작은 Vec 로 보관한다(평탄화 생략, 의미 동일).

use crate::features::{FeatureMatch, TwoViewGeometry, TwoViewGeometryConfig};
use crate::ids::*;
use crate::store::FeatureStore;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

/// 대응 하나: (영상 id, 2D 점 인덱스).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Correspondence {
    /// 영상 id.
    pub image_id: ImageId,
    /// 2D 점 인덱스.
    pub point2d_idx: Point2DIdx,
}

impl Correspondence {
    /// (영상, 2D 점)으로 생성.
    pub fn new(image_id: ImageId, point2d_idx: Point2DIdx) -> Self {
        Self { image_id, point2d_idx }
    }
}

#[derive(Clone, Debug, Default)]
struct GraphImage {
    /// 대응이 하나라도 있는 2D 점 수.
    num_observations: usize,
    /// 짝 매칭 수 합.
    num_correspondences: usize,
    corrs: Vec<Vec<Correspondence>>,
}

#[derive(Clone, Debug)]
struct GraphPair {
    num_correspondences: usize,
    /// 작은 id → 큰 id 방향, 인라이어 목록 제외.
    geometry: TwoViewGeometry,
}

/// 저장소로부터 구성할 때의 옵션.
#[derive(Clone, Debug)]
pub struct MatchGraphOptions {
    /// 짝으로 쓰기 위한 최소 인라이어 매칭 수.
    pub min_num_matches: usize,
    /// 워터마크 짝을 무시할지.
    pub ignore_watermarks: bool,
    /// 비면 전체 영상.
    pub image_names: HashSet<String>,
    /// 거짓이면 "사용 가능한 짝"으로 연결된 영상만 적재.
    pub keep_all_images: bool,
}

impl Default for MatchGraphOptions {
    fn default() -> Self {
        Self { min_num_matches: 15, ignore_watermarks: false, image_names: HashSet::new(), keep_all_images: false }
    }
}

/// 짝 추가 결과 통계(경고 수).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AddPairStats {
    /// 추가한 대응 수.
    pub num_added: usize,
    /// 인덱스가 범위를 벗어나 버린 매칭 수.
    pub num_out_of_range: usize,
    /// 중복이라 버린 매칭 수.
    pub num_duplicates: usize,
    /// 자기 자신과의 짝이었는지.
    pub self_pair: bool,
}

/// 대응 그래프.
#[derive(Clone, Debug, Default)]
pub struct MatchGraph {
    images: HashMap<ImageId, GraphImage>,
    pairs: HashMap<PairId, GraphPair>,
    /// 증분 갱신용 저장소 로그 위치.
    store_cursor: usize,
}

impl MatchGraph {
    /// 빈 그래프.
    pub fn new() -> Self {
        Self::default()
    }

    /// 노드 추가(2D 점 수만큼 빈 대응 목록). 이미 있으면 무시.
    pub fn add_image(&mut self, image_id: ImageId, num_points2d: usize) {
        self.images.entry(image_id).or_insert_with(|| GraphImage { corrs: vec![Vec::new(); num_points2d], ..Default::default() });
    }
    /// 영상 노드가 있는지.
    pub fn exists_image(&self, image_id: ImageId) -> bool {
        self.images.contains_key(&image_id)
    }
    /// 영상 노드 수.
    pub fn num_images(&self) -> usize {
        self.images.len()
    }
    /// 짝 수.
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }
    /// 영상 id 목록(오름차순).
    pub fn image_ids(&self) -> Vec<ImageId> {
        let mut v: Vec<_> = self.images.keys().copied().collect();
        v.sort();
        v
    }
    /// 영상의 2D 점 수.
    pub fn num_points2d(&self, image_id: ImageId) -> usize {
        self.images.get(&image_id).map_or(0, |i| i.corrs.len())
    }
    /// 대응이 하나라도 있는 2D 점 수.
    pub fn observation_count_of(&self, image_id: ImageId) -> usize {
        self.images.get(&image_id).map_or(0, |i| i.num_observations)
    }
    /// 영상이 속한 짝들의 매칭 수 합.
    pub fn match_count_of(&self, image_id: ImageId) -> usize {
        self.images.get(&image_id).map_or(0, |i| i.num_correspondences)
    }
    /// 두 영상 사이 매칭 수.
    pub fn match_count_between(&self, id1: ImageId, id2: ImageId) -> usize {
        pair_id_of(id1, id2).ok().and_then(|p| self.pairs.get(&p)).map_or(0, |p| p.num_correspondences)
    }
    /// 두 영상 사이 짝이 있는지.
    pub fn exists_image_pair(&self, id1: ImageId, id2: ImageId) -> bool {
        pair_id_of(id1, id2).is_ok_and(|p| self.pairs.contains_key(&p))
    }
    /// (짝 id, 매칭 수), 짝 id 오름차순.
    pub fn image_pairs(&self) -> Vec<(PairId, usize)> {
        let mut v: Vec<_> = self.pairs.iter().map(|(k, p)| (*k, p.num_correspondences)).collect();
        v.sort();
        v
    }

    /// 짝 추가. 매칭은 (id1 의 인덱스, id2 의 인덱스).
    /// 두 영상 노드가 있어야 한다(없으면 아무것도 안 함). 기하는 작은 id → 큰 id 방향으로 저장.
    pub fn insert_two_view(&mut self, id1: ImageId, id2: ImageId, tvg: &TwoViewGeometry) -> AddPairStats {
        let mut st = AddPairStats::default();
        if id1 == id2 {
            st.self_pair = true;
            return st;
        }
        let Ok(pid) = pair_id_of(id1, id2) else { return st };
        if !self.images.contains_key(&id1) || !self.images.contains_key(&id2) || self.pairs.contains_key(&pid) {
            return st;
        }
        let n1 = self.images[&id1].corrs.len();
        let n2 = self.images[&id2].corrs.len();
        let mut count = 0usize;
        for m in &tvg.inlier_matches {
            if m.idx1 as usize >= n1 || m.idx2 as usize >= n2 {
                st.num_out_of_range += 1;
                continue;
            }
            let c2 = Correspondence::new(id2, m.idx2);
            {
                let im1 = self.images.get_mut(&id1).expect("존재");
                let list = &mut im1.corrs[m.idx1 as usize];
                if list.contains(&c2) {
                    st.num_duplicates += 1;
                    continue;
                }
                if list.is_empty() {
                    im1.num_observations += 1;
                }
                list.push(c2);
            }
            {
                let im2 = self.images.get_mut(&id2).expect("존재");
                let list = &mut im2.corrs[m.idx2 as usize];
                if list.is_empty() {
                    im2.num_observations += 1;
                }
                list.push(Correspondence::new(id1, m.idx1));
            }
            count += 1;
        }
        self.images.get_mut(&id1).expect("존재").num_correspondences += count;
        self.images.get_mut(&id2).expect("존재").num_correspondences += count;
        let mut geometry = TwoViewGeometry { inlier_matches: Vec::new(), ..tvg.clone() };
        if swap_image_pair(id1, id2) {
            geometry.invert();
        }
        self.pairs.insert(pid, GraphPair { num_correspondences: count, geometry });
        st.num_added = count;
        st
    }

    fn usable(tvg: &TwoViewGeometry, opts: &MatchGraphOptions) -> bool {
        tvg.inlier_matches.len() >= opts.min_num_matches && !(opts.ignore_watermarks && tvg.config == TwoViewGeometryConfig::Watermark)
    }

    /// 저장소에서 구성.
    pub fn from_store(store: &FeatureStore, opts: &MatchGraphOptions) -> Self {
        let mut g = Self::new();
        g.update_from_store(store, opts);
        g
    }

    /// 저장소에 새로 기록된 짝만 반영(증분). 새 영상 노드도 필요 시 추가.
    /// 반환: 새로 추가된 짝 수.
    pub fn update_from_store(&mut self, store: &FeatureStore, opts: &MatchGraphOptions) -> usize {
        let (new_pairs, cursor) = store.two_view_geometries_since(self.store_cursor);
        self.store_cursor = cursor;
        let name_ok =
            |id: ImageId| -> bool { opts.image_names.is_empty() || store.image(id).is_some_and(|im| opts.image_names.contains(&im.name)) };
        if opts.keep_all_images {
            for im in store.images() {
                if name_ok(im.image_id) {
                    self.add_image(im.image_id, store.num_keypoints(im.image_id));
                }
            }
        }
        // 짝 id 순서로 결정적 처리, 중복 기록 제거.
        let mut by_pid: BTreeMap<PairId, _> = BTreeMap::new();
        for (pid, t) in new_pairs {
            by_pid.insert(pid, t);
        }
        let mut added = 0;
        for (pid, t) in by_pid {
            if self.pairs.contains_key(&pid) || !Self::usable(&t, opts) {
                continue;
            }
            let (a, b) = images_of_pair(pid);
            if !name_ok(a) || !name_ok(b) || a == b {
                continue;
            }
            for id in [a, b] {
                if !self.images.contains_key(&id) {
                    self.add_image(id, store.num_keypoints(id));
                }
            }
            self.insert_two_view(a, b, &t);
            added += 1;
        }
        added
    }

    /// (영상, 2D 점)의 직접 대응.
    pub fn find_correspondences(&self, image_id: ImageId, point2d_idx: Point2DIdx) -> &[Correspondence] {
        self.images.get(&image_id).and_then(|i| i.corrs.get(point2d_idx as usize)).map_or(&[], |v| v.as_slice())
    }
    /// (영상, 2D 점)에 대응이 있는지.
    pub fn has_correspondences(&self, image_id: ImageId, point2d_idx: Point2DIdx) -> bool {
        !self.find_correspondences(image_id, point2d_idx).is_empty()
    }
    /// 두 뷰 관측: 대응이 정확히 1개이고 상대도 정확히 1개.
    pub fn in_two_view_track(&self, image_id: ImageId, point2d_idx: Point2DIdx) -> bool {
        let c = self.find_correspondences(image_id, point2d_idx);
        c.len() == 1 && self.find_correspondences(c[0].image_id, c[0].point2d_idx).len() == 1
    }
    /// 두 영상 사이 매칭: 영상1 2D 점 오름차순.
    pub fn matches_between(&self, id1: ImageId, id2: ImageId) -> Vec<FeatureMatch> {
        let mut out = Vec::new();
        if let Some(im) = self.images.get(&id1) {
            for (k, list) in im.corrs.iter().enumerate() {
                for c in list {
                    if c.image_id == id2 {
                        out.push(FeatureMatch::new(k as u32, c.point2d_idx));
                    }
                }
            }
        }
        out
    }
    /// 추이적 대응(너비 우선, 깊이 t). 시작 관측과 이미 본 관측 제외.
    pub fn transitive_matches(&self, image_id: ImageId, point2d_idx: Point2DIdx, depth: usize) -> Vec<Correspondence> {
        let start = Correspondence::new(image_id, point2d_idx);
        let mut seen: HashSet<Correspondence> = HashSet::from([start]);
        let mut out = Vec::new();
        let mut q = VecDeque::from([(start, 0usize)]);
        while let Some((c, d)) = q.pop_front() {
            if d >= depth {
                continue;
            }
            for n in self.find_correspondences(c.image_id, c.point2d_idx) {
                if seen.insert(*n) {
                    out.push(*n);
                    q.push_back((*n, d + 1));
                }
            }
        }
        out
    }
    /// (id1 → id2) 방향 두 뷰 기하(인라이어 제외).
    pub fn two_view_geometry(&self, id1: ImageId, id2: ImageId) -> Option<TwoViewGeometry> {
        let pid = pair_id_of(id1, id2).ok()?;
        let g = &self.pairs.get(&pid)?.geometry;
        Some(if swap_image_pair(id1, id2) { g.inverted() } else { g.clone() })
    }
    /// (id1 → id2) 방향 기하로 갱신.
    pub fn replace_two_view(&mut self, id1: ImageId, id2: ImageId, tvg: &TwoViewGeometry) -> bool {
        let Ok(pid) = pair_id_of(id1, id2) else { return false };
        let Some(p) = self.pairs.get_mut(&pid) else { return false };
        let mut g = TwoViewGeometry { inlier_matches: Vec::new(), ..tvg.clone() };
        if swap_image_pair(id1, id2) {
            g.invert();
        }
        p.geometry = g;
        true
    }

    /// 합집합-찾기 트랙(연결 성분). `pair_filter(작은 id, 큰 id)` 가 참인 짝만 사용.
    /// 각 성분은 (영상, 점) 오름차순, 성분 목록은 첫 원소 오름차순. 길이 1 성분은 없다.
    pub fn union_find_tracks<F>(&self, pair_filter: F) -> Vec<Vec<Correspondence>>
    where
        F: Fn(ImageId, ImageId) -> bool,
    {
        let mut index: HashMap<Correspondence, usize> = HashMap::new();
        let mut nodes: Vec<Correspondence> = Vec::new();
        let mut parent: Vec<usize> = Vec::new();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let mut get = |c: Correspondence, nodes: &mut Vec<Correspondence>, parent: &mut Vec<usize>| -> usize {
            *index.entry(c).or_insert_with(|| {
                nodes.push(c);
                parent.push(parent.len());
                parent.len() - 1
            })
        };
        let mut pairs: Vec<PairId> = self.pairs.keys().copied().collect();
        pairs.sort();
        for pid in pairs {
            let (a, b) = images_of_pair(pid);
            if !pair_filter(a, b) {
                continue;
            }
            for m in self.matches_between(a, b) {
                let x = get(Correspondence::new(a, m.idx1), &mut nodes, &mut parent);
                let y = get(Correspondence::new(b, m.idx2), &mut nodes, &mut parent);
                let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
                if rx != ry {
                    // 사전식으로 큰 쪽 루트를 작은 쪽 아래로.
                    if nodes[rx] < nodes[ry] {
                        parent[ry] = rx;
                    } else {
                        parent[rx] = ry;
                    }
                }
            }
        }
        let mut comps: BTreeMap<usize, Vec<Correspondence>> = BTreeMap::new();
        for (i, node) in nodes.iter().enumerate() {
            let r = find(&mut parent, i);
            comps.entry(r).or_default().push(*node);
        }
        let mut out: Vec<Vec<Correspondence>> = comps
            .into_values()
            .filter(|v| v.len() > 1)
            .map(|mut v| {
                v.sort();
                v
            })
            .collect();
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Camera, CameraModelKind};
    use crate::features::Keypoint;
    use crate::geometry::{Quat, Rigid3, Vec3};

    fn tvg(m: &[(u32, u32)]) -> TwoViewGeometry {
        TwoViewGeometry {
            config: TwoViewGeometryConfig::Calibrated,
            inlier_matches: m.iter().map(|&(a, b)| FeatureMatch::new(a, b)).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn add_pair_rules() {
        let mut g = MatchGraph::new();
        g.add_image(1, 10);
        g.add_image(2, 10);
        g.add_image(3, 10);
        let st = g.insert_two_view(1, 2, &tvg(&[(0, 0), (1, 1), (1, 1), (20, 2), (2, 30), (3, 3)]));
        assert_eq!(st.num_added, 3);
        assert_eq!(st.num_duplicates, 1);
        assert_eq!(st.num_out_of_range, 2);
        assert!(g.insert_two_view(3, 3, &tvg(&[(0, 0)])).self_pair);
        assert_eq!(g.match_count_between(2, 1), 3);
        assert_eq!(g.observation_count_of(1), 3);
        assert_eq!(g.match_count_of(2), 3);
        // reversed direction pair
        g.insert_two_view(3, 2, &tvg(&[(5, 0), (6, 1)]));
        assert_eq!(g.observation_count_of(2), 3);
        assert_eq!(g.match_count_of(2), 5);
        assert_eq!(g.matches_between(2, 3), vec![FeatureMatch::new(0, 5), FeatureMatch::new(1, 6)]);
        assert!(g.in_two_view_track(1, 3));
        assert!(!g.in_two_view_track(1, 0)); // 2:0 has two corrs
        let t = g.transitive_matches(1, 0, 2);
        assert_eq!(t, vec![Correspondence::new(2, 0), Correspondence::new(3, 5)]);
        assert_eq!(g.transitive_matches(1, 0, 1).len(), 1);
    }

    #[test]
    fn geometry_direction() {
        let mut g = MatchGraph::new();
        g.add_image(1, 1);
        g.add_image(2, 1);
        let pose = Rigid3::new(Quat::from_axis_angle(&Vec3::new(1.0, 2.0, 3.0), 0.7), Vec3::new(0.3, -0.2, 1.0));
        let t = TwoViewGeometry { cam1_to_cam2: Some(pose), ..tvg(&[(0, 0)]) };
        g.insert_two_view(2, 1, &t);
        let back = g.two_view_geometry(2, 1).unwrap().cam1_to_cam2.unwrap();
        assert!(back.rotation.angular_distance(&pose.rotation) < 1e-12);
        let inv = g.two_view_geometry(1, 2).unwrap().cam1_to_cam2.unwrap();
        let id = inv * pose;
        assert!(id.rotation.angular_distance(&Quat::IDENTITY) < 1e-12);
        assert!(id.translation.norm() < 1e-12);
    }

    #[test]
    fn incremental_from_store_and_tracks() {
        let s = FeatureStore::new();
        let c = s.add_camera(Camera::from_focal(CameraModelKind::OpenCv, 100.0, 100, 100)).unwrap();
        let ids: Vec<_> = (0..4).map(|k| s.add_image(&format!("{k}.jpg"), c).unwrap()).collect();
        for &i in &ids {
            s.set_keypoints(i, vec![Keypoint::new(0.5, 0.5); 20]);
        }
        let full: Vec<(u32, u32)> = (0..15).map(|k| (k, k)).collect();
        s.put_two_view(ids[0], ids[1], &tvg(&full)).unwrap();
        s.put_two_view(ids[1], ids[2], &tvg(&full)).unwrap();
        s.put_two_view(ids[2], ids[3], &tvg(&full[..14])).unwrap(); // too few
        let opts = MatchGraphOptions::default();
        let mut g = MatchGraph::from_store(&s, &opts);
        assert_eq!(g.num_images(), 3);
        assert_eq!(g.pair_count(), 2);
        // incremental: new pair arrives
        s.put_two_view(ids[3], ids[0], &tvg(&full)).unwrap();
        assert_eq!(g.update_from_store(&s, &opts), 1);
        assert_eq!(g.update_from_store(&s, &opts), 0);
        assert_eq!(g.num_images(), 4);
        let tracks = g.union_find_tracks(|_, _| true);
        assert_eq!(tracks.len(), 15);
        assert!(tracks.iter().all(|t| t.len() == 4));
        let tracks = g.union_find_tracks(|a, b| !(a == ids[1] && b == ids[2]));
        assert!(tracks.iter().all(|t| t.len() == 3));
    }
}
