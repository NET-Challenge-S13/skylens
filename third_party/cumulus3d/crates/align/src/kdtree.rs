/*
 * kdtree.rs
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

//! 3차원 KD-트리(자체 구현). 정밀 점군 근접 마스킹용. 질의는 rayon 병렬.

use rayon::prelude::*;

const LEAF_SIZE: usize = 16;

#[derive(Clone, Copy, Debug)]
enum Node {
    Leaf { start: u32, end: u32 },
    Split { dim: u8, val: f64, left: u32, right: u32 },
}

/// 정적 KD-트리. 점은 트리 순서로 재배열해 저장하고 원래 인덱스를 함께 보관한다.
#[derive(Clone, Debug, Default)]
pub struct KdTree {
    pts: Vec<[f64; 3]>,
    orig: Vec<u32>,
    nodes: Vec<Node>,
}

#[inline]
fn d2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

impl KdTree {
    /// f64 점으로 생성.
    pub fn new(points: impl IntoIterator<Item = [f64; 3]>) -> Self {
        let pts: Vec<[f64; 3]> = points.into_iter().collect();
        assert!(pts.len() < u32::MAX as usize, "점이 너무 많음");
        let mut idx: Vec<u32> = (0..pts.len() as u32).collect();
        let mut nodes = Vec::new();
        if !pts.is_empty() {
            build(&pts, &mut idx, 0, &mut nodes);
        }
        let reordered = idx.iter().map(|&i| pts[i as usize]).collect();
        Self { pts: reordered, orig: idx, nodes }
    }
    /// f32 점(PLY)으로 생성.
    pub fn from_f32(points: &[[f32; 3]]) -> Self {
        Self::new(points.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]))
    }
    /// 점 개수.
    pub fn len(&self) -> usize {
        self.pts.len()
    }
    /// 점이 없으면 참.
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    /// 최근접 점 (원래 인덱스, 거리²). 빈 트리면 None.
    pub fn nearest(&self, q: &[f64; 3]) -> Option<(usize, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = (usize::MAX, f64::INFINITY);
        self.nearest_rec(0, q, &mut best);
        Some((self.orig[best.0] as usize, best.1))
    }

    fn nearest_rec(&self, node: u32, q: &[f64; 3], best: &mut (usize, f64)) {
        match self.nodes[node as usize] {
            Node::Leaf { start, end } => {
                for i in start as usize..end as usize {
                    let d = d2(&self.pts[i], q);
                    if d < best.1 {
                        *best = (i, d);
                    }
                }
            }
            Node::Split { dim, val, left, right } => {
                let diff = q[dim as usize] - val;
                let (near, far) = if diff < 0.0 { (left, right) } else { (right, left) };
                self.nearest_rec(near, q, best);
                if diff * diff < best.1 {
                    self.nearest_rec(far, q, best);
                }
            }
        }
    }

    /// 반경 r 안(거리 ≤ r)에 점이 하나라도 있는가(조기 종료).
    pub fn any_within(&self, q: &[f64; 3], r: f64) -> bool {
        !self.nodes.is_empty() && self.any_rec(0, q, r * r)
    }

    fn any_rec(&self, node: u32, q: &[f64; 3], r2: f64) -> bool {
        match self.nodes[node as usize] {
            Node::Leaf { start, end } => self.pts[start as usize..end as usize].iter().any(|p| d2(p, q) <= r2),
            Node::Split { dim, val, left, right } => {
                let diff = q[dim as usize] - val;
                let (near, far) = if diff < 0.0 { (left, right) } else { (right, left) };
                self.any_rec(near, q, r2) || (diff * diff <= r2 && self.any_rec(far, q, r2))
            }
        }
    }

    /// 반경 r 안의 모든 점(원래 인덱스, 순서 미정).
    pub fn within_radius(&self, q: &[f64; 3], r: f64) -> Vec<usize> {
        let mut out = Vec::new();
        if !self.nodes.is_empty() {
            self.radius_rec(0, q, r * r, &mut out);
        }
        out
    }

    fn radius_rec(&self, node: u32, q: &[f64; 3], r2: f64, out: &mut Vec<usize>) {
        match self.nodes[node as usize] {
            Node::Leaf { start, end } => {
                for i in start as usize..end as usize {
                    if d2(&self.pts[i], q) <= r2 {
                        out.push(self.orig[i] as usize);
                    }
                }
            }
            Node::Split { dim, val, left, right } => {
                let diff = q[dim as usize] - val;
                if diff < 0.0 || diff * diff <= r2 {
                    self.radius_rec(left, q, r2, out);
                }
                if diff >= 0.0 || diff * diff <= r2 {
                    self.radius_rec(right, q, r2, out);
                }
            }
        }
    }

    /// 병렬 최근접 질의.
    pub fn nearest_many(&self, queries: &[[f64; 3]]) -> Vec<Option<(usize, f64)>> {
        queries.par_iter().map(|q| self.nearest(q)).collect()
    }

    /// 병렬 반경 판정(f32 질의). `결과[i]` = 질의 i 의 r 안에 점이 있음.
    pub fn any_within_many_f32(&self, queries: &[[f32; 3]], r: f64) -> Vec<bool> {
        queries.par_iter().map(|q| self.any_within(&[q[0] as f64, q[1] as f64, q[2] as f64], r)).collect()
    }
}

/// idx 구간에 대해 노드를 만들고 그 노드 번호를 반환. `offset` = idx 구간의 전체 내 시작 위치.
fn build(pts: &[[f64; 3]], idx: &mut [u32], offset: usize, nodes: &mut Vec<Node>) -> u32 {
    let me = nodes.len() as u32;
    let n = idx.len();
    if n <= LEAF_SIZE {
        nodes.push(Node::Leaf { start: offset as u32, end: (offset + n) as u32 });
        return me;
    }
    // 범위가 가장 큰 축으로 분할.
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &i in idx.iter() {
        let p = &pts[i as usize];
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let dim = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap_or(0);
    if hi[dim] - lo[dim] <= 0.0 {
        // 모든 점이 같은 위치: 더 나눌 수 없음.
        nodes.push(Node::Leaf { start: offset as u32, end: (offset + n) as u32 });
        return me;
    }
    let mid = n / 2;
    idx.select_nth_unstable_by(mid, |a, b| pts[*a as usize][dim].total_cmp(&pts[*b as usize][dim]));
    let val = pts[idx[mid] as usize][dim];
    // 왼쪽 ≤ val ≤ 오른쪽. 가지치기는 |q_d − val| 하한만 쓰므로 같은 값이 양쪽에 있어도 정확하다.
    nodes.push(Node::Leaf { start: 0, end: 0 }); // 자리 확보
    let (l, r) = idx.split_at_mut(mid);
    let left = build(pts, l, offset, nodes);
    let right = build(pts, r, offset + mid, nodes);
    nodes[me as usize] = Node::Split { dim: dim as u8, val, left, right };
    me
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn nearest_matches_brute_force() {
        let mut rng = cumulus3d_core::ransac::make_rng(Some(21));
        let pts: Vec<[f64; 3]> =
            (0..10_000).map(|_| [rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0), rng.random_range(-5.0..5.0)]).collect();
        let tree = KdTree::new(pts.iter().copied());
        let qs: Vec<[f64; 3]> =
            (0..2000).map(|_| [rng.random_range(-60.0..60.0), rng.random_range(-60.0..60.0), rng.random_range(-10.0..10.0)]).collect();
        let got = tree.nearest_many(&qs);
        for (q, g) in qs.iter().zip(&got) {
            let (bi, bd) = pts.iter().enumerate().map(|(i, p)| (i, d2(p, q))).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            let (gi, gd) = g.unwrap();
            assert_eq!(gd, bd);
            assert!(gi == bi || d2(&pts[gi], q) == bd);
            // 반경 질의도 무차별 대입과 일치.
            let r = 3.0;
            let mut within = tree.within_radius(q, r);
            within.sort_unstable();
            let brute: Vec<usize> = pts.iter().enumerate().filter(|(_, p)| d2(p, q) <= r * r).map(|(i, _)| i).collect();
            assert_eq!(within, brute);
            assert_eq!(tree.any_within(q, r), !brute.is_empty());
        }
    }

    #[test]
    fn duplicates_and_empty() {
        let tree = KdTree::new(std::iter::repeat_n([1.0, 2.0, 3.0], 100));
        assert_eq!(tree.nearest(&[0.0, 0.0, 0.0]).unwrap().1, 14.0);
        assert!(KdTree::new(std::iter::empty()).nearest(&[0.0; 3]).is_none());
        assert!(!KdTree::default().any_within(&[0.0; 3], 1.0));
    }
}
