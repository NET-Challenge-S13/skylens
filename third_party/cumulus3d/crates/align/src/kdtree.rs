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

//! 3D KD-tree (own implementation). For proximity masking against the refined cloud. Queries are parallel with rayon.

use rayon::prelude::*;

const LEAF_SIZE: usize = 16;

#[derive(Clone, Copy, Debug)]
enum Node {
    Leaf { start: u32, end: u32 },
    Split { dim: u8, val: f64, left: u32, right: u32 },
}

/// Static KD-tree. Points are stored rearranged in tree order, together with their original indices.
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
    /// Creates from f64 points.
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
    /// Creates from f32 points (PLY).
    pub fn from_f32(points: &[[f32; 3]]) -> Self {
        Self::new(points.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]))
    }
    /// Number of points.
    pub fn len(&self) -> usize {
        self.pts.len()
    }
    /// True if there are no points.
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    /// Nearest point (original index, distance²). None for an empty tree.
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

    /// Whether any point lies within radius r (distance ≤ r) (early exit).
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

    /// All points within radius r (original index, order unspecified).
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

    /// Parallel nearest-neighbor query.
    pub fn nearest_many(&self, queries: &[[f64; 3]]) -> Vec<Option<(usize, f64)>> {
        queries.par_iter().map(|q| self.nearest(q)).collect()
    }

    /// Parallel radius test (f32 queries). `result[i]` = whether query i has a point within r.
    pub fn any_within_many_f32(&self, queries: &[[f32; 3]], r: f64) -> Vec<bool> {
        queries.par_iter().map(|q| self.any_within(&[q[0] as f64, q[1] as f64, q[2] as f64], r)).collect()
    }
}

/// Builds nodes for the idx range and returns the node index. `offset` = start position of the idx range within the whole.
fn build(pts: &[[f64; 3]], idx: &mut [u32], offset: usize, nodes: &mut Vec<Node>) -> u32 {
    let me = nodes.len() as u32;
    let n = idx.len();
    if n <= LEAF_SIZE {
        nodes.push(Node::Leaf { start: offset as u32, end: (offset + n) as u32 });
        return me;
    }
    // Split along the axis with the largest extent.
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
        // All points at the same position: cannot split further.
        nodes.push(Node::Leaf { start: offset as u32, end: (offset + n) as u32 });
        return me;
    }
    let mid = n / 2;
    idx.select_nth_unstable_by(mid, |a, b| pts[*a as usize][dim].total_cmp(&pts[*b as usize][dim]));
    let val = pts[idx[mid] as usize][dim];
    // Left ≤ val ≤ right. Pruning uses only the |q_d − val| lower bound, so it is exact even if equal values lie on both sides.
    nodes.push(Node::Leaf { start: 0, end: 0 }); // reserve a slot
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
            // Radius queries also match brute force.
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
