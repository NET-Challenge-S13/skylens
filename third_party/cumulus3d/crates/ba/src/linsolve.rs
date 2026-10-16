/*
 * linsolve.rs
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

//! Reduced camera system (Schur complement) storage and three solvers: dense Cholesky, sparse Cholesky (faer), PCG.

use crate::LinearSolverType;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMat};
use faer::{MatMut, Side};
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Upper-triangular (including diagonal) block-row storage of a block-symmetric matrix. Blocks are row-major.
pub(crate) struct BlockSym {
    pub dims: Vec<usize>,
    pub offs: Vec<usize>,
    pub n: usize,
    /// Entry range of block row a (first entry = diagonal).
    pub row_ptr: Vec<usize>,
    pub cols: Vec<u32>,
    pub val_off: Vec<usize>,
    /// Start of the value range of block row a (length = number of rows + 1).
    pub row_val: Vec<usize>,
    pub vals: Vec<f64>,
    /// Direct table (a, b) -> vals offset when the block count is moderate (otherwise binary search).
    table: Vec<u32>,
}

impl BlockSym {
    /// `row_cols[a]` is the sorted, deduplicated column blocks (>= a, including a).
    pub fn new(dims: Vec<usize>, row_cols: Vec<Vec<u32>>) -> Self {
        let mut offs = Vec::with_capacity(dims.len());
        let mut n = 0;
        for &d in &dims {
            offs.push(n);
            n += d;
        }
        let mut row_ptr = vec![0];
        let mut cols = Vec::new();
        let mut val_off = Vec::new();
        let mut row_val = vec![0];
        let mut v = 0;
        for (a, rc) in row_cols.iter().enumerate() {
            debug_assert!(rc.first() == Some(&(a as u32)));
            for &b in rc {
                cols.push(b);
                val_off.push(v);
                v += dims[a] * dims[b as usize];
            }
            row_ptr.push(cols.len());
            row_val.push(v);
        }
        let nb = dims.len();
        let mut table = Vec::new();
        if nb <= 2048 && v < u32::MAX as usize {
            table = vec![u32::MAX; nb * nb];
            for a in 0..nb {
                for e in row_ptr[a]..row_ptr[a + 1] {
                    table[a * nb + cols[e] as usize] = val_off[e] as u32;
                }
            }
        }
        Self { dims, offs, n, row_ptr, cols, val_off, row_val, vals: vec![0.0; v], table }
    }

    pub fn num_rows(&self) -> usize {
        self.dims.len()
    }

    /// Absolute vals offset of block (a, b >= a).
    #[inline]
    pub fn offset(&self, a: usize, b: u32) -> usize {
        if !self.table.is_empty() {
            let o = self.table[a * self.dims.len() + b as usize];
            debug_assert!(o != u32::MAX, "S 구조에 없는 블록");
            return o as usize;
        }
        let r = self.row_ptr[a]..self.row_ptr[a + 1];
        let i = self.cols[r.clone()].binary_search(&b).expect("S 구조에 없는 블록");
        self.val_off[r.start + i]
    }

    pub fn to_dense(&self) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(self.n, self.n);
        for a in 0..self.num_rows() {
            let da = self.dims[a];
            for e in self.row_ptr[a]..self.row_ptr[a + 1] {
                let b = self.cols[e] as usize;
                let db = self.dims[b];
                let blk = &self.vals[self.val_off[e]..self.val_off[e] + da * db];
                for r in 0..da {
                    for c in 0..db {
                        let v = blk[r * db + c];
                        m[(self.offs[a] + r, self.offs[b] + c)] = v;
                        m[(self.offs[b] + c, self.offs[a] + r)] = v;
                    }
                }
            }
        }
        m
    }
}

/// Solver state (the structure-dependent part is built only once).
pub(crate) enum SchurSolver {
    Dense,
    Sparse(Box<SparseState>),
    Iterative(Box<IterState>),
}

pub(crate) struct SparseState {
    sym: SymbolicSparseColMat<usize>,
    symbolic: SymbolicLlt<usize>,
    gather: Vec<usize>,
    values: Vec<f64>,
}

pub(crate) struct IterState {
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    gather: Vec<usize>,
    values: Vec<f64>,
    max_iter: usize,
}

/// Lists upper-triangular scalar entries in (column = scalar row) order: (row i, column j >= i, vals index).
fn upper_scalar_entries(s: &BlockSym, mut f: impl FnMut(usize, usize, usize)) {
    for a in 0..s.num_rows() {
        let da = s.dims[a];
        for r in 0..da {
            let i = s.offs[a] + r;
            for e in s.row_ptr[a]..s.row_ptr[a + 1] {
                let b = s.cols[e] as usize;
                let db = s.dims[b];
                let c0 = if b == a { r } else { 0 };
                for c in c0..db {
                    f(i, s.offs[b] + c, s.val_off[e] + r * db + c);
                }
            }
        }
    }
}

impl SchurSolver {
    pub fn new(kind: LinearSolverType, s: &BlockSym, max_iter: usize) -> Option<Self> {
        Some(match kind {
            LinearSolverType::DenseSchur | LinearSolverType::Auto => SchurSolver::Dense,
            LinearSolverType::SparseSchur => {
                // Upper-triangular row = lower-triangular column (CSC, Side::Lower).
                let mut col_ptr = vec![0usize; s.n + 1];
                let mut row_idx = Vec::new();
                let mut gather = Vec::new();
                upper_scalar_entries(s, |i, j, g| {
                    col_ptr[i + 1] += 1;
                    row_idx.push(j);
                    gather.push(g);
                });
                for i in 0..s.n {
                    col_ptr[i + 1] += col_ptr[i];
                }
                let sym = SymbolicSparseColMat::new_checked(s.n, s.n, col_ptr, None, row_idx);
                let symbolic = SymbolicLlt::try_new(sym.as_ref(), Side::Lower).ok()?;
                let values = vec![0.0; gather.len()];
                SchurSolver::Sparse(Box::new(SparseState { sym, symbolic, gather, values }))
            }
            LinearSolverType::IterativeSchur => {
                // Full symmetric CSR (for row-parallel matrix-vector products).
                let mut rows: Vec<Vec<(u32, usize)>> = vec![Vec::new(); s.n];
                upper_scalar_entries(s, |i, j, g| {
                    rows[i].push((j as u32, g));
                    if j != i {
                        rows[j].push((i as u32, g));
                    }
                });
                let mut row_ptr = vec![0usize];
                let mut col_idx = Vec::new();
                let mut gather = Vec::new();
                for mut r in rows {
                    r.sort_unstable_by_key(|x| x.0);
                    for (j, g) in r {
                        col_idx.push(j);
                        gather.push(g);
                    }
                    row_ptr.push(col_idx.len());
                }
                let values = vec![0.0; gather.len()];
                SchurSolver::Iterative(Box::new(IterState { row_ptr, col_idx, gather, values, max_iter }))
            }
        })
    }

    /// S x = rhs. None on failure (not positive definite or non-finite).
    pub fn solve(&mut self, s: &BlockSym, rhs: &[f64]) -> Option<Vec<f64>> {
        if s.n == 0 {
            return Some(Vec::new());
        }
        let x = match self {
            SchurSolver::Dense => {
                let m = s.to_dense();
                let chol = m.cholesky()?;
                let b = nalgebra::DVector::from_column_slice(rhs);
                chol.solve(&b).as_slice().to_vec()
            }
            SchurSolver::Sparse(st) => {
                let SparseState { sym, symbolic, gather, values } = &mut **st;
                values.par_iter_mut().zip(gather.par_iter()).for_each(|(v, &g)| *v = s.vals[g]);
                let mat = SparseColMatRef::new(sym.as_ref(), values);
                let llt = Llt::try_new_with_symbolic(symbolic.clone(), mat, Side::Lower).ok()?;
                let mut x = rhs.to_vec();
                llt.solve_in_place(MatMut::from_column_major_slice_mut(&mut x, s.n, 1));
                x
            }
            SchurSolver::Iterative(st) => {
                let IterState { row_ptr, col_idx, gather, values, max_iter } = &mut **st;
                values.par_iter_mut().zip(gather.par_iter()).for_each(|(v, &g)| *v = s.vals[g]);
                pcg(s, row_ptr, col_idx, values, rhs, *max_iter)?
            }
        };
        x.iter().all(|v| v.is_finite()).then_some(x)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    // Deterministic sum: add fixed-size chunk sums in order.
    let parts: Vec<f64> = a.par_chunks(4096).zip(b.par_chunks(4096)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum()).collect();
    parts.iter().sum()
}

/// Block-Jacobi (Schur diagonal blocks) preconditioned conjugate gradient.
/// Termination: quadratic-model relative decrease (Nash-Sofer, η = 0.1, library LM default) or residual <= 1e-12‖b‖ or iteration cap.
fn pcg(s: &BlockSym, row_ptr: &[usize], col_idx: &[u32], vals: &[f64], b: &[f64], max_iter: usize) -> Option<Vec<f64>> {
    const ETA: f64 = 0.1;
    let n = s.n;
    // Diagonal block inverses.
    let pinv: Vec<DMatrix<f64>> = (0..s.num_rows())
        .into_par_iter()
        .map(|a| {
            let d = s.dims[a];
            let e = s.row_ptr[a];
            let blk = &s.vals[s.val_off[e]..s.val_off[e] + d * d];
            let m = DMatrix::from_row_slice(d, d, blk);
            m.cholesky().map(|c| c.inverse()).unwrap_or_else(|| DMatrix::identity(d, d))
        })
        .collect();
    let precond = |r: &[f64], z: &mut [f64]| {
        for a in 0..s.num_rows() {
            let (o, d) = (s.offs[a], s.dims[a]);
            let p = &pinv[a];
            for i in 0..d {
                let mut acc = 0.0;
                for j in 0..d {
                    acc += p[(i, j)] * r[o + j];
                }
                z[o + i] = acc;
            }
        }
    };
    let matvec = |x: &[f64], y: &mut [f64]| {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let mut acc = 0.0;
            for k in row_ptr[i]..row_ptr[i + 1] {
                acc += vals[k] * x[col_idx[k] as usize];
            }
            *yi = acc;
        });
    };
    let bnorm = dot(b, b).sqrt();
    let mut x = vec![0.0; n];
    if bnorm == 0.0 {
        return Some(x);
    }
    let mut r = b.to_vec();
    let mut z = vec![0.0; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0; n];
    let mut q0 = 0.0; // Q(x) = ½xᵀAx − bᵀx = −½ xᵀ(b + r)
    for it in 1..=max_iter.max(1) {
        matvec(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap.is_nan() || pap <= 0.0 || !rz.is_finite() {
            if it == 1 {
                return None;
            }
            break;
        }
        let alpha = rz / pap;
        x.iter_mut().zip(&p).for_each(|(xi, pi)| *xi += alpha * pi);
        r.iter_mut().zip(&ap).for_each(|(ri, api)| *ri -= alpha * api);
        let rnorm = dot(&r, &r).sqrt();
        if rnorm <= 1e-12 * bnorm {
            break;
        }
        let xbr: f64 = {
            let br: Vec<f64> = b.iter().zip(&r).map(|(bi, ri)| bi + ri).collect();
            dot(&x, &br)
        };
        let q1 = -0.5 * xbr;
        if q1 < 0.0 && (it as f64) * (q1 - q0) / q1 < ETA {
            break;
        }
        q0 = q1;
        precond(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        p.iter_mut().zip(&z).for_each(|(pi, zi)| *pi = zi + beta * *pi);
    }
    Some(x)
}
