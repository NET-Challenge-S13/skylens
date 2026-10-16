/*
 * cache.rs
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

//! Depth map cache: (image name + pose/intrinsics/size hash, settings hash) → final depth, normal and validity mask.
//! Keeps overlapping zones from recomputing images with the same pose.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// One cached view (final scale): pre-filter depth (input to other views' geometric runs) and final depth, normal and cost (fusion input).
#[derive(Clone, Debug)]
pub struct CachedDepth {
    /// Width (pixels).
    pub width: usize,
    /// Height (pixels).
    pub height: usize,
    /// Final depth before filtering (input to other views' geometric runs).
    pub raw_depth: Vec<f32>,
    /// Depth after the filter (0 = invalid).
    pub depth: Vec<f32>,
    /// Normal in the reference camera coordinates.
    pub normal: Vec<[f32; 3]>,
    /// Final aggregated cost.
    pub cost: Vec<f32>,
}

type Entries = (HashMap<(u64, u64), Arc<CachedDepth>>, VecDeque<(u64, u64)>);

/// Depth map cache (thread-safe; when over capacity, the oldest entries are dropped first).
pub struct DepthMapCache {
    inner: Mutex<Entries>,
    capacity: usize,
}

impl Default for DepthMapCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DepthMapCache {
    /// Capacity of 512 views.
    pub fn new() -> Self {
        Self::with_capacity(512)
    }
    /// Creates a cache with the given capacity (minimum 1).
    pub fn with_capacity(capacity: usize) -> Self {
        Self { inner: Mutex::new((HashMap::new(), VecDeque::new())), capacity: capacity.max(1) }
    }
    /// Looks up by key (image geometry hash, settings hash).
    pub fn get(&self, key: (u64, u64)) -> Option<Arc<CachedDepth>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).0.get(&key).cloned()
    }
    /// Inserts. When over capacity, the oldest entries are dropped first.
    pub fn insert(&self, key: (u64, u64), v: Arc<CachedDepth>) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.0.insert(key, v).is_none() {
            g.1.push_back(key);
        }
        while g.1.len() > self.capacity {
            if let Some(k) = g.1.pop_front() {
                g.0.remove(&k);
            }
        }
    }
    /// Number of stored views.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).0.len()
    }
    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
