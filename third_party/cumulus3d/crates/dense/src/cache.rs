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

//! 깊이맵 캐시: (영상 이름 + 자세·내부 매개변수·크기 해시, 설정 해시) → 최종 깊이·법선·유효 표시.
//! 겹치는 구역에서 같은 자세의 영상을 다시 계산하지 않게 한다.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// 캐시된 뷰 하나(최종 스케일): 필터 전 깊이(다른 뷰의 기하 실행 입력)와 최종 깊이·법선·비용(융합 입력).
#[derive(Clone, Debug)]
pub struct CachedDepth {
    /// 너비(픽셀).
    pub width: usize,
    /// 높이(픽셀).
    pub height: usize,
    /// 필터 전 최종 깊이(다른 뷰의 기하 실행 입력).
    pub raw_depth: Vec<f32>,
    /// 필터 통과 깊이(무효 0).
    pub depth: Vec<f32>,
    /// 기준 카메라 좌표 법선.
    pub normal: Vec<[f32; 3]>,
    /// 최종 집계 비용.
    pub cost: Vec<f32>,
}

type Entries = (HashMap<(u64, u64), Arc<CachedDepth>>, VecDeque<(u64, u64)>);

/// 깊이맵 캐시(스레드 안전, 용량 초과 시 오래된 것부터 버림).
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
    /// 용량 512 뷰.
    pub fn new() -> Self {
        Self::with_capacity(512)
    }
    /// 용량을 정해 만든다(최소 1).
    pub fn with_capacity(capacity: usize) -> Self {
        Self { inner: Mutex::new((HashMap::new(), VecDeque::new())), capacity: capacity.max(1) }
    }
    /// 열쇠 (영상 기하 해시, 설정 해시) 로 찾는다.
    pub fn get(&self, key: (u64, u64)) -> Option<Arc<CachedDepth>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).0.get(&key).cloned()
    }
    /// 넣는다. 용량을 넘으면 가장 오래된 항목부터 버린다.
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
    /// 저장된 뷰 수.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).0.len()
    }
    /// 비어 있는지.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
