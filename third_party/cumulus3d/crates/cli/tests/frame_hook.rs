/*
 * frame_hook.rs
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

//! 프레임 훅(`on_frame`): 디코딩된 입력 프레임이 위치 처리 전에, 카메라 한 장씩 전달되는지 시험한다.

mod common;
use common::{make_dataset, NPOS};
use cumulus3d_cli::declare::{Recon, Sinks};
use cumulus3d_cli::events::{Event, EventKind};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn dataset(tag: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("frame_hook_{tag}"));
    let _ = std::fs::remove_dir_all(&d);
    make_dataset(&d);
    d
}

#[test]
fn frames_arrive_before_processing() {
    let src = dataset("on");
    let frames = Arc::new(Mutex::new(Vec::new()));
    let order = Arc::new(Mutex::new(Vec::new()));
    let (f, o) = (frames.clone(), order.clone());
    Recon::declare()
        .input(&src)
        .preset("aerial-formation")
        .stride(1)
        .zones(4, 1)
        .no_dense()
        .seed(7)
        .sinks(Sinks::none())
        .on_frame(move |fr| f.lock().unwrap().push(fr.clone()))
        .on_any(move |e: &Event| match e {
            Event::FrameDecoded { frame, .. } => o.lock().unwrap().push(("frame", frame.position, e.meta().version)),
            Event::FeaturesExtracted { position, .. } => o.lock().unwrap().push(("features", *position, e.meta().version)),
            _ => {}
        })
        .build()
        .unwrap()
        .run()
        .unwrap();

    let frames = frames.lock().unwrap();
    assert_eq!(frames.len(), NPOS * 3, "위치 × 카메라 수만큼");
    for fr in frames.iter() {
        assert!(fr.width > 0 && fr.height > 0);
        assert_eq!(fr.rgb.len(), (fr.width * fr.height * 3) as usize);
        assert!(fr.path.exists());
        assert!(fr.name.starts_with(&fr.camera));
    }
    // 위치마다 프레임 3장이 모두 그 위치의 특징 추출보다 먼저(버전이 작게) 나온다.
    let order = order.lock().unwrap();
    for p in 0..NPOS {
        let of = |k: &str| order.iter().filter(|t| t.0 == k && t.1 == p).map(|t| t.2).collect::<Vec<_>>();
        let (fr, ft) = (of("frame"), of("features"));
        assert_eq!(fr.len(), 3, "위치 {p} 프레임 수");
        assert!(!ft.is_empty(), "위치 {p} 특징 추출");
        assert!(fr.iter().max() < ft.iter().min(), "위치 {p}: 프레임 {fr:?} < 특징 {ft:?}");
    }
}

#[test]
fn no_decoding_without_frame_hook() {
    let src = dataset("off");
    let n = Arc::new(Mutex::new(0usize));
    let c = n.clone();
    Recon::declare()
        .input(&src)
        .preset("aerial-formation")
        .stride(1)
        .zones(4, 1)
        .no_dense()
        .seed(7)
        .sinks(Sinks::none())
        .on_any(move |e: &Event| {
            if e.kind() == EventKind::FrameDecoded {
                *c.lock().unwrap() += 1;
            }
        })
        .build()
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(*n.lock().unwrap(), 0);
}
