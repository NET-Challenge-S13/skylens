/*
 * bench_contention.rs
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

//! 같은 GPU 에 다른 작업이 있을 때 희소 단계 호출(관측 검사)이 얼마나 기다리는지 잰다.
//!
//! `cargo run --release -p cumulus3d-cuda --example bench_contention -- [부하 커널 ms] [한 번에 쌓는 커널 수] [블록 수]`
//!
//! 부하 스레드는 보통 우선순위 스트림에 바쁜 커널을 묶음으로 넣고(조밀화처럼 동기화 없이), 측정 스레드는
//! 20만 관측 검사를 되풀이하며 걸린 시간을 찍는다. 부하 없음 → 부하 있음 → 부하 + 다른 스레드의 드라이버
//! 호출 순서로 잰다. 마지막 단계의 호출은 환경 변수 `MODE` 로 고른다:
//!
//! - (기본) 동기 할당·해제 `cuMemAlloc`/`cuMemFree`
//! - `hostfree`: 고정 메모리 할당·해제 `cuMemHostAlloc`/`cuMemFreeHost`
//! - `freeasync`: 비동기 할당·해제 후 스트림 동기화(cudarc `CudaSlice` 의 길)
//! - `memcpy`: 기본 스트림의 동기 내려 받기
//! - `ctxsync`: 문맥 동기화
//! - `alloc`: 동기 할당만
//!
//! 측정(RTX 4050 Laptop, WDDM): 커널 부하만으로는 거의 늘지 않고(8ms 안팎), `cuMemFree`·`cuMemFreeHost` 는
//! 다른 스레드의 호출을 장치 대기열이 빌 때까지 세운다(450ms). 데모에서 같은 일을 한 것은 조밀화가 앞선 커널을
//! 기다리며 일반 메모리로 내려 받는 `cuMemcpyDtoHAsync` 였다(`device::SyncedCopy` 참조).

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cumulus3d_core::{Camera, CameraModelKind, Rigid3, Vec2, Vec3};
use cumulus3d_cuda::CudaSparse;
use cumulus3d_sfm::ObsQuery;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BUSY: &str = r#"
extern "C" __global__ void busy(float* out, long long spins) {
    float v = threadIdx.x;
    for (long long i = 0; i < spins; ++i) v = v * 1.0000001f + 0.5f;
    out[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
"#;

fn probe(sp: &CudaSparse, label: &str, rounds: usize) {
    let cams = vec![Camera::new(0, CameraModelKind::Pinhole, 1920, 1080, vec![1500.0, 1500.0, 960.0, 540.0]).expect("카메라")];
    let poses: Vec<Rigid3> =
        (0..30).map(|k| Rigid3::from_rotation_matrix(&cumulus3d_core::Mat3::identity(), Vec3::new(k as f64 * 0.1, 0.0, 10.0))).collect();
    let queries: Vec<ObsQuery> = (0..200_000u32)
        .map(|i| ObsQuery {
            camera: 0,
            pose: i % 30,
            xy: Vec2::new((i % 1920) as f64, (i % 1080) as f64),
            xyz: Vec3::new((i % 13) as f64 - 6.0, (i % 7) as f64 - 3.0, (i % 17) as f64),
        })
        .collect();
    let mut t: Vec<f64> = Vec::new();
    for _ in 0..rounds {
        let t0 = Instant::now();
        sp.check_observations_gpu(&cams, &poses, &queries).expect("관측 검사");
        t.push(t0.elapsed().as_secs_f64() * 1e3);
        std::thread::sleep(Duration::from_millis(20));
    }
    t.sort_by(|a, b| a.total_cmp(b));
    println!("{label}: 중앙값 {:.1}ms, 90% {:.1}ms, 최대 {:.1}ms", t[t.len() / 2], t[t.len() * 9 / 10], t[t.len() - 1]);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kernel_ms: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20.0);
    let depth: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    {
        let c = CudaContext::new(0).expect("문맥");
        use cudarc::driver::sys::CUdevice_attribute as A;
        println!(
            "메모리 풀 지원 {:?}, 통합 주소 {:?}",
            c.attribute(A::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED),
            c.attribute(A::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)
        );
        if std::env::var("ONLY_ATTR").is_ok() {
            return;
        }
    }
    let sp = CudaSparse::try_default().expect("CUDA");
    probe(&sp, "준비", 5);
    probe(&sp, "부하 없음", 40);

    let ctx = CudaContext::new(0).expect("문맥");
    let s = ctx.new_stream().expect("스트림");
    let ptx = cudarc::nvrtc::compile_ptx(BUSY).expect("컴파일");
    let f = ctx.load_module(ptx).expect("모듈").load_function("busy").expect("함수");
    let blocks: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let mut out = s.alloc_zeros::<f32>(blocks as usize * 256).expect("버퍼");
    // 커널 하나가 대략 kernel_ms 가 되도록 맞춘다.
    let mut spins: i64 = 1 << 12;
    loop {
        let t0 = Instant::now();
        let mut l = s.launch_builder(&f);
        l.arg(&mut out).arg(&spins);
        unsafe { l.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.expect("실행");
        s.synchronize().expect("동기화");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms >= kernel_ms * 0.8 || spins > 1 << 30 {
            spins = (spins as f64 * kernel_ms / ms.max(0.01)) as i64;
            break;
        }
        spins *= 2;
    }
    println!("부하: 커널 하나 약 {kernel_ms}ms(블록 {blocks}), 한 번에 {depth}개씩 쌓음");
    let stop = Arc::new(AtomicBool::new(false));
    let st = stop.clone();
    let h = std::thread::spawn(move || {
        while !st.load(Ordering::Relaxed) {
            for _ in 0..depth {
                let mut l = s.launch_builder(&f);
                l.arg(&mut out).arg(&spins);
                unsafe { l.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.expect("실행");
            }
            s.synchronize().expect("동기화");
        }
    });
    std::thread::sleep(Duration::from_millis(300));
    probe(&sp, "부하 있음", 40);
    // 다른 스레드가 동기 할당·해제(cuMemAlloc/cuMemFree)를 되풀이할 때. cuMemFree 는 장치 전체가
    // 쉴 때까지 기다린다(UNet 의 libtorch 캐시 할당기가 메모리를 돌려줄 때와 같은 호출).
    let ctx2 = ctx.clone();
    let st2 = stop.clone();
    let h2 = std::thread::spawn(move || {
        ctx2.bind_to_thread().expect("문맥");
        while !st2.load(Ordering::Relaxed) {
            match std::env::var("MODE").as_deref() {
                Ok("ctxsync") => ctx2.synchronize().expect("동기화"),
                Ok("hostfree") => unsafe {
                    let p = cudarc::driver::result::malloc_host(16 << 20, 0).expect("고정 할당");
                    cudarc::driver::result::free_host(p).expect("고정 해제");
                },
                Ok("freeasync") => {
                    // cudarc 의 CudaSlice 와 같은 길: 비동기 할당·해제 후 그 스트림 동기화.
                    let s3 = ctx2.new_stream().expect("스트림");
                    for _ in 0..4 {
                        let b = s3.alloc_zeros::<u8>(64 << 20).expect("할당");
                        drop(b);
                        s3.synchronize().expect("동기화");
                    }
                }
                Ok("memcpy") => unsafe {
                    // 기본(legacy) 스트림의 동기 복사(libtorch 의 .to(CPU) 와 같은 호출).
                    let p = cudarc::driver::result::malloc_sync(16 << 20).expect("할당");
                    let mut h = vec![0u8; 16 << 20];
                    cudarc::driver::result::memcpy_dtoh_sync(&mut h, p).expect("복사");
                    std::mem::forget(p);
                },
                Ok("alloc") => unsafe {
                    let p = cudarc::driver::result::malloc_sync(64 << 20).expect("할당");
                    std::mem::forget(p);
                },
                _ => unsafe {
                    let p = cudarc::driver::result::malloc_sync(64 << 20).expect("할당");
                    cudarc::driver::result::free_sync(p).expect("해제");
                },
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    probe(&sp, "부하 + 동기 할당·해제", 40);
    stop.store(true, Ordering::Relaxed);
    h.join().expect("부하 스레드");
    h2.join().expect("할당 스레드");
}
