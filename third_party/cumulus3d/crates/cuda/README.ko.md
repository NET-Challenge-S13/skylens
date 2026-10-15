[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/cuda/README.md) | 한국어

# cumulus3d-cuda

GPU 백엔드: 다중 뷰 스테레오(PatchMatch), SIFT 기술자 매칭, SIFT 스케일 공간을 CUDA 로 계산한다.
cudarc(드라이버 API + NVRTC)로 CUDA C 커널을 실행 시 컴파일하며, 다른 크레이트가 정의한 백엔드 트레이트를 구현해 끼워 넣는다.

**파이프라인 단계**: 특징 추출(`cumulus3d_features::SiftEngine` → `CudaSift`), 매칭(`cumulus3d_matching::MatcherBackend` → `CudaMatcher`),
조밀화(`cumulus3d_dense::PatchMatchBackend` → `CudaPatchMatch`)의 계산 부분. `cumulus3d stream --gpu` 가 이 백엔드들을 쓴다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `is_available` | 드라이버·NVRTC 라이브러리와 장치가 있는지 확인(없으면 거짓, panic 없음) | `()` → `bool` |
| `CudaPatchMatch::try_default` | 장치 0, 기본 옵션 PatchMatch 백엔드 | `()` → `Result<CudaPatchMatch, GpuError>` |
| `CudaPatchMatch` | `PatchMatchBackend` 구현. `cumulus3d_dense::densify` 에 넘긴다 | `&DenseScene` … → `DenseOutput` (dense 경유) |
| `CudaMatcher::try_default` | 장치 0 기술자 매칭 백엔드 | `()` → `Result<CudaMatcher, GpuError>` |
| `CudaMatcher` | `MatcherBackend` 구현(정수 내적 행·열 top-2) | 기술자 두 묶음 → `(Vec<Top2>, Vec<Top2>)` |
| `CudaSift::try_default` | 장치 0 SIFT 엔진 | `()` → `Result<CudaSift, GpuError>` |
| `CudaSift` | `SiftEngine` 구현(GPU 스케일 공간 + CPU 검출·기술자, `CpuSift` 와 비트 동일) | `&GrayImage`, `&SiftOptions` → `Result<SiftOutput>` |
| `CudaDevice::new` | 장치 `ordinal` 의 문맥·전용 스트림 생성(여러 백엔드가 공유) | `usize` → `Result<Arc<CudaDevice>, GpuError>` |

## 공개 항목

하위 모듈은 비공개이고 모든 항목이 크레이트 루트(`cumulus3d_cuda::`)에 재노출된다.

| 항목 | 종류 | 역할 |
|---|---|---|
| `is_available` | fn | CUDA 사용 가능 여부 |
| `GpuError` | enum | `NotAvailable`, `Driver`(cudarc 드라이버 오류), `Compile`(NVRTC 실패) |
| `CudaDevice` | struct | 한 GPU 의 문맥과 전용 스트림, 계산 능력에 맞춘 NVRTC 아키텍처 |
| `CudaDevice::new` | fn | 장치 열기 → `Arc<CudaDevice>` |
| `CudaDevice::name` | fn | 장치 이름 |
| `CudaPatchMatch` | struct | `cumulus3d_dense::PatchMatchBackend` 구현(스케일별 실행·평가·판독·상향 표본·중앙값 필터) |
| `CudaPatchMatch::new` / `try_default` | fn | 장치·옵션으로 생성 / 장치 0·기본 옵션 |
| `CudaPatchMatch::device` / `options` | fn | 사용 장치 / 옵션 |
| `CudaPatchMatchOptions` | struct | `hw_interp`(true), `batch_pixels`(2,200,000), `max_batch`(64), `min_blocks`(2), `fast_math`(true) |
| `CudaMatcher` | struct | `cumulus3d_matching::MatcherBackend` 구현 |
| `CudaMatcher::new` / `try_default` | fn | 장치로 생성 / 장치 0 |
| `CudaMatcher::top2_gpu` | fn | GPU 행·열 top-2(오류를 그대로 돌려줌; 트레이트 `top2` 는 오류 시 경고 후 CPU 로 계산) |
| `CudaSift` | struct | `cumulus3d_features::SiftEngine` 구현(장치 오류 시 경고 후 `CpuSift` 로 계산) |
| `CudaSift::new` / `try_default` | fn | 장치로 생성 / 장치 0 |

## 사용 예

크레이트 문서의 doc-test 와 같은 코드. CUDA 가 없으면 아무것도 하지 않는다.

```rust
use cumulus3d_cuda::{is_available, CudaMatcher};
use cumulus3d_matching::MatcherBackend;

// CUDA 가 없는 기계에서는 거짓이므로 아무것도 하지 않는다.
if is_available() {
    let matcher = CudaMatcher::try_default().expect("CUDA 장치 0");
    let (d1, d2) = (vec![1u8; 128 * 4], vec![1u8; 128 * 3]); // 128바이트 SIFT 기술자 4개, 3개
    let (rows, cols) = matcher.top2(&d1, 4, &d2, 3);
    assert_eq!((rows.len(), cols.len()), (4, 3));
}
```

조밀화에 GPU 백엔드 넘기기:

```rust,no_run
use cumulus3d_cuda::CudaPatchMatch;
use cumulus3d_dense::{densify, DenseScene, DensifyOptions};

fn run(scene: &DenseScene) -> Result<(), Box<dyn std::error::Error>> {
    let pm = CudaPatchMatch::try_default()?; // 장치 0, 기본 옵션
    let out = densify(scene, &DensifyOptions::default(), &pm, None)?;
    println!("조밀 점 {}개", out.cloud.len());
    Ok(())
}
```

## 기능 플래그·하드웨어

- 기능 플래그 없음.
- 실행에는 NVIDIA GPU 와 **CUDA 12.x 드라이버**(`libcuda`)·NVRTC(`libnvrtc`)가 필요하다. 라이브러리 경로는
  `LD_LIBRARY_PATH` 에 있어야 한다(예: `/usr/local/cuda-12.4/lib64`).
- cudarc 를 `dynamic-loading` 으로 쓰므로 **CUDA 가 없는 기계에서도 빌드·테스트된다**. 장치가 없으면 `is_available()` 이 거짓이고
  GPU 테스트(`tests/*_gpu.rs`)는 건너뛴다.
- 커널은 장치당 한 번 NVRTC 로 컴파일한다(약 1 s). 알려진 계산 능력: 7.0 ~ 9.0.
- 환경 변수: `CUMULUS3D_PM_MIN_BLOCKS`, `CUMULUS3D_PM_FAST_MATH`(PatchMatch 옵션 기본값),
  `CUMULUS3D_CUDA_TRACE=1`(커널 자원·호출별 시간 출력).
- `unsafe` 는 cudarc 가 unsafe 로 둔 곳(커널 발사, 고정 호스트 메모리, 텍스처 객체, `DeviceRepr`)에만 쓴다.
- PatchMatch 는 같은 장치·입력이면 비트 단위로 같은 결과를 낸다(카운터 기반 난수).
