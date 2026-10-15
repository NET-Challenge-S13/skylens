[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/features/README.md) | 한국어

# cumulus3d-features

영상 읽기(디코딩·EXIF·회색 변환·축소·방향 회전), 카메라 초기 파라미터 결정, 결정적 SIFT 특징 추출을 맡는 크레이트.

**파이프라인 단계**: 첫 단계(특징 추출). 영상 묶음을 받아 카메라를 배정하고 키포인트·기술자를
`cumulus3d_core::FeatureStore` 에 증분 기록한다. 이 저장소를 `cumulus3d-matching` 이 읽는다.
SIFT 계산은 `SiftEngine` 트레이트 뒤에 있어 GPU 백엔드(`cumulus3d-cuda` 의 `CudaSift`)로 바꿔 끼울 수 있다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `FeatureExtractor::new()` / `with_backend(Arc<dyn SiftEngine>)` | 추출기 생성(CPU 기본 / 백엔드 지정) | 백엔드 → `FeatureExtractor` |
| `FeatureExtractor::extract_inputs` | 디코딩된 영상들을 저장소에 추가(위치 하나 단위 호출용) | `&FeatureStore`, `Vec<ImageSource>`, `&ExtractionOptions` → `Result<Vec<ImageReport>>` |
| `FeatureExtractor::extract_files` | 파일 목록을 읽어(EXIF + 디코딩, 병렬) 저장소에 추가 | `&FeatureStore`, 루트 폴더, 이름 목록, `&ExtractionOptions` → `Result<Vec<ImageReport>>` |
| `ExtractionOptions` / `ReaderOptions` / `SiftOptions` | 추출·읽기·SIFT 옵션 | `Default` 로 시작해 필드 수정 |
| `CpuSift::new()` + `SiftEngine::extract` | 회색 영상 한 장에서 SIFT 추출 | `&GrayImage`, `&SiftOptions` → `Result<SiftOutput>` |
| `read_gray` | 영상 파일 → 8비트 회색 영상 | 경로 → `Result<GrayImage>` |
| `ExifInfo::read` / `from_bytes` | EXIF 요약(초점·센서·방향·GPS) | 경로 또는 바이트 → `ExifInfo` |
| `init_camera` | EXIF/기본값으로 새 카메라 초기화 | 모델, W, H, `&ExifInfo`, 파라미터(선택), 계수 → `Result<Camera>` |
| `extract_for_camera` | 축소 → EXIF 방향 회전 → SIFT → 카메라 크기 좌표로 환산 | 백엔드, `&GrayImage`, 방향, 카메라 크기, 최대 크기, `&SiftOptions` → `Result<(Vec<Keypoint>, SiftOutput)>` |
| `read_image_list` | 영상 목록 파일 읽기 | 경로 → `Result<Vec<String>>` |

## 공개 항목

표의 "루트" 표시는 크레이트 루트에서 `cumulus3d_features::이름` 으로 재노출된 항목이다.

### `extract`

| 항목 | 종류 | 역할 |
|---|---|---|
| `FeatureExtractor` (루트) | struct | SIFT 백엔드 + 저장소 기록. `Clone` 비용 낮음 |
| `FeatureExtractor::new` / `with_backend` / `backend` | fn | CPU 백엔드로 생성 / 지정 백엔드로 생성 / 사용 중인 백엔드 |
| `FeatureExtractor::extract_files` | fn | 파일 목록(이름 사전순 처리)을 읽어 추출·기록. `Existing(N)` 에서 카메라 N 이 없으면 즉시 오류, 영상별 실패는 보고서에 |
| `FeatureExtractor::extract_inputs` | fn | 디코딩된 영상 목록을 추출·기록(이름 사전순으로 id 발급) |
| `ExtractionOptions` (루트) | struct | `reader`, `sift`, `sequential_images` |
| `ReaderOptions` (루트) | struct | 카메라 모델(기본 `OpenCv`)·묶기 방식·고정 파라미터·기본 초점 계수(1.2)·최대 영상 크기(3200)·엄격 크기 검사·EXIF GPS 사전 위치 기록 |
| `CameraMode` (루트) | enum | 카메라 묶기: `PerImage`, `Single`, `PerFolder`(기본), `Existing(CameraId)` |
| `ImageSource` (루트) | struct | 메모리 입력 영상: `name`, `gray`, `exif` |
| `ImageReport` (루트) | struct | 영상 하나의 처리 보고: `name`, `status` |
| `ImageStatus` (루트) | enum | `Extracted{image_id, camera_id, num_features}`, `AlreadyExists{image_id}`, `Failed{error}` |
| `extract_for_camera` (루트) | fn | 영상 한 장 추출 후 키포인트를 카메라 크기 좌표로 환산 |
| `read_image_list` (루트) | fn | 목록 파일: 줄마다 공백 제거, 빈 줄 건너뜀 |
| `image_folder` (루트) | fn | 이름의 부모 폴더(`"camF/x.jpg"` → `"camF"`) |

### `gray`

| 항목 | 종류 | 역할 |
|---|---|---|
| `GrayImage` (루트) | struct | 행 우선 8비트 회색 영상(`width`, `height`, `data`) |
| `GrayImage::new` / `filled` / `from_f32` | fn | 버퍼로 생성(길이 검사) / 단색 / `[0,1]` 실수 함수로 생성 |
| `GrayImage::get` / `rotate_ccw` / `resized` | fn | 화소 값 / 반시계 90°×k 회전 / Lanczos3 재표본화 |
| `read_gray` (루트) | fn | 파일 디코딩 → 회색(EXIF 자동 회전 없음) |
| `to_gray` | fn | `image::DynamicImage` → 회색(알파 버림, 16비트 → 8비트) |
| `rgb_to_gray` (루트) | fn | `round(0.2126 R + 0.7152 G + 0.0722 B)` |
| `limited_size` (루트) | fn | 최대 크기 제한 후 크기 |
| `rotate_keypoint_ccw` (루트) | fn | 키포인트를 반시계 90°×k 회전 |
| `orientation_to_rotation` | fn | EXIF Orientation → 반시계 회전 횟수 |
| `orientation_gravity` | fn | EXIF Orientation → 영상 좌표 중력 방향 |

### `exif_info`

| 항목 | 종류 | 역할 |
|---|---|---|
| `ExifInfo` (루트) | struct | 제조사·모델·초점(mm, 35mm 환산)·초점면 해상도·방향·GPS(위도, 경도, 고도) |
| `ExifInfo::read` / `from_bytes` | fn | 파일 / 메모리 바이트에서 읽기(EXIF 없으면 빈 값) |

### `camera_init`

| 항목 | 종류 | 역할 |
|---|---|---|
| `FocalEstimate` (루트) | struct | 초점거리(`focal`)와 EXIF 유래 여부(`prior`) |
| `infer_focal` (루트) | fn | 초점 규칙: 35mm 환산 → 초점면 해상도 → 센서 폭 표 → 계수 × max(W, H) |
| `init_camera` (루트) | fn | 새 카메라: 주점 (W/2, H/2), 왜곡 0, 파라미터를 주면 그대로 |
| `lookup_sensor_width` (루트) | fn | 내장 표에서 제조사·모델로 센서 폭(mm) 조회 |

### `sift`

| 항목 | 종류 | 역할 |
|---|---|---|
| `SiftEngine` (루트) | trait | SIFT 백엔드: `name()`, `extract(&GrayImage, &SiftOptions)` |
| `CpuSift` (루트) | struct | CPU(rayon) 백엔드, 피라미드 버퍼를 영상 간 재사용. `new()` |
| `SiftOptions` (루트) | struct | 최대 특징 수(8192), 첫 옥타브(−1), 옥타브 수(자동), 레벨 수(3), 임계값, 방향 수, 정규화, 선택 방식 등 |
| `SiftFeature` (루트) | struct | 특징 하나(x, y, scale, orientation, octave, level, response). `keypoint()` 은 아핀 키포인트 |
| `SiftOutput` (루트) | struct | `features` + `descriptors`(행 i ↔ 특징 i). `keypoints()`, `len()`, `is_empty()` |
| `FeatureSelection` (루트) | enum | 최대 특징 수 제한: `CompatLevels`(레벨 단위, 기본), `TopK`, `SpatialGrid{cells}` |
| `DescriptorNormalization` (루트) | enum | 기술자 정규화: `L1Root`(기본), `L2` |

### `sift::pyramid` (저수준)

| 항목 | 종류 | 역할 |
|---|---|---|
| `BufferPool` | struct | f32 버퍼 재사용 풀: `take`, `give` |
| `gaussian_kernel` | fn | 1D 가우시안 커널(폭 [5, 33], 합 1) |
| `gaussian_blur` | fn | 분리형 가우시안 블러(가장자리 복제) |
| `upsample2` / `decimate2` | fn | 2배 업샘플 / 짝수 행·열 1/2 축소 |
| `Octave` | struct | 옥타브 하나의 가우시안 레벨들. `level(l)` |
| `ScaleSpace` | struct | 스케일 공간 상수(S, k, σ₀). `new`, `sigma`, `sigma_inc` |
| `auto_num_octaves` | fn | 옥타브 수 자동 결정 |
| `build_pyramid` | fn | 가우시안 피라미드 구성 |

### `sift::detect` (저수준)

| 항목 | 종류 | 역할 |
|---|---|---|
| `Candidate` | struct | 검출 점(옥타브 좌표, σ, 보정된 DoG 응답) |
| `DetectParams` | struct | 검출 매개변수(임계값, 보정 반복, 특이 헤시안 처리, σ₀, k) |
| `detect_level` | fn | 검출 레벨 하나에서 DoG 극값 검출·엣지 억제·부분화소 보정 |
| `dog` | fn | 가우시안 차(DoG) 계산 |

### `sift::orient` (저수준)

| 항목 | 종류 | 역할 |
|---|---|---|
| `Gradient` | struct | 가우시안 레벨 기울기. `lazy`(즉석), `precomputed`(선계산, 결과 비트 동일), `width`, `at` |
| `OrientParams` | struct | 방향 할당 매개변수 |
| `quantize_angle` / `dequantize_angle` | fn | 각도 ↔ 16비트 양자화 |
| `orientations` | fn | 키포인트의 주 방향들 |
| `descriptor` | fn | 128차원 uint8 기술자 |
| `normalize_quantize` | fn | 기술자 정규화·양자화 |

## 사용 예

합성 영상으로 SIFT 를 돌리고 저장소에 기록한다(크레이트 문서의 doc-test 와 같은 코드).

```rust
use cumulus3d_core::FeatureStore;
use cumulus3d_features::{
    CpuSift, ExifInfo, ExtractionOptions, FeatureExtractor, GrayImage, ImageSource, ImageStatus, SiftEngine,
    SiftOptions,
};

// 합성 영상: 밝기 블롭 세 개.
let blob = |x: usize, y: usize, cx: f32, cy: f32, s: f32| {
    let (dx, dy) = (x as f32 - cx, y as f32 - cy);
    (-(dx * dx + dy * dy) / (2.0 * s * s)).exp()
};
let img = GrayImage::from_f32(320, 240, |x, y| {
    0.2 + 0.6 * (blob(x, y, 80.0, 60.0, 6.0) + blob(x, y, 200.0, 150.0, 9.0) + blob(x, y, 260.0, 70.0, 5.0))
});

// 1) SIFT 백엔드 직접 호출: 특징 i ↔ 기술자 행 i.
let out = CpuSift::new().extract(&img, &SiftOptions::default())?;
assert!(!out.is_empty());
assert_eq!(out.features.len(), out.descriptors.len());

// 2) 저장소에 기록(카메라 배정 + 키포인트·기술자).
let store = FeatureStore::new();
let src = ImageSource { name: "camF/0001.jpg".into(), gray: img, exif: ExifInfo::default() };
let reports = FeatureExtractor::new().extract_inputs(&store, vec![src], &ExtractionOptions::default())?;
assert!(matches!(reports[0].status, ImageStatus::Extracted { .. }));
assert_eq!(store.num_images(), 1);
```

파일에서 읽을 때는 `FeatureExtractor::extract_files(&store, root, &names, &opts)` 를 쓴다.
벤치마크: `cargo run --release -p cumulus3d-features --example bench_sift -- <영상|synthetic> 2048 1152`.

## 기능 플래그·하드웨어

- 기능 플래그 없음. 순수 CPU(rayon) 구현이라 특별한 하드웨어가 필요 없다.
- GPU 로 SIFT 를 돌리려면 `cumulus3d-cuda` 의 `CudaSift` 를 `FeatureExtractor::with_backend` 에 넘긴다(CUDA 12.x 드라이버 필요).

## 동작 메모

- 출력은 완전 결정적이다: 옥타브 오름차순 → 레벨 오름차순 → 행 우선 검출 순, 두 번째 방향은 바로 뒤.
- 결과를 바꾸는 개선 옵션은 모두 기본 꺼짐: `truncate_width_to_4 = false`, `refinement_iterations = 5`,
  `reject_singular_refinement`, `orientation_bin_interpolation`, `FeatureSelection::TopK`/`SpatialGrid`, `num_octaves = Some(n)`.
- 이름이 이미 있고 키포인트·기술자가 모두 있으면 건너뛴다(`AlreadyExists`). 행만 있으면 특징만 채운다.
- `read_pose_priors` 가 켜져 있고 EXIF GPS 가 있으면 저장소에 사전 위치(위도, 경도, 고도)를 기록한다.
- 저장소에는 rig/frame 행이 없다. 재구성 단계에서 `Reconstruction::add_image_own_frame` 으로 영상마다 프레임을 만든다.
