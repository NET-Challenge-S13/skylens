[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/matching/README.md) | 한국어

# cumulus3d-matching

SIFT 기술자 매칭과 두 뷰 기하(E/F/H) 추정·검증, 그리고 상대 자세 분해를 맡는 크레이트.

**파이프라인 단계**: 특징 추출(`cumulus3d-features`) 다음 단계. 짝 목록의 영상 쌍마다 기술자를 매칭하고
E/F/H LO-RANSAC 으로 기하를 검증해 원시 매칭과 `TwoViewGeometry` 를 `FeatureStore` 에 기록한다.
SfM(`cumulus3d-sfm`)은 이 결과와 `pose` 모듈의 분해 함수를 쓴다. 매칭 커널은 `MatcherBackend`
트레이트 뒤에 있어 GPU 백엔드(`cumulus3d-cuda` 의 `CudaMatcher`)로 바꿀 수 있다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `match_pairs` | 짝 목록을 매칭·검증해 저장소에 기록(이미 처리된 짝은 건너뜀) | `&FeatureStore`, `&[(ImageId, ImageId)]`, `&PairMatchingOptions`, `&dyn MatcherBackend` → `Result<MatchingStats>` |
| `match_pair_list_file` | 짝 목록 파일을 읽어 `match_pairs` 수행 | `&FeatureStore`, 경로, 옵션, 백엔드 → `Result<(PairList, MatchingStats)>` |
| `estimate_two_view` | 두 영상의 기하 추정(E/F/H LO-RANSAC + 구성 판정) | 카메라·키포인트 ×2, `&[FeatureMatch]`, `&TwoViewOptions` → `TwoViewGeometry` |
| `verify_pair` | 저장소의 두 영상과 매칭으로 기하 검증(최소 인라이어 규칙 적용) | `&FeatureStore`, id1, id2, 매칭, `&TwoViewOptions` → `TwoViewGeometry` |
| `CpuMatcher` + `MatcherBackend::match_descriptors` | 기술자 무차별 매칭(비율·거리·교차 검사) | `&Descriptors` ×2, `&DescriptorMatchOptions`, 최대 수 → `Vec<FeatureMatch>` |
| `PairMatchingOptions` / `TwoViewOptions` / `DescriptorMatchOptions` | 매칭·기하·기술자 옵션 | `Default` 로 시작해 필드 수정 |
| `read_pair_list` / `parse_pair_list` | 짝 목록 파일/텍스트 파싱 | 경로·텍스트, 이름 → id 조회 함수 → `PairList` |
| `recover_two_view_pose` | 두 뷰 기하에서 상대 자세·삼각측량 각 채우기 | 카메라·키포인트, `&mut TwoViewGeometry` → `bool` |
| `pose_from_essential` | E → 상대 자세 + 3D 점(cheirality) | `&Mat3`, 광선 ×2 → `Option<(Rigid3, Vec<Vec3>)>` |
| `essential_five_point` / `fundamental_seven_point` / `homography_dlt` | 최소 해법 | 대응점 → 행렬 후보 |

## 공개 항목

"루트" 표시는 크레이트 루트에서 `cumulus3d_matching::이름` 으로 재노출된 항목이다.

### `pipeline`

| 항목 | 종류 | 역할 |
|---|---|---|
| `match_pairs` (루트) | fn | 짝 매칭 → 검증 → 기록. 블록 단위·짝 단위 병렬, 기록은 입력 순서. 매칭만 있으면 검증만, 기하만 있으면 다시 매칭 |
| `verify_pair` (루트) | fn | 짝 하나 기하 검증(저장소에서 카메라·키포인트 읽음) |
| `match_pair_list_file` (루트) | fn | 짝 목록 파일 읽기 + `match_pairs` |
| `PairMatchingOptions` (루트) | struct | `sift`, `geometry`, `max_num_matches`(32768, 실제값은 저장소 최대 키포인트 수로 제한), `block_size`(1225), `skip_geometric_verification` |
| `MatchingStats` (루트) | struct | 건너뜀·매칭·검증만·유효 기하 짝 수 |

### `pairs`

| 항목 | 종류 | 역할 |
|---|---|---|
| `PairList` (루트) | struct | `pairs`(파일 순서, 무순서 중복·자기 짝 제외), `missing_names` |
| `parse_pair_list` (루트) | fn | 텍스트 파싱: 공백 제거, 빈 줄·`#` 무시, 구분자는 공백 하나(탭 아님) |
| `read_pair_list` (루트) | fn | 파일 읽기 + 파싱 |

### `descriptor`

| 항목 | 종류 | 역할 |
|---|---|---|
| `MatcherBackend` (루트) | trait | 매칭 커널: `top2`(필수, 행/열 top-2), `match_descriptors`(기본 구현) |
| `CpuMatcher` (루트) | struct | CPU 백엔드: u8×u8→u32 정수 내적, 행 블록 rayon 병렬 + 열 타일(`row_block`, `col_tile`) |
| `DescriptorMatchOptions` (루트) | struct | `max_ratio`(0.8), `max_distance`(0.7 rad), `cross_check`(참), `rule` |
| `AcceptRule` (루트) | enum | 경계 처리: `Gpu`(기본, 엄격 <), `CpuBruteForce`(≤) |
| `Top2` (루트) | struct | 행(열)별 최댓값·인덱스·두 번째 값. `push`, `merge` |
| `DOT_NORM` | const | 정규화 기술자 노름의 제곱(512²) |
| `dot_to_angle` | fn | 내적 → 각도(f32 `acos`) |
| `apply_tests` | fn | top-2 결과에 비율·거리·교차 검사 적용 |
| `top2_naive` | fn | 순차 기준 구현(테스트용, `CpuMatcher` 와 비트 일치) |

### `two_view`

| 항목 | 종류 | 역할 |
|---|---|---|
| `estimate_two_view` (루트) | fn | 두 뷰 기하 추정. 두 카메라 모두 사전 초점이면 보정 경로(E/F/H), 아니면 비보정(F/H). 다중 모델·H 강제·정지 매칭 필터 지원 |
| `finalize_geometry` (루트) | fn | 인라이어가 최소 수 미만이면 기본값(UNDEFINED) |
| `TwoViewOptions` (루트) | struct | 최소 인라이어(15), 비율 임계, 워터마크, 다중 모델, 상대 자세 계산, RANSAC 매개변수 등 |
| `decide_calibrated` (루트) | fn | 보정 경로 구성 판정(E/F/H 결과 → `Decision`) |
| `decide_uncalibrated` (루트) | fn | 비보정 경로 구성 판정(F/H) |
| `Decision` (루트) | struct | 판정 구성 + 사용할 마스크 |
| `MaskChoice` (루트) | enum | `E`, `F`, `H` 인라이어 마스크 |
| `ModelOutcome` (루트) | struct | RANSAC 하나의 성공 여부·인라이어 수 |
| `is_watermark` (루트) | fn | 영상 가장자리 고정 매칭(워터마크) 검출 |
| `derive_seed` | fn | 모델 종류별 RANSAC 시드 파생(재현용) |

### `pose`

| 항목 | 종류 | 역할 |
|---|---|---|
| `decompose_essential` (루트) | fn | E → 4개 (R, t) 후보 |
| `pose_from_essential` (루트) | fn | E → 상대 자세 + 카메라1 좌표 3D 점 |
| `decompose_homography` (루트) | fn | H → (R, t, n) 후보 |
| `pose_from_homography` (루트) | fn | H → 상대 자세, 평면 법선, 3D 점 |
| `triangulate_midpoint` (루트) | fn | 중점 삼각측량(깊이 ≤ ε 이면 None) |
| `recover_two_view_pose` (루트) | fn | 기하 → `cam1_to_cam2`·`tri_angle`·평면/회전 구성 확정 |
| `refit_and_estimate_relative_pose` (루트) | fn | 전역 SfM 준비용: 행렬이 없으면 인라이어로 다시 맞춘 뒤 분해, t 단위화 |
| `median_triangulation_angle` | fn | 삼각측량 각 중앙값 |
| `inlier_rays` | fn | 인라이어 매칭의 단위 광선 |

### `essential`

| 항목 | 종류 | 역할 |
|---|---|---|
| `essential_five_point` (루트) | fn | 5점(N ≥ 5) E 해들(최대 10개, 노름 1) |
| `essential_eight_point` (루트) | fn | 8점 이상 E(단위 광선, rank-2 강제) |
| `epipolar_row` | fn | 에피폴라 제약 한 행 |

### `estimators` — `cumulus3d_core::ransac::Estimator` 구현

| 항목 | 종류 | 역할 |
|---|---|---|
| `fundamental_seven_point` (루트) | fn | 7점 F(최대 3개) |
| `fundamental_eight_point` (루트) | fn | 정규화 8점 F(rank-2) |
| `homography_dlt` (루트) | fn | DLT H(4점 LU / N점 SVD, 선택적 하틀리 정규화) |
| `sampson_error_sq` (루트) | fn | Sampson 제곱 오차(분모 0 → ∞) |
| `homography_transfer_error_sq` (루트) | fn | H 단방향 전이 제곱 오차 |
| `EssentialFivePointEstimator` (루트) | struct | 5점 E 추정기(단위 광선 입력) |
| `Fundamental7PtEstimator` (루트) | struct | 7점 F 추정기 |
| `FundamentalEightPointEstimator` (루트) | struct | 정규화 8점 F 국소 추정기 |
| `HomographyEstimator` (루트) | struct | DLT H 추정기(`normalize` 옵션) |
| `TranslationEstimator` (루트) | struct | 2D 평행이동 추정기(워터마크 검출용) |

### `linalg`

| 항목 | 종류 | 역할 |
|---|---|---|
| `null_space_9` | fn | N×9 제약 행렬의 영공간 기저 |
| `singular_values_9` | fn | 9열 행렬의 특이값과 최소 우특이벡터 |
| `mat3_from_row_major` | fn | 행 우선 9-벡터 → 3×3 |
| `svd3` | fn | `cumulus3d_core::linalg::svd3` 재노출 |
| `null_vector3` | fn | 3×3 의 근사 영벡터 |
| `hartley_normalize` | fn | 하틀리 정규화(점, 변환 T) |
| `solve8` | fn | 8×8 부분 피벗 LU |

### `poly`

| 항목 | 종류 | 역할 |
|---|---|---|
| `roots_companion` | fn | 동반행렬 고유값으로 다항식 복소 근 |
| `solve_cubic_monic` | fn | 모닉 3차식 실근(해석적) |
| `poly_mul` / `poly_add` / `poly_eval` | fn | 다항식 곱 / 합·차 / 값 |

## 사용 예

기술자 매칭과 두 뷰 기하를 합성 데이터로 실행한다(크레이트 문서의 doc-test 와 같은 코드).

```rust
use cumulus3d_core::{Camera, CameraModelKind, Descriptors, FeatureMatch, Keypoint, TwoViewGeometryConfig};
use cumulus3d_matching::{estimate_two_view, CpuMatcher, DescriptorMatchOptions, MatcherBackend, TwoViewOptions};

// 1) 기술자 매칭: 기술자 i 는 성분 4i..4i+4 만 255 인 서로 직교하는 벡터.
let make = |order: &[usize]| {
    let mut d = Descriptors::new();
    for &i in order {
        let mut row = [0u8; 128];
        row[4 * i..4 * i + 4].fill(255);
        d.push(&row);
    }
    d
};
let d1 = make(&(0..32).collect::<Vec<_>>());
let d2 = make(&(0..32).rev().collect::<Vec<_>>());
let m = CpuMatcher::default().match_descriptors(&d1, &d2, &DescriptorMatchOptions::default(), 1000);
assert_eq!(m.len(), 32);
assert!(m.iter().all(|x| x.idx2 == 31 - x.idx1));

// 2) 두 뷰 기하: 같은 핀홀 카메라가 x 축으로 1 만큼 이동한 두 영상.
let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 500.0, 640, 480);
cam.focal_from_prior = true; // 초점을 알면 보정(E) 경로를 쓴다.
let (mut kps1, mut kps2, mut matches) = (Vec::new(), Vec::new(), Vec::new());
for i in 0..60u32 {
    let (x, y, z) = (((i * 37) % 23) as f64 * 0.2 - 2.2, ((i * 17) % 19) as f64 * 0.2 - 1.8, 4.0 + ((i * 7) % 11) as f64 * 0.3);
    let px = |cx: f64| Keypoint::new((500.0 * cx / z + 320.0) as f32, (500.0 * y / z + 240.0) as f32);
    kps1.push(px(x));
    kps2.push(px(x - 1.0));
    matches.push(FeatureMatch::new(i, i));
}
let tvg = estimate_two_view(&cam, &kps1, &cam, &kps2, &matches, &TwoViewOptions::default());
assert_eq!(tvg.config, TwoViewGeometryConfig::Calibrated);
assert!(tvg.inlier_matches.len() >= 50);
```

저장소 단위로는 `match_pairs(&store, &pairs, &PairMatchingOptions::default(), &CpuMatcher::default())` 한 번이면
매칭·검증·기록이 끝난다. 벤치마크: `cargo run --release -p cumulus3d-matching --example bench_match`(8192² 매칭).

## 기능 플래그·하드웨어

- 기능 플래그 없음. 순수 CPU(rayon) 구현.
- `MatcherBackend` 를 `cumulus3d-cuda` 의 `CudaMatcher` 로 바꾸면 기술자 매칭(정수 내적 GEMM + top-2)을 GPU 에서 한다
  (CUDA 12.x 드라이버 필요). GPU 백엔드는 `top2` 만 구현하고 검사 규칙은 이 크레이트의 기본 구현을 공유한다.

## 동작 메모

- 재현성: `TwoViewOptions::ransac.random_seed = Some(s)` 이면 (s, 짝 id, 모델 종류)로 파생한 시드를 쓴다.
- 원시 매칭 또는 인라이어가 `min_num_inliers`(기본 15) 미만이면 빈 매칭·기본 기하(UNDEFINED)로 기록한다. 처리한 짝은 항상 두 행(매칭·기하)을 기록한다.
- 실패한 RANSAC 의 모델은 기록하지 않는다(성공한 E/F/H 만).
- 하틀리 정규화 H(`normalize_homography`)는 결과가 미세하게 달라지므로 기본 꺼짐.
- `skip_geometric_verification` 이면 원시 매칭만 기록한다(기하 행 없음).
- 영상 자료(카메라·키포인트·기술자) 누락은 처리 전에 `Error::NotFound`.
