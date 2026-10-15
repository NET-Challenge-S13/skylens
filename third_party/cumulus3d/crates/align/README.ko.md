[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/align/README.md) | 한국어

# cumulus3d-align

좌표 정렬과 점군 후처리: GPS(WGS84) ↔ ENU 변환, 재구성을 GPS ENU 에 맞추는 강건 Sim3 정렬,
두 재구성의 공유 3D 점 정렬, 구역 좌표계 연쇄(reanchor), 점군 마스킹·추출·스냅샷 합성.
모든 Sim3 는 core 규약 `new_from_old` (`X' = sRX + t`).

**파이프라인 단계**: SfM 다음. 구역 정밀본을 GPS 기준 ENU(정밀 지도)에 맞추고(`align_to_gps`),
초벌을 정밀본 좌표계로 옮기며(`align_reconstructions`), 구역 간 좌표계를 이어(`AnchorChain`)
단계별 스냅샷 점군(`compose_snapshot`)을 만든다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `align_to_gps` | 등록 영상 투영 중심을 GPS ENU 에 강건 Sim3 로 맞추고 모델에 적용(실패 시 모델 불변) | `&mut Reconstruction`, `&[GpsRecord]`, `&ModelAlignerOptions` → `Result<GpsAlignment>` |
| `align_to_gps_file` | 위와 같고 GPS 목록을 파일에서 읽음 | `&mut Reconstruction`, 경로, `&ModelAlignerOptions` → `Result<GpsAlignment>` |
| `estimate_gps_alignment` | 정렬 Sim3 만 추정(모델 불변) | `&Reconstruction`, `&[GpsRecord]`, `&ModelAlignerOptions` → `Result<GpsAlignment>` |
| `align_reconstructions` | 공유 3D 점 + 견고 Umeyama 로 두 재구성 사이 Sim3 추정 | `&Reconstruction`×2, `&SharedPointOptions`, `&RobustUmeyamaOptions` → `Result<(RobustUmeyamaResult, 대응 쌍)>` |
| `AnchorChain` | 구역별 "최신 좌표계 ← 구역 좌표계" 변환 연쇄 | `push_model(prev, new, ..)` → `zone_to_latest(k)` |
| `compose_snapshot` | 정밀 구역 + (겹치지 않는) 초벌 구역 합성 후 1/N 추출 | `&[&PointCloud]`, `&[&PointCloud]`, `&SnapshotOptions` → `PointCloud` |
| `EnuFrame` | 고정 원점 ENU 좌표계(LLA/ECEF/ENU 상호 변환) | `new(lat, lon, alt)` → `lla_to_enu`, `enu_to_lla`, … |
| `umeyama` | 대응점 Sim3(또는 강체) 닫힌 해 | `&[Vec3]`, `&[Vec3]`, `bool` → `Option<Sim3>` |
| `robust_umeyama` | 중앙값 기반 반복 재적합 Umeyama | `&[Vec3]`, `&[Vec3]`, `&RobustUmeyamaOptions` → `Option<RobustUmeyamaResult>` |
| `estimate_sim3_ransac` | LO-RANSAC 견고 Sim3 | `&[Vec3]`, `&[Vec3]`, `&RansacParams`, `RankCheck` → `RansacReport<Sim3>` |
| `transform_cloud` | 점군에 Sim3 적용(점·법선) | `&mut PointCloud`, `&Sim3` → `()` |
| `KdTree` | 정적 3D KD-트리(최근접·반경 질의, 병렬판 포함) | 점 목록 → 질의 결과 |

## 공개 항목

"루트" 열이 ✓ 인 항목은 크레이트 루트(`cumulus3d_align::`)에도 재노출된다.

### `cloud` — 점군 후처리(core `PointCloud`)

| 항목 | 종류 | 루트 | 역할 |
|---|---|---|---|
| `transform_cloud` | fn | ✓ | 점 sRx+t, 법선 Rn 적용(f64 계산) |
| `select` | fn | | 마스크가 참인 점만 남긴 새 점군 |
| `remove_near` | fn | ✓ | 기준 KD-트리와 거리 ≤ r 인 점 제거 |
| `decimate` | fn | ✓ | 1/stride 간격 추출(인덱스 0, N, 2N, …) |
| `merge_clouds` | fn | ✓ | 이어 붙이기(없는 법선은 0, 없는 색은 흰색으로 채움) |
| `SnapshotOptions` | struct | ✓ | `mask_radius`(기본 1.5 m), `stride`(기본 6) |
| `compose_snapshot` | fn | ✓ | 정밀 전부 + 정밀과 겹치지 않는 초벌, 그 뒤 1/N 추출(입력은 같은 좌표계) |

### `error`

| 항목 | 종류 | 루트 | 역할 |
|---|---|---|---|
| `AlignError` | enum | ✓ | `Core`, `BadMaxError`, `InvalidArgument`, `TooFewReferences`, `TooFewCommonImages`, `TooFewCorrespondences`, `RansacFailed`, `Degenerate` |
| `Result<T>` | type | ✓ | `std::result::Result<T, AlignError>` |

### `geodesy` — WGS84 LLA ↔ ECEF ↔ ENU

| 항목 | 종류 | 루트 | 역할 |
|---|---|---|---|
| `WGS84_A`, `WGS84_F`, `WGS84_B`, `WGS84_E2` | const | ✓ | 장반경, 편평률, 단반경, 이심률² |
| `lla_to_ecef` | fn | ✓ | (위도°, 경도°, 타원체고 m) → ECEF |
| `ecef_to_lla` | fn | ✓ | ECEF → (위도°, 경도°, 타원체고 m)(반복법, 극점은 닫힌 해) |
| `EnuFrame` | struct | ✓ | 원점 고정 ENU 좌표계. 필드 `origin_lla`, `origin_ecef`, `rotation` |
| `EnuFrame::new` | fn | | 원점 LLA 로 생성 |
| `EnuFrame::ecef_to_enu` / `enu_to_ecef` | fn | | ECEF ↔ ENU |
| `EnuFrame::lla_to_enu` / `enu_to_lla` | fn | | LLA ↔ ENU |

### `kdtree`

| 항목 | 종류 | 루트 | 역할 |
|---|---|---|---|
| `KdTree` | struct | ✓ | 정적 KD-트리(원래 인덱스 보관) |
| `KdTree::new` / `from_f32` | fn | | f64 / f32(PLY) 점으로 생성 |
| `KdTree::len` / `is_empty` | fn | | 점 개수 / 비었는지 |
| `KdTree::nearest` | fn | | 최근접 (원래 인덱스, 거리²) |
| `KdTree::any_within` / `within_radius` | fn | | 반경 r 안 점 존재 여부 / 모든 점 |
| `KdTree::nearest_many` / `any_within_many_f32` | fn | | 병렬 질의판 |

### `model_aligner` — GPS ENU 정렬

| 항목 | 종류 | 루트 | 역할 |
|---|---|---|---|
| `EnuOrigin` | enum | ✓ | `FirstRecord`(기본: GPS 목록 첫 줄), `Explicit { lat, lon, alt }`(세션 고정 원점) |
| `ModelAlignerOptions` | struct | ✓ | `max_error`(3 m), `min_common_images`(3), `origin`, `ransac`, `rank_check`(`Uncentered`) |
| `GpsAlignment` | struct | ✓ | `sim3`(model_to_enu), `enu`, `common`, `inlier_mask`, `num_inliers`, `num_trials`, `errors`, `mean_error`, `median_error` |
| `gps_to_enu` | fn | ✓ | GPS 목록 → (ENU 좌표계, (이름, ENU) 목록) |
| `estimate_gps_alignment` | fn | ✓ | 정렬 Sim3 추정(모델 불변). 실패: `max_error ≤ 0`, 기준 < 3, 공통 영상 부족, RANSAC 실패 |
| `align_to_gps` | fn | ✓ | 추정 후 성공하면 `rec.transform(sim3)` |
| `align_to_gps_file` | fn | ✓ | GPS 파일 경로판 |

### `reanchor`

| 항목 | 종류 | 루트 | 역할 |
|---|---|---|---|
| `AnchorChain` | struct | ✓ | 구역 좌표계 연쇄 Sim3 묶음 |
| `AnchorChain::new` / `len` / `is_empty` | fn | | 생성 / 구역 수 / 비었는지 |
| `AnchorChain::push` | fn | | `Option<new_from_prev>` 로 새 구역 추가(Some 이면 이전 구역 전부에 합성) |
| `AnchorChain::push_model` | fn | | 직전·새 정밀 모델의 공유 점으로 연결 Sim3 를 구해 추가(실패해도 구역은 추가, 오류 반환) |
| `AnchorChain::zone_to_latest` / `is_linked` | fn | | 구역 → 최신 좌표 변환 / 직전 구역과 연결 여부 |

### `shared` — 두 재구성의 공유 3D 점

| 항목 | 종류 | 루트 | 역할 |
|---|---|---|---|
| `NameFilter` | type | | 영상 이름 필터 `Arc<dyn Fn(&str) -> bool + Send + Sync>` |
| `SharedPointOptions` | struct | ✓ | `position_range`(위치 인덱스 범위), `name_filter` |
| `position_index_from_name` | fn | ✓ | 파일 줄기의 마지막 숫자열(`camF/camF_0012.jpg` → 12) |
| `shared_point_correspondences` | fn | ✓ | 같은 이름 영상·같은 2D 인덱스로 (A 점 id, B 점 id) 쌍(정렬·중복 제거) |
| `align_reconstructions` | fn | ✓ | 공유 점 + 견고 Umeyama → `src_to_dst` |

### `umeyama` — Sim3 추정

| 항목 | 종류 | 루트 | 역할 |
|---|---|---|---|
| `RankCheck` | enum | ✓ | 퇴화 검사: `Uncentered`(기본), `Centered`(공선 배치만 거부), `None` |
| `umeyama` | fn | ✓ | Umeyama 닫힌 해(src_to_dst; 3점 미만·퇴화면 None) |
| `Sim3Estimator` | struct | ✓ | core `ransac::Estimator` 구현(최소 3쌍, 잔차 ‖y − Tx‖²) |
| `estimate_sim3_ransac` | fn | ✓ | LO-RANSAC 견고 Sim3(`max_error` = 거리 임계) |
| `RobustUmeyamaOptions` | struct | ✓ | `iterations`(5), `factor`(3), `min_threshold`(0.3), `estimate_scale`(true) |
| `RobustUmeyamaResult` | struct | ✓ | `sim3`, `median_residual`, `num_inliers`, `inlier_mask`, `residuals` |
| `robust_umeyama` | fn | ✓ | 반복 재적합: 임계 = max(factor·중앙 잔차, min_threshold) |

## 사용 예

크레이트 문서의 doc-test 와 같은 코드(합성 데이터).

```rust
use cumulus3d_align::{compose_snapshot, umeyama, EnuFrame, SnapshotOptions};
use cumulus3d_core::io::PointCloud;
use cumulus3d_core::{Quat, Sim3, Vec3};

// GPS(WGS84) ↔ ENU: 원점 기준 동쪽·북쪽·위 미터 좌표.
let enu = EnuFrame::new(37.5, 127.0, 10.0);
let p = enu.lla_to_enu(37.5001, 127.0001, 10.0);
let (lat, lon, _alt) = enu.enu_to_lla(&p);
assert!((lat - 37.5001).abs() < 1e-9 && (lon - 127.0001).abs() < 1e-9);

// 대응점 집합 사이의 Sim3(src → dst) 추정.
let truth = Sim3::new(2.0, Quat::from_axis_angle(&Vec3::z(), 0.3), Vec3::new(1.0, 2.0, 3.0));
let src: Vec<Vec3> = (0..10).map(|i| Vec3::new(i as f64, (i * i) as f64 * 0.1, (i as f64).sin())).collect();
let dst: Vec<Vec3> = src.iter().map(|x| truth.transform_point(x)).collect();
let est = umeyama(&src, &dst, true).unwrap();
assert!((est.scale - 2.0).abs() < 1e-9);

// 스냅샷: 정밀 점과 1.5 m 안에 있는 초벌 점을 지우고 합친 뒤 1/stride 추출.
let fine = PointCloud { positions: vec![[0.0, 0.0, 0.0]], ..Default::default() };
let coarse = PointCloud { positions: vec![[0.5, 0.0, 0.0], [10.0, 0.0, 0.0]], ..Default::default() };
let snap = compose_snapshot(&[&fine], &[&coarse], &SnapshotOptions { mask_radius: 1.5, stride: 1 });
assert_eq!(snap.len(), 2);
```

재구성을 GPS 기준 ENU 로 정렬:

```rust,no_run
use cumulus3d_align::{align_to_gps_file, ModelAlignerOptions};
let mut rec = cumulus3d_core::interop::read_model("model/0").unwrap();
let a = align_to_gps_file(&mut rec, "gps_ref.txt", &ModelAlignerOptions::default()).unwrap();
println!("인라이어 {}/{}, 중앙 오차 {:.2} m", a.num_inliers, a.common.len(), a.median_error);
```

## 기능 플래그·하드웨어

- 기능 플래그 없음. CPU 전용(점군 처리·KD-트리 질의는 rayon 병렬).
- GPS 파일 형식(`이름 위도 경도 고도` 줄)은 `cumulus3d_core::io::read_gps_file` 이 읽는다.
