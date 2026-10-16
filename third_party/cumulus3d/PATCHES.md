# cumulus3d 패치

GitHub `AH100-1/cumulus3d` 리비전 `b9f4a9bda714fa31fc0fd395f821114b64fd2c1f`(0.3.1 이후 git head)를 그대로 가져와 SkyLens에 필요한 것만 고친다. 루트 `Cargo.toml`의 `[patch."https://github.com/AH100-1/cumulus3d"]`가 이 사본으로 바꿔 끼운다. 원본 라이선스는 Apache-2.0(`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`).

| 변경 | 이유 |
|---|---|
| (진행 중) 위치마다 도는 단계(SIFT 검출·기술자, 기하 검증, 정합, 삼각측량, 번들 조정)를 CUDA로 | 라이브 복원에서 GPU가 맡는 일이 블러·매칭·조밀화뿐이라, 묶음 하나에 약 4초가 CPU에서 걸렸다(2026-10-16 데모 측정) |
| `crates/cuda`의 `CudaSift`가 SIFT 전 과정을 GPU에서 한다. 스케일 공간에 더해 DoG, 극값 검출(부분화소 보정, 엣지·대비 억제), 방향 할당(키포인트당 여러 방향), 최대 특징 수 선택(`CompatLevels`·`TopK`·`SpatialGrid`), 128차원 기술자(L1-ROOT/L2, 0.2 절단, ×512 양자화)를 커널로 옮기고, 최종 특징·기술자와 레벨별 개수만 내려받는다. 식·매개변수·좌표 규칙·출력 순서는 `CpuSift`와 같지만 합산 순서와 초월함수 구현이 달라 비트 단위로 같지는 않다(같은 품질, 실행마다 같은 결과). 비트 동일을 요구하던 `tests/sift_gpu.rs`는 지우고 `src/sift.rs` 안에 CPU 대비 동등성·회전 축소 사본 매칭 시험을 두었다. CUDA가 실패하면 예전처럼 경고 후 `CpuSift`로 계산한다 | 예전 `CudaSift`는 블러만 GPU에서 하고 옥타브를 내려받아 검출·방향·기술자를 CPU에서 계산해, 1920 화소 영상 3장 묶음에 특징 단계가 약 1.6초 걸렸다 |
| 번들 조정을 CUDA로(`crates/cuda/src/ba.rs`, `kernels/ba.cu`). `cumulus3d-ba`에 풀이 백엔드 트레이트(`BaBackend`, 기본 CPU)와 `bundle_adjust_with`를 두고, 세션의 구역 정밀 BA와 전역 매퍼의 BA 단계가 계획의 `[sparse] ba_backend`(빌더 `gpu()`면 `cuda`)로 백엔드를 고른다. 관측별 잔차·야코비안·견고 손실, 정규방정식, Schur 축소 계통 조립, 축소 계통 풀이(512차원 이하는 내려받아 CPU 밀집 촐레스키, 그보다 크면 GPU 블록 야코비 PCG), 점 역대입을 GPU에서 f64로 하고 LM 루프는 CPU에 둔다. GPU가 실패하면 경고를 찍고 CPU로 다시 푼다. 회전 평균·전역 위치 추정과 점만 정제하는 BA(`refine_points`)는 CPU 그대로다 | 라이브 경로에서 구역 정밀 BA가 CPU로 구역마다 7.7~22초 걸렸다 |
| 희소 단계 백엔드: `TwoViewBackend`(matching), `AbsPoseBackend`·`TriangulationBackend`(sfm)와 CUDA 구현 `CudaSparse`(cuda, `kernels/sparse.cu`). 두 뷰 기하 검증은 블록 안 모든 짝의 E/F/H LO-RANSAC 가설을 라운드마다 커널 한 번으로 채점하고(`core::ransac::BatchedRansac`), 정합은 P3P 가설을 묶어 채점하고, 삼각측량은 후보 트랙마다 스레드 하나가 RANSAC 삼각측량(2점 DLT, 다중 뷰 LO, 깊이·각도 검사)을 하고 재투영 오차·깊이 필터 값을 한꺼번에 계산한다. 가설 생성·LO·모델 갱신은 CPU. 계획의 `sparse.backend`(cpu, cuda), `gpu()` 가 cuda 로 켠다 | 위치 하나(3장)에서 매칭·검증 약 0.56초, 정합 약 0.2초, 삼각측량 약 1.7초가 CPU에서 걸렸다. 채점 커널은 곱셈-덧셈 융합을 끄고 잔차를 CPU와 같은 순서로 계산해 인라이어 수가 같다(잔차 합은 블록 안에서 합치는 순서가 달라 마지막 자리가 다를 수 있다) |
| 희소 단계 GPU 대기 줄이기. `CudaSparse` 스트림을 가장 높은 우선순위로 만들고(`CudaDevice::new_high_priority`), 업로드·내려받기를 고정 메모리 스테이징과 늘기만 하는 장치 버퍼로 한다. `BatchedRansac::propose` 의 묶음 크기를 min_batch 에서 시작해 두 배씩 늘린다(처음부터 512개를 내면 CPU가 수십 번에 멈출 RANSAC 을 수백 번 채점한다). 채점 커널은 스레드마다 모아 블록 끝에서 한 번 합친다. 삼각측량 문제는 관측 수 순으로 정렬해 올린다 | 같은 입력의 CPU 시간과 견주면(`CUMULUS3D_CUDA_COMPARE=1`) 삼각측량·관측 검사·기하 검증이 GPU에서 3~7배 느렸다 |
| 장치에서 내려받는 모든 복사(`memcpy_dtoh`, `clone_dtoh`)를 스트림을 먼저 기다린 뒤 하도록(`device::SyncedCopy`) | 일반 메모리로 내려받는 `cuMemcpyDtoHAsync` 는 앞선 커널을 기다리는 동안 드라이버 전역 잠금을 쥔다. 조밀화가 수백 ms 커널 뒤에서 내려받으면 같은 프로세스의 UNet·희소 단계 CUDA 호출이 모두 그만큼 멈췄다(Nsight Systems 로 확인, 재현은 `examples/bench_contention.rs`) |

cumulus3d를 새 리비전으로 올릴 때는 원본에 같은 기능이 들어갔는지 확인하고, 들어갔으면 패치를 지운다.
