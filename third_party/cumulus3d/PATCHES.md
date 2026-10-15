# cumulus3d 패치

GitHub `AH100-1/cumulus3d` 리비전 `b9f4a9bda714fa31fc0fd395f821114b64fd2c1f`(0.3.1 이후 git head)를 그대로 가져와 SkyLens에 필요한 것만 고친다. 루트 `Cargo.toml`의 `[patch."https://github.com/AH100-1/cumulus3d"]`가 이 사본으로 바꿔 끼운다. 원본 라이선스는 Apache-2.0(`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`).

| 변경 | 이유 |
|---|---|
| (진행 중) 위치마다 도는 단계(SIFT 검출·기술자, 기하 검증, 정합, 삼각측량, 번들 조정)를 CUDA로 | 라이브 복원에서 GPU가 맡는 일이 블러·매칭·조밀화뿐이라, 묶음 하나에 약 4초가 CPU에서 걸렸다(2026-10-16 데모 측정) |
| 희소 단계 백엔드: `TwoViewBackend`(matching), `AbsPoseBackend`·`TriangulationBackend`(sfm)와 CUDA 구현 `CudaSparse`(cuda, `kernels/sparse.cu`). 두 뷰 기하 검증은 블록 안 모든 짝의 E/F/H LO-RANSAC 가설을 라운드마다 커널 한 번으로 채점하고(`core::ransac::BatchedRansac`), 정합은 P3P 가설을 묶어 채점하고, 삼각측량은 후보 트랙마다 스레드 하나가 RANSAC 삼각측량(2점 DLT, 다중 뷰 LO, 깊이·각도 검사)을 하고 재투영 오차·깊이 필터 값을 한꺼번에 계산한다. 가설 생성·LO·모델 갱신은 CPU. 계획의 `sparse.backend`(cpu, cuda), `gpu()` 가 cuda 로 켠다 | 위치 하나(3장)에서 매칭·검증 약 0.56초, 정합 약 0.2초, 삼각측량 약 1.7초가 CPU에서 걸렸다. 채점 커널은 곱셈-덧셈 융합을 끄고 CPU와 같은 순서로 더해서 같은 시드면 CPU와 같은 인라이어를 낸다 |

cumulus3d를 새 리비전으로 올릴 때는 원본에 같은 기능이 들어갔는지 확인하고, 들어갔으면 패치를 지운다.
