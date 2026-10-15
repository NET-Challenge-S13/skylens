# cumulus3d 패치

GitHub `AH100-1/cumulus3d` 리비전 `b9f4a9bda714fa31fc0fd395f821114b64fd2c1f`(0.3.1 이후 git head)를 그대로 가져와 SkyLens에 필요한 것만 고친다. 루트 `Cargo.toml`의 `[patch."https://github.com/AH100-1/cumulus3d"]`가 이 사본으로 바꿔 끼운다. 원본 라이선스는 Apache-2.0(`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`).

| 변경 | 이유 |
|---|---|
| (진행 중) 위치마다 도는 단계(SIFT 검출·기술자, 기하 검증, 정합, 삼각측량, 번들 조정)를 CUDA로 | 라이브 복원에서 GPU가 맡는 일이 블러·매칭·조밀화뿐이라, 묶음 하나에 약 4초가 CPU에서 걸렸다(2026-10-16 데모 측정) |
| 번들 조정을 CUDA로(`crates/cuda/src/ba.rs`, `kernels/ba.cu`). `cumulus3d-ba`에 풀이 백엔드 트레이트(`BaBackend`, 기본 CPU)와 `bundle_adjust_with`를 두고, 세션의 구역 정밀 BA와 전역 매퍼의 BA 단계가 계획의 `[sparse] ba_backend`(빌더 `gpu()`면 `cuda`)로 백엔드를 고른다. 관측별 잔차·야코비안·견고 손실, 정규방정식, Schur 축소 계통 조립, 축소 계통 풀이(512차원 이하는 내려받아 CPU 밀집 촐레스키, 그보다 크면 GPU 블록 야코비 PCG), 점 역대입을 GPU에서 f64로 하고 LM 루프는 CPU에 둔다. GPU가 실패하면 경고를 찍고 CPU로 다시 푼다. 회전 평균·전역 위치 추정과 점만 정제하는 BA(`refine_points`)는 CPU 그대로다 | 라이브 경로에서 구역 정밀 BA가 CPU로 구역마다 7.7~22초 걸렸다 |

cumulus3d를 새 리비전으로 올릴 때는 원본에 같은 기능이 들어갔는지 확인하고, 들어갔으면 패치를 지운다.
