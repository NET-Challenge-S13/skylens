# asrun — 실제로 돌린 스크립트 원본

`RESULTS.md`의 모든 수치를 만든 스크립트를 **손대지 않고 그대로** 넣었다.
2026-08-09 ~ 08-16, 로컬 GPU에서 실행된 것들이다.

상위 폴더의 `pipeline/`·`experiments/`는 이걸 저장소용으로 일반화한 것이다.
**그쪽은 그 형태 그대로 끝까지 돌려본 적이 없다** — 문법 검사와 파이썬 도구 단위 검증만 거쳤다.
결과를 재현하거나 "정확히 무엇을 돌렸나"를 확인해야 할 때는 이쪽이 정본이다.

## 주의

- **경로가 `/root/skylens` 로 하드코딩돼 있다.** 다른 환경에서 그대로 돌아가지 않는다.
- 실험용 임시 코드라 정리돼 있지 않다. 읽는 용도지 실행 용도가 아니다.
- `incr_trainer.py` 는 gsplat `examples/` 안에서 도는 것을 전제한다
  (`simple_trainer` 를 임포트해 패치한다).

## 무엇이 무엇을 만들었나

| 파일 | 만든 결과 |
|---|---|
| `corridor_hq_local.sh` | 프레임 540장 추출 + SIFT COLMAP 재구성 → 실험 1의 SIFT 쪽 |
| `corridor_aliked.sh` | ALIKED + LightGlue 재구성 → 실험 1의 ALIKED 쪽, 이후 모든 실험의 기반 모델 |
| `chain_aliked_train.sh` | 위 재구성 완료를 기다렸다가 학습 체인 |
| `train_aliked.sh` | 30,000스텝 학습 + LOD 체크포인트 11단계 → **실험 2** |
| `chain_pose_sweep.sh` | 포즈 오차 8조건 → **실험 3** |
| `baseline_sweep.sh` | 편대 6조건 → **실험 4** (평가 세트 문제는 `RESULTS.md` 참조) |
| `incr_trainer.py` | 웜스타트 래퍼 — `create_splats_with_optimizers` 를 패치해 이전 체크포인트를 초기값으로 쓴다 → **실험 5** |

## 실험 5가 실패한 지점

`incr_trainer.py` 자체는 의도대로 동작했다. 문제는 gsplat 데이터 로더가
`normalize=True` 일 때 카메라 집합으로부터 좌표 변환을 매번 새로 계산한다는 것이었다.
75장 기준과 150장 기준 변환이 87.8° 틀어져 있어서, 이어받은 가우시안이
회전된 공간에 떨어졌다. 자세한 것은 `../README.md` §5.
