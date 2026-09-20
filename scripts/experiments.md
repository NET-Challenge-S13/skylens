# 실험 실행 명령

각 실험을 어떤 명령으로 돌렸는지 적는다. 수치는 실험 PR이 갖고, 여기는 **그 수치를 다시 만드는 방법**만 둔다.

모두 저장소 루트(또는 실험 워크트리)에서 돌린다. `data/`는 메인 체크아웃 것을 자동으로 찾으므로 따로 지정하지 않는다.

| 실험 | 브랜치 | PR |
|---|---|---|
| 기준선 | `develop/v1` | 없음 (`runs/skylens/checkpoint-14100`) |
| 세그 클래스 가중치 | `experiment/seg-class-weights` | [#24](https://github.com/NET-Challenge-S13/skylens/pull/24) |
| road 오버샘플링 | `experiment/road-oversample` | [#26](https://github.com/NET-Challenge-S13/skylens/pull/26) |
| 구성비 보존 오버샘플링 | `experiment/balanced-oversample` | [#27](https://github.com/NET-Challenge-S13/skylens/pull/27) |
| 표본 + 가중치 결합 | `experiment/balanced-plus-weights` | [#28](https://github.com/NET-Challenge-S13/skylens/pull/28) |
| 구성비 보존 5 에폭 | `experiment/balanced-5ep` | 이 PR |

## 기준선

가중치도 오버샘플링도 없다. `runs/skylens/checkpoint-14100`이 이 설정의 결과다.

```bash
uv run python scripts/train_experiment.py --run-name baseline --epochs 2
```

## 세그 클래스 가중치 (#24)

네 클래스 모두에 역제곱근 가중치. 기각됐다.

```bash
uv run python scripts/train_experiment.py --run-name seg-class-weights --epochs 2 \
    --danger-class-weights 1.0 10.19 4.67 7.52
```

## road 오버샘플링 (#26)

`road_blocked` 픽셀이 있는 RescueNet 이미지 561장을 4배로. `road_blocked`가 처음 0을 벗어났다.

```bash
uv run python scripts/train_experiment.py --run-name road-oversample --epochs 2 \
    --road-oversample 4
```

## 구성비 보존 오버샘플링 (#27)

위에 더해 fire_seg도 4배로 불려 세그 데이터셋 비율을 유지한다.

```bash
uv run python scripts/train_experiment.py --run-name balanced-oversample --epochs 2 \
    --road-oversample 4 --balance-fire-seg
```

## 표본 + 가중치 결합 (#28)

`#27`의 표본 구성에 `road_blocked`만 8.0 가중치. 기각됐다.

```bash
uv run python scripts/train_experiment.py --run-name balanced-plus-weights --epochs 2 \
    --road-oversample 4 --balance-fire-seg --danger-class-weights 1.0 1.0 1.0 8.0
```

## 구성비 보존 5 에폭 (이 실험)

`#27`과 완전히 같은 설정을 5 에폭으로. 에폭 수 외에 바뀌는 것이 없다.

```bash
uv run python scripts/train_experiment.py --run-name balanced-5ep --epochs 5 \
    --road-oversample 4 --balance-fire-seg
```

## 실행에 관한 사실

- GPU는 RTX 4050 Laptop(VRAM 6GB)이고 512px batch 8에서 약 3.1GB를 쓴다. 동시에 두 실험을 돌릴 수 없다.
- 2 에폭 실행이 약 2시간이다. 오버샘플링을 켜면 한 에폭의 step 수가 2,905에서 3,545로 늘어 더 걸린다.
- 평가는 에폭 경계에서만 돈다. 학습 도중에는 손실만 보인다.
- 결과는 끝난 뒤 `rt.log()`로 PR의 YAML 블록에, 에폭별 전체 지표 표는 PR 본문 `측정값 전체` 절에 자동으로 올라간다.

## sard-person-tiles

```
python scripts/train_experiment.py --run-name sard-person-tiles --epochs 5 --road-oversample 4 --balance-fire-seg --dice-loss-weight 1.0 --visdrone-tiles --sard-tiles --offset-head --num-workers 8
```

## eval-size-relative-match

```
python scripts/eval_deploy.py --ckpt <run>/final --data-root <data> --bs 4
```

## person-head-stride2-v4

```
python scripts/train_experiment.py --run-name person-head-stride2-v4 --epochs 5 --road-oversample 4 --balance-fire-seg --dice-loss-weight 1.0 --visdrone-tiles --offset-head --sard tiles --person-head-stride 2 --num-workers 8
```

## stride2-radius-combo

```
python scripts/train_experiment.py --run-name stride2-radius-combo --epochs 5 --road-oversample 4 --balance-fire-seg --dice-loss-weight 1.0 --visdrone-tiles --offset-head --sard tiles --person-head-stride 2 --radius-rounding round --num-workers 8
```

## combo-wh-weight

```
python scripts/train_experiment.py --run-name combo-wh-weight --epochs 5 --road-oversample 4 --balance-fire-seg --dice-loss-weight 1.0 --visdrone-tiles --offset-head --sard tiles --person-head-stride 2 --radius-rounding round --wh-loss-weight 1.0 --num-workers 8
```
