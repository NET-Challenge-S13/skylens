# skylens_model

SkyLens의 AI·복원 모델 코드를 담는 Python 패키지.

1. **Detection** — 4채널 입력(RGB + 열화상) 위의 UNet 백본 + 이중 헤드
   (위험구역 세그멘테이션 / 사람 점 검출). Depth Map 레이캐스팅으로
   2D 탐지를 3D 세계좌표에 투영한다. → `models/skylensnet/`
2. **Reconstruction** — 3DGS 실시간 3D 복원 (GLOMAP 포즈추정 → gsplat 학습 →
   Open3D ICP 멀티드론 융합). → `models/skylens/` — 구현됨, 해당 README 참조

학습은 루트의 `train.ipynb` 에서 돌린다.

> 관련 문서: [ARCHITECTURE.md](../../res/docs/ARCHITECTURE.md) §3-A (모델·융합 파이프라인) ·
> [DATASETS.md](../../res/docs/DATASETS.md) (데이터셋 조사) · [IDEA.md](../../res/docs/IDEA.md) (기획)

---

# 설계 철학 (결정 사항)

> 이 절은 **왜 이렇게 설계했는가**를 기록한다. 구현이 바뀌더라도 여기 적힌 판단 근거는
> 유지되어야 하며, 근거가 무효화되면 결정도 함께 재검토한다.

## 0. 최상위 원칙 — 레이어를 분리한다

SkyLens의 AI는 **하나의 큰 모델이 전부 하는 구조가 아니다.** 서로 다른 성격의 문제를
서로 다른 레이어가 나눠 맡고, 각 레이어는 자기 문제만 푼다.

| 레이어 | 성격 | 무엇을 아는가 |
|---|---|---|
| ① 탐지 (UNet) | **학습됨** · stateless | 이미지 1장. 그게 전부 |
| ② 투영 (레이캐스팅) | 결정론적 기하 | 카메라 포즈 + depth |
| ③ 랜드마크 융합 | **학습 안 됨** · stateful | 3D 공간 전체, 시간 누적 |
| ④ 소리 보정 (YAMNet) | 학습됨(사전) · Late Fusion | 환경음 → confidence 변조 |

이 분리가 이 문서 대부분의 결정을 파생시킨다.

### 원칙 1 — 탐지기는 프레임 1장만 본다 (stateless)

탐지 모델은 **전체 공간에 대한 이해를 갖지 않는다.** 입력은 프레임 한 장,
출력은 그 프레임의 2D 결과. 메모리도, 시계열 입력도, 이전 프레임과의 연결도 없다.
순수 함수로 취급한다.

**왜:**
- 시공간 일관성은 ③이 이미 담당한다. 모델까지 시계열을 보면 역할이 중복된다.
- 시계열 모델(3D conv, temporal attention)은 무겁고 데이터를 훨씬 많이 요구하는데,
  우리는 [DATASETS.md](../../res/docs/DATASETS.md)에서 확인했듯 데이터가 부족하고 파편화돼 있다.
- stateless면 프레임을 드론 구분 없이 큐에 넣고 배치 추론할 수 있어 GPU 효율이 좋다.

**예상 질문 대응** — "왜 모델이 시계열을 안 보나요?"
→ *"탐지는 프레임별로 하고, 시공간 일관성은 지오메트리 기반 랜드마크 융합이 담당합니다."*

### 원칙 2 — 프레임 간 "라벨 합의"는 하지 않는다. 3D 좌표로 합친다

같은 물체가 여러 프레임에 걸쳐 잡힐 때, 2D 외관으로 재식별(Re-ID)하지 않는다.
**각 탐지를 3D 세계좌표로 투영한 뒤 좌표 근접성으로 묶는다.**

포즈(RTK/IMU)가 이미 알려져 있으므로 재투영 좌표를 신뢰할 수 있고, 따라서
"어느 프레임의 어느 픽셀이 다른 프레임의 어느 픽셀과 같은 물체냐"를
외관 매칭 없이 **기하학적으로** 풀 수 있다.

프레임마다 라벨이 흔들려도(0.6 → 0.4 → 미탐) 문제가 되지 않는다. 같은 3D 위치에
누적되는 **증거**로 처리하면 confidence가 자연히 수렴한다. 자세한 알고리즘은 §2.

### 원칙 3 — 모달리티 결손은 예외가 아니라 정상 상태

열화상 카메라가 고장나거나, 드론 일부만 열화상을 달고 있거나, 데이터셋에 한쪽
모달리티만 있는 상황은 **드물지 않다.** 모델은 RGB만, 열화상만, 둘 다 —
세 경우 모두에서 동작해야 한다. 자세한 방법은 [skylensnet §2.2](models/skylensnet/README.md).

ARCHITECTURE.md §3-A가 파이프라인 분리로 SPOF를 회피한 것과 같은 논리를
모델 내부에도 적용한다.

---

> **모델 구조·입력 규약·학습 전략**(UNet 채택 근거, 이중 헤드, modality dropout,
> 프리트레인 스택)은 모델 폴더로 옮겼다 →
> [`models/skylensnet/README.md`](models/skylensnet/README.md)

---

## 1. 멀티드론 처리

**드론이 3대여도 모델 구조는 바뀌지 않는다.** 탐지기 입장에서는
출처 구분 없는 프레임 큐일 뿐이다. 가중치는 하나, 입력만 여러 소스에서 온다.

| 레이어 | 드론 3대의 영향 |
|---|---|
| ① 탐지 | **없음.** 섞어서 배치 추론 |
| ② 투영 | 각 프레임이 자기 드론 pose를 씀. 로직 동일 |
| ③ 융합 | **여기서 이득 발생** ↓ |

**멀티드론의 진짜 가치는 ③에 있다.** 드론 A가 t=10s에, 드론 B가 t=11s에 다른 각도에서
같은 3D 좌표를 잡으면 융합 레이어가 자동으로 같은 랜드마크로 묶는다.
이때 **서로 다른 드론의 근접 시각 관측(시차 있음)이 같은 드론의 재방문보다
훨씬 강한 증거**다 — 두 시점에서 동시에 잡혔다는 건 노이즈가 아닐 확률이 높다.

→ 융합 시 증거 가중치를 다르게 준다: `다른 드론 동시관측 > 같은 드론 재방문`.

---

## 2. 랜드마크 융합 레이어 (신규 컴포넌트)

ARCHITECTURE.md의 파이프라인에 **명시적으로 추가되어야 할 컴포넌트**다.
학습되는 모델이 아니라 **알고리즘**이라는 점이 중요하다.

```
① 영상 AI (UNet, 프레임별 stateless)
② 2D→3D 투영 (Depth Map 레이캐스팅, 프레임별)
③ 랜드마크 융합  ← NEW · 학습 안 됨
     - 3D 거리 기반 클러스터링/게이팅
     - log-odds 베이지안 confidence 누적
     - bidirectional(사전검증) / causal-cursor(실시간) 두 모드
④ 소리 confidence 보정 (Late Fusion, 기존과 동일)
```

### 2.1 알고리즘

```
새 탐지 D (프레임 t) → pose·depth로 3D 투영 → 좌표 X
  └─ 기존 랜드마크 중 |X - Xᵢ| < 임계값(사람 스케일 ~1m) 인 것이 있나?
       ├─ 있음 → 같은 물체로 간주, log-odds 누적 갱신
       │          (이번에도 잡히면 +, 관측 범위에 있었는데 미탐이면 소폭 −)
       └─ 없음 → 새 랜드마크 생성 (confidence = 이번 탐지값)
```

SLAM의 landmark occupancy 업데이트, OctoMap의 log-odds 갱신과 같은 방식이다.
클래스가 어쩌다 흔들려도 누적이 걸러준다.

### 2.2 bidirectional → cursor 전환

원래 "슬라이딩 윈도우를 bidirectional로 만든 뒤 cursor로 전환한다"는 아이디어가
적용되는 지점은 **탐지 모델이 아니라 이 융합 레이어**다.

| 모드 | 용도 | 동작 |
|---|---|---|
| **bidirectional** | 사전 검증 (오프라인) | 한 구간의 모든 재방문 프레임(과거+미래)을 다 놓고 최적 confidence 계산 → 실시간 버전의 검증 기준(pseudo-GT) |
| **cursor (causal)** | 실시간 데모·최종평가 | 현재까지 들어온 프레임만 사용. 재방문마다 log-odds가 한 스텝씩 갱신 |

이 전환은 최적화 목표가 **정확도 → 지연시간**으로 바뀌는 지점이다.
두 버전이 왜 모두 필요한지를 이렇게 설명하면 근거가 명확해진다.

### 2.3 자료구조 (미확정)

voxel hash / KD-tree 등 후보가 있으나 아직 결정하지 않음. §6 참조.

---

## 3. 스플래팅 레벨과 탐지의 관계

**중요: 3DGS 정제 레벨과 탐지 모델은 인과관계가 없다.**
ARCHITECTURE.md §1의 흐름도에서 `RX → 현황판`과 `RX → AI`가 **병렬 분기**인 것이 정확하다.
프레임이 도착하면 UNet은 스플랫 상태와 무관하게 즉시 추론할 수 있다.

레벨과 실제로 엮이는 것은 **투영 정밀도뿐**이다:

| 요소 | 스플래팅 레벨 의존성 |
|---|---|
| 탐지 실행 여부 | ❌ 없음 — 프레임 도착 즉시 |
| 탐지 품질 | ❌ 없음 |
| 투영된 3D 좌표의 **정밀도** | ✅ depth map 품질에 의존 |
| 랜드마크 confidence 수렴 | ❌ 레벨이 아니라 **재방문 횟수·시점 다양성**에 의존 |

즉 confidence가 올라가는 이유는 "스플랫이 고밀도라서"가 아니라
**"같은 대상을 여러 각도에서 반복 관측했기 때문"** 이다.
저밀도(Level 0) depth로도 거친 좌표는 뽑을 수 있고, 레벨이 올라가면
같은 랜드마크의 좌표 정밀도가 함께 개선된다.

---

## 4. 기반 라이브러리

**결정: PyTorch 기반 + `transformers` 인코더 + 자체 UNet 디코더/헤드**

### 4.1 왜 transformers인가

**주의: "transformers를 쓴다"가 "Transformer 모델을 쓴다"는 뜻이 아니다.**
`AutoBackbone`으로 ResNet·ConvNeXt 같은 **CNN 인코더**를 불러오므로
[skylensnet §1.1](models/skylensnet/README.md)의 UNet/CNN 채택 결정과 모순되지 않는다.

- HF [Backbone API](https://huggingface.co/docs/transformers/backbones)가 명시적으로
  **backbone / neck / head 분해**를 전제로 설계돼 있어, 우리 구조
  (공유 인코더 + 이중 헤드)와 철학이 일치한다.
- `AutoBackbone.from_pretrained(..., out_indices=(0,1,2,3))` 가
  **멀티스케일 feature map**을 반환한다 — UNet 디코더 스킵 커넥션에 필요한 것 그 자체.
- `TimmBackbone`으로 timm 생태계까지 흡수 → 인코더 교체가 한 줄.
- [skylensnet §1.1](models/skylensnet/README.md)에서 로드맵으로 남긴 **TransUNet·Mask2Former 비교 실험**이 같은 API로 확장된다.

### 4.2 감수하는 비용

| 필요한 것 | transformers 제공 | 대응 |
|---|---|---|
| 멀티스케일 인코더 | ✅ `AutoBackbone` | 그대로 사용 |
| **UNet 디코더** | ❌ | **자체 구현** (~100줄) |
| 이중 헤드 | ❌ | 자체 구현 (어차피 커스텀) |
| **4채널 입력** | ❌ `in_channels` 파라미터 없음 | 첫 conv 수동 확장 ([skylensnet §3.1](models/skylensnet/README.md)) |
| 4채널 전처리 | ❌ `AutoImageProcessor`는 RGB PIL 전제 | 자체 작성 (radiometric TIFF라 어차피 필요) |

**`segmentation_models_pytorch`(smp)를 택하지 않은 이유**: smp는 UNet 디코더와
`in_channels=4`가 기성품이라 시작은 빠르다. 그러나 이중 헤드·modality dropout·
4채널 전처리를 **어차피 전부 직접 구현**해야 하므로 기성품 이점이 작고,
생태계·확장성에서 transformers가 앞선다. (smp는 초기 베이스라인 대조군으로는 유용)

### 4.3 의존성 스택

```
torch, torchvision       기반
transformers             인코더 (AutoBackbone · CNN 계열) — 버전 pin 필수
timm                     TimmBackbone 경유 인코더 확장
albumentations           증강 — 4채널 + mask + bbox 동시 변환 ★
tifffile (or rasterio)   radiometric thermal TIFF 읽기 (FLAME 3)
numpy, opencv-python     전처리
accelerate               (선택) 분산 · mixed precision
```

**albumentations가 중요한 이유** — 임의 채널 수를 지원하면서 이미지·세그 마스크·
bbox를 **한 번에 같이 변환**한다. 4채널 + 세그 마스크 + 사람 bbox를 동시에 다루는
우리 구조에서, 이게 없으면 증강 단계에서 정합이 깨지기 쉽다.

**학습 루프는 `Trainer` 대신 커스텀**을 쓴다 — 멀티 데이터셋을 배치마다 다른 헤드
loss로 돌리고 modality dropout까지 얹는 구조라, `Trainer`에 맞추는 비용이
직접 구현하는 비용보다 크다.

---

## 5. 채택하지 않은 대안

| 대안 | 기각 사유 |
|---|---|
| TransUNet 기본 채택 | 데이터 부족 상황에서 CNN 대비 불리. 로드맵으로 보류 ([skylensnet §1.1](models/skylensnet/README.md)) |
| VLM 기반 Open-Vocabulary 분할 | 연산 무거워 30초 준실시간 위협 + 프롬프트 민감성 (ARCHITECTURE.md §3-A) |
| 시계열/비디오 입력 모델 | 시공간 일관성은 ③이 담당. 역할 중복 + 데이터 요구량 (§0 원칙 1) |
| 3채널·4채널 모델 2개 분리 | AIResQ 등 단일 모달리티 데이터를 버리게 됨 ([skylensnet §2.2](models/skylensnet/README.md)) |
| 2D 외관 기반 Re-ID로 프레임 간 매칭 | 포즈를 알고 있으므로 기하학적 매칭이 더 정확·단순 (§0 원칙 2) |
| latent(중간) 융합 | 3DGS는 결합할 latent feature가 없고, 매칭된 멀티모달 재난 데이터셋도 없음 (ARCHITECTURE.md §3-A) |
| AIDER를 세그 헤드 학습에 사용 | **분류 라벨만 있고 세그 마스크 없음.** 웹 수집이라 품질 편차 큼 → RescueNet으로 대체 |
| 고정 FPS 추론 | 정지 비행 시 낭비, 고속 이동 시 누락 ([skylensnet §2.4](models/skylensnet/README.md)) |
| `segmentation_models_pytorch` 단독 사용 | 커스텀 비중이 커서 기성품 이점이 작음. 생태계·확장성에서 transformers 우위 (§4.2). 베이스라인 대조군으로는 유지 |
| HF `Trainer` | 멀티 데이터셋 · 헤드별 loss · modality dropout 구조에 맞추는 비용이 큼 (§4.3) |

---

## 6. 미결정 사항

- [ ] 랜드마크 융합의 자료구조 (voxel hash vs KD-tree)
- [ ] log-odds 갱신 파라미터 (증거 가중치, 미탐 페널티 크기)
- [ ] 클래스 통합 스키마 확정 — `{정상 / 화재 / 붕괴 / 도로차단}` + `{사람}`
- [ ] 결손 채널 인코딩 방식 선택 (정규화 예약 vs validity mask 5번째 채널)
- [ ] keyframe 임계값 실측 튜닝 (이동 0.3~0.5m / 회전 10~15°는 초기 추정치)
- [ ] AI Hub 야간 IR 데이터셋 실물 확인 (RGB 페어 여부·시점·라벨 형식)
- [ ] modality dropout 확률 (0.25/0.25/0.5는 초기 제안치)
- [ ] 인코더 확정 (ResNet-50 vs ConvNeXt vs timm 계열) 및 `transformers` 버전 pin

---

# 폴더 구조

```
skylens_model/
  README.md           파이프라인 설계 철학 (이 문서)
  models/
    skylensnet/       ★ 4채널 인지 모델 (transformers 표준)
      README.md         모델 설계 결정 — 구조·입력 규약·학습 전략
      configuration_skylensnet.py
      modeling_skylensnet.py
    skylens/          3DGS 복원 (COLMAP → gsplat, 셸 파이프라인 + 도구)
  datasets/
    README.md         데이터셋 API 계약 · 다운로드 판정
    base.py           SkyLensDatasetBase + 통합 스키마 + 다운로드 유틸
    llvip.py · rescuenet.py · visdrone.py · sard.py · flame.py · airesq.py
  utils/
    collate.py        CenterNet 타겟 인코딩
    trainer.py · training_args.py · callbacks.py · metrics.py
    geo.py            ENU ↔ GPS 변환, src/skylens_core/geo.ts 의 미러
  tests/
    test_datasets.py  데이터셋 계약 검증
```

# 클라이언트와의 연결

TypeScript 클라이언트(`src/skylens_core/`)는 이 패키지를 직접 호출하지 않는다.
`src/skylens_core/protocol.ts`에 정의된 와이어 프로토콜로 결과만 소비한다.
이 패키지의 역할은 결국 그 스키마로 직렬화될 값을 **생산**하는 것이다.

- `SkyLensForDisasterPerception` 의 출력(세그 맵 + 점 검출)이 Depth Map 레이캐스팅을
  거쳐 `protocol.ts` 의 `DetectionResult`(`category`, `gps`, `confidence`, `label`)로 직렬화된다.

좌표는 `src/skylens_core/geo.ts`에 정의되고 `skylens_model/utils/geo.py`에 미러된
**ENU(East/North/Up) 규약**으로 공유한다 — GPS in, 로컬 미터 out.
덕분에 이 패키지가 내놓는 탐지 결과와 스플랫 청크가 클라이언트가 렌더하는
동일한 씬 프레임에 정렬된다.
