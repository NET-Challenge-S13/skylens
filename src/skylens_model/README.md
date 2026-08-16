# skylens_model

SkyLens의 AI·복원 모델 코드를 담는 Python 패키지.

1. **Detection** — 4채널 입력(RGB + 열화상) 위의 UNet 백본 + 이중 헤드
   (위험구역 세그멘테이션 / 사람 점 검출). Depth Map 레이캐스팅으로
   2D 탐지를 3D 세계좌표에 투영한다.
2. **Reconstruction** — 3DGS 실시간 3D 복원 (GLOMAP 포즈추정 → gsplat 학습 →
   Open3D ICP 멀티드론 융합).

현재는 **스캐폴드 상태**다 — dataclass와 인터페이스 스텁만 있고, 학습된 가중치도
텐서 연산도 서드파티 ML 의존성도 없다.

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
누적되는 **증거**로 처리하면 confidence가 자연히 수렴한다. 자세한 알고리즘은 §5.

### 원칙 3 — 모달리티 결손은 예외가 아니라 정상 상태

열화상 카메라가 고장나거나, 드론 일부만 열화상을 달고 있거나, 데이터셋에 한쪽
모달리티만 있는 상황은 **드물지 않다.** 모델은 RGB만, 열화상만, 둘 다 —
세 경우 모두에서 동작해야 한다. 자세한 방법은 §3.

ARCHITECTURE.md §3-A가 파이프라인 분리로 SPOF를 회피한 것과 같은 논리를
모델 내부에도 적용한다.

---

## 1. 모델 구조

### 1.1 UNet 채택 (TransUNet 보류)

**결정: 순수 UNet 계열** (ResNet/EfficientNet 인코더 + UNet 디코더).
TransUNet은 데이터 확보 후 실험 로드맵으로 남긴다.

**왜:**
- **데이터가 적고 파편화됨.** 사용 가능한 데이터셋이 수천~수만 장 규모에 도메인도
  제각각이다. Transformer는 CNN보다 데이터 요구량이 크고, 데이터가 적으면 오히려
  성능이 떨어지거나 학습이 불안정해진다. UNet은 지역성(locality) inductive bias
  덕에 적은 데이터로도 안정적으로 수렴한다.
- **태스크 특성.** 화재·연기·잔해·사람은 지역적 텍스처·형태 패턴이 강한 대상이라,
  Transformer의 강점인 장거리 문맥이 결정적이지 않다.
- **자원 경쟁.** Core HPC GPU를 3DGS 학습과 나눠 쓰므로, 더 가볍고 추론 시간이
  예측 가능한 쪽이 스케줄링에 유리하다.
- ARCHITECTURE.md §3-A가 VLM 기반 Open-Vocabulary 분할을 배제한 근거
  ("재난 대응은 최신성보다 신뢰성·실시간성")가 TransUNet에도 그대로 적용된다.

> ⚠️ ARCHITECTURE.md·IDEA.md에는 아직 "UNet / TransUNet"이 병기돼 있다.
> **UNet이 기본값**임을 반영하는 문서 수정이 필요하다.

### 1.2 단일 백본 + 이중 헤드

4채널을 공유 인코더가 인코딩한 뒤 두 헤드로 분기한다. 두 태스크가 같은 특징을
공유하므로 멀티태스크 구조가 연산상 효율적이다.

| 헤드 | 출력 | 대상 | 형태 |
|---|---|---|---|
| **세그멘테이션 헤드** | per-pixel 클래스 맵 | 위험구역 (stuff: 붕괴·화재·도로차단) | 영역 마스크 |
| **점 검출 헤드** | 중심점 히트맵 + (w,h) 회귀 | 사람 (인스턴스) | **점 + confidence** |

**왜 위험구역은 영역이고 사람은 점인가:**
- 위험구역은 전경 객체가 아니라 일반 배경과 구분되는 '2차 배경(stuff)'이라
  영역 분할이 적합하다.
- 사람은 **이후 단계가 점을 요구한다.** 핀홀 역투영 식이 $(u, v, Z)$ 를 받으므로
  투영 단계에서 어차피 점 하나로 줄여야 하고, 소리 confidence 보정(④)도
  '점 위치 + confidence' 단위로 동작한다.

### 1.3 "인스턴스 헤드" → **점 검출 헤드** (명칭 정정)

기존 문서·코드의 "인스턴스 헤드"라는 이름은 통상 instance segmentation
(인스턴스별 마스크)을 뜻하므로 **오해를 부른다.** 실제로 하는 일은 중심점 검출이다.

> 🔧 **TODO**: `models/detection.py` 독스트링의 "instance head for people"과
> ARCHITECTURE.md §3-A의 "인스턴스 헤드"를 **"점 검출 헤드"**로 정정할 것.

### 1.4 bbox와 점의 관계 — 둘 다 쓴다

혼동하기 쉬운 지점이라 명시한다. **bbox는 라벨 쪽, 점은 투영 입력 쪽이다.**

```
학습 라벨:        bbox           (SARD·KAIST·LLVIP·VisDrone 모두 bbox 제공)
      ↓ 학습 시 변환 (중심에 가우시안 → GT 히트맵, w·h는 회귀 타깃)
모델 출력:  ① 중심점 히트맵  → peak = (u,v)   ← 투영에 사용
            ② (w, h) 회귀    → bbox 복원
      ↓
투영:       (u,v) → depth 조회 → 3D 세계좌표
```

**CenterNet 방식을 쓰는 이유:**
- UNet 디코더 출력과 히트맵이 **같은 dense map 형태**라 헤드 구조가 대칭이 되고,
  anchor·NMS 같은 부속물이 필요 없다.
- 사람이 초소형 객체(SARD 기준 중앙값 면적이 전체의 **0.1% 미만**)라
  anchor 기반보다 히트맵이 유리하다.
- bbox 라벨을 버리지 않고 전부 활용한다.

**bbox를 남겨두는 실질적 이유** — 중심점 픽셀 하나의 depth만 읽으면,
그 지점이 잔해에 가려졌거나 스플랫이 비어 있을 때 마커가 공중에 뜨거나 벽 뒤로 박힌다.
bbox가 있으면 **박스 영역 내 depth의 median**을 취해 robust하게 뽑을 수 있다.
(ARCHITECTURE.md §3-A의 "지표면 강제 투영" 트릭보다 근본적인 해법)

### 1.5 소형 객체 대응

드론 고도에서 사람은 수십 픽셀에 불과하다. 따라서:
- **고해상도 입력 유지** — 과도한 다운샘플링 금지
- **얕은 층 스킵 커넥션 적극 활용** — UNet 구조의 본래 강점
- 필요 시 **타일 단위 추론** 검토

---

## 2. 입력 규약

### 2.1 4채널 Early Fusion

RGB(3ch) ⊕ 열화상(1ch) = 4채널 concat → 단일 백본.
가시광이 약한 야간·연기 상황을 열화상이 보완하므로 입력단에서 합쳐 함께 학습시킨다.
(ARCHITECTURE.md §3-A의 Early Fusion 결정 유지)

### 2.2 대칭 Modality Dropout — 3채널/4채널을 **한 모델로** 커버

모델을 둘로 나누지 않는다. 학습 중 모달리티를 확률적으로 누락시켜
하나의 가중치가 세 가지 입력 조합을 모두 처리하게 한다.

```
학습 시 매 배치마다:
  p=0.25   열화상 제거 → RGB-only 모드
  p=0.25   RGB 제거    → thermal-only 모드
  p=0.50   둘 다 사용  → 4채널 모드

추론 시: 있는 모달리티만 넣는다. 가중치는 하나.
```

**왜 이게 이 프로젝트에 특히 중요한가 — 데이터 파편화가 해결된다:**

| 데이터셋 | 모달리티 | 단일 모델 학습 |
|---|---|---|
| SARD, RescueNet, FLAME, VisDrone | RGB만 | ✅ RGB-only 모드 |
| AIResQ (9,788장 고해상도 IR) | **열화상만** | ✅ thermal-only 모드 |
| FLAME 3, KAIST, LLVIP | RGB+열 페어 | ✅ 4채널 모드 |

모델을 둘로 나누면 열화상 단독인 AIResQ는 어느 쪽에도 못 넣고 **버려진다.**
대칭 dropout이면 [DATASETS.md](../../res/docs/DATASETS.md)의 모든 데이터셋을 하나의 모델에 태울 수 있다.

**운용상 이점** — 열화상 카메라 고장이나 기종 혼용 시에도 graceful degradation.
모달리티 하나가 죽어도 시스템은 계속 돈다.

**트레이드오프 (솔직히 기록)** — dropout 모델은 4채널 전용 모델 대비
완전체 모드 성능이 소폭 낮다(보통 몇 %p). 다만 데이터 총량 증가로 상쇄되거나
역전될 가능성이 높다. 데이터 부족이 주 병목이므로 이 교환은 유리하다고 판단했다.

### 2.3 결손 채널 인코딩

누락 채널을 그냥 0으로 채우면 안 된다. FLAME 3의 radiometric thermal은
**픽셀당 실제 온도값**이라 0이 유효한 값일 수 있고, 모델이
"열화상 없음"과 "열화상이 차갑다"를 구분하지 못한다.

- **(간단)** 열화상을 `[0.1, 1.0]`으로 정규화하고 `0`을 "없음" 전용으로 예약
- **(명확)** 5번째 채널로 validity mask(0/1) 추가 — 비용 거의 없고 모호성이 사라짐

### 2.4 추론 주기 — motion-gated keyframe

**고정 FPS로 넣지 않는다.** 드론이 정지 비행하면 연속 프레임이 거의 동일해서
새로운 3D 증거가 늘지 않는데 GPU만 소모한다.

```
새 프레임 도착
  └─ 직전 keyframe 대비 이동 ≥ 0.3~0.5m 또는 회전 ≥ 10~15°?
       ├─ 예   → keyframe 채택 → 포즈 추정 + UNet 추론 둘 다 여기에 태움
       └─ 아니오 → 폐기
          (단, 마지막 keyframe 이후 2초 초과 시 정지 비행 대비 강제 채택)
```

**핵심: keyframe 선택을 한 번만 하고 3D 복원(GLOMAP→gsplat)과 AI 탐지가 공유한다.**
어차피 포즈 추정이 "이 프레임이 새로운 시점을 제공하는가"를 판단해야 하므로,
탐지기는 그 판단에 무임승차한다. 별도 파이프라인을 만들지 않는다.

이로써 "몇 Hz로 넣을까"는 **"복원용 keyframe 임계값을 얼마로 둘까"** 라는
기존 파라미터 튜닝 문제로 흡수된다.

> 지휘관에게 보여주는 **라이브 영상은 원본 풀프레임 그대로**다.
> keyframe 서브샘플링은 모델 입력에만 적용된다.

---

## 3. 멀티드론 처리

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

## 4. 랜드마크 융합 레이어 (신규 컴포넌트)

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

### 4.1 알고리즘

```
새 탐지 D (프레임 t) → pose·depth로 3D 투영 → 좌표 X
  └─ 기존 랜드마크 중 |X - Xᵢ| < 임계값(사람 스케일 ~1m) 인 것이 있나?
       ├─ 있음 → 같은 물체로 간주, log-odds 누적 갱신
       │          (이번에도 잡히면 +, 관측 범위에 있었는데 미탐이면 소폭 −)
       └─ 없음 → 새 랜드마크 생성 (confidence = 이번 탐지값)
```

SLAM의 landmark occupancy 업데이트, OctoMap의 log-odds 갱신과 같은 방식이다.
클래스가 어쩌다 흔들려도 누적이 걸러준다.

### 4.2 bidirectional → cursor 전환

원래 "슬라이딩 윈도우를 bidirectional로 만든 뒤 cursor로 전환한다"는 아이디어가
적용되는 지점은 **탐지 모델이 아니라 이 융합 레이어**다.

| 모드 | 용도 | 동작 |
|---|---|---|
| **bidirectional** | 사전 검증 (오프라인) | 한 구간의 모든 재방문 프레임(과거+미래)을 다 놓고 최적 confidence 계산 → 실시간 버전의 검증 기준(pseudo-GT) |
| **cursor (causal)** | 실시간 데모·최종평가 | 현재까지 들어온 프레임만 사용. 재방문마다 log-odds가 한 스텝씩 갱신 |

이 전환은 최적화 목표가 **정확도 → 지연시간**으로 바뀌는 지점이다.
두 버전이 왜 모두 필요한지를 이렇게 설명하면 근거가 명확해진다.

### 4.3 자료구조 (미확정)

voxel hash / KD-tree 등 후보가 있으나 아직 결정하지 않음. §7 참조.

---

## 5. 스플래팅 레벨과 탐지의 관계

**중요: 3DGS 정제 레벨과 탐지 모델은 인과관계가 없다.**
ARCHITECTURE.md §1의 흐름도에서 `RX → RECON`과 `RX → AI`가 **병렬 분기**인 것이 정확하다.
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

## 6. 학습 전략

### 6.1 프리트레인 스택

```
백본        ResNet-50 (또는 EfficientNet-B3), ImageNet 프리트레인
            └─ 첫 conv 4채널 inflation:
                 weight[:, :3] = 프리트레인 RGB 가중치 (그대로)
                 weight[:, 3]  = 프리트레인 RGB 가중치.mean(dim=1)
               ※ 4번째 채널을 RGB 평균으로 채우면 초기 활성 스케일이 유지된다.
                 0이나 랜덤보다 안정적. smp의 in_channels 자동 처리는
                 버전에 따라 랜덤 초기화될 수 있으므로 직접 확장할 것.

1차 워밍업   VisDrone (8,629장) 으로 파인튜닝  ← 중요
            └─ 드론 시점 + 초소형 사람(이미지당 ~70개)에 백본을 먼저 적응
               ImageNet(지상 일반 사진) → 재난 데이터로 바로 점프하면 도메인 갭이
               크므로, "드론 시점 + 초소형 사람"이라는 중간 다리를 놓는다.

2차 학습     modality dropout 켜고 본 학습
            ├─ 세그 헤드:    FLAME, RescueNet
            └─ 점 검출 헤드: SARD, AIResQ, KAIST, LLVIP

최종 검증    FLAME 3 (RGB + radiometric thermal 페어, 드론 항공)
            └─ 실제 운용 조건에 가장 가까운 데이터
```

### 6.2 알려진 프리트레인 갭

**열화상 채널은 사실상 scratch다.** RGB-Thermal 4채널로 대규모 사전학습된 공개
백본은 없다. 4채널 중 앞 3개만 프리트레인 이득을 받는다.

이는 §2.2의 modality dropout과 잘 맞물린다 — dropout이 두 경로가 각자 독립적으로도
동작하도록 강제하므로, 약한 쪽(열화상)이 강한 쪽에 얹혀가며 학습을 게을리하는
현상이 줄어든다.

**SatlasPretrain 검토 결과: 우선순위 낮음.** highres 모델이 0.5~2.0 m/pixel 기준인데
우리 저고도 드론은 cm/pixel 급이라 GSD 갭이 한 자릿수 이상이다.
"항공"이라는 단어는 같지만 실질은 위성 도메인.

**소리(확장)는 이미 해결** — YAMNet은 AudioSet 521개 환경음 클래스로 사전학습돼 있고
`Crying/Yell/Groan/Knock` 클래스를 보유해 추가 학습 없이 zero-shot 시연 가능.

### 6.3 헤드별 분리 학습

재난+사람이 **동시에** 라벨링된 데이터가 없으므로, 배치마다 해당 헤드의 loss만
계산하는 방식으로 학습한다. 공유 백본은 두 데이터 모두로 학습되므로 이득을 본다.

---

## 7. 기반 라이브러리

**결정: PyTorch 기반 + `transformers` 인코더 + 자체 UNet 디코더/헤드**

### 7.1 왜 transformers인가

**주의: "transformers를 쓴다"가 "Transformer 모델을 쓴다"는 뜻이 아니다.**
`AutoBackbone`으로 ResNet·ConvNeXt 같은 **CNN 인코더**를 불러오므로
§1.1의 UNet/CNN 채택 결정과 모순되지 않는다.

- HF [Backbone API](https://huggingface.co/docs/transformers/backbones)가 명시적으로
  **backbone / neck / head 분해**를 전제로 설계돼 있어, 우리 구조
  (공유 인코더 + 이중 헤드)와 철학이 일치한다.
- `AutoBackbone.from_pretrained(..., out_indices=(0,1,2,3))` 가
  **멀티스케일 feature map**을 반환한다 — UNet 디코더 스킵 커넥션에 필요한 것 그 자체.
- `TimmBackbone`으로 timm 생태계까지 흡수 → 인코더 교체가 한 줄.
- §1.1에서 로드맵으로 남긴 **TransUNet·Mask2Former 비교 실험**이 같은 API로 확장된다.

### 7.2 감수하는 비용

| 필요한 것 | transformers 제공 | 대응 |
|---|---|---|
| 멀티스케일 인코더 | ✅ `AutoBackbone` | 그대로 사용 |
| **UNet 디코더** | ❌ | **자체 구현** (~100줄) |
| 이중 헤드 | ❌ | 자체 구현 (어차피 커스텀) |
| **4채널 입력** | ❌ `in_channels` 파라미터 없음 | 첫 conv 수동 확장 (§6.1 공식) |
| 4채널 전처리 | ❌ `AutoImageProcessor`는 RGB PIL 전제 | 자체 작성 (radiometric TIFF라 어차피 필요) |

**`segmentation_models_pytorch`(smp)를 택하지 않은 이유**: smp는 UNet 디코더와
`in_channels=4`가 기성품이라 시작은 빠르다. 그러나 이중 헤드·modality dropout·
4채널 전처리를 **어차피 전부 직접 구현**해야 하므로 기성품 이점이 작고,
생태계·확장성에서 transformers가 앞선다. (smp는 초기 베이스라인 대조군으로는 유용)

### 7.3 의존성 스택

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

## 8. 채택하지 않은 대안

| 대안 | 기각 사유 |
|---|---|
| TransUNet 기본 채택 | 데이터 부족 상황에서 CNN 대비 불리. 로드맵으로 보류 (§1.1) |
| VLM 기반 Open-Vocabulary 분할 | 연산 무거워 30초 준실시간 위협 + 프롬프트 민감성 (ARCHITECTURE.md §3-A) |
| 시계열/비디오 입력 모델 | 시공간 일관성은 ③이 담당. 역할 중복 + 데이터 요구량 (§0 원칙 1) |
| 3채널·4채널 모델 2개 분리 | AIResQ 등 단일 모달리티 데이터를 버리게 됨 (§2.2) |
| 2D 외관 기반 Re-ID로 프레임 간 매칭 | 포즈를 알고 있으므로 기하학적 매칭이 더 정확·단순 (§0 원칙 2) |
| latent(중간) 융합 | 3DGS는 결합할 latent feature가 없고, 매칭된 멀티모달 재난 데이터셋도 없음 (ARCHITECTURE.md §3-A) |
| AIDER를 세그 헤드 학습에 사용 | **분류 라벨만 있고 세그 마스크 없음.** 웹 수집이라 품질 편차 큼 → RescueNet으로 대체 |
| 고정 FPS 추론 | 정지 비행 시 낭비, 고속 이동 시 누락 (§2.4) |
| `segmentation_models_pytorch` 단독 사용 | 커스텀 비중이 커서 기성품 이점이 작음. 생태계·확장성에서 transformers 우위 (§7.2). 베이스라인 대조군으로는 유지 |
| HF `Trainer` | 멀티 데이터셋 · 헤드별 loss · modality dropout 구조에 맞추는 비용이 큼 (§7.3) |

---

## 9. 미결정 사항

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
  README.md           이 문서 (설계 철학 + 스캐폴드 안내)
  __init__.py         패키지 버전/독스트링
  models/
    detection.py      HumanDetector 인터페이스: Frame, Detection, HumanDetector
    splat.py          SplatReconstructor 인터페이스: SplatChunkSpec, SplatAlign
  datasets/
    README.md         데이터셋 배치 규약
  utils/
    README.md
    geo.py            ENU ↔ GPS 변환, src/skylens_core/geo.ts 의 미러
```

# 클라이언트와의 연결

TypeScript 클라이언트(`src/skylens_core/`)는 이 패키지를 직접 호출하지 않는다.
`src/skylens_core/protocol.ts`에 정의된 와이어 프로토콜로 결과만 소비한다.
이 패키지의 역할은 결국 그 스키마로 직렬화될 값을 **생산**하는 것이다.

- `HumanDetector.infer()` → `Detection` 객체. `protocol.ts`의 `DetectionResult`와
  1:1 대응 (`category`, `gps`, `confidence`, `label`)
- `SplatReconstructor.export_chunk()` → `SplatChunkSpec`. `protocol.ts`의 `SplatChunk`
  (`id`, `url`, `align`)와 대응하며 `align`은 `SplatAlign`
  (`anchor?`, `position`, `rotation`, `scale`)의 미러

좌표는 `src/skylens_core/geo.ts`에 정의되고 `skylens_model/utils/geo.py`에 미러된
**ENU(East/North/Up) 규약**으로 공유한다 — GPS in, 로컬 미터 out.
덕분에 이 패키지가 내놓는 탐지 결과와 스플랫 청크가 클라이언트가 렌더하는
동일한 씬 프레임에 정렬된다.
