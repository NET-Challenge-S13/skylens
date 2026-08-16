# skylens_model.datasets

torchvision 스타일 Dataset API. **데이터는 이 저장소에 포함되지 않는다** — 각 클래스는
이미 내려받아 둔 사본을 `root` 아래에서 찾고, 소스가 허용하는 경우에만
`download=True`로 직접 가져온다.

> 설계 근거: [`../README.md`](../README.md) §1.4(bbox↔점) · §2(4채널·modality dropout) ·
> §6.3(헤드별 분리 학습) · 데이터셋 조사: [`../../../res/docs/DATASETS.md`](../../../res/docs/DATASETS.md)

---

## 1. 고정 계약

### 1.1 `__getitem__` 반환값

모든 Dataset이 **동일한 dict**를 돌려준다. 이 계약이 여러 데이터셋을 한 모델에
태울 수 있게 하는 유일한 접착제다.

```python
{
  "image":        np.ndarray,   # (H, W, C) uint8 | float32
                                #   C=3 RGB · C=4 RGB+thermal · C=1 thermal-only
  "has_rgb":      bool,
  "has_thermal":  bool,
  "danger_mask":  np.ndarray | None,   # (H, W) uint8
  "person_boxes": np.ndarray | None,   # (N, 4) float32, xyxy 픽셀
}
```

`None`은 오류가 아니라 **정상값**이다. 재난+사람이 동시에 라벨링된 공개 데이터가
없으므로(README §6.3), 세그 전용 데이터셋은 `person_boxes=None`, 사람 전용
데이터셋은 `danger_mask=None`을 낸다.

### 1.2 통합 클래스 스키마

| 값 | 이름 | 출처 |
|---|---|---|
| 0 | `normal` | 그 외 전부 |
| 1 | `fire` | FLAME 화재 마스크 |
| 2 | `collapse` | RescueNet building major damage / total destruction |
| 3 | `road_blocked` | RescueNet road-blocked |
| 255 | `ignore` | unlabeled / void |

이것으로 상위 README §9의 미결정 항목 "클래스 통합 스키마 확정"이 코드 레벨에서
확정된다(`base.DangerClass`).

**RescueNet 11 → 4 매핑** (`rescuenet.RESCUENET_TO_SKYLENS`):

| 원본 id | 원본 클래스 | → |
|---|---|---|
| 0 | unlabeled | 255 ignore |
| 1 | water | 0 |
| 2 | building-no-damage | 0 |
| 3 | building-medium-damage | 0 |
| 4 | building-major-damage | **2 collapse** |
| 5 | building-total-destruction | **2 collapse** |
| 6 | vehicle | 0 |
| 7 | road-clear | 0 |
| 8 | road-blocked | **3 road_blocked** |
| 9 | tree | 0 |
| 10 | pool | 0 |

*medium damage를 붕괴에 넣지 않은 이유*: 지휘관이 "진입 금지"를 판단해야 하는 대상은
major damage 이상이다. medium까지 넣으면 양성 클래스가 과포화되어 경보가 무뎌진다.

**VisDrone**: 12개 카테고리 중 `pedestrian(1)`, `people(2)`만 person으로 추출한다.
`people`은 서 있지 않은 자세의 사람을 가리키는 클래스라 오히려 우리 시나리오
(쓰러진 사람)에 직결된다. `score==0` 행은 ignored region이므로 버린다.

### 1.3 배치 dict (`SkyLensCollator`)

| 키 | shape / dtype | 비고 |
|---|---|---|
| `pixel_values` | (B, C, H, W) float32 | C=4, `validity_channel=True`면 5 |
| `modality_mask` | (B, 2) bool | `[rgb_present, thermal_present]` |
| `danger_labels` | (B, H, W) int64 | ignore=255. 배치 전체가 `None`이면 **키 자체 없음** |
| `person_heatmap` | (B, 1, H/s, W/s) float32 | CenterNet 가우시안 |
| `person_wh` | (B, 2, H/s, W/s) float32 | (w, h), **출력 stride 단위** |
| `person_reg_mask` | (B, 1, H/s, W/s) float32 | 1 = 유효 중심 |

키 생략이 §6.3의 헤드별 분리 학습을 구현하는 방식이다 — 학습 루프는 키 존재
여부만 보고 어느 loss를 계산할지 정한다.

채널 레이아웃 (§2.3):

```
0:3  RGB, [0, 1] 스케일
3    thermal, [0.1, 1.0] 정규화 — 정확히 0.0 = "없음" 예약값
4    (선택) validity mask, thermal이 실제로 있으면 1.0
```

FLAME 3의 radiometric thermal은 픽셀당 실제 온도라 raw 0이 유효값이다. 그래서
"차갑다"와 "센서 없음"이 구분되도록 0.0을 비워 둔다.

**타겟 인코딩**은 CenterNet 방식이다. bbox 중심을 stride로 나눈 좌표에 가우시안을
splat하고, 반경은 CornerNet `gaussian_radius`(세 경우의 최소근), 겹치면
`np.maximum`으로 합친다. 오프셋 회귀는 계약에 없어 생략했다 — 필요해지면
`person_offset` 키를 추가하는 방향으로 확장한다.

---

## 2. 자동 다운로드 가능 여부

> 판정은 **{자동 / 계정필요 / 수동전용}** 3단계다. 각 클래스의 `availability`
> 속성에 `"auto" / "account" / "manual"`로 박혀 있고, 실패 시 `download=True`는
> **URL·필요 계정·압축 해제 후 기대 경로를 담은 `ManualDownloadRequired`를
> 던진다.** 조용히 실패하지 않는다.

| 데이터셋 | 판정 | 경로 | 근거 |
|---|---|---|---|
| **FLAME 3** | 계정필요 | [IEEE DataPort](https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management) | open-access지만 다운로드에 무료 IEEE 계정 로그인 필요. 안정적 직접 URL 없음 |
| **FLAME** (원본) | 계정필요 | [IEEE DataPort](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) | 동일. 세그용은 항목 9(Images)+10(Masks) |
| **RescueNet** | 수동전용 | [GitHub (BinaLab)](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation) | 저자가 Google Drive/Dropbox 링크로 배포. 대용량 Drive는 바이러스 검사 인터스티셜 때문에 `requests`로 못 받음 → `gdown` 시도 or 브라우저 |
| **SARD** | 계정필요 | Kaggle / Roboflow Universe 미러 | 공식 직접 URL 없음. **HF에는 미러 없음(확인함)** — `?search=SARD` 결과는 전부 동명이인(SW 취약점 SARD, 사르데냐어 등). Kaggle API 토큰(`~/.kaggle/kaggle.json`) 또는 Roboflow API 키 필요. slug는 `SARD.kaggle_slug`로 주입 |
| **VisDrone** | 계정필요 | [GitHub](https://github.com/VisDrone/VisDrone-Dataset) + HF 미러 | 공식은 Google Drive/OneDrive라 wget 불가. **HF 미러 `Voxel51/VisDrone2019-DET` 존재 확인함** → `download=True` 자동화 가능. 단 FiftyOne 포맷이라 배치 재정리 필요 |
| **LLVIP** | **자동** | HF `jsonhash/LLVIP` | **확인함**: public·non-gated, `LLVIP.zip`(~4GB) + `coco_annotations.7z`. `huggingface_hub`로 받아진다. 단 repo가 단일 zip이라 압축 해제는 별도. 공식 페이지는 Drive/Baidu라 불가 |
| **AIResQ** | 계정필요 | [Sci Data 논문](https://www.nature.com/articles/s41597-026-07663-9)의 Data Records DOI | DOI가 Zenodo/Figshare로 풀리면 **직접 URL로 자동 가능**. 해당 URL을 `AIResQ.direct_url`에 넣으면 `download=True` 동작 |
| **KAIST Multispectral** | 계정필요 | [프로젝트 페이지](https://soonminhwang.github.io/rgbt-ped-detection/) | Drive/FTP 배포. 로더 미구현 |
| **AIDER** | 계정필요 | [GitHub (ckyrkou)](https://github.com/ckyrkou/AIDER) | Drive 배포. 분류 라벨뿐이라 세그 헤드에 못 씀 → 로더 미구현 |
| **AI Hub 야간 IR** | 수동전용 | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=497) | 한국 계정 + 활용 신청·승인 절차. 자동화 경로 없음 |

**확인 방법**: HF는 `https://huggingface.co/api/datasets?search=<term>`을 직접 조회해
repo 존재·gated 여부를 확인했다. 위 표에서 "확인함"이라고 쓴 항목만 실제로
검증된 것이고, 나머지(IEEE DataPort·Kaggle slug·AIResQ DOI·AI Hub 절차)는
**공개된 배포 방식으로부터의 추론**이다. 특히 `SARD.kaggle_slug`의 기본값은
**미검증 추정치**다.

**결론: 우리 핵심 4종(FLAME·RescueNet·SARD·VisDrone) 중 완전 무인 자동
다운로드가 되는 것은 없다.** 전부 계정·토큰이 한 번은 개입한다. 자동화가 실증된
것은 LLVIP(HF) 하나, 조건부로 VisDrone(HF 미러)이다. 그래서 이 패키지는
"자동 다운로드"를 약속하지 않고, **정확한 수동 절차를 에러 메시지로 알려주는 것**을
계약으로 삼는다.

> ⚠️ 위 표의 HF repo id / Kaggle slug는 **클래스 속성으로 빼 두었다**
> (`hf_repo_id`, `kaggle_slug`). 미러는 이름이 자주 바뀌므로, 실제로 확인한
> 값으로 덮어쓴 뒤 사용할 것:
> ```python
> VisDronePerson.hf_repo_id = "확인한/repo-id"
> ds = VisDronePerson(root, "train", download=True)
> ```
> 기본값은 검증되지 않은 **추정치**이므로 그대로 믿지 말 것.

---

## 3. 구현된 클래스

| 클래스 | 모듈 | 헤드 | 모달리티 | 출력 |
|---|---|---|---|---|
| `FlameSegmentation` | `flame.py` | 세그 | RGB | `danger_mask` {0,1} |
| `Flame3Pairs` | `flame.py` | (검증용) | **RGB+thermal** | 4채널 `image` |
| `RescueNetSegmentation` | `rescuenet.py` | 세그 | RGB | `danger_mask` {0,2,3,255} |
| `SARD` | `sard.py` | 점 검출 | RGB | `person_boxes` |
| `VisDronePerson` | `visdrone.py` | 점 검출(워밍업) | RGB | `person_boxes` |
| `LLVIP` | `llvip.py` | 점 검출 | **RGB+thermal** | 4채널 + `person_boxes` |
| `AIResQ` | `airesq.py` | 점 검출 | **thermal only** | 1채널 + `person_boxes` |

지원 모듈: `base.py`(공통 베이스·스키마·IO) · `collate.py`(배치+타겟 인코딩) ·
`download.py`(HTTP/HF/Kaggle 유틸) · `_selftest.py`(계약 검증).

`albumentations`와 `tifffile`은 **lazy import**다. 미설치 상태에서도 모듈은
정상적으로 로드되고, 실제로 필요한 경로에 들어갔을 때만 설치 안내와 함께 실패한다.

---

## 4. 사용법

```python
from torch.utils.data import DataLoader
from skylens_model.datasets import RescueNetSegmentation, SARD, SkyLensCollator

seg = RescueNetSegmentation("data/rescuenet", split="train")
det = SARD("data/sard", split="train")

collate = SkyLensCollator(
    person_head_stride=4,
    validity_channel=False,      # True면 pixel_values가 5채널
    modality_dropout=(0.25, 0.25),  # 학습 시 §2.2 권장값 (남은 0.5가 4채널 모드)
)

loader = DataLoader(seg, batch_size=8, collate_fn=collate)
batch = next(iter(loader))
# batch: pixel_values, modality_mask, danger_labels  (person_* 없음)
```

`transforms`는 샘플 dict를 받아 dict를 돌려주는 callable이다. albumentations
파이프라인은 얇은 어댑터로 감싼다 — 4채널 + mask + bbox를 한 번에 변환할 수 있는
것이 이 라이브러리를 고른 이유다(상위 README §7.3).

```python
import albumentations as A

aug = A.Compose(
    [A.RandomCrop(512, 512), A.HorizontalFlip()],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)

def adapter(s):
    boxes = s["person_boxes"]
    out = aug(image=s["image"],
              mask=s["danger_mask"] if s["danger_mask"] is not None else None,
              bboxes=boxes.tolist() if boxes is not None else [],
              labels=[0] * (0 if boxes is None else len(boxes)))
    s["image"] = out["image"]
    if s["danger_mask"] is not None:
        s["danger_mask"] = out["mask"]
    if boxes is not None:
        s["person_boxes"] = np.asarray(out["bboxes"], np.float32).reshape(-1, 4)
    return s
```

> `SkyLensCollator`는 배치 내 모든 이미지가 같은 `(H, W)`일 것을 요구하고, 아니면
> 명확한 에러를 낸다. 리사이즈/크롭은 `transforms`에서 처리할 것.

---

## 5. 검증

실데이터 없이 합성 더미로 전 경로를 돌린다:

```bash
PYTHONPATH=src python -m skylens_model.datasets._selftest
```

검사 항목: 7개 Dataset의 `__getitem__` → collate shape/dtype 일치 ·
RescueNet 11→4 매핑 · VisDrone person 필터링 · thermal `[0.1,1.0]` 예약 0 인코딩 ·
세그+검출 혼합 배치 · `ManualDownloadRequired` 메시지 내용.

---

## 6. 디렉터리 배치 규약

각 클래스의 `expected_layout` 속성이 정본이며, 데이터가 없을 때 에러 메시지에
그대로 출력된다. `raw/`·`processed/` 아래 실데이터는 **절대 커밋하지 않는다** —
이 폴더는 코드와 문서만 담는다.
