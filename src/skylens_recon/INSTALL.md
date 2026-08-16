# 설치 — COLMAP · gsplat

이 파이프라인은 **파이썬 패키지 두 개와 시스템 바이너리 두 개**를 요구한다.
`uv sync`만으로는 끝나지 않는 부분이 있어 순서와 함정을 함께 적는다.

밟았던 함정은 전부 실제로 시간을 잃은 것들이다 (§4). 같은 자리를 다시 밟지 않기 위한 기록이다.

---

## 1. 시스템 바이너리

| 도구 | 용도 | 설치 |
|---|---|---|
| **ffmpeg** | 영상 → 프레임 | `apt install ffmpeg` |
| **COLMAP 4.1+** | SfM (카메라 포즈 역산) | 아래 |

COLMAP은 파이썬 패키지가 아니라 `pyproject.toml`에 넣을 수 없다.
**CUDA 빌드가 필요하다** — ALIKED 특징점과 LightGlue 매처는 GPU에서만 돈다.

```bash
conda create -n colmap-cuda -c conda-forge colmap=4.1 'libfaiss=1.10.0=cuda129_openblas*'
conda activate colmap-cuda
colmap feature_extractor --help | grep ALIKED   # 확인
```

`libfaiss`는 **반드시 openblas 변형**을 쓴다. MKL 변형은 `libmkl_*.so.2`를
못 찾고 죽는다.

ALIKED가 ONNX Runtime CUDA 프로바이더를 못 올리면 (`libcufft.so.11` 없음):

```bash
conda install -c conda-forge libcufft libcublas libcudnn cuda-version=12.9
```

---

## 2. 파이썬 의존성

`pyproject.toml`의 **`recon` 그룹**에 정리돼 있다. 기본 설치에는 들어가지 않는다
(`default-groups = ["dev"]`) — gsplat이 CUDA 툴체인을 요구하기 때문에
복원 작업을 하지 않는 팀원이 `uv sync`에서 막히면 안 된다.

```bash
uv sync --group recon
```

### 기존 의존성과 겹치거나 충돌하는 것

`recon` 그룹에 **일부러 넣지 않은** 항목들이다.

| 패키지 | 이유 |
|---|---|
| `torch` · `torchvision` | 이미 최상위 `dependencies`에 있다. CUDA 인덱스 설정도 그대로 쓴다 |
| `numpy` | 이미 있다. gsplat 예제는 `numpy<2.0.0`을 요구하지만 그건 보수적인 핀이고, 여기서 상한을 걸면 저장소 전체가 묶인다 |
| `opencv-python` | **충돌.** 최상위에 `opencv-python-headless`가 있고 둘 다 `cv2`를 제공한다. headless 쪽을 유지한다 |
| `tensorboard` | 이미 `train` 그룹에 있다 |
| `tqdm` · `requests` | 이미 최상위에 있다 |

### git 소스 세 개

`[tool.uv.sources]`에 커밋 해시로 고정돼 있다. 전부 PyPI에 없다.

- **`pycolmap`** — `rmbrualla` 포크. gsplat 예제 로더가 이걸 요구한다.
  ⚠️ **공식 `pycolmap`과 모듈 이름이 같다.** 같이 설치하면 서로를 덮어쓴다.
  이 포크에는 `Reconstruction` API가 없고 `SceneManager`만 있다 —
  `colmap_io.py`가 바이너리를 직접 파싱하는 이유다.
- **`nerfview`** — 학습 중 뷰어
- **`fused-ssim`** — 학습 손실. CUDA 확장이라 빌드가 필요하다

### gsplat 체크아웃

pip 패키지에는 `examples/` 트레이너가 들어 있지 않다.
**설치된 gsplat 버전과 같은 태그**로 따로 받는다 — API가 버전 간에 바뀐다.

```bash
git clone --recursive https://github.com/nerfstudio-project/gsplat
cd gsplat && git checkout v1.5.3
export GSPLAT_EXAMPLES=$PWD/examples
```

`--recursive`가 빠지면 GLM 서브모듈이 없어 빌드가 실패한다.

---

## 3. gsplat CUDA 빌드 환경

소스 빌드가 필요한 경우(프리빌트 휠이 없는 GPU 아키텍처) 아래를 그대로 쓴다.
플래그가 달라지면 **CUDA JIT 재컴파일이 다시 돈다 — 50분짜리다.**

```bash
export TORCH_CUDA_ARCH_LIST="12.0"        # GPU 아키텍처에 맞게
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export CUDAHOSTCXX=$CXX
export MAX_JOBS=24

pip install --no-build-isolation -e .
```

---

## 4. 밟았던 함정

| 증상 | 진짜 원인 | 해결 |
|---|---|---|
| 매퍼가 "No images with matches" | COLMAP 4.x에서 옵션 이름이 바뀌었는데 **경고 없이 무시**된다 | `--SiftExtraction.*` → `--FeatureExtraction.*` |
| gsplat 빌드 실패 ① | pip 빌드 격리 환경에 torch가 없다 | `--no-build-isolation` |
| gsplat 빌드 실패 ② | conda gcc 14 > CUDA 12.8 지원 상한 | gcc 13 설치 |
| gsplat 빌드 실패 ③ | `--depth 1` 클론이 GLM 서브모듈을 빠뜨림 | `git submodule update --init --recursive` |
| gsplat 빌드 실패 ④ | `cusolverDn.h` 없음 | `export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include` |
| 초점거리가 46% 과대 | 내부 파라미터를 안 주고 자기보정만 끔 → 기본값 2457.6에 고정 | `--ImageReader.camera_params` 명시 + 보정 ON |
| 학습 결과 가우시안 226개 | 매퍼가 `sparse/0`(11장)과 `sparse/1`(540장)을 만들었는데 0번을 집음 | `02_reconstruct.sh`가 큰 모델을 0번으로 승격 |
| 부분집합 학습이 `KeyError` | `images.bin`만 걸러 `points3D.bin`에 삭제된 image_id가 남음 | `subset_model.py`가 track도 정리 |
| 웜스타트가 6.5dB 열세 | 좌표 정규화 변환이 배치마다 재계산 (87.8° 차이) | 첫 배치 변환 고정 (미구현) |
| `fused_ssim` import 실패 | gsplat 1.5.3이 최상위에서 import한다 | git 소스로 설치 (§2) |

### 빌드를 중단시킬 때

`pkill -f`로 잡지 말 것. 자기 SSH 세션까지 죽이면서 nvcc 자식 프로세스를
고아로 남기고, ninja가 부분 `.o`를 지워 `.ninja_log`가 불완전해진다.
그 상태에서 재시도하면 **처음부터 다시 컴파일**한다.

---

## 5. 확인

```bash
python3 -c "import gsplat, torch; print(gsplat.__version__, torch.cuda.is_available())"
colmap feature_extractor --help | grep -c ALIKED     # 0이면 CUDA 빌드가 아니다
python3 -c "from skylens_recon import colmap_io; print('ok')"
```
