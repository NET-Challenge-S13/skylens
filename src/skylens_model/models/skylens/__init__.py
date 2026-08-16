"""SkyLens — 3D Gaussian Splatting 복원.

멀티드론 영상을 하나의 3D 장면으로 복원한다. COLMAP 이 각 프레임의 촬영 위치를
역산하고, gsplat 이 그 카메라 배치를 근거로 공간을 가우시안으로 채운다.

``skylensnet`` 과 성격이 다르다는 점을 먼저 알아둘 것. 3DGS 는 학습된 가중치가
형상을 추론하는 방식이 아니라 **장면마다 처음부터 도는 최적화**라, 사전학습
체크포인트가 없고 새 장면마다 전 과정을 다시 돌린다. 그래서 이 패키지에는
``modeling_*.py`` / ``configuration_*.py`` 가 없고, 대신 외부 바이너리(ffmpeg,
COLMAP)를 순서대로 부르는 셸 파이프라인과 그 파이프라인이 쓰는 도구가 들어 있다.

파이썬 도구는 모듈로 직접 실행한다::

    python3 -m skylens_model.models.skylens.subset_model --src ... --dst ... --keep cam2037
    python3 -m skylens_model.models.skylens.ckpt_to_ply --ckpt ... --out ... --light

- 설계 결정과 근거: ``README.md``
- 측정값(실험 5건): ``RESULTS.md``
- 설치와 함정: ``INSTALL.md``
- 파이프라인: ``pipeline/01_extract_frames.sh`` → ``02_reconstruct.sh`` → ``03_train.sh``

의존성은 ``pyproject.toml`` 의 ``recon`` 그룹에 있고 기본 설치에는 들어가지 않는다
(``uv sync --group recon``). gsplat 과 fused-ssim 이 CUDA 툴체인을 요구하기 때문이다.
"""

from __future__ import annotations

__all__: list[str] = []
