"""SkyLensGSplat — 3D Gaussian Splatting 복원.

**이 자리는 비어 있다.** 복원 파이프라인은 ``src/skylens_recon`` 에 구현돼 있다.

여기 두지 않은 이유: 3DGS 는 학습된 모델이 형상을 추론하는 방식이 아니라
장면마다 처음부터 도는 최적화라, ``models/`` 의 다른 패키지들(백본·헤드·체크포인트)
과 성격이 다르다. 게다가 파이프라인의 본체가 ffmpeg 와 COLMAP 바이너리를
순서대로 부르는 셸 스크립트다.

- 설계 결정과 근거: ``src/skylens_recon/README.md``
- 측정값: ``src/skylens_recon/RESULTS.md``
- 설치: ``src/skylens_recon/INSTALL.md``
"""

from __future__ import annotations

__all__: list[str] = []
