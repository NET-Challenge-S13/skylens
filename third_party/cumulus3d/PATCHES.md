# cumulus3d 패치

GitHub `AH100-1/cumulus3d` 리비전 `b9f4a9bda714fa31fc0fd395f821114b64fd2c1f`(0.3.1 이후 git head)를 그대로 가져와 SkyLens에 필요한 것만 고친다. 루트 `Cargo.toml`의 `[patch."https://github.com/AH100-1/cumulus3d"]`가 이 사본으로 바꿔 끼운다. 원본 라이선스는 Apache-2.0(`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`).

| 변경 | 이유 |
|---|---|
| (진행 중) 위치마다 도는 단계(SIFT 검출·기술자, 기하 검증, 정합, 삼각측량, 번들 조정)를 CUDA로 | 라이브 복원에서 GPU가 맡는 일이 블러·매칭·조밀화뿐이라, 묶음 하나에 약 4초가 CPU에서 걸렸다(2026-10-16 데모 측정) |

cumulus3d를 새 리비전으로 올릴 때는 원본에 같은 기능이 들어갔는지 확인하고, 들어갔으면 패치를 지운다.
