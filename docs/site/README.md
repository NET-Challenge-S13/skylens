# docs/site: GitHub Pages 가이드북

SkyLens를 처음 보는 사람에게 설명하는 **정적 가이드북**이다. 같은 `docs/` 안에 있지만 설계 문서와 목적이 다르다. `docs/`는 만드는 사람이 읽고, `docs/site/`는 보는 사람이 읽는다.

## 구성

| 파일 | 내용 |
|---|---|
| `index.html` | 홈: SkyLens가 뭔지, 왜 필요한지 (문제 정의·홍제동 사례) |
| `how-it-works.html` | 3단계 흐름: 분할 탐색 → 고속망으로 모아 3D 복원 + AI 감지 → 3D 상황판 |
| `screens.html` | 관제탑·현황판 화면 소개, 딜레이 패턴(수준 1~4) |
| `architecture.html` | 컴포넌트 8개·데이터 흐름·포트 맵 + ResearchTree 연구 기록 |
| `run.html` | 로컬에서 돌리는 법, 쿼리 옵션, VWorld 키 |
| `model.html` | 4채널 UNet·이중 헤드·Depth Map 레이캐스팅·3DGS 복원 실측치 |
| `assets/guide.css` | 스타일 하나. 토큰 블록은 `src/shared/viewer/style.css`에서 그대로 가져왔다 |
| `assets/guide.js` | 모바일 메뉴 토글 + 현재 페이지 표시. 그게 전부다 |
| `assets/*.jpg · *.png` | `docs/figures/`에서 **복사해 온** 이미지. 원본 경로를 참조하지 않는다 |

## 규약

- **빌드 스텝이 없다.** 프레임워크도 번들러도 쓰지 않는다. `docs/site/`를 그대로 올린다.
- **경로는 전부 상대경로다.** Pages base path가 `/skylens/`라도 그대로 동작해야 한다.
- **디자인 토큰은 관제탑 UI가 출처다.** 색·폰트·반경·그림자를 새로 만들지 말고 `src/shared/viewer/style.css`에서 가져온다. 단 본문 타이포 스케일과 `--text-muted`는 가독성·대비비(4.5:1) 때문에 가이드북에서 따로 정의한다.
- **다이어그램은 인라인 SVG로 직접 그린다.** mermaid 같은 런타임 의존을 붙이지 않는다.
- 본문은 한국어, 해요체.

## 로컬에서 보기

빌드가 없으니 정적 서버 아무거나 띄우면 된다.

```bash
npx serve site          # 또는
python -m http.server 4173 -d site
```

`http://localhost:4173/` 로 접속한다. 파일을 직접 열어도(`file://`) 대부분 동작하지만, 상대경로 확인은 서버로 하는 편이 정확하다.

## 배포

`.github/workflows/pages.yml` 이 `main` 브랜치 push 때 `docs/site/`를 GitHub Pages에 올린다(`actions/upload-pages-artifact` → `actions/deploy-pages`). 저장소 Settings → Pages 에서 **Source 를 "GitHub Actions"** 로 한 번 설정해 둬야 한다.
