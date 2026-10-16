# pointblitz 패치

crates.io `pointblitz` 0.2.0 · `pointblitz-core` 0.2.0 · `pointblitz-io` 0.2.0 을 그대로 가져와 SkyLens에 필요한 것만 더했다. 루트 `Cargo.toml`의 `[patch.crates-io]`가 이 사본으로 바꿔 끼운다. 원본: https://github.com/i3months/pointblitz (Apache-2.0 또는 MIT).

| 변경 | 이유 |
|---|---|
| `pointblitz-core/src/layers.rs` 신규: `begin_pass`, `FrameStyle`, `Fog` | 관제탑은 점·메시·선을 한 패스, 한 깊이 버퍼에 그려야 서로 가린다. 원본 `Renderer::render`는 패스를 직접 열고 검정으로 지우며 깊이를 버린다(`Discard`). `begin_pass`는 배경색(#0a1128)으로 지우고 깊이를 0(reversed-Z)으로 지운 뒤 **저장**해서 이후 패스도 깊이를 쓸 수 있게 한다 |
| `layers.rs` `MeshRenderer` (+ `src/mesh.wgsl`) | 위성 텍스처를 입힌 지형, 돌출 건물 프리즘, 지상 패드를 그린다. id별 GPU 메시, 위치는 점과 같은 방식으로 CPU에서 f64로 `원점 - 카메라` 를 계산해 넘긴다(원점은 메시 bbox 중심, 정점은 그 기준 f32). three.js `MeshLambertMaterial`(반구광 + 태양광 + emissive, r155 이후 물리 단위라 `diffuse / pi`)과 `MeshBasicMaterial`(Unlit)을 같은 셰이더로 표현한다. Opaque/Alpha/Additive 블렌딩, 깊이 테스트는 항상, 깊이 쓰기는 불투명만, 양면 옵션, 반투명은 카메라에서 먼 것부터 그린다(three.js 정렬과 같다). `set_material`은 지오메트리 재업로드 없이 표시 모드만 바꾼다 |
| 법선이 없으면 생성 (`flat` 이면 면 법선, 아니면 면적 가중 평균) | 건물 프리즘은 면이 각져 보여야 하고(면마다 정점을 따로 둔다), 지형은 매끈해야 한다. 면적 가중은 three.js `computeVertexNormals`와 같다. 퇴화 삼각형은 NaN 대신 +Z |
| 텍스처: `Rgba8UnormSrgb`, 선형 필터 + 밉맵(CPU에서 선형광 기준 2x2 평균), clamp, `max_texture_dimension_2d` 초과 시 절반씩 축소 | three.js 기본값이 밉맵 사용이라 비스듬히 본 위성 지도가 같은 모양으로 보이려면 필요하다. WebGL2 보장 한도가 2048이라 큰 정사영상은 줄여서 올린다. 업로드는 `create_texture_with_data` 한 번 |
| `layers.rs` `LineRenderer` (+ `src/lines.wgsl`) | 건물 윤곽선(시안, Additive), 드론 궤적(정점별 알파로 페이드), 금색 경로선과 수직선. 1 px 선, List/Strip, 정점 색 또는 단색, 블렌딩 3종, 깊이 테스트 끄기(항상 위). 깊이 테스트를 끈 세트는 마지막에 그린다 |
| 색 공간 규칙: 입력 색은 모두 sRGB 표시값(CSS hex / 255) | three.js는 선형 공간에서 곱하고 조명한 뒤 sRGB로 내보낸다. 메시·선 셰이더도 입력을 선형으로 풀고 계산한 뒤, 색 타깃이 `*Srgb` 포맷이 아니면 셰이더에서 다시 인코딩한다. 그래서 `Rgba8Unorm` 캡처와 sRGB 서피스 양쪽에서 같은 화면이 나온다. 점은 원본 동작대로 바이트를 그대로 쓴다 |
| `Fog` 를 모든 셰이더에 같은 식으로 적용 (`src/color.wgsl` 공용) | three.js `Fog`는 `smoothstep(near, far, 시선 깊이)` 로 섞는다. 시선 깊이는 `clip.w` 이고, 정점에서 `t = (깊이 - near) / (far - near)` 를 만들어 넘기면 보간이 정확하다. 프래그먼트는 `smoothstep(0, 1, t)` 로 섞는다. 섞는 위치는 sRGB 인코딩 뒤(three.js 순서)이고, 안개색은 배경색과 같은 공간이라 먼 곳이 배경으로 정확히 녹아든다 |
| `renderer.rs` `set_point_size` | `Pixels` 는 원본 동작(`point_px`). `Metres` 는 three.js `sizeAttenuation` 처럼 거리에 따라 줄어든다: 화면 지름 = 미터 x 초점거리(px) / 시선 깊이. 초점거리는 CPU에서 `높이/2 / tan(fov/2)` 로 넣는다. 눈앞의 점이 화면을 덮지 않게 1~512 px로 자른다 |
| `renderer.rs` `prepare` / `draw` | 현황판은 복원 구간마다 `Scene` 하나를 두고 한 프레임에 여러 개를 그린다. `prepare` 가 장면 순서대로 청크 슬롯을 이어 붙여 쓰고 장면별 시작 위치를 기억한다. `draw` 는 호출자가 연 패스에 그린다. `render` 도 같은 경로를 쓰므로 원본 테스트·벤치가 그대로 동작한다 |
| 장면별 알파 (`draw` 의 `alpha`) | 구간이 드러날 때 페이드 인. 알파는 버퍼가 아니라 패스의 블렌드 상수(`set_blend_constant`, 계수 `Constant / OneMinusConstant`)로 넘겨서 그리기 사이에 버퍼 쓰기가 없다. `None` 이나 1 이상이면 원본과 같은 불투명 파이프라인(블렌딩 없음), 0 이하면 그리지 않는다 |
| `points.wgsl` 에 안개와 미터 크기 추가, `CameraUniform` 112 바이트로 확장 | 점도 메시·선과 같은 안개를 받아야 화면이 이어진다. 남는 필드(`_pad`)를 `point_m` 으로 쓰고 안개색·안개 계수·초점거리를 붙였다 |
| WebGL2 대응 | 유니폼은 정점 단계에서만 읽고(프래그먼트는 varying으로 받는다), 스토리지 버퍼는 쓰지 않는다. 물체별 유니폼은 한 버퍼의 슬롯으로 두고 `min_uniform_buffer_offset_alignment` 에 맞춘 동적 오프셋으로 고른다. 바인딩 하나가 슬롯 하나(최대 112 바이트)라 16 KiB 한도와 무관하다. `@interpolate(flat)` 은 GLSL로 옮길 수 없는 `first` 샘플링이 기본이라 쓰지 않고, 그리기마다 상수인 값을 보통 varying으로 넘긴다. varying은 6개(한도 15) |
| 버전·가시성 기록 | `upsert` 마다 버전이 올라가고, `prepare` 가 만든 그리기 목록은 버전이 다른(그 사이 교체된) 물체를 건너뛴다. 슬롯 버퍼가 커지면 그 프레임 목록을 비운다. 교체해도 `set_visible` 상태는 유지한다 |
| `headless.rs` `Headless::capture_with` | 호출자가 기록한 프레임(`begin_pass` + 레이어)을 캡처 타깃에 그려 픽셀로 돌려준다. GPU 테스트와 서버 측 프레임용 |
| `tests/layers.rs`, 각 모듈 단위 테스트 | 순수 로직(법선, 안개, 버퍼 패킹, 밉맵, 그리기 목록, 점 크기, 장면 슬롯, 알파 판정)은 GPU 없이 검사한다. GPU 테스트는 어댑터가 없으면 건너뛴다 |

pointblitz를 새 버전으로 올릴 때는 새 원본을 받아 이 표의 변경을 다시 적용한다.
