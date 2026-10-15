[English](COMPOSE.md) | 한국어

# 실시간 합성 (`cumulus3d_cli::compose`)

`Recon::declare()…build()?.run()` 은 실행 전에 정해진 폴더 하나를 끝까지 재구성한다. `compose` 는 입력이 계속 들어오는 경우
(디코더, 네트워크 수신기, 촬영 루프)를 위한 짝이다. **유지되는 노드**가 세션·파이프라인·받아들인 프레임 기록을 갖고, 호출하는 쪽은
**지금의 입력 상태만 선언**한다. 노드는 선언을 이미 받아들인 것과 비교해 필요한 증분 작업만 한다. `Pipeline::push` 를 직접 부를 필요가 없다.

```rust
use cumulus3d_cli::compose::{CompositionHost, FrameKey, FrameState, LiveFrameSet, ReconNode, ReconcilePolicy};
use cumulus3d_cli::declare::Dense;

let host = CompositionHost::new();
let recon = ReconNode::declare()
    .images("live/images")                         // 파일 입력의 기준 폴더, 메모리 입력은 이 안에 저장
    .cameras(["camF", "camR", "camL"])
    .zones(12, 2)
    .dense(Dense::profile("fast").fusion_min_views(3))
    .reconcile(ReconcilePolicy::AppendOnly)
    .on_zone_preview(|range, cloud| println!("초벌 {}: 점 {}", range.zone, cloud.len()))
    .build(&host)?;

recon.compose(FrameState::ready(frame_set_a))?;   // 새 식별자: 위치 하나 처리
recon.compose(FrameState::ready(frame_set_b))?;
recon.compose(FrameState::ready(frame_set_b))?;   // 같은 식별자·리비전: 아무것도 안 함
let closed = recon.close()?;                      // 남은 구역 닫기, 정밀 작업 대기, 마지막 이벤트
```

## 1. 프레임 식별자와 입력

| 형식 | 뜻 |
|---|---|
| `FrameKey { source_id, sequence, revision }` | 동기화된 묶음 하나의 식별자. `sequence` 가 위치 순서, `revision` 은 정정 버전(0 = 처음 받은 것). |
| `LiveFrameSet { key, timestamp, images, gps }` | 카메라마다 한 장 + GPS. `.file(cam, rel)`, `.rgb8(cam, w, h, pixels)`, `.encoded(cam, bytes, format)`, `.gps_fix(cam, lat, lon, alt)`, `.at(time)`. |
| `FramePayload::File { relative_name }` | 영상 폴더 안의 기존 파일. 다 쓴 뒤에 선언해야 한다. |
| `FramePayload::Rgb8 { width, height, pixels }` | 디코딩된 화소. 스풀 폴더에 PNG 로 저장. |
| `FramePayload::Encoded { bytes, format }` | JPEG/PNG 바이트. 스풀 폴더에 그대로 저장. |

메모리 입력은 `<images>/<spool>/<camera>/<source>_<sequence>.<ext>` 에 임시 파일 → 이름 바꾸기로 저장되어(스풀 기본 `.live`,
`.spool(rel)` 로 변경), 재구성이 덜 쓴 영상을 읽는 일이 없다. 카메라 이름(`"camF"`)으로 된 GPS 기록은 그 카메라의 저장 이름으로 바뀐다.

## 2. 선언한 상태와 노드 동작

| 선언 | 결과(`Outcome`) |
|---|---|
| 다음 순번의 `FrameState::Ready(set)` | `Appended { position, skipped: 0 }` — 위치 하나 처리 |
| 같은 키·리비전 재선언 | `Duplicate { position }` — 특징·매칭·위치 추가 없음 |
| `FrameState::Pending { key, missing_cameras, .. }` 또는 카메라가 빠진 묶음 | `Waiting { missing }` — 다 모일 때까지 처리 안 함(`ReconNode::waiting()`) |
| 순번 건너뜀(프레임 유실) | `Appended { position, skipped: n }` + 경고 |
| 처음 보는 이전 순번(늦게 도착·순서 뒤바뀜) | 정책에 따름(아래) |
| 받아들인 순번의 더 높은 리비전(정정) | 정책에 따름(아래) |
| 받아들인 것보다 낮은 리비전 | `Ignored(StaleRevision)` |
| 노드와 다른 `source_id` | `Ignored(ForeignSource)` |
| 모르는·중복 카메라, 없는 파일, 화소 버퍼 크기 오류 | `Err(InvalidFrameSet)` |
| `FrameState::End` | `Ended` — 노드를 닫고, `close()` 가 보관한 요약을 돌려줌 |

받아들인 순번 하나가 위치 하나가 되며 순번 순서를 따른다. 입력을 버리거나 순서를 바꾼 결정은 `Event::Warning`(`compose: …`)으로도
나가므로 훅·구독 채널·`run.log` 에 남는다.

## 3. 조정 정책(`ReconcilePolicy`)

| 정책 | 늦은 도착 | 정정 | 순번 건너뜀 |
|---|---|---|---|
| `AppendOnly`(기본) | 무시 | 무시 | 받음 |
| `ReplayWindow { positions: w }` | 최근 `w` 위치 안이면 되감기(`ResetFrom`) 후 끼워 넣은 묶음과 그 뒤 묶음을 다시 처리 | 정정 위치부터 같은 방식 | 받음 |
| `InvalidateAffectedZones` | 무시 | 정정 화소를 원래 이름으로 저장하고 그 위치를 포함한 도착 구역을 다시 만든다(`InvalidateZone`). 특징·매칭·자세·GPS 는 다시 계산하지 않음 | 받음 |
| `Strict` | 오류 | 오류 | 오류 |

`ReplayWindow` 는 세션에 되감기 지점 `w + 1` 개, 다시 처리용 묶음 `w` 개를 보관한다. 범위보다 오래된 것은 `Ignored(OutsideReplayWindow)`.

## 4. 호스트와 조회

`CompositionHost` 는 노드를 id 로 보관한다(빌더 `.id("main")`, 기본 `"default"`). `reconstruct(&host, "main", state)` 로 노드를
들고 있지 않아도 입력마다 다시 선언할 수 있다. `host.close(id)` 는 노드 하나를 닫고 빼고, `host.close_all()` 은 전부.

## 5. 생산자 스레드와 역압

`node.spawn()` 은 노드를 큐 뒤의 전용 스레드에서 돌리고 `ComposeHandle` 을 돌려준다.

| 메서드 | 동작 |
|---|---|
| `send(state)` | 큐에 넣음. `Block` 이면 자리가 날 때까지 기다림 |
| `try_send(state)` | 기다리지 않음. `Block` 큐가 차면 `Err(SendError::Full(state))` (비동기 코드에서 사용) |
| `send_all(iter)` | 반복자·채널 수신기의 각 항목을 `send` |
| `stats()` | 넣은 수, 버린 수, 처리 수, 현재·최대 깊이 |
| `close()` | `End` 를 보내고 큐가 빌 때까지 기다린 뒤 상태별 결과와 종료 결과를 돌려줌 |

| `Backpressure` | 큐가 찼을 때 |
|---|---|
| `Block { max_pending }`(기본 8) | 생산자가 기다림 |
| `KeepLatest { max_pending }` | 가장 오래된 대기 묶음을 버림(순번 건너뜀으로 나타남) |
| `Unbounded` | 차지 않음 |

종류별 훅 큐 정책은 `.event_policy(EventKind::ZonePreview, QueuePolicy::LatestPerKey)`.

## 6. 빌더(`ReconNode::declare()`)

재구성 설정은 `Plan`(`declare` 와 같은 필드·TOML, `.plan(plan)` 으로 교체)이다. 계획의 위치 수·간격·GPS 파일·출력 설정은 쓰지 않는다.

`.id`, `.images`, `.cameras`, `.zones`, `.align`, `.fixed_enu_origin`, `.features`, `.match_backend`, `.gpu`,
`.incremental_triangulation`, `.dense`, `.no_dense`(기본), `.seed`, `.reconcile`, `.backpressure`, `.source`, `.spool`,
`.output(dir, SinkOptions)`(기본 출력 파일), 훅 `.on`, `.on_event`, `.on_frame`, `.on_zone_preview`, `.on_zone_refined`,
`.on_message`, `.event_policy`, `.sync_hooks`, 그리고 설정 문제를 한 번에 모아 알려 주는 `.build(&host)`(`ComposeError::Config`).

노드 조회: `positions()`, `journal()`(위치별 순번·리비전), `waiting()`, `stats()`, `subscribe()`, `poll()`, `with_session(|s| …)`.

## 7. 보장(시험으로 확인)

`crates/cli/tests/compose_synthetic.rs` 가 합성 장면으로 모든 정책을 돌린다: 재선언은 아무것도 하지 않고, 대기·불완전 묶음은 처리하지
않으며, 순번 건너뜀을 알리고, 늦은 도착과 정정은 범위 안에서 다시 처리하고 밖이면 무시하며, 정정은 해당 구역만 다시 만들고,
`Strict` 는 다음 순번 외에는 거부하고, 메모리 입력은 덜 쓴 파일 없이 저장되며, `KeepLatest` 로 띄운 노드는 대기 중인 묶음만 버린다.
배치 API(`Recon::declare().run()`)는 바뀌지 않았다.
