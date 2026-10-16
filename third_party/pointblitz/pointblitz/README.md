# PointBlitz

**Point clouds at GPU speed — one Rust core, every target.**

PointBlitz is a point-cloud renderer written in Rust on top of [wgpu](https://wgpu.rs).
The same render core runs in three places:

| Target | Backend | What the client gets |
|---|---|---|
| Browser | wasm + WebGPU (WebGL2 fallback) | GPU-ready point chunks |
| Native | Vulkan / DX12 / Metal | GPU-ready point chunks |
| Server | headless wgpu + hardware video encoder | a video stream |

It is built for people who hit the limits of three.js with large point clouds:
full-file downloads before the first frame, main-thread parsing, and re-uploading the whole cloud every time it grows.

> **Status: measured on one machine.** One PC (Windows 11, RTX 4070, i7-13700F, 60 Hz display, Chrome 155),
> server and client on the same PC (loopback), reference dataset flight-01 (14 snapshots, 2.5 M points at the end),
> 2026-10-09. Every comparison is same-session, alternating run by run, under the validity rules C1–C4, and judged by
> the median ratio with a bootstrap 95 % interval (decision [0049](docs/decisions/0049-remove-remaining-ambiguity.md)).
> Other hardware is estimated with a cost model, not measured ([cost-model.md](docs/bench/cost-model.md)).

> **What it is compared with.** Not a deployed system: SkyLens and skyrecon do not display this point cloud today.
> The comparisons use three.js 0.185 (the version SkyLens uses) done well
> ([decision 0048](docs/decisions/0048-incremental-baselines.md)): **SkyLens-style incremental three.js** — PLY
> snapshots, preview snapshots fetched as appended records only, parsed in a Web Worker — and **three.js on the same
> data** — the very chunk stream PointBlitz receives, put on the GPU without parsing.
> Browser and native draw on the user's GPU; server video draws on the server's GPU.

## Results

**A new refined snapshot is on screen in about 35–43 ms in the browser (WebGPU) and about 49 ms as server video**
— the target is ≤ 50 ms — where it took about 115–127 ms before the latency work, about 3× sooner. That first
reflection shows every eighth point of the new snapshot (coarse first); the rest fills in by about 95–106 ms in the
browser, whose floor is the time the browser takes to receive about 30 MB, and arrives with the first frame as
server video (about 49 ms). Sources: [latency-structure.md](docs/bench/latency-structure.md),
[coarse-first.md](docs/bench/coarse-first.md), [read-overlap.md](docs/bench/read-overlap.md), [pass-chunks.md](docs/bench/pass-chunks.md).

| refined snapshot, ×60 replay, p50 | before the latency work (builds 0404a79 … ff05817) | now (3dce69f) | source |
|---|---:|---:|---|
| browser WebGPU — first reflection | 115–127 ms (= complete) | **43.3 ms** (34.5–37.2 ms in the two builds before) | [pass-chunks.md](docs/bench/pass-chunks.md), [read-overlap.md](docs/bench/read-overlap.md), [coarse-first.md](docs/bench/coarse-first.md) |
| browser WebGPU — complete | 115–127 ms | 94.8 ms (101–106 ms in the two builds before) | same |
| server video (on screen) — first and complete | 115 ms | **48.8 ms** | [latency-structure.md](docs/bench/latency-structure.md), [pass-chunks.md](docs/bench/pass-chunks.md) |
| bytes received per replay (browser, native) | 180 MB | 135 MB | [coarse-first.md](docs/bench/coarse-first.md) |

"Before" values are the medians of the earlier builds in the sessions that compared them (115 ms: 0404a79 in P4.11; 119 ms:
ff05817 in P4.14; 125 ms: ee1609e in P4.11; 127 ms: session p48); "now" is the final build's own session (P4.16).
Values from different sessions are shown side by side, not judged against each other. Changes behind it: the server converts a snapshot when it is announced and serves it from
memory, the video server reads the data on its machine directly (decision [0050](docs/decisions/0050-latency-structure.md)),
chunk positions are 16-bit and refined snapshots are sent coarse first (decision [0051](docs/decisions/0051-fewer-bytes-coarse-first.md)).

**Against three.js** (session p48/p49 of 2026-10-09, with the PointBlitz build of that time — before the coarse-first
work; [comparison-incremental.md](docs/bench/comparison-incremental.md)):

| ×60 replay p50 | SkyLens-style three.js | three.js, same data | PointBlitz browser WebGPU (then) |
|---|---:|---:|---:|
| new preview on screen | 36.0 ms | 30.0 ms | 29.3 ms |
| new refined snapshot on screen | 162 ms | 117 ms | 127 ms |
| cold start, last snapshot | 184 ms | 130 ms | 134 ms |
| renderer memory | 354 MB | 199 MB | 183 MB |

"three.js, same data" takes, per row, the better of its two upload variants (decision 0048): the refined value
(117 ms) is variant (b) — positions and colours split on the CPU — and the other rows are variant (a) — the chunk
bytes as they are.

What Rust, wgpu and wasm themselves add — same data, same graphics API
([comparison-incremental.md](docs/bench/comparison-incremental.md) §7):

- **WebGPU: PointBlitz draws a frame 1.5× faster** than three.js (1.41 vs 2.13 ms, 30 frames per GPU sync, GPU
  timestamps agree), and its renderer uses 1.57× less memory (183 vs 287 MB). The new preview is 1.17× sooner;
  refined and cold show no difference.
- **WebGL2: three.js draws about 3.2× faster** than PointBlitz (1.56 vs 5.00 ms per frame) and shows a new preview
  sooner. The 5.00 ms includes a 1-pixel copy per frame that only PointBlitz's measurement makes (about 0.9 ms);
  without it about 3.9 ms (about 2.5×). Most of the remaining gap, about 1.8 ms, is in ANGLE on Direct3D 11: on
  ANGLE's Vulkan backend the two draw alike (§8).
- Most of the gain over a plain three.js viewer comes from the structure (incremental, GPU-ready, coarse-first
  deliveries), which three.js on the same data gets as well.

Since 0.2.0 the chunk format is v2 (16-bit positions): 0.1.0 clients and servers cannot read it ([CHANGELOG.md](CHANGELOG.md)).

## Why

The first user is [SkyLens](https://github.com/NET-Challenge-S13/skylens), a drone mapping system whose
reconstruction ([skyrecon](https://github.com/AH100-1/skyrecon)) streams a growing point cloud during flight —
from 150 k to 2.5 M points (67.5 MB) over 14 snapshots, 530 MB in total.
Preview snapshots only add points; refined snapshots recompute the whole cloud.
PointBlitz appends what is new, swaps whole generations without a blank frame, and never parses on the main thread.

## Install

PointBlitz 0.2.0 is on [crates.io](https://crates.io/crates/pointblitz) (Rust 1.99 or newer); changes in [CHANGELOG.md](CHANGELOG.md).

```sh
cargo add pointblitz                 # library: render core + PLY / chunk format (pointblitz-core, pointblitz-io)
cargo install pointblitz-native      # window viewer that follows a replay server
cargo install pointblitz-server      # headless render + hardware video (needs an NVIDIA GPU and driver, Windows or Linux)
```

The browser module is not installed from crates.io: build it from this repository with `bash web/build.sh` (below). The crate `pointblitz-web` on crates.io is its Rust source, for building your own wasm package.

## Try it

PointBlitz replays a directory of skyrecon snapshot PLYs (not included in this repository) as if a flight were
streaming them. You need Rust (see `rust-toolchain.toml`), and for the browser target
`cargo install wasm-bindgen-cli --version 0.2.129 --locked`.

```sh
cargo build --release -p pointblitz-bench -p pointblitz-native -p pointblitz-server
bash web/build.sh                # browser module (WebGPU + WebGL2 fallback); `bash web/build.sh webgpu` for WebGPU only

# 1. replay server: serves the snapshots, the chunk stream and the demo pages (127.0.0.1 only)
target/release/pointblitz-bench replay --data <ply dir> --web . --port 8700
```

| Target | Start | Notes |
|---|---|---|
| browser | open `http://127.0.0.1:8700/static/web/index.html?scenario=replay&speed=60` | `&backend=webgl` forces WebGL2, `&pkg=webgpu` loads the WebGPU-only module, `&scenario=still` shows only the last snapshot |
| native | `target/release/pointblitz-native --server http://127.0.0.1:8700 --scenario replay` | Vulkan / DX12 / Metal, whatever wgpu picks |
| server video | `target/release/pointblitz-server serve --replay http://127.0.0.1:8700 --port 8720 --wait-for-client`, then open `http://127.0.0.1:8700/static/web/video.html` | Windows or Linux with an NVIDIA GPU (NVENC, from the driver) and a browser with WebCodecs |

Everywhere: drag to orbit, wheel to zoom, keys 1–8 for the fixed viewpoints. `speed` is the replay speed factor
(60 = snapshots made 97–2,053 s into the flight, replayed from the first one in about 32.6 s).

Server video options and limits (decision 0042):

- Frames are sent the moment they are encoded; on a 60 Hz display 2 of 5 rotation runs still showed an occasional
  doubled frame, because the server's 60 Hz clock and the display drift apart.
- `serve --phase-lock on` moves the server's tick to the browser's display so that drift goes away. It follows
  **one** client (the first that reports), and it cannot match a display that is not 60 Hz. It is off by default:
  it did not meet its acceptance bound (see [docs/bench/server-video.md](docs/bench/server-video.md) §2.5).
- `?pacing=adaptive` (or `buffer`) on the video page holds one frame back and evens out frame timing, at the cost
  of up to one frame of latency (decisions 0039, 0040). It is meant for displays that are not 60 Hz, but it has
  **not been measured on such displays**.

## Documents

| | |
|---|---|
| [INTENT.md](INTENT.md) | why this exists |
| [SPEC.md](SPEC.md) | what it must do: inputs, architecture, metrics, comparison table |
| [PROJECT.md](PROJECT.md) | how it is built and reviewed |
| [docs/decisions/](docs/decisions/) | every technical choice and the reason for it |
| [docs/data/flight-01.md](docs/data/flight-01.md) | analysis of the reference dataset |

## Roadmap

| Phase | |
|---|---|
| P0 | measure the current three.js approach on the reference dataset |
| P1 | core + native renderer |
| P2 | browser (wasm, WebGPU / WebGL2) |
| P3 | server-side rendering with hardware video encoding |
| P4 | comparison with three.js, hardware cost model on two machines, crates.io 0.1.0 and 0.2.0, latency work (results above) |
| P5 | meshes (terrain, buildings, textured surfaces), Python bindings |

## 한국어

PointBlitz 는 Rust + wgpu 로 만드는 점군 렌더러입니다. 같은 렌더 코어를 브라우저(wasm), 네이티브, 서버(영상 전송)
세 곳에서 돌리고, three.js 기준 방식과 같은 데이터·같은 시점으로 성능을 비교합니다.
문서는 [INTENT](INTENT.md) → [SPEC](SPEC.md) → [PROJECT](PROJECT.md) → [결정 기록](docs/decisions/) 순서로 보세요.

**결과(2026-10-09, 한 PC — Windows 11·RTX 4070·i7-13700F·Chrome 155·같은 PC 안 연결, flight-01)** — 모든 비교는 같은 세션에서 번갈아 재고 측정 유효성 규칙 C1–C4 를 지킨 것이며, 중앙값 비와 부트스트랩 95 % 구간으로 판정합니다([결정 0049](docs/decisions/0049-remove-remaining-ambiguity.md)).

> **무엇과 비교했나.** 실제 운영 중인 시스템과의 비교가 아닙니다(SkyLens·skyrecon 은 지금 이 점군을 화면에 표시하지 않습니다). 기준은 SkyLens 가 쓰는 three.js 0.185 를 잘 쓴 두 가지입니다([결정 0048](docs/decisions/0048-incremental-baselines.md)): **SkyLens 식 증분 three.js**(PLY 를 받고, 미리보기는 새로 붙은 레코드만 받아 Web Worker 에서 해석)와 **같은 데이터를 받는 three.js**(PointBlitz 와 같은 청크를 해석 없이 GPU 에 올림). 브라우저·native 는 사용자 GPU, 서버 영상만 서버 GPU 가 그립니다.

- **새 정밀 스냅샷이 화면에 나오기까지(첫 반영)**: 브라우저(WebGPU) 약 35–43 ms(최종 빌드 3dce69f 세션 43.3 ms, 그 앞 두 빌드 34.5·37.2 ms), 서버 영상 약 49 ms(48.8 ms) — 목표 ≤ 50 ms 통과. 지연 작업 전에는 약 115–127 ms 로, 약 3 배 빨라졌습니다. 첫 반영은 새 스냅샷의 8 개마다 하나(거친 것 먼저)이고, 나머지까지 다 보이는 **완전 반영**은 브라우저 약 95–106 ms(브라우저가 약 30 MB 를 받는 시간이 바닥), 서버 영상은 첫 반영과 같은 약 49 ms 입니다. 출처: [latency-structure.md](docs/bench/latency-structure.md), [coarse-first.md](docs/bench/coarse-first.md), [read-overlap.md](docs/bench/read-overlap.md), [pass-chunks.md](docs/bench/pass-chunks.md).
- 재생 1 회에 받는 데이터: 180 → 135 MB(브라우저·native, 청크 좌표 16 비트).
- **three.js 대비**(2026-10-09 세션 p48, 그때의 PointBlitz 빌드 — 거친 것 먼저 이전, [comparison-incremental.md](docs/bench/comparison-incremental.md)): 정밀 스냅샷이 화면에 SkyLens 식 three.js 162 ms, 같은 데이터 three.js 117 ms, PointBlitz 127 ms; 미리보기 36.0 / 30.0 / 29.3 ms; 렌더러 메모리 354 / 199 / 183 MB. "같은 데이터 three.js" 는 지표마다 두 올리기 방식 중 좋은 쪽입니다(결정 0048): 정밀 117 ms 는 (b) CPU 로 나눠 복사, 나머지는 (a) 청크 바이트 그대로.
- **Rust·wgpu·wasm 자체의 몫**(같은 데이터·같은 그래픽 API, 같은 문서 §7): **WebGPU 에서는 PointBlitz 가 그리기 1.5 배 빠르고**(1.41 대 2.13 ms, 30 프레임마다 동기화, GPU 타임스탬프도 같은 방향) **메모리 1.57 배 적습니다**(183 대 287 MB). **WebGL2 에서는 three.js 가 그리기 약 3.2 배 빠릅니다**(1.56 대 5.00 ms). 5.00 ms 에는 PointBlitz 측정에만 있는 매 프레임 1 픽셀 복사(약 0.9 ms)가 들어 있어, 빼면 약 3.9 ms(약 2.5 배)입니다. 남은 차이의 대부분(약 1.8 ms)은 ANGLE 의 Direct3D 11 쪽입니다 — ANGLE 의 Vulkan 백엔드에서는 둘이 같습니다(§8). 일반 three.js 뷰어 대비 이득의 대부분은 구조(증분·GPU 형식·거친 것 먼저)에서 오며, 같은 데이터를 받는 three.js 도 그 이득을 얻습니다.
- 0.2.0 부터 청크 형식이 v2(좌표 16 비트)라 0.1.0 의 클라이언트·서버와 호환되지 않습니다([CHANGELOG.md](CHANGELOG.md)).
- 다른 하드웨어 값은 실측이 아니라 비용 모델 추정입니다([cost-model.md](docs/bench/cost-model.md)).

설치: `cargo add pointblitz`(라이브러리), `cargo install pointblitz-native`(창 뷰어), `cargo install pointblitz-server`(서버 영상, NVIDIA GPU·드라이버 필요, Windows·Linux) — crates.io 0.2.0, Rust 1.99 이상. 브라우저 모듈은 crates.io 가 아니라 저장소에서 `bash web/build.sh` 로 빌드합니다.

직접 돌려 보려면 위 [Try it](#try-it) 를 보세요 — 재생 서버를 띄운 뒤 브라우저·native·서버 영상 중 하나로 봅니다(드래그 회전, 휠 확대, 1–8 고정 시점). 서버 영상은 Windows 또는 Linux + NVIDIA GPU(드라이버의 NVENC)가 필요하고, `--phase-lock on` 은 클라이언트 하나·60 Hz 화면에서만 맞으며(기본 끔), 영상 페이지의 `?pacing=adaptive` 는 한 프레임을 쌓아 두어 프레임 간격을 고르게 합니다(지연 최대 한 프레임) — 60 Hz 가 아닌 화면을 위한 것이지만 그런 화면에서는 재 보지 않았습니다.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
