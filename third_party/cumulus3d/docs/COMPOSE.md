English | [한국어](COMPOSE.ko.md)

# Live composition (`cumulus3d_cli::compose`)

`Recon::declare()…build()?.run()` reconstructs a finite folder that is known before the run. `compose` is the counterpart for
input that keeps arriving (a decoder, a network receiver, a capture loop): a **retained node** owns the session, the pipeline and
the journal of accepted frames, and the caller only **declares the current input state**. The node reconciles every declaration
against what it already accepted and does only the incremental work that is needed. Direct `Pipeline::push` is not needed.

```rust
use cumulus3d_cli::compose::{CompositionHost, FrameKey, FrameState, LiveFrameSet, ReconNode, ReconcilePolicy};
use cumulus3d_cli::declare::Dense;

let host = CompositionHost::new();
let recon = ReconNode::declare()
    .images("live/images")                         // file payloads are relative to it; in-memory ones are spooled in it
    .cameras(["camF", "camR", "camL"])
    .zones(12, 2)
    .dense(Dense::profile("fast").fusion_min_views(3))
    .reconcile(ReconcilePolicy::AppendOnly)
    .on_zone_preview(|range, cloud| println!("preview {}: {} points", range.zone, cloud.len()))
    .build(&host)?;

recon.compose(FrameState::ready(frame_set_a))?;   // new identity: one position is ingested
recon.compose(FrameState::ready(frame_set_b))?;
recon.compose(FrameState::ready(frame_set_b))?;   // same identity and revision: no-op
let closed = recon.close()?;                      // close open zones, drain refinement, final events
```

## 1. Frame identity and payloads

| Type | Meaning |
|---|---|
| `FrameKey { source_id, sequence, revision }` | Identity of one synchronized set. `sequence` orders positions; `revision` versions corrections (0 = original). |
| `LiveFrameSet { key, timestamp, images, gps }` | One frame per camera plus GPS fixes. Builders: `.file(cam, rel)`, `.rgb8(cam, w, h, pixels)`, `.encoded(cam, bytes, format)`, `.gps_fix(cam, lat, lon, alt)`, `.at(time)`. |
| `FramePayload::File { relative_name }` | Existing image under the image folder. The producer must finish writing it before declaring it. |
| `FramePayload::Rgb8 { width, height, pixels }` | Decoded pixels; stored as PNG in the spool folder. |
| `FramePayload::Encoded { bytes, format }` | JPEG/PNG bytes; stored as-is in the spool folder. |

In-memory payloads are written to `<images>/<spool>/<camera>/<source>_<sequence>.<ext>` (spool default `.live`, set with
`.spool(rel)`) through a temporary file and a rename, so the reconstruction never reads a partial image. A GPS fix named after a
camera (`"camF"`) is renamed to that camera's stored image name.

## 2. Declared states and what the node does

| Declaration | Result (`Outcome`) |
|---|---|
| `FrameState::Ready(set)` with the next sequence | `Appended { position, skipped: 0 }` — one position ingested |
| same key and revision again | `Duplicate { position }` — no extraction, matching or position |
| `FrameState::Pending { key, missing_cameras, .. }` or a ready set with cameras missing | `Waiting { missing }` — nothing ingested until the set is complete (`ReconNode::waiting()`) |
| sequence jump (dropped frames) | `Appended { position, skipped: n }` + warning |
| older, never-seen sequence (late / out of order) | per policy (below) |
| higher revision of an accepted sequence (correction) | per policy (below) |
| lower revision than the accepted one | `Ignored(StaleRevision)` |
| other `source_id` than the node's | `Ignored(ForeignSource)` |
| unknown or repeated camera, missing file, wrong pixel buffer size | `Err(InvalidFrameSet)` |
| `FrameState::End` | `Ended` — the node closes; `close()` returns the kept summary |

Each accepted sequence becomes one position, in sequence order. Decisions that drop or reorder input are also emitted as
`Event::Warning` (`compose: …`), so they appear in hooks, the subscription channel and `run.log`.

## 3. Reconciliation policies (`ReconcilePolicy`)

| Policy | Late arrival | Correction | Gap |
|---|---|---|---|
| `AppendOnly` (default) | ignored | ignored | accepted |
| `ReplayWindow { positions: w }` | inserted if within the last `w` positions: rewind (`ResetFrom`) and replay the inserted set and every later set | same, from the corrected position | accepted |
| `InvalidateAffectedZones` | ignored | the corrected pixels are stored under the original names and every arrived zone containing the position is rebuilt (`InvalidateZone`); features, matches, poses and GPS are not recomputed | accepted |
| `Strict` | error | error | error |

`ReplayWindow` keeps `w + 1` rewind points in the session and the last `w` frame sets for replay. Anything older than the window is
`Ignored(OutsideReplayWindow)`.

## 4. Host and lookup

`CompositionHost` keeps the nodes by id (`.id("main")` on the builder, default `"default"`).
`reconstruct(&host, "main", state)` declares a state on that node, so a caller can re-declare on every input without holding the node.
`host.close(id)` closes and removes one node, `host.close_all()` all of them.

## 5. Producer thread and backpressure

`node.spawn()` runs the node on its own thread behind a queue and returns a `ComposeHandle`:

| Method | Behavior |
|---|---|
| `send(state)` | queue; under `Block` waits for room |
| `try_send(state)` | never waits; `Err(SendError::Full(state))` when a `Block` queue is full (usable from async code) |
| `send_all(iter)` | `send` for each item of an iterator or channel receiver |
| `stats()` | submitted, dropped, processed, current and largest depth |
| `close()` | sends `End`, waits for the queue to drain, returns the per-state outcomes and the close result |

| `Backpressure` | When the queue is full |
|---|---|
| `Block { max_pending }` (default 8) | the producer waits |
| `KeepLatest { max_pending }` | the oldest queued ready set is dropped (it then shows up as a sequence gap) |
| `Unbounded` | never full |

Per-kind hook queue policies are set with `.event_policy(EventKind::ZonePreview, QueuePolicy::LatestPerKey)`.

## 6. Builder (`ReconNode::declare()`)

Reconstruction settings are a `Plan` (same fields and TOML as `declare`; `.plan(plan)` replaces them). Positions, stride, the GPS
file and sinks of the plan are not used.

`.id`, `.images`, `.cameras`, `.zones`, `.align`, `.fixed_enu_origin`, `.features`, `.match_backend`, `.gpu`,
`.incremental_triangulation`, `.dense`, `.no_dense` (default), `.seed`, `.reconcile`, `.backpressure`, `.source`, `.spool`,
`.output(dir, SinkOptions)` (standard output files), hooks `.on`, `.on_event`, `.on_frame`, `.on_zone_preview`, `.on_zone_refined`,
`.on_message`, `.event_policy`, `.sync_hooks`, and `.build(&host)`, which reports every configuration problem at once
(`ComposeError::Config`).

Node queries: `positions()`, `journal()` (sequence, revision per position), `waiting()`, `stats()`, `subscribe()`, `poll()`,
`with_session(|s| …)`.

## 7. Guarantees (covered by tests)

`crates/cli/tests/compose_synthetic.rs` runs every policy on the synthetic scene: duplicate declarations are no-ops, pending and
incomplete sets ingest nothing, gaps are reported, late arrivals and corrections are replayed inside the window and ignored outside
it, corrections rebuild exactly the affected zones, `Strict` rejects anything but the next sequence, in-memory payloads are spooled
without partial files, and a spawned node with `KeepLatest` drops only queued ready sets. The batch API
(`Recon::declare().run()`) is unchanged.
