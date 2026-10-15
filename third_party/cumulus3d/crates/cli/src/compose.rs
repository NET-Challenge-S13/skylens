/*
 * compose.rs
 *
 * Copyright (c) 2026 ParkSangWoo
 *
 * Author(s):
 *
 *      ParkSangWoo <dev.parksangwoo@gmail.com>
 *
 *
 * This file is part of cumulus3d.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Declarative incremental composition for live frame sources.
//!
//! [`crate::declare::Recon`] runs a finite, pre-discovered folder to completion. This module is the live counterpart:
//! a retained node ([`ReconNode`]) owns a [`Session`] and its [`Pipeline`], and the caller only *declares* the current
//! input state ([`FrameState`]). The node reconciles each declaration against what it has already accepted and does
//! the minimal incremental work:
//!
//! | declared state | node action |
//! |---|---|
//! | new synchronized set (next sequence) | ingest one position |
//! | same key and revision again | nothing ([`Outcome::Duplicate`]) |
//! | set with cameras missing, or [`FrameState::Pending`] | wait, ingest nothing ([`Outcome::Waiting`]) |
//! | sequence jump (dropped frames) | ingest, report the gap ([`Outcome::Appended`]`{ skipped }`) |
//! | late arrival of an older sequence | per [`ReconcilePolicy`] |
//! | higher revision of an accepted sequence | per [`ReconcilePolicy`] |
//! | [`FrameState::End`] / [`ReconNode::close`] | close open zones, drain refinement, final events |
//!
//! Frames are identified by [`FrameKey`] (`source_id`, `sequence`, `revision`), not by path. Each accepted sequence
//! becomes one position, in sequence order. Payloads can be files under the image folder or in-memory images
//! ([`FramePayload`]); in-memory images are written atomically to a spool folder inside the image folder, because the
//! reconstruction core reads images by name.
//!
//! Events are the ordinary pipeline events (hooks registered on the builder, or [`ReconNode::subscribe`]).
//! Reconciliation decisions that drop or reorder input are also reported as [`Event::Warning`].
//!
//! # Example
//!
//! ```no_run
//! use cumulus3d_cli::compose::{reconstruct, CompositionHost, FrameKey, FrameState, LiveFrameSet, ReconNode, ReconcilePolicy};
//! use cumulus3d_cli::declare::Dense;
//!
//! fn main() -> Result<(), cumulus3d_cli::compose::ComposeError> {
//!     let host = CompositionHost::new();
//!     let recon = ReconNode::declare()
//!         .images("live/images")
//!         .cameras(["camF", "camR", "camL"])
//!         .zones(12, 2)
//!         .dense(Dense::profile("fast").fusion_min_views(3))
//!         .reconcile(ReconcilePolicy::ReplayWindow { positions: 4 })
//!         .on_zone_preview(|range, cloud| println!("preview {}: {} points", range.zone, cloud.len()))
//!         .build(&host)?;
//!
//!     let set = |seq: u64| {
//!         LiveFrameSet::new(FrameKey::new("drone-1", seq, 0))
//!             .file("camF", format!("camF/camF_{:04}.jpg", seq * 3))
//!             .file("camR", format!("camR/camR_{:04}.jpg", seq * 3))
//!             .file("camL", format!("camL/camL_{:04}.jpg", seq * 3))
//!     };
//!     recon.compose(FrameState::ready(set(0)))?;
//!     recon.compose(FrameState::ready(set(1)))?;
//!     recon.compose(FrameState::ready(set(1)))?; // no-op: same identity and revision
//!     // The same node, found by id (positional-memoization style):
//!     reconstruct(&host, "default", FrameState::ready(set(2)))?;
//!     let closed = recon.close()?;
//!     println!("{} events, {:?}", closed.summary.events, closed.stats);
//!     Ok(())
//! }
//! ```
//!
//! For a producer thread (decoder, network receiver) use [`ReconNode::spawn`]: the node then runs on its own thread
//! behind a bounded queue with an explicit [`Backpressure`] policy, and the producer only calls
//! [`ComposeHandle::send`] / [`ComposeHandle::try_send`] (both usable from async code; `try_send` never blocks).

use crate::declare::{AlignFrame, Dense, Features, Plan};
use crate::densewrap::{make_match_backend, make_sift_backend, parse_profile};
use crate::events::{Command, Event, EventKind, Frame, ZoneRange};
use crate::pipeline::{Pipeline, QueuePolicy, Summary};
use crate::session::{FrameSet, Input, Session, SessionConfig};
use crate::sinks::{self, SinkOptions};
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::io::GpsRecord;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::SystemTime;

// ============================================================================================
// Input model
// ============================================================================================

/// Stable identity of one synchronized frame set.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameKey {
    /// Source (stream) identity. A node accepts one source.
    pub source_id: String,
    /// Synchronized position sequence number (increasing along the stream).
    pub sequence: u64,
    /// Correction/replacement version of this sequence (0 = original).
    pub revision: u32,
}

impl FrameKey {
    /// Key from its three parts.
    pub fn new(source_id: impl Into<String>, sequence: u64, revision: u32) -> Self {
        Self { source_id: source_id.into(), sequence, revision }
    }
}

impl fmt::Display for FrameKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}r{}", self.source_id, self.sequence, self.revision)
    }
}

/// Encoded image container for [`FramePayload::Encoded`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodedFormat {
    /// JPEG bytes.
    Jpeg,
    /// PNG bytes.
    Png,
}

impl EncodedFormat {
    fn extension(self) -> &'static str {
        match self {
            EncodedFormat::Jpeg => "jpg",
            EncodedFormat::Png => "png",
        }
    }
}

/// Where the pixels of one camera frame come from.
#[derive(Clone)]
pub enum FramePayload {
    /// Existing file, relative to the node's image folder. The producer must finish writing it before declaring it.
    File {
        /// Path relative to the image folder (also the image name in models and GPS records).
        relative_name: String,
    },
    /// Decoded RGB8 pixels (row-major, `width * height * 3` bytes). Spooled as PNG.
    Rgb8 {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Pixels.
        pixels: Arc<[u8]>,
    },
    /// Encoded image bytes. Spooled as-is.
    Encoded {
        /// Image bytes.
        bytes: Arc<[u8]>,
        /// Container format.
        format: EncodedFormat,
    },
}

impl fmt::Debug for FramePayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FramePayload::File { relative_name } => f.debug_struct("File").field("relative_name", relative_name).finish(),
            FramePayload::Rgb8 { width, height, .. } => f.debug_struct("Rgb8").field("width", width).field("height", height).finish(),
            FramePayload::Encoded { bytes, format } => {
                f.debug_struct("Encoded").field("bytes", &bytes.len()).field("format", format).finish()
            }
        }
    }
}

/// One camera frame of a synchronized set.
#[derive(Clone, Debug)]
pub struct LiveImage {
    /// Camera name (one of the node's cameras).
    pub camera: String,
    /// Pixels.
    pub payload: FramePayload,
}

/// One synchronized frame set (one camera frame per camera) with its identity.
#[derive(Clone, Debug)]
pub struct LiveFrameSet {
    /// Identity.
    pub key: FrameKey,
    /// Capture time.
    pub timestamp: SystemTime,
    /// Camera frames, any order.
    pub images: Vec<LiveImage>,
    /// GPS fixes for this set. A record named after a camera (e.g. `"camF"`) is renamed to that camera's stored
    /// image name; a record already named after the stored image is kept.
    pub gps: Vec<GpsRecord>,
}

impl LiveFrameSet {
    /// Empty set with this key, stamped now.
    pub fn new(key: FrameKey) -> Self {
        Self { key, timestamp: SystemTime::now(), images: Vec::new(), gps: Vec::new() }
    }
    /// Adds a frame.
    pub fn image(mut self, camera: impl Into<String>, payload: FramePayload) -> Self {
        self.images.push(LiveImage { camera: camera.into(), payload });
        self
    }
    /// Adds a file frame (path relative to the image folder).
    pub fn file(self, camera: impl Into<String>, relative_name: impl Into<String>) -> Self {
        self.image(camera, FramePayload::File { relative_name: relative_name.into() })
    }
    /// Adds an RGB8 frame.
    pub fn rgb8(self, camera: impl Into<String>, width: u32, height: u32, pixels: impl Into<Arc<[u8]>>) -> Self {
        self.image(camera, FramePayload::Rgb8 { width, height, pixels: pixels.into() })
    }
    /// Adds an encoded frame.
    pub fn encoded(self, camera: impl Into<String>, bytes: impl Into<Arc<[u8]>>, format: EncodedFormat) -> Self {
        self.image(camera, FramePayload::Encoded { bytes: bytes.into(), format })
    }
    /// Adds a GPS fix for one camera (`name` = camera name, see [`LiveFrameSet::gps`]).
    pub fn gps_fix(mut self, camera: impl Into<String>, lat: f64, lon: f64, alt: f64) -> Self {
        self.gps.push(GpsRecord { name: camera.into(), lat, lon, alt });
        self
    }
    /// Sets the capture time.
    pub fn at(mut self, t: SystemTime) -> Self {
        self.timestamp = t;
        self
    }
}

/// Declared input state.
#[derive(Clone, Debug)]
pub enum FrameState {
    /// A synchronized set is available. If cameras are missing it is treated like [`FrameState::Pending`].
    Ready(LiveFrameSet),
    /// A set is being assembled; nothing is ingested until it is declared ready.
    Pending {
        /// Identity of the set being assembled.
        key: FrameKey,
        /// Capture time.
        timestamp: SystemTime,
        /// Cameras not yet received.
        missing_cameras: Vec<String>,
    },
    /// The stream ended: same as [`ReconNode::close`], the summary is kept for a later `close`.
    End,
}

impl FrameState {
    /// [`FrameState::Ready`].
    pub fn ready(set: LiveFrameSet) -> Self {
        FrameState::Ready(set)
    }
    /// [`FrameState::Pending`] stamped now.
    pub fn pending<S: Into<String>>(key: FrameKey, missing_cameras: impl IntoIterator<Item = S>) -> Self {
        FrameState::Pending { key, timestamp: SystemTime::now(), missing_cameras: missing_cameras.into_iter().map(Into::into).collect() }
    }
}

// ============================================================================================
// Policies and results
// ============================================================================================

/// What to do with input that is not a plain append (late arrivals and corrected revisions).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReconcilePolicy {
    /// Append new sequences only. Late arrivals and corrections are ignored (with a warning). Gaps are accepted.
    #[default]
    AppendOnly,
    /// Like `AppendOnly`, but a late arrival or a correction within the last `positions` positions rewinds the session
    /// to that position (`Command::ResetFrom`) and replays the corrected/inserted set and every later set.
    /// Older ones are ignored. Costs memory for `positions + 1` rewind points.
    ReplayWindow {
        /// Replay window in positions.
        positions: usize,
    },
    /// A correction keeps the registered geometry and rebuilds the outputs (preview and refined clouds) of every
    /// arrived zone containing the position, with the corrected pixels (`Command::InvalidateZone`). The corrected
    /// frame is stored under the original image name; features, matches and poses are not recomputed and GPS fixes of
    /// the correction are not applied (use `ReplayWindow` for those). Late arrivals are ignored.
    InvalidateAffectedZones,
    /// Accept only the exact next sequence; everything else (gap, late, correction, foreign source) is an error.
    Strict,
}

/// Queue policy between a producer and a spawned node ([`ReconNode::spawn`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backpressure {
    /// Never drop; the queue grows without bound.
    Unbounded,
    /// `send` blocks while `max_pending` states are queued (`try_send` returns [`SendError::Full`]).
    Block {
        /// Queue capacity.
        max_pending: usize,
    },
    /// Keep the newest states: when `max_pending` are queued, the oldest queued ready set is dropped (it then shows up
    /// as a sequence gap). Capture never waits for reconstruction.
    KeepLatest {
        /// Queue capacity.
        max_pending: usize,
    },
}

impl Default for Backpressure {
    fn default() -> Self {
        Backpressure::Block { max_pending: 8 }
    }
}

/// Why a declared state was not applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IgnoreReason {
    /// Revision lower than the accepted one.
    StaleRevision {
        /// Accepted revision.
        accepted: u32,
    },
    /// Higher revision of an accepted sequence, but the policy does not apply corrections.
    CorrectionNotApplied,
    /// Older sequence than the newest accepted one, never seen before, and the policy does not insert.
    Late,
    /// Correction or late arrival older than the replay window (or no rewind point left).
    OutsideReplayWindow {
        /// Position it would have to rewind to.
        position: usize,
    },
    /// Different `source_id` than the node's source.
    ForeignSource {
        /// The node's source.
        expected: String,
    },
    /// The node is closed.
    Closed,
}

impl fmt::Display for IgnoreReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IgnoreReason::StaleRevision { accepted } => write!(f, "stale revision (accepted revision {accepted})"),
            IgnoreReason::CorrectionNotApplied => write!(f, "correction not applied by this policy"),
            IgnoreReason::Late => write!(f, "late arrival not inserted by this policy"),
            IgnoreReason::OutsideReplayWindow { position } => write!(f, "outside replay window (position {position})"),
            IgnoreReason::ForeignSource { expected } => write!(f, "foreign source (node source is {expected})"),
            IgnoreReason::Closed => write!(f, "node closed"),
        }
    }
}

/// Result of one declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Ingested as a new position. `skipped` = sequences missing between the previous accepted one and this one.
    Appended {
        /// New position.
        position: usize,
        /// Missing sequences before this one.
        skipped: u64,
    },
    /// Same key and revision as an accepted set; nothing done.
    Duplicate {
        /// Its position.
        position: usize,
    },
    /// Set incomplete; waiting for these cameras.
    Waiting {
        /// Missing cameras.
        missing: Vec<String>,
    },
    /// Rewound to `from` and replayed `positions` sets (correction or late arrival under `ReplayWindow`).
    Replayed {
        /// Rewind position.
        from: usize,
        /// Sets pushed again (including the new one).
        positions: usize,
    },
    /// Correction applied by rebuilding these zones (`InvalidateAffectedZones`).
    Invalidated {
        /// Corrected position.
        position: usize,
        /// Rebuilt zones (empty if no zone containing the position had arrived yet).
        zones: Vec<usize>,
    },
    /// Not applied.
    Ignored(IgnoreReason),
    /// The stream ended and the node closed.
    Ended,
}

/// Errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComposeError {
    /// Builder configuration problems (all of them).
    Config(Vec<String>),
    /// Malformed frame set (unknown or duplicate camera, missing file, bad pixel buffer).
    InvalidFrameSet(String),
    /// Rejected by [`ReconcilePolicy::Strict`] (or a gap/late/correction under it).
    Rejected(String),
    /// File I/O while spooling.
    Io(String),
    /// The reconstruction session stopped on a fatal error.
    Session(String),
    /// No node with this id on the host.
    UnknownNode(String),
    /// A node with this id already exists on the host.
    DuplicateNode(String),
    /// The node was already closed and its summary taken.
    Closed,
}

impl fmt::Display for ComposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComposeError::Config(v) => write!(f, "invalid node configuration: {}", v.join("; ")),
            ComposeError::InvalidFrameSet(m) => write!(f, "invalid frame set: {m}"),
            ComposeError::Rejected(m) => write!(f, "rejected: {m}"),
            ComposeError::Io(m) => write!(f, "i/o: {m}"),
            ComposeError::Session(m) => write!(f, "session failed: {m}"),
            ComposeError::UnknownNode(id) => write!(f, "no node {id:?} on this host"),
            ComposeError::DuplicateNode(id) => write!(f, "node {id:?} already exists on this host"),
            ComposeError::Closed => write!(f, "node already closed"),
        }
    }
}

impl std::error::Error for ComposeError {}

/// Reconciliation counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComposeStats {
    /// Declarations received (all states).
    pub declared: u64,
    /// Sets ingested as new positions.
    pub appended: u64,
    /// Duplicate declarations (no-ops).
    pub duplicates: u64,
    /// Pending/incomplete declarations.
    pub waiting: u64,
    /// Sequences missing between accepted ones (dropped frames).
    pub skipped_sequences: u64,
    /// Rewind-and-replay operations.
    pub replays: u64,
    /// Sets pushed again by replays.
    pub replayed_positions: u64,
    /// Corrections applied by zone invalidation.
    pub invalidations: u64,
    /// Declarations ignored.
    pub ignored: u64,
}

/// What [`ReconNode::close`] returns.
pub struct Closed {
    /// Final session state.
    pub session: Session,
    /// Pipeline summary (event counts, hook calls).
    pub summary: Summary,
    /// Reconciliation counters.
    pub stats: ComposeStats,
}

impl fmt::Debug for Closed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Closed")
            .field("positions", &self.session.positions())
            .field("summary", &self.summary)
            .field("stats", &self.stats)
            .finish()
    }
}

// ============================================================================================
// Host
// ============================================================================================

/// Owner of retained nodes, addressed by id. Cheap to clone (shared).
#[derive(Clone, Default)]
pub struct CompositionHost {
    nodes: Arc<Mutex<BTreeMap<String, ReconNode>>>,
}

impl fmt::Debug for CompositionHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompositionHost").field("nodes", &self.ids()).finish()
    }
}

impl CompositionHost {
    /// Empty host.
    pub fn new() -> Self {
        Self::default()
    }
    /// Node by id.
    pub fn node(&self, id: &str) -> Option<ReconNode> {
        lock(&self.nodes).get(id).cloned()
    }
    /// Ids of the open nodes.
    pub fn ids(&self) -> Vec<String> {
        lock(&self.nodes).keys().cloned().collect()
    }
    /// Declares a state on node `id` ([`ReconNode::compose`]).
    pub fn compose(&self, id: &str, state: FrameState) -> Result<Outcome, ComposeError> {
        self.node(id).ok_or_else(|| ComposeError::UnknownNode(id.to_string()))?.compose(state)
    }
    /// Closes node `id` and removes it from the host.
    pub fn close(&self, id: &str) -> Result<Closed, ComposeError> {
        let node = lock(&self.nodes).remove(id).ok_or_else(|| ComposeError::UnknownNode(id.to_string()))?;
        node.close_inner()
    }
    /// Closes every node (id order) and removes them.
    pub fn close_all(&self) -> Vec<(String, Result<Closed, ComposeError>)> {
        let nodes = std::mem::take(&mut *lock(&self.nodes));
        nodes.into_iter().map(|(id, n)| (id, n.close_inner())).collect()
    }
    fn register(&self, node: &ReconNode) -> Result<(), ComposeError> {
        let mut m = lock(&self.nodes);
        if m.contains_key(&node.id) {
            return Err(ComposeError::DuplicateNode(node.id.clone()));
        }
        m.insert(node.id.clone(), node.clone());
        Ok(())
    }
}

/// Declares `state` on the host's node `id`: the declarative entry point
/// (`reconstruct(&host, "main", FrameState::ready(set))` on every new input).
pub fn reconstruct(host: &CompositionHost, id: &str, state: FrameState) -> Result<Outcome, ComposeError> {
    host.compose(id, state)
}

// ============================================================================================
// Builder
// ============================================================================================

type Attach = Box<dyn FnOnce(Pipeline<Session>) -> Pipeline<Session> + Send>;

/// Records a node configuration; [`NodeBuilder::build`] checks it and creates the node on a host.
///
/// The reconstruction settings are a [`Plan`] (same fields and TOML as [`crate::declare`]); the input folder is the
/// image root, positions/stride are not used (input is declared live), and GPS comes with each set.
pub struct NodeBuilder {
    id: String,
    plan: Plan,
    policy: ReconcilePolicy,
    backpressure: Backpressure,
    source: Option<String>,
    spool: PathBuf,
    output: Option<(PathBuf, SinkOptions)>,
    attach: Vec<Attach>,
    frames: bool,
}

impl fmt::Debug for NodeBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeBuilder").field("id", &self.id).field("policy", &self.policy).field("hooks", &self.attach.len()).finish()
    }
}

impl ReconNode {
    /// Starts a node declaration with the default plan (`aerial-formation` cameras, zones 12/2), dense off, no output files.
    pub fn declare() -> NodeBuilder {
        let mut plan = Plan::default();
        plan.dense.enabled = false;
        plan.gps = None;
        NodeBuilder {
            id: "default".to_string(),
            plan,
            policy: ReconcilePolicy::default(),
            backpressure: Backpressure::default(),
            source: None,
            spool: PathBuf::from(".live"),
            output: None,
            attach: Vec::new(),
            frames: false,
        }
    }
}

impl NodeBuilder {
    /// Node id on the host (default `"default"`).
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }
    /// Replaces the reconstruction settings with a plan (e.g. read from TOML). Input positions/stride and
    /// the GPS file are ignored; sinks are ignored (use [`NodeBuilder::output`]).
    pub fn plan(mut self, plan: Plan) -> Self {
        self.plan = plan;
        self
    }
    /// Image folder: file payloads are relative to it, in-memory payloads are spooled inside it.
    pub fn images(mut self, dir: impl Into<PathBuf>) -> Self {
        self.plan.input.images = dir.into();
        self
    }
    /// Cameras of one synchronized set, in position order.
    pub fn cameras<S: Into<String>>(mut self, cams: impl IntoIterator<Item = S>) -> Self {
        self.plan.input.cameras = cams.into_iter().map(Into::into).collect();
        self
    }
    /// Zone size and overlap in positions.
    pub fn zones(mut self, span: usize, overlap: usize) -> Self {
        self.plan.zones.span = span;
        self.plan.zones.overlap = overlap;
        self
    }
    /// Coordinate alignment (ENU needs GPS fixes in the sets).
    pub fn align(mut self, frame: AlignFrame) -> Self {
        self.plan.align.frame = frame;
        self
    }
    /// Fixes the ENU origin at the first GPS fix.
    pub fn fixed_enu_origin(mut self, on: bool) -> Self {
        self.plan.align.fixed_origin = on;
        self
    }
    /// Feature extraction settings.
    pub fn features(mut self, f: Features) -> Self {
        self.plan.features = f;
        self
    }
    /// Descriptor matching backend (`cpu` | `cuda`).
    pub fn match_backend(mut self, name: &str) -> Self {
        self.plan.matching.backend = name.to_string();
        self
    }
    /// CUDA for features, matching and densification.
    pub fn gpu(mut self) -> Self {
        self.plan.features.backend = "cuda".into();
        self.plan.matching.backend = "cuda".into();
        self.plan.dense.backend = "cuda".into();
        self
    }
    /// Triangulate only newly registered images.
    pub fn incremental_triangulation(mut self, on: bool) -> Self {
        self.plan.sparse.incremental_triangulation = on;
        self
    }
    /// Densification settings (enables it).
    pub fn dense(mut self, d: Dense) -> Self {
        self.plan.dense = Dense { enabled: true, ..d };
        self
    }
    /// No densification (default).
    pub fn no_dense(mut self) -> Self {
        self.plan.dense.enabled = false;
        self
    }
    /// RANSAC seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.plan.seed = Some(seed);
        self
    }
    /// Reconciliation policy (default [`ReconcilePolicy::AppendOnly`]).
    pub fn reconcile(mut self, p: ReconcilePolicy) -> Self {
        self.policy = p;
        self
    }
    /// Producer queue policy for [`ReconNode::spawn`] (default `Block { max_pending: 8 }`).
    pub fn backpressure(mut self, b: Backpressure) -> Self {
        self.backpressure = b;
        self
    }
    /// Accept only this source id (default: the first declared source).
    pub fn source(mut self, id: impl Into<String>) -> Self {
        self.source = Some(id.into());
        self
    }
    /// Spool folder for in-memory payloads, relative to the image folder (default `.live`).
    pub fn spool(mut self, rel: impl Into<PathBuf>) -> Self {
        self.spool = rel.into();
        self
    }
    /// Writes the standard output files (timeline, run.log, zone PLYs, snapshots) to `dir`.
    pub fn output(mut self, dir: impl Into<PathBuf>, opts: SinkOptions) -> Self {
        self.output = Some((dir.into(), opts));
        self
    }
    fn hook(mut self, f: impl FnOnce(Pipeline<Session>) -> Pipeline<Session> + Send + 'static) -> Self {
        self.attach.push(Box::new(f));
        self
    }
    /// Hook for one event kind ([`Pipeline::on`]). `FrameDecoded` turns frame decoding on.
    pub fn on<F>(mut self, kind: EventKind, f: F) -> Self
    where
        F: FnMut(&Event) + Send + 'static,
    {
        self.frames |= kind == EventKind::FrameDecoded;
        self.hook(move |p| p.on(kind, f))
    }
    /// Hook for every event ([`Pipeline::on_any`]).
    pub fn on_event<F>(self, f: F) -> Self
    where
        F: FnMut(&Event) + Send + 'static,
    {
        self.hook(move |p| p.on_any(f))
    }
    /// Decoded input frames ([`Pipeline::on_frame`]); turns frame decoding on.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: FnMut(&Arc<Frame>) + Send + 'static,
    {
        self.frames = true;
        self.hook(move |p| p.on_frame(f))
    }
    /// Zone preview clouds ([`Pipeline::on_zone_preview`]).
    pub fn on_zone_preview<F>(self, f: F) -> Self
    where
        F: FnMut(ZoneRange, &Arc<PointCloud>) + Send + 'static,
    {
        self.hook(move |p| p.on_zone_preview(f))
    }
    /// Zone refined clouds ([`Pipeline::on_zone_refined`]).
    pub fn on_zone_refined<F>(self, f: F) -> Self
    where
        F: FnMut(ZoneRange, &Arc<PointCloud>) + Send + 'static,
    {
        self.hook(move |p| p.on_zone_refined(f))
    }
    /// Warnings and errors ([`Pipeline::on_message`]); includes reconciliation warnings.
    pub fn on_message<F>(self, f: F) -> Self
    where
        F: FnMut(EventKind, &str) + Send + 'static,
    {
        self.hook(move |p| p.on_message(f))
    }
    /// Per-kind hook queue policy ([`Pipeline::policy`]), e.g. `LatestPerKey` for `ZonePreview`.
    pub fn event_policy(self, kind: EventKind, p: QueuePolicy) -> Self {
        self.hook(move |pl| pl.policy(kind, p))
    }
    /// Runs hooks synchronously on the composing thread ([`Pipeline::sync`]).
    pub fn sync_hooks(self, on: bool) -> Self {
        self.hook(move |pl| pl.sync(on))
    }

    /// Checks the configuration, creates the session and pipeline, and registers the node on `host`.
    /// Creates the image folder's spool folder; nothing else is written until frames are declared.
    pub fn build(self, host: &CompositionHost) -> Result<ReconNode, ComposeError> {
        let NodeBuilder { id, plan, policy, backpressure, source, spool, output, attach, frames } = self;
        let mut pr = Vec::new();
        let cams = &plan.input.cameras;
        if cams.is_empty() {
            pr.push("cameras: empty".to_string());
        }
        let mut seen = std::collections::BTreeSet::new();
        for c in cams {
            if c.is_empty() || c.contains(['/', '\\']) {
                pr.push(format!("cameras: invalid name {c:?}"));
            } else if !seen.insert(c.as_str()) {
                pr.push(format!("cameras: duplicate {c}"));
            }
        }
        let z = plan.zones;
        if z.span == 0 {
            pr.push("zones.span: must be at least 1".into());
        } else if z.overlap >= z.span {
            pr.push(format!("zones.overlap: must be smaller than the span ({}), got {}", z.span, z.overlap));
        }
        if !plan.input.images.is_dir() {
            pr.push(format!("images: folder not found: {}", plan.input.images.display()));
        }
        if spool.is_absolute() || spool.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
            pr.push(format!("spool: must be a relative path inside the image folder, got {}", spool.display()));
        }
        if let ReconcilePolicy::ReplayWindow { positions: 0 } = policy {
            pr.push("reconcile: replay window must be at least 1 position".into());
        }
        match backpressure {
            Backpressure::Block { max_pending: 0 } | Backpressure::KeepLatest { max_pending: 0 } => {
                pr.push("backpressure: max_pending must be at least 1".into())
            }
            _ => {}
        }
        let sift = make_sift_backend(&plan.features.backend).map_err(|e| pr.push(format!("features.backend: {e}"))).ok();
        let matcher = make_match_backend(&plan.matching.backend).map_err(|e| pr.push(format!("matching.backend: {e}"))).ok();
        if plan.dense.enabled {
            if let Err(e) = parse_profile(&plan.dense.profile) {
                pr.push(format!("dense.profile: {e}"));
            }
        }
        let dense = if plan.dense.enabled { plan.dense.config().map_err(|e| pr.push(format!("dense: {e}"))).ok() } else { None };
        if !pr.is_empty() {
            return Err(ComposeError::Config(pr));
        }
        if host.node(&id).is_some() {
            return Err(ComposeError::DuplicateNode(id));
        }

        let mut sc = SessionConfig::new(plan.input.images.clone());
        sc.span = z.span;
        sc.overlap = z.overlap;
        sc.total_positions = None;
        sc.incremental_triangulation = plan.sparse.incremental_triangulation;
        sc.fixed_enu_origin = plan.align.fixed_origin;
        sc.seed = plan.seed;
        sc.extraction.sift.max_num_features = plan.features.max_num_features;
        sc.extraction.reader.max_image_size = plan.features.max_image_size;
        sc.sift = sift.expect("checked");
        sc.matcher = Arc::from(matcher.expect("checked"));
        sc.dense = dense;
        sc.history = match policy {
            ReconcilePolicy::ReplayWindow { positions } => positions + 1,
            _ => 0,
        };
        sc.decode_frames = frames;

        let mut pl = Pipeline::new(Session::new(sc));
        if let Some((dir, opts)) = &output {
            pl = sinks::attach(pl, dir, opts).map_err(|e| ComposeError::Io(format!("{}: {e}", dir.display())))?;
        }
        for a in attach {
            pl = a(pl);
        }
        let core = Core {
            cameras: plan.input.cameras.clone(),
            image_root: plan.input.images.clone(),
            spool,
            policy,
            source,
            accepted: BTreeMap::new(),
            order: Vec::new(),
            recent: VecDeque::new(),
            waiting: BTreeMap::new(),
            stats: ComposeStats::default(),
            pipeline: Some(pl),
            ended: None,
        };
        let node = ReconNode { id, backpressure, core: Arc::new(Mutex::new(core)) };
        host.register(&node)?;
        Ok(node)
    }
}

// ============================================================================================
// Node
// ============================================================================================

/// A retained reconstruction node: session, pipeline, accepted-frame journal and lifecycle.
/// Cheap to clone (shared); all clones address the same node.
#[derive(Clone)]
pub struct ReconNode {
    id: String,
    backpressure: Backpressure,
    core: Arc<Mutex<Core>>,
}

impl fmt::Debug for ReconNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReconNode").field("id", &self.id).finish()
    }
}

#[derive(Clone, Copy, Debug)]
struct Accepted {
    revision: u32,
    position: usize,
}

struct Core {
    cameras: Vec<String>,
    image_root: PathBuf,
    spool: PathBuf,
    policy: ReconcilePolicy,
    source: Option<String>,
    /// sequence -> accepted revision and position.
    accepted: BTreeMap<u64, Accepted>,
    /// sequence at each position (increasing).
    order: Vec<u64>,
    /// Resolved sets of the last positions (replay buffer), aligned to the end of `order`.
    recent: VecDeque<FrameSet>,
    /// Sets being assembled: sequence -> missing cameras.
    waiting: BTreeMap<u64, Vec<String>>,
    stats: ComposeStats,
    pipeline: Option<Pipeline<Session>>,
    /// Summary of a node closed by `FrameState::End`, until `close` takes it.
    ended: Option<Closed>,
}

impl ReconNode {
    /// Node id on its host.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Declares the current input state and reconciles it (see the module table).
    pub fn compose(&self, state: FrameState) -> Result<Outcome, ComposeError> {
        lock(&self.core).compose(state)
    }

    /// Declares several states in order, stopping at the first error.
    pub fn compose_all(&self, states: impl IntoIterator<Item = FrameState>) -> Result<Vec<Outcome>, ComposeError> {
        states.into_iter().map(|s| self.compose(s)).collect()
    }

    /// Collects finished background (refinement) events without new input.
    pub fn poll(&self) -> usize {
        lock(&self.core).pipeline.as_mut().map_or(0, |p| p.poll())
    }

    /// Channel receiving every event of this node (see [`Pipeline::subscribe`]). `None` after close.
    pub fn subscribe(&self) -> Option<Receiver<Event>> {
        lock(&self.core).pipeline.as_ref().map(|p| p.subscribe())
    }

    /// Reconciliation counters so far.
    pub fn stats(&self) -> ComposeStats {
        lock(&self.core).stats.clone()
    }

    /// Positions ingested so far.
    pub fn positions(&self) -> usize {
        lock(&self.core).order.len()
    }

    /// Accepted (sequence, revision) per position.
    pub fn journal(&self) -> Vec<(u64, u32)> {
        let c = lock(&self.core);
        c.order.iter().map(|s| (*s, c.accepted[s].revision)).collect()
    }

    /// Sets currently waiting for cameras: (sequence, missing cameras).
    pub fn waiting(&self) -> Vec<(u64, Vec<String>)> {
        lock(&self.core).waiting.iter().map(|(k, v)| (*k, v.clone())).collect()
    }

    /// Read access to the session (e.g. arrived zones, models).
    pub fn with_session<T>(&self, f: impl FnOnce(&Session) -> T) -> Option<T> {
        lock(&self.core).pipeline.as_ref().map(|p| f(p.reducer()))
    }

    /// Closes the node: closes open zones, waits for background refinement and emits the final events.
    /// The node stays listed on its host; [`CompositionHost::close`] also removes it.
    pub fn close(&self) -> Result<Closed, ComposeError> {
        self.close_inner()
    }

    fn close_inner(&self) -> Result<Closed, ComposeError> {
        let mut c = lock(&self.core);
        if let Some(done) = c.ended.take() {
            return Ok(done);
        }
        c.finish().ok_or(ComposeError::Closed)
    }

    /// Moves the node onto its own thread behind a queue with the builder's [`Backpressure`] policy.
    /// The node stays registered on its host; composing on it directly while spawned is serialized with the worker.
    pub fn spawn(&self) -> ComposeHandle {
        let queue = Arc::new(Queue { st: Mutex::new(QueueState::default()), cv: Condvar::new() });
        let (q, node) = (Arc::clone(&queue), self.clone());
        let worker = std::thread::Builder::new()
            .name(format!("compose-{}", self.id))
            .spawn(move || {
                let mut outcomes = Vec::new();
                loop {
                    let next = {
                        let mut st = lock(&q.st);
                        loop {
                            if let Some(s) = st.items.pop_front() {
                                q.cv.notify_all();
                                break Some(s);
                            }
                            if st.closed {
                                break None;
                            }
                            st = q.cv.wait(st).unwrap_or_else(|p| p.into_inner());
                        }
                    };
                    let Some(state) = next else { break };
                    let end = matches!(state, FrameState::End);
                    let r = node.compose(state);
                    lock(&q.st).processed += 1;
                    let fatal = matches!(r, Err(ComposeError::Session(_)));
                    outcomes.push(r);
                    if end || fatal {
                        break;
                    }
                }
                let mut st = lock(&q.st);
                st.closed = true;
                st.items.clear();
                q.cv.notify_all();
                outcomes
            })
            .expect("compose worker thread");
        ComposeHandle { node: self.clone(), queue, policy: self.backpressure, worker: Some(worker) }
    }
}

impl Core {
    fn pl(&mut self) -> &mut Pipeline<Session> {
        self.pipeline.as_mut().expect("open pipeline")
    }

    fn warn(&mut self, message: String) {
        if let Some(p) = self.pipeline.as_mut() {
            p.reducer().emit_external(|meta| Event::Warning { meta, message });
            p.poll();
        }
    }

    fn ignore(&mut self, key: &FrameKey, reason: IgnoreReason) -> Result<Outcome, ComposeError> {
        if self.policy == ReconcilePolicy::Strict && reason != IgnoreReason::Closed {
            return Err(ComposeError::Rejected(format!("{key}: {reason}")));
        }
        self.stats.ignored += 1;
        self.warn(format!("compose: {key} ignored: {reason}"));
        Ok(Outcome::Ignored(reason))
    }

    fn compose(&mut self, state: FrameState) -> Result<Outcome, ComposeError> {
        self.stats.declared += 1;
        if self.pipeline.is_none() {
            return match state {
                FrameState::End => Ok(Outcome::Ended),
                _ => Ok(Outcome::Ignored(IgnoreReason::Closed)),
            };
        }
        match state {
            FrameState::End => {
                let done = self.finish().ok_or(ComposeError::Closed)?;
                self.ended = Some(done);
                Ok(Outcome::Ended)
            }
            FrameState::Pending { key, missing_cameras, .. } => {
                if let Some(r) = self.check_source(&key) {
                    return r;
                }
                self.stats.waiting += 1;
                if !self.accepted.contains_key(&key.sequence) {
                    self.waiting.insert(key.sequence, missing_cameras.clone());
                }
                Ok(Outcome::Waiting { missing: missing_cameras })
            }
            FrameState::Ready(set) => self.ready(set),
        }
    }

    fn check_source(&mut self, key: &FrameKey) -> Option<Result<Outcome, ComposeError>> {
        match &self.source {
            None => {
                self.source = Some(key.source_id.clone());
                None
            }
            Some(s) if *s == key.source_id => None,
            Some(s) => {
                let expected = s.clone();
                Some(self.ignore(key, IgnoreReason::ForeignSource { expected }))
            }
        }
    }

    fn ready(&mut self, set: LiveFrameSet) -> Result<Outcome, ComposeError> {
        let key = set.key.clone();
        // Shape checks first: a malformed set is an error under every policy.
        let mut by_cam: HashMap<&str, &LiveImage> = HashMap::new();
        for im in &set.images {
            if !self.cameras.contains(&im.camera) {
                return Err(ComposeError::InvalidFrameSet(format!("{key}: unknown camera {}", im.camera)));
            }
            if by_cam.insert(im.camera.as_str(), im).is_some() {
                return Err(ComposeError::InvalidFrameSet(format!("{key}: camera {} given twice", im.camera)));
            }
        }
        if let Some(r) = self.check_source(&key) {
            return r;
        }
        let missing: Vec<String> = self.cameras.iter().filter(|c| !by_cam.contains_key(c.as_str())).cloned().collect();
        if !missing.is_empty() && !self.accepted.contains_key(&key.sequence) {
            self.stats.waiting += 1;
            self.waiting.insert(key.sequence, missing.clone());
            return Ok(Outcome::Waiting { missing });
        }

        if let Some(acc) = self.accepted.get(&key.sequence).copied() {
            if key.revision == acc.revision {
                self.stats.duplicates += 1;
                return Ok(Outcome::Duplicate { position: acc.position });
            }
            if key.revision < acc.revision {
                return self.ignore(&key, IgnoreReason::StaleRevision { accepted: acc.revision });
            }
            if !missing.is_empty() {
                return Err(ComposeError::InvalidFrameSet(format!("{key}: correction misses cameras {}", missing.join(", "))));
            }
            return self.correct(set, acc);
        }

        let newest = self.order.last().copied();
        match newest {
            Some(n) if key.sequence < n => self.late(set),
            _ => {
                let skipped = newest.map_or(0, |n| key.sequence - n - 1);
                if skipped > 0 && self.policy == ReconcilePolicy::Strict {
                    return Err(ComposeError::Rejected(format!("{key}: {skipped} sequence(s) missing before it")));
                }
                let fs = self.resolve(&set, None)?;
                self.waiting.remove(&key.sequence);
                let position = self.order.len();
                self.push(fs)?;
                self.order.push(key.sequence);
                self.accepted.insert(key.sequence, Accepted { revision: key.revision, position });
                self.stats.appended += 1;
                if skipped > 0 {
                    self.stats.skipped_sequences += skipped;
                    self.warn(format!("compose: {key} appended at position {position} after {skipped} missing sequence(s)"));
                }
                Ok(Outcome::Appended { position, skipped })
            }
        }
    }

    /// Late arrival of an unseen older sequence.
    fn late(&mut self, set: LiveFrameSet) -> Result<Outcome, ComposeError> {
        let key = set.key.clone();
        let ReconcilePolicy::ReplayWindow { positions: w } = self.policy else {
            return self.ignore(&key, IgnoreReason::Late);
        };
        let p = self.order.partition_point(|&s| s < key.sequence);
        if !self.in_window(p, w) {
            return self.ignore(&key, IgnoreReason::OutsideReplayWindow { position: p });
        }
        let fs = self.resolve(&set, None)?;
        let tail_sets: Vec<FrameSet> = self.recent.iter().skip(self.recent.len() - (self.order.len() - p)).cloned().collect();
        let mut tail = vec![fs];
        tail.extend(tail_sets);
        let mut seqs = vec![key.sequence];
        seqs.extend(self.order[p..].iter().copied());
        self.waiting.remove(&key.sequence);
        let n = self.replay(p, tail, &seqs, (key.sequence, key.revision))?;
        self.warn(format!("compose: late {key} inserted at position {p}, replayed {n} position(s)"));
        Ok(Outcome::Replayed { from: p, positions: n })
    }

    /// Higher revision of an accepted sequence.
    fn correct(&mut self, set: LiveFrameSet, acc: Accepted) -> Result<Outcome, ComposeError> {
        let key = set.key.clone();
        let p = acc.position;
        match self.policy {
            ReconcilePolicy::AppendOnly | ReconcilePolicy::Strict => self.ignore(&key, IgnoreReason::CorrectionNotApplied),
            ReconcilePolicy::ReplayWindow { positions: w } => {
                if !self.in_window(p, w) {
                    return self.ignore(&key, IgnoreReason::OutsideReplayWindow { position: p });
                }
                let fs = self.resolve(&set, None)?;
                let start = self.recent.len() - (self.order.len() - p);
                let mut tail: Vec<FrameSet> = self.recent.iter().skip(start).cloned().collect();
                tail[0] = fs;
                let seqs: Vec<u64> = self.order[p..].to_vec();
                let n = self.replay(p, tail, &seqs, (key.sequence, key.revision))?;
                self.warn(format!("compose: correction {key} applied from position {p}, replayed {n} position(s)"));
                Ok(Outcome::Replayed { from: p, positions: n })
            }
            ReconcilePolicy::InvalidateAffectedZones => {
                // Store the corrected pixels under the original names so rebuilt outputs read them.
                let names: Vec<String> = self.pl().reducer().frames()[p].iter().map(|f| f.name.clone()).collect();
                self.resolve(&set, Some(&names))?;
                let zones: Vec<usize> =
                    self.pl().reducer().arrived_zones().into_iter().filter(|z| z.lo <= p && p < z.hi).map(|z| z.zone).collect();
                for &z in &zones {
                    self.pl().command(Command::InvalidateZone(z));
                }
                self.fail_check()?;
                self.accepted.insert(key.sequence, Accepted { revision: key.revision, position: p });
                self.stats.invalidations += 1;
                Ok(Outcome::Invalidated { position: p, zones })
            }
        }
    }

    /// Whether position `p` can be rewound to with window `w`.
    fn in_window(&self, p: usize, w: usize) -> bool {
        let npos = self.order.len();
        npos - p <= w && npos - p <= self.recent.len()
    }

    /// Rewinds to `p` and pushes `tail` (sequences `seqs`), updating the journal and the replay buffer.
    /// `changed` = the (sequence, revision) that differs from the journal (corrected or inserted set).
    fn replay(&mut self, p: usize, tail: Vec<FrameSet>, seqs: &[u64], changed: (u64, u32)) -> Result<usize, ComposeError> {
        let npos = self.order.len();
        self.pl().command(Command::ResetFrom(p));
        if self.pl().reducer().positions() != p {
            return Err(ComposeError::Session(format!("rewind to position {p} was refused by the session")));
        }
        let revs: HashMap<u64, u32> = self.order[p..].iter().map(|s| (*s, self.accepted[s].revision)).collect();
        for s in &self.order[p..] {
            self.accepted.remove(s);
        }
        self.order.truncate(p);
        let keep = self.recent.len() - (npos - p);
        self.recent.truncate(keep);
        let n = tail.len();
        for (fs, &seq) in tail.into_iter().zip(seqs) {
            let revision = if seq == changed.0 { changed.1 } else { revs[&seq] };
            let position = self.order.len();
            self.push(fs)?;
            self.order.push(seq);
            self.accepted.insert(seq, Accepted { revision, position });
        }
        self.stats.replays += 1;
        self.stats.replayed_positions += n as u64;
        Ok(n)
    }

    /// Pushes one resolved set and keeps it in the replay buffer.
    fn push(&mut self, fs: FrameSet) -> Result<(), ComposeError> {
        let cap = match self.policy {
            ReconcilePolicy::ReplayWindow { positions } => positions,
            _ => 0,
        };
        if cap > 0 {
            self.recent.push_back(fs.clone());
            while self.recent.len() > cap {
                self.recent.pop_front();
            }
        }
        self.pl().push(Input::Frames(fs));
        self.fail_check()
    }

    fn fail_check(&mut self) -> Result<(), ComposeError> {
        match self.pl().reducer().failed() {
            Some(e) => Err(ComposeError::Session(e.to_string())),
            None => Ok(()),
        }
    }

    /// Materializes payloads (spooling in-memory ones) and builds the session's frame set in camera order.
    /// `names` forces the stored names (corrections under `InvalidateAffectedZones`).
    fn resolve(&self, set: &LiveFrameSet, names: Option<&[String]>) -> Result<FrameSet, ComposeError> {
        let key = &set.key;
        let mut fs = FrameSet::default();
        for (i, cam) in self.cameras.iter().enumerate() {
            let im = set.images.iter().find(|x| &x.camera == cam).expect("complete set");
            let forced = names.map(|n| n[i].clone());
            let name = match &im.payload {
                FramePayload::File { relative_name } => {
                    let src = self.image_root.join(relative_name);
                    if !src.is_file() {
                        return Err(ComposeError::InvalidFrameSet(format!("{key}: file not found: {}", src.display())));
                    }
                    match forced {
                        Some(n) if n != *relative_name => {
                            let bytes = std::fs::read(&src).map_err(|e| ComposeError::Io(format!("{}: {e}", src.display())))?;
                            self.store_encoded(&n, &bytes, ext_of(relative_name))?;
                            n
                        }
                        _ => relative_name.clone(),
                    }
                }
                FramePayload::Rgb8 { width, height, pixels } => {
                    if pixels.len() != *width as usize * *height as usize * 3 || *width == 0 || *height == 0 {
                        return Err(ComposeError::InvalidFrameSet(format!(
                            "{key}: camera {cam}: {} bytes for {width}x{height} RGB8",
                            pixels.len()
                        )));
                    }
                    let n = forced.unwrap_or_else(|| self.spool_name(key, cam, "png"));
                    let img = image::RgbImage::from_raw(*width, *height, pixels.to_vec()).expect("size checked");
                    self.store_image(&n, &image::DynamicImage::ImageRgb8(img))?;
                    n
                }
                FramePayload::Encoded { bytes, format } => {
                    let n = forced.unwrap_or_else(|| self.spool_name(key, cam, format.extension()));
                    self.store_encoded(&n, bytes, Some(format.extension()))?;
                    n
                }
            };
            fs.images.push(crate::session::FrameImage { camera: cam.clone(), name });
        }
        for g in &set.gps {
            let mut g = g.clone();
            if let Some(i) = self.cameras.iter().position(|c| *c == g.name) {
                g.name = fs.images[i].name.clone();
            }
            fs.gps.push(g);
        }
        Ok(fs)
    }

    fn spool_name(&self, key: &FrameKey, cam: &str, ext: &str) -> String {
        let src: String = key.source_id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        let p = self.spool.join(cam).join(format!("{src}_{:010}.{ext}", key.sequence));
        p.to_string_lossy().replace('\\', "/")
    }

    /// Writes encoded bytes to `name`; re-encodes when the target extension differs from the bytes' format.
    fn store_encoded(&self, name: &str, bytes: &[u8], ext: Option<&str>) -> Result<(), ComposeError> {
        if ext.is_some() && ext == ext_of(name) {
            return self.atomic_write(name, |path| std::fs::write(path, bytes).map_err(|e| e.to_string()));
        }
        let img = image::load_from_memory(bytes).map_err(|e| ComposeError::InvalidFrameSet(format!("{name}: {e}")))?;
        self.store_image(name, &img)
    }

    fn store_image(&self, name: &str, img: &image::DynamicImage) -> Result<(), ComposeError> {
        let fmt = image::ImageFormat::from_path(name).map_err(|e| ComposeError::Io(format!("{name}: {e}")))?;
        self.atomic_write(name, |path| img.save_with_format(path, fmt).map_err(|e| e.to_string()))
    }

    /// Writes through a temporary file in the same folder and renames, so readers never see a partial image.
    fn atomic_write(&self, name: &str, write: impl FnOnce(&Path) -> Result<(), String>) -> Result<(), ComposeError> {
        let dst = self.image_root.join(name);
        let dir = dst.parent().unwrap_or(&self.image_root);
        std::fs::create_dir_all(dir).map_err(|e| ComposeError::Io(format!("{}: {e}", dir.display())))?;
        let file = dst.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let ext = ext_of(name).unwrap_or("tmp");
        let tmp = dir.join(format!(".{file}.partial.{ext}"));
        write(&tmp).map_err(|e| ComposeError::Io(format!("{}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &dst).map_err(|e| ComposeError::Io(format!("{}: {e}", dst.display())))
    }

    fn finish(&mut self) -> Option<Closed> {
        let pl = self.pipeline.take()?;
        let (session, summary) = pl.finish();
        Some(Closed { session, summary, stats: self.stats.clone() })
    }
}

fn ext_of(name: &str) -> Option<&str> {
    Path::new(name).extension().and_then(|e| e.to_str())
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

// ============================================================================================
// Spawned node
// ============================================================================================

#[derive(Default)]
struct QueueState {
    items: VecDeque<FrameState>,
    closed: bool,
    submitted: u64,
    dropped: u64,
    processed: u64,
    max_depth: usize,
}

struct Queue {
    st: Mutex<QueueState>,
    cv: Condvar,
}

/// Why a state could not be queued.
#[derive(Debug)]
pub enum SendError {
    /// `try_send` under `Block`: the queue is full. The state is handed back.
    Full(FrameState),
    /// The worker stopped (closed, or the session failed). The state is handed back.
    Closed(FrameState),
}

impl fmt::Display for SendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SendError::Full(_) => write!(f, "compose queue full"),
            SendError::Closed(_) => write!(f, "compose worker stopped"),
        }
    }
}

impl std::error::Error for SendError {}

/// Queue counters of a spawned node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueueStats {
    /// States accepted into the queue.
    pub submitted: u64,
    /// Ready sets dropped by `KeepLatest`.
    pub dropped: u64,
    /// States composed by the worker.
    pub processed: u64,
    /// Current queue depth.
    pub depth: usize,
    /// Largest queue depth seen.
    pub max_depth: usize,
}

/// Producer side of a spawned node ([`ReconNode::spawn`]).
pub struct ComposeHandle {
    node: ReconNode,
    queue: Arc<Queue>,
    policy: Backpressure,
    worker: Option<JoinHandle<Vec<Result<Outcome, ComposeError>>>>,
}

impl fmt::Debug for ComposeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComposeHandle").field("node", &self.node.id).field("policy", &self.policy).field("stats", &self.stats()).finish()
    }
}

impl ComposeHandle {
    /// The node.
    pub fn node(&self) -> &ReconNode {
        &self.node
    }

    /// Queues a state. Under `Block` waits for room; never waits otherwise.
    pub fn send(&self, state: FrameState) -> Result<(), SendError> {
        self.enqueue(state, true)
    }

    /// Queues a state without waiting (`Block` returns [`SendError::Full`] when full).
    pub fn try_send(&self, state: FrameState) -> Result<(), SendError> {
        self.enqueue(state, false)
    }

    /// Queues every state of a source (iterator, channel receiver, ...) with [`ComposeHandle::send`].
    pub fn send_all(&self, states: impl IntoIterator<Item = FrameState>) -> Result<(), SendError> {
        states.into_iter().try_for_each(|s| self.send(s))
    }

    fn enqueue(&self, state: FrameState, wait: bool) -> Result<(), SendError> {
        let q = &self.queue;
        let mut st = lock(&q.st);
        if st.closed {
            return Err(SendError::Closed(state));
        }
        // End is never dropped or refused for room.
        let end = matches!(state, FrameState::End);
        match self.policy {
            Backpressure::Unbounded => {}
            Backpressure::Block { max_pending } => {
                while !end && st.items.len() >= max_pending && !st.closed {
                    if !wait {
                        return Err(SendError::Full(state));
                    }
                    st = q.cv.wait(st).unwrap_or_else(|p| p.into_inner());
                }
                if st.closed {
                    return Err(SendError::Closed(state));
                }
            }
            Backpressure::KeepLatest { max_pending } => {
                while !end && st.items.len() >= max_pending {
                    match st.items.iter().position(|s| matches!(s, FrameState::Ready(_))) {
                        Some(i) => {
                            st.items.remove(i);
                            st.dropped += 1;
                        }
                        None => {
                            st.items.pop_front();
                        }
                    }
                }
            }
        }
        st.items.push_back(state);
        st.submitted += 1;
        st.max_depth = st.max_depth.max(st.items.len());
        q.cv.notify_all();
        Ok(())
    }

    /// Queue counters.
    pub fn stats(&self) -> QueueStats {
        let st = lock(&self.queue.st);
        QueueStats { submitted: st.submitted, dropped: st.dropped, processed: st.processed, depth: st.items.len(), max_depth: st.max_depth }
    }

    /// Sends `End`, waits for the worker to compose everything queued, and closes the node.
    /// Returns the per-state outcomes (in queue order) and the close result.
    pub fn close(mut self) -> (Vec<Result<Outcome, ComposeError>>, Result<Closed, ComposeError>) {
        let _ = self.enqueue(FrameState::End, false);
        let outcomes = self.worker.take().map(|w| w.join().unwrap_or_default()).unwrap_or_default();
        (outcomes, self.node.close_inner())
    }
}

impl Drop for ComposeHandle {
    fn drop(&mut self) {
        if let Some(w) = self.worker.take() {
            {
                let mut st = lock(&self.queue.st);
                st.closed = true;
                self.queue.cv.notify_all();
            }
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spool_names_are_sanitized_and_stable() {
        let c = Core {
            cameras: vec!["camF".into()],
            image_root: PathBuf::from("/tmp/x"),
            spool: PathBuf::from(".live"),
            policy: ReconcilePolicy::AppendOnly,
            source: None,
            accepted: BTreeMap::new(),
            order: Vec::new(),
            recent: VecDeque::new(),
            waiting: BTreeMap::new(),
            stats: ComposeStats::default(),
            pipeline: None,
            ended: None,
        };
        let k = FrameKey::new("rtsp://drone 1", 42, 3);
        assert_eq!(c.spool_name(&k, "camF", "png"), ".live/camF/rtsp___drone_1_0000000042.png");
    }

    #[test]
    fn key_display() {
        assert_eq!(FrameKey::new("d", 7, 1).to_string(), "d#7r1");
    }
}
