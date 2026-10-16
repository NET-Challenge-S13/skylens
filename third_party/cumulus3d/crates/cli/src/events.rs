/*
 * events.rs
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

//! Pipeline events. Each time a new frame bundle is processed, the result decomposed by stage is emitted.
//! This file is the shared contract of three modules (session, pipeline, sinks). Adding fields is allowed; changing or removing existing
//! fields is forbidden.

use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::{ImageId, Reconstruction, Rigid3, Sim3};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Header attached to every event.
#[derive(Clone, Debug)]
pub struct Meta {
    /// Session version (a monotonically increasing number of the state that emitted the event).
    pub version: u64,
    /// Time of occurrence.
    pub at: SystemTime,
}

/// Zone index and position range [lo, hi).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoneRange {
    /// Zone index.
    pub zone: usize,
    /// First position (inclusive).
    pub lo: usize,
    /// End position (exclusive).
    pub hi: usize,
}

/// A decoded input frame (one camera image). It is delivered first as [`Event::FrameDecoded`], before processing (features, matching,
/// registration).
#[derive(Clone, Debug)]
pub struct Frame {
    /// Position index.
    pub position: usize,
    /// Camera name (e.g. `camF`).
    pub camera: String,
    /// Image name (path relative to `image_root`).
    pub name: String,
    /// Image file path.
    pub path: std::path::PathBuf,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGB8 pixels (row-major, length `width * height * 3`). Shared between hooks without copying.
    pub rgb: Arc<[u8]>,
    /// GPS record of this image (if any).
    pub gps: Option<cumulus3d_core::io::GpsRecord>,
}

/// Pipeline events (per-stage results). Every variant carries a [`Meta`].
#[derive(Clone, Debug)]
pub enum Event {
    /// The frame bundle of one position has arrived.
    FrameIngested {
        /// Common header.
        meta: Meta,
        /// Position index.
        position: usize,
        /// Image names of this position.
        images: Vec<String>,
    },
    /// Decoding of one input frame is complete. Comes first, before the position is processed (only when `SessionConfig::decode_frames`
    /// is on).
    FrameDecoded {
        /// Common header.
        meta: Meta,
        /// Decoded frame.
        frame: Arc<Frame>,
    },
    /// Session start (once, at the first input). `positions` = total number of known positions. (added by session)
    Started {
        /// Common header.
        meta: Meta,
        /// Total number of known positions (None if unknown).
        positions: Option<usize>,
    },
    /// One run.log line. If `stamped`, it is written with a `[HH:MM:SS] ` prefix (meta.at local time). (added by session)
    Log {
        /// Common header.
        meta: Meta,
        /// Line to record.
        line: String,
        /// Whether to add the `[HH:MM:SS] ` prefix.
        stamped: bool,
    },
    /// Feature extraction of one image is complete.
    FeaturesExtracted {
        /// Common header.
        meta: Meta,
        /// Image name.
        image: String,
        /// Number of extracted features.
        count: usize,
        /// Position index.
        position: usize,
        /// Image id.
        image_id: ImageId,
    },
    /// Result of matching new pairs.
    /// `pairs` = pairs matched at this position, `verified` = pairs with valid geometry, `matched` = pairs for which descriptor matching
    /// was performed.
    PairsMatched {
        /// Common header.
        meta: Meta,
        /// Pairs matched at this position.
        pairs: usize,
        /// Pairs with valid geometry.
        verified: usize,
        /// Position index.
        position: usize,
        /// Pairs for which descriptor matching was performed.
        matched: usize,
    },
    /// The first model has been built.
    ModelInitialized {
        /// Common header.
        meta: Meta,
        /// Number of registered images.
        registered: usize,
        /// Position index.
        position: usize,
        /// Copy of the relevant model.
        model: Arc<Reconstruction>,
    },
    /// An image has been registered to the model.
    FrameRegistered {
        /// Common header.
        meta: Meta,
        /// Position index.
        position: usize,
        /// Image name.
        image: String,
        /// Image id.
        image_id: ImageId,
        /// Registered pose (world -> camera).
        cam_from_world: Rigid3,
    },
    /// Registration and triangulation of one position are finished.
    PositionDone {
        /// Common header.
        meta: Meta,
        /// Position index.
        position: usize,
        /// Number of registered images.
        registered: usize,
        /// Number of images expected up to this position.
        expected: usize,
    },
    /// The last position of a zone has arrived.
    /// `model` = copy of the chain model at that point (before alignment).
    ZoneArrived {
        /// Common header.
        meta: Meta,
        /// Zone range.
        range: ZoneRange,
        /// Copy of the relevant model.
        model: Arc<Reconstruction>,
    },
    /// Zone preview point cloud (no BA).
    /// `model` = preview model (zone 0 is the GPS-aligned one), `dense` = false means densification was not run or failed and `cloud` is
    /// an empty placeholder.
    /// `frame` = point-cloud coordinate frame ("gps" | "refined_K" | "unaligned").
    ZonePreview {
        /// Common header.
        meta: Meta,
        /// Zone range.
        range: ZoneRange,
        /// Point cloud (an empty placeholder if there is no densification).
        cloud: Arc<PointCloud>,
        /// Copy of the relevant model.
        model: Arc<Reconstruction>,
        /// Whether densification was actually performed.
        dense: bool,
        /// Point-cloud coordinate frame ("gps" | "refined_K" | "unaligned").
        frame: String,
    },
    /// Refinement job start (background). (added by session)
    RefineStarted {
        /// Common header.
        meta: Meta,
        /// Zone index.
        zone: usize,
    },
    /// Model right after BA of the refinement job (before GPS alignment). (added by session)
    ZoneAdjusted {
        /// Common header.
        meta: Meta,
        /// Zone index.
        zone: usize,
        /// Copy of the relevant model.
        model: Arc<Reconstruction>,
    },
    /// Zone refined pose finalized (BA + GPS alignment).
    /// `model` = refined model after BA + GPS alignment.
    ZoneRefinedPose {
        /// Common header.
        meta: Meta,
        /// Zone range.
        range: ZoneRange,
        /// Number of registered images.
        registered: usize,
        /// Mean reprojection error (pixels).
        mean_reproj_px: f64,
        /// Copy of the relevant model.
        model: Arc<Reconstruction>,
    },
    /// Zone refined point cloud. It replaces the preview of the same zone.
    /// `dense` = false means `cloud` is an empty placeholder.
    ZoneRefined {
        /// Common header.
        meta: Meta,
        /// Zone range.
        range: ZoneRange,
        /// Point cloud (an empty placeholder if there is no densification).
        cloud: Arc<PointCloud>,
        /// Copy of the relevant model.
        model: Arc<Reconstruction>,
        /// Whether densification was actually performed.
        dense: bool,
    },
    /// The refined model was adopted as the reference for subsequent registration.
    BaseAdopted {
        /// Common header.
        meta: Meta,
        /// Zone index.
        zone: usize,
    },
    /// Coordinate-frame transform of an already sent zone (the point cloud is not sent again): new_from_old.
    Reanchored {
        /// Common header.
        meta: Meta,
        /// Zone index.
        zone: usize,
        /// new_from_old transform.
        transform: Sim3,
    },
    /// Snapshot for display (event sequence number and composite point cloud).
    Snapshot {
        /// Common header.
        meta: Meta,
        /// Event sequence number.
        index: usize,
        /// Point cloud (an empty placeholder if there is no densification).
        cloud: Arc<PointCloud>,
    },
    /// All positions processed.
    AllPositionsDone {
        /// Common header.
        meta: Meta,
    },
    /// All refinement jobs complete.
    /// `chain` = final chain model.
    AllRefinedDone {
        /// Common header.
        meta: Meta,
        /// Final chain model.
        chain: Option<Arc<Reconstruction>>,
    },
    /// Session end summary (the last of all events). `stage_times` = (stage, accumulated time, count) in stage-name order,
    /// `pipeline_secs` = start to all refinements complete, `chain_stats` = statistics line of the final chain model. (added by session)
    Finished {
        /// Common header.
        meta: Meta,
        /// Start to all refinements complete (seconds).
        pipeline_secs: f64,
        /// (stage, accumulated time, count) in stage-name order.
        stage_times: Vec<(String, Duration, usize)>,
        /// Statistics line of the final chain model.
        chain_stats: Vec<String>,
    },
    /// `Command::ResetFrom` was handled. (added by session)
    Reset {
        /// Common header.
        meta: Meta,
        /// Position index.
        position: usize,
    },
    /// `Command::InvalidateZone` was handled. (added by session)
    ZoneInvalidated {
        /// Common header.
        meta: Meta,
        /// Zone index.
        zone: usize,
    },
    /// Warning.
    Warning {
        /// Common header.
        meta: Meta,
        /// Message.
        message: String,
    },
    /// Error (including hook panics).
    Error {
        /// Common header.
        meta: Meta,
        /// Message.
        message: String,
    },
}

/// Event kind (for hook registration and filtering).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// [`Event::FrameIngested`].
    FrameIngested,
    /// [`Event::FrameDecoded`].
    FrameDecoded,
    /// [`Event::Started`].
    Started,
    /// [`Event::Log`].
    Log,
    /// [`Event::RefineStarted`].
    RefineStarted,
    /// [`Event::ZoneAdjusted`].
    ZoneAdjusted,
    /// [`Event::Finished`].
    Finished,
    /// [`Event::Reset`].
    Reset,
    /// [`Event::ZoneInvalidated`].
    ZoneInvalidated,
    /// [`Event::FeaturesExtracted`].
    FeaturesExtracted,
    /// [`Event::PairsMatched`].
    PairsMatched,
    /// [`Event::ModelInitialized`].
    ModelInitialized,
    /// [`Event::FrameRegistered`].
    FrameRegistered,
    /// [`Event::PositionDone`].
    PositionDone,
    /// [`Event::ZoneArrived`].
    ZoneArrived,
    /// [`Event::ZonePreview`].
    ZonePreview,
    /// [`Event::ZoneRefinedPose`].
    ZoneRefinedPose,
    /// [`Event::ZoneRefined`].
    ZoneRefined,
    /// [`Event::BaseAdopted`].
    BaseAdopted,
    /// [`Event::Reanchored`].
    Reanchored,
    /// [`Event::Snapshot`].
    Snapshot,
    /// [`Event::AllPositionsDone`].
    AllPositionsDone,
    /// [`Event::AllRefinedDone`].
    AllRefinedDone,
    /// [`Event::Warning`].
    Warning,
    /// [`Event::Error`].
    Error,
}

impl Event {
    /// Event kind.
    pub fn kind(&self) -> EventKind {
        use Event::*;
        match self {
            FrameIngested { .. } => EventKind::FrameIngested,
            FrameDecoded { .. } => EventKind::FrameDecoded,
            Started { .. } => EventKind::Started,
            Log { .. } => EventKind::Log,
            RefineStarted { .. } => EventKind::RefineStarted,
            ZoneAdjusted { .. } => EventKind::ZoneAdjusted,
            Finished { .. } => EventKind::Finished,
            Reset { .. } => EventKind::Reset,
            ZoneInvalidated { .. } => EventKind::ZoneInvalidated,
            FeaturesExtracted { .. } => EventKind::FeaturesExtracted,
            PairsMatched { .. } => EventKind::PairsMatched,
            ModelInitialized { .. } => EventKind::ModelInitialized,
            FrameRegistered { .. } => EventKind::FrameRegistered,
            PositionDone { .. } => EventKind::PositionDone,
            ZoneArrived { .. } => EventKind::ZoneArrived,
            ZonePreview { .. } => EventKind::ZonePreview,
            ZoneRefinedPose { .. } => EventKind::ZoneRefinedPose,
            ZoneRefined { .. } => EventKind::ZoneRefined,
            BaseAdopted { .. } => EventKind::BaseAdopted,
            Reanchored { .. } => EventKind::Reanchored,
            Snapshot { .. } => EventKind::Snapshot,
            AllPositionsDone { .. } => EventKind::AllPositionsDone,
            AllRefinedDone { .. } => EventKind::AllRefinedDone,
            Warning { .. } => EventKind::Warning,
            Error { .. } => EventKind::Error,
        }
    }
    /// Common header.
    pub fn meta(&self) -> &Meta {
        use Event::*;
        match self {
            FrameIngested { meta, .. }
            | FeaturesExtracted { meta, .. }
            | PairsMatched { meta, .. }
            | ModelInitialized { meta, .. }
            | FrameRegistered { meta, .. }
            | PositionDone { meta, .. }
            | ZoneArrived { meta, .. }
            | ZonePreview { meta, .. }
            | ZoneRefinedPose { meta, .. }
            | ZoneRefined { meta, .. }
            | BaseAdopted { meta, .. }
            | Reanchored { meta, .. }
            | Snapshot { meta, .. }
            | AllPositionsDone { meta }
            | AllRefinedDone { meta, .. }
            | FrameDecoded { meta, .. }
            | Started { meta, .. }
            | Log { meta, .. }
            | RefineStarted { meta, .. }
            | ZoneAdjusted { meta, .. }
            | Finished { meta, .. }
            | Reset { meta, .. }
            | ZoneInvalidated { meta, .. }
            | Warning { meta, .. }
            | Error { meta, .. } => meta,
        }
    }
}

impl Event {
    /// Event text for timeline.txt (None if there is none). The wording and format are the same as the existing record.
    pub fn timeline_text(&self) -> Option<String> {
        use Event::*;
        Some(match self {
            Started { positions, .. } => match positions {
                Some(n) => format!("start NPOS={n}"),
                None => "start".to_string(),
            },
            ModelInitialized { registered, .. } => format!("init_model Registered images: {registered}"),
            BaseAdopted { zone, .. } => format!("adopt refined {zone} as base"),
            PositionDone { position, registered, expected, .. } => format!("pos {position} registered {registered}/{expected}"),
            ZoneArrived { range, .. } => format!("region {} arrived pos[{},{})", range.zone, range.lo, range.hi),
            ZonePreview { range, .. } => format!("preview {} ready", range.zone),
            RefineStarted { zone, .. } => format!("refined {zone} ba_start"),
            ZoneRefinedPose { range, registered, mean_reproj_px, .. } => {
                format!("refined {} pose_ready Registered images: {registered} Mean reprojection error: {mean_reproj_px:.6}px ", range.zone)
            }
            ZoneRefined { range, .. } => format!("refined {} ready", range.zone),
            AllPositionsDone { .. } => "all positions done".to_string(),
            AllRefinedDone { .. } => "all refined done".to_string(),
            _ => return None,
        })
    }
}

/// Commands fed into the reducer.
#[derive(Clone, Debug)]
pub enum Command {
    /// Reprocess from this position (discard later state).
    ResetFrom(usize),
    /// Invalidate the zone result and have it rebuilt.
    InvalidateZone(usize),
}
