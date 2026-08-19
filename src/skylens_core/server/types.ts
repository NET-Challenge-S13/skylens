// Core-local types. Deliberately NOT in shared/protocol.ts: none of this
// crosses a component boundary — it is how the core remembers what it has seen
// and what it still owes the model API.

import type {
  DroneStation,
  DroneTelemetry,
  Envelope,
  LinkMode,
  SplatChunk,
} from '../../shared/protocol.ts';

/** One hop's stamps, appended by the gateway/proxy. The core reads them for
 *  per-hop latency and never rewrites them. Mirrors skylens_gateway/types.ts —
 *  duplicated on purpose so neither transport becomes a core dependency. */
export interface HopStamp {
  at: string;
  rx: number;
  tx: number;
  via?: string;
}

export type StampedEnvelope<T> = Envelope<T> & { path?: HopStamp[] };

/** A video slice as the core stores it: exactly what a recon job needs. */
export interface SourceSlice {
  uri: string;
  poses: DroneTelemetry[];
  droneId: number;
  seq: number;
  bytes: number;
  receivedAt: number;
}

export type SegmentState = 'open' | 'queued' | 'processing' | 'settled' | 'final';

/**
 * A span of the assigned route (or, without a route, of the drone's odometer).
 * A segment is a PLACE, not a period: flying the route back and forth revisits
 * the same segment indices, which is why `passes` exists.
 */
export interface SegmentRecord {
  index: number;
  /** Arc-length window along the route, meters. */
  spanStartM: number;
  spanEndM: number;
  sources: SourceSlice[];
  /** Times the drone has traversed this span. */
  passes: number;
  /** Bumped on every close; a job from an older generation is stale input. */
  generation: number;
  firstSeenAt: number;
  lastClosedAt: number | null;
  /** Highest level actually delivered to viewers. 0 = nothing yet. */
  deliveredLevel: number;
  deliveredSteps: number;
  label: string;
  state: SegmentState;
  chunk: SplatChunk | null;
  /** Set once a detect job has been dispatched for this generation. */
  detectGeneration: number;
}

export type JobKind = 'recon' | 'detect';

export interface QueuedJob {
  id: string;
  kind: JobKind;
  segment: number;
  /** Recon only; detect jobs carry 0. */
  level: number;
  steps: number;
  label: string;
  /** Segment generation this job was cut from. */
  generation: number;
  enqueuedAt: number;
  attempts: number;
  /** Set while the job is backing off after a transport failure. */
  notBefore: number;
}

export interface DroneRecord {
  droneId: number;
  /** Formation station — the aircraft's name on the operator's screens. */
  station: DroneStation;
  model: string;
  mode: LinkMode;
  connectedAt: number;
  lastSeenAt: number;
  last: DroneTelemetry | null;
  history: DroneTelemetry[];
  /** Meters travelled, used when no route is assigned. */
  odometer: number;
  /** Segment index the drone is currently inside, null before first fix. */
  currentSegment: number | null;
  /** Video slices received from this drone. */
  slices: number;
  bytes: number;
  /**
   * True once the drone sent `drone-hello`. Telemetry alone does NOT set it:
   * a drone in transit streams its position while it flies to the site, and
   * counting that as "arrived" would collapse the drone-connection wait the
   * mission is built around (COMPONENTS.md §5.2 steps 3-4).
   */
  announced: boolean;
}
