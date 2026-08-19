// Gateway-local wire additions.
//
// These deliberately do NOT live in src/shared/protocol.ts. They describe
// (a) how a transport hop annotates traffic it carries and (b) how the gateway
// brokers hole punching. Neither is part of the cross-component message contract:
// a receiver that ignores both still sees a well-formed Envelope<UplinkMessage>.

import type { ComponentId, Envelope, LinkMode } from '../shared/protocol.ts';

/** One hop's timestamps, appended as the message passes through. */
export interface HopStamp {
  at: ComponentId;
  /** Unix ms when this hop received the message. */
  rx: number;
  /** Unix ms when this hop handed it on. */
  tx: number;
  /** Egress the hop chose, when it had a choice (proxy core path). */
  via?: string;
}

/**
 * An Envelope plus the hops it has crossed. Purely additive — `path` is optional
 * and `payload` / `originTs` / `seq` / `from` are never rewritten, so the core can
 * treat this as a plain Envelope and read `path` only if it wants per-hop latency.
 */
export type StampedEnvelope<T> = Envelope<T> & { path?: HopStamp[] };

/** Append a hop without mutating the caller's object. */
export function stamp<T>(env: StampedEnvelope<T>, hop: HopStamp): StampedEnvelope<T> {
  return { ...env, path: [...(env.path ?? []), hop] };
}

// ---------------------------------------------------------------------------
// Hole-punching signalling (webrtc mode only)
// ---------------------------------------------------------------------------
//
// The gateway never sees media in this mode. It pairs one drone with the proxy,
// pumps SDP/ICE between them, and steps out. Frames travel as Envelope<SignalFrame>
// so the transport framing stays identical to the relay path.

export type SignalKind =
  /** drone → gateway: I want a direct path. */
  | 'signal-hello'
  /** drone → proxy */
  | 'signal-offer'
  /** proxy → drone */
  | 'signal-answer'
  /** either direction */
  | 'signal-ice'
  /** proxy → drone: punch succeeded, here is where to send media. */
  | 'signal-ready'
  /** either direction */
  | 'signal-bye';

export interface SignalFrame {
  kind: SignalKind;
  /** Pairing key created by the gateway, echoed by both ends. */
  sessionId: string;
  droneId: number;
  mode: LinkMode;
  /** SDP blob for offer/answer. */
  sdp?: string;
  /** ICE candidate, opaque to the gateway. */
  candidate?: unknown;
  /** Direct proxy endpoint handed to the drone with signal-ready. */
  direct?: string;
  reason?: string;
}
