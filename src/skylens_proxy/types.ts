// Proxy-local wire additions — the mirror image of src/skylens_gateway/types.ts.
//
// Deliberately duplicated rather than pushed into src/shared/protocol.ts:
// hop stamping and hole-punch signalling are transport concerns between two
// components, not part of the cross-component message contract. A receiver that
// ignores both still sees a well-formed Envelope<UplinkMessage>.

import type { ComponentId, Envelope, LinkMode } from '../shared/protocol.ts';

export interface HopStamp {
  at: ComponentId;
  /** Unix ms when this hop received the message. */
  rx: number;
  /** Unix ms when this hop handed it on. */
  tx: number;
  /** Egress the hop chose — for the proxy, the core path that was active. */
  via?: string;
}

export type StampedEnvelope<T> = Envelope<T> & { path?: HopStamp[] };

export function stamp<T>(env: StampedEnvelope<T>, hop: HopStamp): StampedEnvelope<T> {
  return { ...env, path: [...(env.path ?? []), hop] };
}

export type SignalKind =
  | 'signal-hello'
  | 'signal-offer'
  | 'signal-answer'
  | 'signal-ice'
  | 'signal-ready'
  | 'signal-bye';

export interface SignalFrame {
  kind: SignalKind;
  sessionId: string;
  droneId: number;
  mode: LinkMode;
  sdp?: string;
  candidate?: unknown;
  /** Direct proxy endpoint handed to the drone with signal-ready. */
  direct?: string;
  reason?: string;
}
