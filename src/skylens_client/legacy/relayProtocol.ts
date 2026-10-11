// Board ↔ client-server relay framing.
//
// The client server (src/skylens_client/server) is a RELAY, not a producer: it
// holds one upstream WebSocket to the core (ws://localhost:8080/viewer) and
// pushes every `ViewerMessage` it receives, VERBATIM, down to every connected
// situation board. Boards therefore speak the same wire contract as the core
// (shared/protocol.ts) — nothing is translated on the way.
//
// Two extra frames exist that the core never sends, because they describe the
// RELAY itself and the board needs them to tell "core is quiet" apart from
// "core is unreachable":
//   relay-hello   — sent once on connect (board id, peer signalling coordinates)
//   relay-status  — upstream link state + counters, sent on every change
// Their `kind` values are disjoint from every ViewerMessage kind, so a receiver
// switches over one flat union.
//
// Pure data: no DOM, no Node built-ins — imported by both halves.

import type { ViewerMessage } from '../shared/protocol.ts';

/** State of the relay's UPSTREAM link (client server → core). */
export type UpstreamState = 'connecting' | 'online' | 'offline';

export interface RelayHello {
  kind: 'relay-hello';
  /** Id the relay assigned this board connection. */
  boardId: string;
  /** Unix ms on the relay, so the board can estimate skew. */
  serverTime: number;
  upstream: UpstreamState;
  /** Where PeerJS signalling is mounted, for the planned board↔board P2P. */
  peerPath: string;
  /** Room token this board was admitted to. */
  room: string;
  /**
   * PeerJS id this board should register under IF it ever joins the P2P mesh
   * (COMPONENTS.md §2.3). Nothing registers it today — the relay hands it out so
   * ids are allocated in one place when redistribution is switched on.
   */
  peerId: string;
  /** How many frames the relay replayed from its cache right after this hello. */
  replayed: number;
}

export interface RelayStatus {
  kind: 'relay-status';
  upstream: UpstreamState;
  /** Operator-facing Korean line for the waiting badge. */
  detail: string;
  /** Unix ms the current upstream state began. */
  since: number;
  /** Boards currently attached to this relay. */
  boards: number;
  /** ViewerMessages relayed since the relay started. */
  relayed: number;
  /** Failed upstream connect attempts since the last success. */
  retries: number;
}

/** Anything a board can receive on the relay socket. */
export type RelayFrame = ViewerMessage | RelayHello | RelayStatus;

export function isRelayFrame(v: unknown): v is RelayFrame {
  return typeof v === 'object' && v !== null && typeof (v as { kind?: unknown }).kind === 'string';
}

/** ViewerMessage kinds, as a runtime set — the relay and the board both need to
 *  tell a core frame apart from a relay-only frame over one flat union. */
export const VIEWER_KINDS: ReadonlySet<string> = new Set([
  'splat-chunk',
  'detection',
  'telemetry',
  'mission-status',
  'server-status',
  'link-status',
]);

export function isViewerMessage(v: unknown): v is ViewerMessage {
  return (
    typeof v === 'object' &&
    v !== null &&
    VIEWER_KINDS.has(String((v as { kind?: unknown }).kind))
  );
}

/** WebSocket path boards connect to on the client server. */
export const RELAY_STREAM_PATH = '/stream';

/** Express mount point of the PeerJS signalling server. */
export const RELAY_PEER_PATH = '/peerjs';

/** Default listen port of the client server (COMPONENTS.md §3.6). */
export const RELAY_DEFAULT_PORT = 8090;
