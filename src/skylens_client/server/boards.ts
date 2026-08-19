/// <reference types="node" />
// DOWNSTREAM fan-out: client server → situation boards (browsers).
//
// One upstream socket to the core feeds N boards. This module owns the board
// registry and the REPLAY CACHE, which is the piece that makes a board reload
// (or a late-joining second board) land on a full picture instead of an empty
// scene: the core streams state once, so a relay that only forwarded live
// frames would leave every reconnecting board blank until the next chunk.
//
// The cache is deliberately tiny and is the SAME shape as the wire — the newest
// frame per key, never a derived model:
//   splat-chunk    → highest level seen per segment (the delay pattern's own rule)
//   detection      → by id
//   telemetry      → by droneId
//   link-status    → by hop
//   mission-status → the last one
//   server-status  → the last one
// Nothing here schedules or synthesizes anything; scheduling lives in the core
// (COMPONENTS.md §3.4).

import type { WebSocket } from 'ws';
import type {
  DetectionResult,
  DroneTelemetry,
  LinkStatus,
  MissionStatus,
  ServerStatus,
  SplatChunk,
  ViewerMessage,
} from '../../shared/protocol.ts';
import type { RelayHello, RelayStatus, UpstreamState } from '../relayProtocol.ts';

export interface BoardCounters {
  connected: number;
  seen: number;
  /** ViewerMessages pushed to at least one board. */
  relayed: number;
  /** Individual frames written across all board sockets (relayed × boards). */
  writes: number;
  /** Frames replayed from the cache to joining boards. */
  replayed: number;
  dropped: number;
  byKind: Record<string, number>;
  cached: {
    segments: number;
    detections: number;
    drones: number;
    links: number;
    mission: boolean;
    serverStatus: boolean;
  };
  rooms: Record<string, number>;
}

export interface BoardHub {
  /** Adopt a freshly upgraded board socket. */
  attach(ws: WebSocket, room: string): void;
  /** Push one core frame to every board (and remember it for replay). */
  broadcast(msg: ViewerMessage): void;
  /** Upstream link changed — every board gets a relay-status immediately. */
  setUpstream(state: UpstreamState, detail: string, retries: number): void;
  counters(): BoardCounters;
  close(): void;
}

interface Board {
  id: string;
  room: string;
  ws: WebSocket;
}

const OPEN = 1; // ws.OPEN, without importing the value form.

export function createBoardHub(peerPath: string): BoardHub {
  const boards = new Set<Board>();
  let nextId = 1;
  let seen = 0;
  let relayed = 0;
  let writes = 0;
  let replayed = 0;
  let dropped = 0;
  const byKind: Record<string, number> = {};

  // Replay cache.
  const chunks = new Map<number, SplatChunk>();
  const detections = new Map<string, DetectionResult>();
  const telemetry = new Map<number, DroneTelemetry>();
  const links = new Map<string, LinkStatus>();
  let mission: MissionStatus | null = null;
  let serverStatus: ServerStatus | null = null;

  let upstream: UpstreamState = 'connecting';
  let upstreamDetail = '코어 연결 시도 중';
  let upstreamSince = Date.now();
  let upstreamRetries = 0;

  function send(board: Board, frame: unknown): boolean {
    if (board.ws.readyState !== OPEN) return false;
    try {
      board.ws.send(JSON.stringify(frame));
      writes += 1;
      return true;
    } catch {
      dropped += 1;
      return false;
    }
  }

  function relayStatusFrame(): RelayStatus {
    return {
      kind: 'relay-status',
      upstream,
      detail: upstreamDetail,
      since: upstreamSince,
      boards: boards.size,
      relayed,
      retries: upstreamRetries,
    };
  }

  /** Everything the cache holds, in the order a board wants to apply it. */
  function replayFrames(): ViewerMessage[] {
    const out: ViewerMessage[] = [];
    if (mission) out.push(mission);
    for (const l of links.values()) out.push(l);
    for (const t of telemetry.values()) out.push(t);
    // Segments ascending, so the board's ingest queue sees the same order the
    // core produced them in.
    for (const seg of [...chunks.keys()].sort((a, b) => a - b)) {
      out.push(chunks.get(seg)!);
    }
    for (const d of detections.values()) out.push(d);
    if (serverStatus) out.push(serverStatus);
    return out;
  }

  function remember(msg: ViewerMessage): void {
    switch (msg.kind) {
      case 'splat-chunk': {
        const prev = chunks.get(msg.segment);
        // A superseded level must never survive in the cache, or a reloading
        // board would be handed the coarse version after the fine one.
        if (!prev || msg.level >= prev.level) chunks.set(msg.segment, msg);
        break;
      }
      case 'detection':
        detections.set(msg.id, msg);
        break;
      case 'telemetry':
        telemetry.set(msg.droneId, msg);
        break;
      case 'link-status':
        links.set(msg.hop, msg);
        break;
      case 'mission-status':
        mission = msg;
        break;
      case 'server-status':
        serverStatus = msg;
        break;
    }
  }

  return {
    attach(ws: WebSocket, room: string): void {
      const board: Board = { id: `board-${nextId++}`, room, ws };
      boards.add(board);
      seen += 1;

      const frames = replayFrames();
      const hello: RelayHello = {
        kind: 'relay-hello',
        boardId: board.id,
        serverTime: Date.now(),
        upstream,
        peerPath,
        room,
        peerId: `skylens-${room}-${board.id}`,
        replayed: frames.length,
      };
      send(board, hello);
      send(board, relayStatusFrame());
      for (const f of frames) {
        if (send(board, f)) replayed += 1;
      }

      const drop = (): void => {
        boards.delete(board);
      };
      ws.on('close', drop);
      ws.on('error', drop);
      // Boards are receive-only today. Anything they send is ignored on purpose:
      // control goes control-tower → core, never board → core.
      ws.on('message', () => {});
    },

    broadcast(msg: ViewerMessage): void {
      remember(msg);
      relayed += 1;
      byKind[msg.kind] = (byKind[msg.kind] ?? 0) + 1;
      for (const b of boards) send(b, msg);
    },

    setUpstream(state: UpstreamState, detail: string, retries: number): void {
      upstream = state;
      upstreamDetail = detail;
      upstreamSince = Date.now();
      upstreamRetries = retries;
      const frame = relayStatusFrame();
      for (const b of boards) send(b, frame);
    },

    counters(): BoardCounters {
      const rooms: Record<string, number> = {};
      for (const b of boards) rooms[b.room] = (rooms[b.room] ?? 0) + 1;
      return {
        connected: boards.size,
        seen,
        relayed,
        writes,
        replayed,
        dropped,
        byKind: { ...byKind },
        cached: {
          segments: chunks.size,
          detections: detections.size,
          drones: telemetry.size,
          links: links.size,
          mission: mission !== null,
          serverStatus: serverStatus !== null,
        },
        rooms,
      };
    },

    close(): void {
      for (const b of boards) {
        try {
          b.ws.close(1001, 'relay shutting down');
        } catch {
          /* ignore */
        }
      }
      boards.clear();
    },
  };
}
