// The board's ONLY data source.
//
// Everything on the situation board — drone poses, geometry, markers, panels —
// arrives here, over one socket to this component's own server (§3.6), which is
// relaying what the core pushed (§3.4). The board is a subscriber: it never
// schedules the refinement ladder, never simulates a drone, and never talks to
// the control tower (COMPONENTS.md §8, "관제탑↔현황판 P2P는 폐기한다").
//
// Two link states matter and the operator must be able to tell them apart:
//   relay     browser → client server (this page's socket)
//   upstream  client server → core    (reported in the relay-status frame)
// "코어 대기 중" with the relay online is a very different situation from "중계
// 서버 연결 끊김", and a board that shows one frozen screen for both is lying.
//
// The segment ladder is NOT reconstructed here. `SegmentStatus[]` is rendered
// exactly as the core sends it, including `levels` — the ladder height is the
// core's (skylens_core/server/ladder.ts), and a client-side guess at it would be
// a second opinion about a thing that has exactly one owner.

import type {
  DetectionResult,
  DroneTelemetry,
  LinkStatus,
  MissionStatus,
  ServerStatus,
  SplatChunk,
  ViewerMessage,
} from '../../shared/protocol.ts';
import { createViewerStream } from '../../shared/viewer/serverSource.ts';
import type { StreamState } from '../../shared/viewer/serverSource.ts';
import type { RelayHello, RelayStatus, UpstreamState } from '../relayProtocol.ts';
import { RELAY_DEFAULT_PORT, RELAY_STREAM_PATH } from '../relayProtocol.ts';

export interface FeedStatus {
  /** Board → client server. */
  relay: StreamState;
  /** Client server → core, as the relay reports it. */
  upstream: UpstreamState;
  /** Korean one-liner for the badge — describes whichever link is the problem. */
  detail: string;
  /** True once geometry/markers are actually flowing. */
  receiving: boolean;
  /** Last ServerStatus the core sent, or a locally-counted stand-in. */
  server: ServerStatus;
  mission: MissionStatus | null;
  links: LinkStatus[];
  /** Boards attached to the same relay (from relay-status). */
  boards: number;
  /** Id the relay assigned this page. */
  boardId: string | null;
}

type Cb<T> = (v: T) => void;

export interface RelayClient {
  onSplatChunk(cb: Cb<SplatChunk>): void;
  onDetection(cb: Cb<DetectionResult>): void;
  onTelemetry(cb: Cb<DroneTelemetry>): void;
  onMission(cb: Cb<MissionStatus>): void;
  onStatus(cb: Cb<FeedStatus>): void;
  start(): void;
  stop(): void;
  readonly status: FeedStatus;
  readonly url: string;
}

/**
 * Where the relay lives. Served from :8090 the board is same-origin and needs no
 * configuration at all; opened straight off the Vite dev server (:5173) it falls
 * back to the relay's default port on the same host. `?relay=` overrides both
 * (a full ws:// URL, or just a port number).
 */
export function resolveRelayUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const q = new URLSearchParams(window.location.search).get('relay');
  if (q) {
    if (/^wss?:\/\//.test(q)) return q;
    if (/^\d+$/.test(q)) return `${proto}://${window.location.hostname}:${q}${RELAY_STREAM_PATH}`;
  }
  const host =
    window.location.port === '5173'
      ? `${window.location.hostname}:${RELAY_DEFAULT_PORT}`
      : window.location.host;
  return `${proto}://${host}${RELAY_STREAM_PATH}`;
}

function roomFromQuery(): string {
  const r = new URLSearchParams(window.location.search).get('room');
  return r && r.trim() ? r.trim() : 'default';
}

function emptyServerStatus(): ServerStatus {
  return {
    kind: 'server-status',
    connected: false,
    receiving: false,
    chunks: 0,
    detections: 0,
    lastSeq: 0,
    latencyMs: null,
    segments: [],
  };
}

export function createRelayClient(url = resolveRelayUrl()): RelayClient {
  const chunkCbs: Cb<SplatChunk>[] = [];
  const detCbs: Cb<DetectionResult>[] = [];
  const teleCbs: Cb<DroneTelemetry>[] = [];
  const missionCbs: Cb<MissionStatus>[] = [];
  const statusCbs: Cb<FeedStatus>[] = [];

  const full = `${url}?room=${encodeURIComponent(roomFromQuery())}`;
  const stream = createViewerStream({ url: full, label: 'relay' });

  let coreStatus: ServerStatus | null = null;
  let chunks = 0;
  let detections = 0;

  const status: FeedStatus = {
    relay: 'connecting',
    upstream: 'connecting',
    detail: '중계 서버 연결 시도 중',
    receiving: false,
    server: emptyServerStatus(),
    mission: null,
    links: [],
    boards: 0,
    boardId: null,
  };

  function describe(): string {
    if (status.relay !== 'online') {
      return status.relay === 'connecting'
        ? '중계 서버 연결 시도 중'
        : '중계 서버 연결 끊김 · 재연결 대기';
    }
    if (status.upstream === 'online') {
      return status.receiving ? '코어 스트림 수신 중' : '코어 연결됨 · 데이터 대기';
    }
    return status.upstream === 'connecting' ? '코어 연결 시도 중' : '코어 응답 없음 · 재연결 대기';
  }

  function publish(): void {
    status.detail = describe();
    const base = coreStatus ?? emptyServerStatus();
    status.server = {
      ...base,
      connected: status.relay === 'online' && status.upstream === 'online',
      receiving: status.receiving,
      // Counters fall back to what this page has actually seen, so the panel
      // still reads true before the core's first server-status lands.
      chunks: Math.max(base.chunks, chunks),
      detections: Math.max(base.detections, detections),
    };
    const snap: FeedStatus = {
      ...status,
      server: { ...status.server, segments: status.server.segments.map((s) => ({ ...s })) },
      links: [...status.links],
    };
    for (const cb of statusCbs) cb(snap);
  }

  function onMessage(msg: ViewerMessage): void {
    switch (msg.kind) {
      case 'splat-chunk':
        chunks += 1;
        status.receiving = true;
        for (const cb of chunkCbs) cb(msg);
        publish();
        break;
      case 'detection':
        detections += 1;
        status.receiving = true;
        for (const cb of detCbs) cb(msg);
        publish();
        break;
      case 'telemetry':
        status.receiving = true;
        for (const cb of teleCbs) cb(msg);
        break;
      case 'mission-status':
        status.mission = msg;
        for (const cb of missionCbs) cb(msg);
        publish();
        break;
      case 'server-status':
        coreStatus = msg;
        publish();
        break;
      case 'link-status': {
        const i = status.links.findIndex((l) => l.hop === msg.hop);
        if (i >= 0) status.links[i] = msg;
        else status.links.push(msg);
        publish();
        break;
      }
    }
  }

  function onOther(frame: Record<string, unknown>): void {
    if (frame.kind === 'relay-hello') {
      const hello = frame as unknown as RelayHello;
      status.boardId = hello.boardId;
      status.upstream = hello.upstream;
      publish();
      return;
    }
    if (frame.kind === 'relay-status') {
      const rs = frame as unknown as RelayStatus;
      status.upstream = rs.upstream;
      status.boards = rs.boards;
      // The core going away must stop claiming a live stream, but the geometry
      // already on the board stays — it was reconstructed, not simulated.
      if (rs.upstream !== 'online') status.receiving = false;
      publish();
    }
  }

  stream.onMessage(onMessage);
  stream.onOther(onOther);
  stream.onState((s: StreamState) => {
    status.relay = s;
    if (s !== 'online') status.receiving = false;
    publish();
  });

  return {
    onSplatChunk: (cb) => void chunkCbs.push(cb),
    onDetection: (cb) => void detCbs.push(cb),
    onTelemetry: (cb) => void teleCbs.push(cb),
    onMission: (cb) => void missionCbs.push(cb),
    onStatus: (cb) => {
      statusCbs.push(cb);
      publish();
    },
    start: () => stream.start(),
    stop: () => stream.stop(),
    get status() {
      return status;
    },
    url: full,
  };
}
