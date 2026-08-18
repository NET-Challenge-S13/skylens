// WebRTC transport via PeerJS.
//
// Signaling uses the PUBLIC PeerJS broker cloud (0.peerjs.com) — no self-hosted
// server. NAT traversal uses Google's public STUN server. Once the peers hand
// shake through the broker, live state flows over the P2P DataChannel directly.
//
// Only the LISTENER needs a stable id so the other side can find it: CONTROL
// registers `skylens-<room>-control`; STATUS uses a RANDOM id and connects out to
// CONTROL (so STATUS can never collide). If CONTROL's id is already taken on the shared
// public broker (usually a stale registration), CONTROL recreates its peer after a
// backoff to self-heal. Use a distinct `?room=` to avoid clashing with others.

import { Peer } from 'peerjs';
import type { DataConnection } from 'peerjs';

export type PeerRole = 'control' | 'status';

export type PeerStatus =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'error';

export interface Transport {
  /** Send a payload to the other peer (no-op until connected). */
  send(data: unknown): void;
  /** Register a handler for inbound payloads. */
  onData(cb: (data: unknown) => void): void;
  /** Register a handler for connection status changes. */
  onStatus(cb: (s: PeerStatus, detail?: string) => void): void;
  readonly status: PeerStatus;
  dispose(): void;
}

const STUN = { urls: 'stun:stun.l.google.com:19302' };
const RETRY_MS = 2000;

function peerId(room: string, role: PeerRole): string {
  return `skylens-${room}-${role}`;
}

export function createTransport(role: PeerRole, room = 'default'): Transport {
  const dataCbs: Array<(d: unknown) => void> = [];
  const statusCbs: Array<(s: PeerStatus, detail?: string) => void> = [];
  let status: PeerStatus = 'idle';
  let conn: DataConnection | null = null;
  let disposed = false;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let idRetryTimer: ReturnType<typeof setTimeout> | null = null;
  let peer: Peer;

  function setStatus(s: PeerStatus, detail?: string): void {
    status = s;
    for (const cb of statusCbs) cb(s, detail);
  }

  function wireConnection(c: DataConnection): void {
    conn = c;
    c.on('open', () => setStatus('connected'));
    c.on('data', (d) => {
      for (const cb of dataCbs) cb(d);
    });
    c.on('close', () => {
      if (conn === c) conn = null;
      if (!disposed) {
        setStatus('disconnected');
        scheduleConnect();
      }
    });
    c.on('error', (e) => setStatus('error', String(e)));
  }

  function connectToControl(): void {
    if (disposed || conn) return;
    setStatus('connecting');
    // STATUS initiates; reliable+ordered so visited deltas can't be lost.
    const c = peer.connect(peerId(room, 'control'), {
      reliable: true,
      serialization: 'json',
      metadata: { from: 'status' },
    });
    wireConnection(c);
  }

  function scheduleConnect(): void {
    if (disposed || role !== 'status') return;
    if (retryTimer) clearTimeout(retryTimer);
    retryTimer = setTimeout(connectToControl, RETRY_MS);
  }

  function makePeer(): void {
    if (disposed) return;
    // CONTROL = deterministic listener id; STATUS = random id (it only connects out).
    peer =
      role === 'control'
        ? new Peer(peerId(room, 'control'), { config: { iceServers: [STUN] }, debug: 1 })
        : new Peer({ config: { iceServers: [STUN] }, debug: 1 });

    peer.on('open', () => {
      if (role === 'status') connectToControl();
      else setStatus('connecting'); // CONTROL waits for STATUS to connect
    });

    // CONTROL side: accept the inbound connection from STATUS.
    peer.on('connection', (c) => {
      wireConnection(c);
    });

    peer.on('disconnected', () => {
      if (disposed) return;
      setStatus('disconnected');
      // PeerJS lost its broker socket; try to get it back so re-connect can happen.
      try {
        peer.reconnect();
      } catch {
        /* ignore */
      }
    });

    peer.on('error', (e: unknown) => {
      const msg = (e as { type?: string; message?: string }) ?? {};
      // Target (CONTROL) not online yet — retry quietly.
      if (msg.type === 'peer-unavailable') {
        scheduleConnect();
        return;
      }
      // CONTROL's fixed id is taken (usually a stale registration) — recreate to heal.
      if (msg.type === 'unavailable-id') {
        setStatus('connecting', 'id taken — retrying');
        try {
          peer.destroy();
        } catch {
          /* ignore */
        }
        if (idRetryTimer) clearTimeout(idRetryTimer);
        idRetryTimer = setTimeout(makePeer, 3000);
        return;
      }
      setStatus('error', msg.message ?? String(e));
    });
  }
  makePeer();

  return {
    send(data: unknown): void {
      if (conn && status === 'connected') conn.send(data);
    },
    onData(cb): void {
      dataCbs.push(cb);
    },
    onStatus(cb): void {
      statusCbs.push(cb);
      cb(status);
    },
    get status(): PeerStatus {
      return status;
    },
    dispose(): void {
      disposed = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (idRetryTimer) clearTimeout(idRetryTimer);
      try {
        conn?.close();
      } catch {
        /* ignore */
      }
      try {
        peer.destroy();
      } catch {
        /* ignore */
      }
    },
  };
}
