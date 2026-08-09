// WebRTC transport via PeerJS.
//
// Signaling uses the PUBLIC PeerJS broker cloud (0.peerjs.com) — no self-hosted
// server. NAT traversal uses Google's public STUN server. Once the peers hand
// shake through the broker, live state flows over the P2P DataChannel directly.
//
// Deterministic ids let the two sides find each other without exchanging ids
// out of band: given a room token, SIM registers `skylens-<room>-sim` and RECON
// connects to it. Change the room via `?room=` to avoid collisions on the shared
// public broker.

import { Peer } from 'peerjs';
import type { DataConnection } from 'peerjs';

export type PeerRole = 'sim' | 'recon';

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

  const peer = new Peer(peerId(room, role), {
    config: { iceServers: [STUN] },
    debug: 1,
  });

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

  function connectToSim(): void {
    if (disposed || conn) return;
    setStatus('connecting');
    // RECON initiates; reliable+ordered so visited deltas can't be lost.
    const c = peer.connect(peerId(room, 'sim'), {
      reliable: true,
      serialization: 'json',
      metadata: { from: 'recon' },
    });
    wireConnection(c);
  }

  function scheduleConnect(): void {
    if (disposed || role !== 'recon') return;
    if (retryTimer) clearTimeout(retryTimer);
    retryTimer = setTimeout(connectToSim, RETRY_MS);
  }

  peer.on('open', () => {
    if (role === 'recon') connectToSim();
    else setStatus('connecting'); // SIM waits for RECON to connect
  });

  // SIM side: accept the inbound connection from RECON.
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
    // 'peer-unavailable' just means SIM isn't online yet — retry quietly.
    if (msg.type === 'peer-unavailable') {
      scheduleConnect();
      return;
    }
    setStatus('error', msg.message ?? String(e));
  });

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
