/// <reference types="node" />
// Browser-facing WebRTC CONNECTION RELAY — the signalling half.
//
// COMPONENTS.md §3.6 makes this component responsible for "코어와 현황판 사이의
// WebRTC 연결 중계(시그널링)". Two honest statements about what that means today:
//
//   LIVE NOW      The board's data comes over the plain WebSocket at
//                 RELAY_STREAM_PATH. Node has no built-in WebRTC, so the core
//                 pushes to this component over WebSocket and this component
//                 pushes to the browsers the same way (COMPONENTS.md §8).
//   PREPARED FOR  §2.3 — board↔board P2P redistribution of splat results. When a
//                 dozen boards are attached, having the core upload every chunk
//                 to every one of them is the uplink bottleneck; boards that
//                 already hold a segment can forward it to boards that don't.
//                 That handshake needs a broker, and this IS the broker: a real
//                 PeerJS signalling server, mounted, running, and counted in
//                 /health. Nothing registers against it yet.
//
// Room/peer bookkeeping lives here so ids are allocated in one place: the relay
// hands each board a `peerId` of the form `skylens-<room>-board-<n>` in its
// relay-hello, and this module derives the room back out of any id that
// registers, so /health can show the mesh per room the moment it is switched on.

import type { IncomingMessage, Server as HttpServer } from 'node:http';
import type { Duplex } from 'node:stream';
import { WebSocketServer } from 'ws';
import type { Express } from 'express';
import { ExpressPeerServer } from 'peer';

export interface PeerCounters {
  path: string;
  /** WebSocket path the PeerJS browser client must hit. */
  wsPath: string;
  /** Signalling is up and accepting registrations. */
  live: boolean;
  /** Peers currently registered on the broker. */
  peers: number;
  /** Peers registered since start. */
  peersSeen: number;
  /** Signalling messages brokered (OFFER/ANSWER/CANDIDATE/…). */
  messages: number;
  /** room → registered peer ids. */
  rooms: Record<string, string[]>;
  /** What the mesh is for, echoed into /health so an operator isn't guessing. */
  purpose: string;
}

export interface PeerRelay {
  counters(): PeerCounters;
  /** Route an upgrade whose path matched `wsPath`. Node hands the raw stream as
   *  a Duplex, which is exactly what ws' handleUpgrade wants. */
  handleUpgrade(req: IncomingMessage, socket: Duplex, head: Buffer): void;
  readonly wsPath: string;
}

const PURPOSE =
  'board↔board P2P splat redistribution (COMPONENTS.md §2.3) — signalling is live, no peers register yet';

/** `skylens-<room>-board-3` → `<room>`. Ids we did not mint land in `unknown`. */
function roomOf(peerId: string): string {
  const m = /^skylens-(.+)-board-\d+$/.exec(peerId);
  return m ? m[1] : 'unknown';
}

/**
 * Mount PeerJS signalling under `mountPath` on an existing express app + http
 * server. The internal WebSocketServer is created in `noServer` mode so it does
 * NOT install its own 'upgrade' listener — this process routes upgrades itself
 * (the board stream and the broker share one port).
 */
export function mountPeerRelay(app: Express, server: HttpServer, mountPath: string): PeerRelay {
  let wss: WebSocketServer | null = null;
  let wsPath = `${mountPath}/peerjs`;
  const peers = new Map<string, string>(); // peerId → room
  let peersSeen = 0;
  let messages = 0;

  const peerServer = ExpressPeerServer(server, {
    path: '/',
    // Keep dead registrations from lingering in the bookkeeping.
    alive_timeout: 30_000,
    expire_timeout: 10_000,
    allow_discovery: true,
    createWebSocketServer: (options) => {
      // `options.path` is the mount path joined with the peer path, i.e. the
      // full URL the browser client opens.
      if (typeof options.path === 'string') wsPath = options.path;
      wss = new WebSocketServer({ noServer: true });
      return wss;
    },
  });

  peerServer.on('connection', (client) => {
    const id = client.getId();
    peers.set(id, roomOf(id));
    peersSeen += 1;
  });
  peerServer.on('disconnect', (client) => {
    peers.delete(client.getId());
  });
  peerServer.on('message', () => {
    messages += 1;
  });
  peerServer.on('error', (err) => {
    console.warn('[client/peer]', err.message);
  });

  // Mounting is what triggers the peer server's own initialization (it listens
  // for express' 'mount' event), so this must happen before any upgrade lands.
  app.use(mountPath, peerServer);

  return {
    get wsPath() {
      return wsPath;
    },
    handleUpgrade(req, socket, head): void {
      if (!wss) {
        socket.destroy();
        return;
      }
      const server = wss;
      server.handleUpgrade(req, socket, head, (ws) => server.emit('connection', ws, req));
    },
    counters(): PeerCounters {
      const rooms: Record<string, string[]> = {};
      for (const [id, room] of peers) (rooms[room] ??= []).push(id);
      return {
        path: mountPath,
        wsPath,
        live: wss !== null,
        peers: peers.size,
        peersSeen,
        messages,
        rooms,
        purpose: PURPOSE,
      };
    },
  };
}
