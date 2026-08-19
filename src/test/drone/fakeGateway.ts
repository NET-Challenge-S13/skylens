// A scripted stand-in for skylens_gateway (+ the proxy behind it), used to drive
// the drone end to end without bringing the real KOREN chain up.
//
// It mirrors the REAL endpoints so the drone needs no special casing:
//
//   relay   ws /uplink   — receives Envelope<UplinkMessage>, pushes control down
//   webrtc  ws /signal   — greets with signal-hello, answers signal-offer with
//                          signal-answer + signal-ice + signal-ready{direct}
//           ws /direct   — the endpoint signal-ready hands over; the uplink
//                          arrives here and NOTHING arrives on /signal after
//
// It also plays the core's part: once the drone says hello it assigns a route
// (loop: true), and it can push ManualControl to test the takeover path.
//
//   npx tsx src/test/drone/fakeGateway.ts --port=8081
//   npx tsx src/test/drone/fakeGateway.ts --port=8081 --mode=webrtc

import http from 'node:http';
import process from 'node:process';
import { WebSocketServer, type WebSocket } from 'ws';
import type {
  AssignRoute,
  Envelope,
  LinkMode,
  ManualControl,
  UplinkMessage,
} from '../../shared/protocol.ts';
import { DEMO_ROUTE } from '../../skylens_drone/core/config.ts';

/** Mirrors the gateway's own signalling dialect (skylens_gateway/types.ts). */
interface SignalFrame {
  kind: 'signal-hello' | 'signal-offer' | 'signal-answer' | 'signal-ice' | 'signal-ready' | 'signal-bye';
  sessionId: string;
  droneId: number;
  mode: LinkMode;
  sdp?: string;
  candidate?: unknown;
  direct?: string;
  reason?: string;
}

export interface FakeGatewayOptions {
  port: number;
  /** Which endpoint layout to serve. Default relay. */
  mode?: LinkMode;
  /** Delay from the drone's hello to the "core" assigning a route. */
  routeDelayMs?: number;
  route?: AssignRoute;
  onFrame?: (env: Envelope<UplinkMessage>, via: 'uplink' | 'direct') => void;
  onLog?: (line: string) => void;
}

export interface FakeGateway {
  close(): Promise<void>;
  /** Push a downlink message to every uplink-carrying socket. */
  send(msg: unknown): void;
  readonly frames: Envelope<UplinkMessage>[];
  /** Frames that arrived on the signalling socket — must stay 0 in webrtc mode. */
  readonly signalMedia: number;
}

let sessionCounter = 0;

export function startFakeGateway(opts: FakeGatewayOptions): Promise<FakeGateway> {
  const log = opts.onLog ?? ((line: string) => console.log(line));
  const mode: LinkMode = opts.mode ?? 'relay';
  const frames: Envelope<UplinkMessage>[] = [];
  /** Sockets the uplink actually rides — the only ones we push control down. */
  const uplinks = new Set<WebSocket>();
  let signalMedia = 0;
  let seq = 0;
  let routeTimer: ReturnType<typeof setTimeout> | null = null;

  const server = http.createServer((req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ component: 'fake-gateway', mode, uplinks: uplinks.size, path: req.url }));
  });
  const wssUplink = new WebSocketServer({ noServer: true });
  const wssSignal = new WebSocketServer({ noServer: true });
  const wssDirect = new WebSocketServer({ noServer: true });

  const uplinkPath = mode === 'webrtc' ? '/direct' : '/uplink';
  const signalPath = '/signal';

  const envelope = (payload: unknown): string => {
    seq += 1;
    return JSON.stringify({ seq, originTs: Date.now(), from: 'gateway', payload });
  };

  const push = (msg: unknown): void => {
    const text = envelope(msg);
    for (const s of uplinks) if (s.readyState === 1) s.send(text);
  };

  const assignAfterHello = (droneId: number): void => {
    if (routeTimer) return;
    const delay = opts.routeDelayMs ?? 2000;
    log(`[gateway] hello received — the core assigns a route in ${delay} ms`);
    routeTimer = setTimeout(() => {
      const route: AssignRoute =
        opts.route ?? { kind: 'assign-route', droneId, waypoints: DEMO_ROUTE, loop: true };
      log(`[gateway] -> assign-route (${route.waypoints.length} waypoints, loop=${route.loop})`);
      push(route);
    }, delay);
  };

  const acceptUplink = (socket: WebSocket, via: 'uplink' | 'direct'): void => {
    uplinks.add(socket);
    log(`[gateway] ${via} socket connected`);
    socket.on('message', (raw) => {
      let env: Envelope<UplinkMessage>;
      try {
        env = JSON.parse(String(raw)) as Envelope<UplinkMessage>;
      } catch {
        log(`[gateway] unparseable frame: ${String(raw).slice(0, 120)}`);
        return;
      }
      frames.push(env);
      opts.onFrame?.(env, via);
      const payload = env.payload as { kind?: string; droneId?: number };
      if (payload?.kind === 'drone-hello') assignAfterHello(payload.droneId ?? 0);
    });
    socket.on('close', () => {
      uplinks.delete(socket);
      log(`[gateway] ${via} socket closed`);
    });
  };

  wssUplink.on('connection', (socket) => acceptUplink(socket, 'uplink'));
  wssDirect.on('connection', (socket) => acceptUplink(socket, 'direct'));

  wssSignal.on('connection', (socket) => {
    sessionCounter += 1;
    const sessionId = `s${sessionCounter}-${Date.now().toString(36)}`;
    log(`[gateway] signalling session ${sessionId} opened`);
    const reply = (frame: SignalFrame) => {
      if (socket.readyState === 1) socket.send(envelope(frame));
      log(`[gateway] -> ${frame.kind}${frame.direct ? ` direct=${frame.direct}` : ''}`);
    };
    reply({ kind: 'signal-hello', sessionId, droneId: 0, mode: 'webrtc' });

    socket.on('message', (raw) => {
      let env: Envelope<SignalFrame>;
      try {
        env = JSON.parse(String(raw)) as Envelope<SignalFrame>;
      } catch {
        return;
      }
      const frame = env?.payload;
      if (!frame || typeof frame.kind !== 'string') return;
      if (!frame.kind.startsWith('signal-')) {
        // Media on the signalling socket would mean the drone never left the
        // gateway; count it so the test can assert it never happens.
        signalMedia += 1;
        log(`[gateway] !! media on the signalling socket: ${frame.kind}`);
        return;
      }
      log(`[gateway] <- ${frame.kind} (session ${frame.sessionId})`);
      if (frame.kind === 'signal-offer') {
        reply({ kind: 'signal-answer', sessionId, droneId: frame.droneId, mode: 'webrtc', sdp: 'v=0\r\n' });
        reply({
          kind: 'signal-ice',
          sessionId,
          droneId: frame.droneId,
          mode: 'webrtc',
          candidate: { candidate: 'fake-host', sdpMid: '0', sdpMLineIndex: 0 },
        });
        reply({
          kind: 'signal-ready',
          sessionId,
          droneId: frame.droneId,
          mode: 'webrtc',
          direct: `ws://127.0.0.1:${opts.port}${uplinkPath}`,
        });
      }
    });
    socket.on('close', () => log(`[gateway] signalling session ${sessionId} closed`));
  });

  server.on('upgrade', (req, socket, head) => {
    const path = new URL(req.url ?? '/', 'http://localhost').pathname;
    const target =
      path === uplinkPath ? wssUplink : path === signalPath && mode === 'webrtc' ? wssSignal : null;
    if (!target) {
      log(`[gateway] rejected upgrade on ${path} (mode ${mode} serves ${uplinkPath})`);
      socket.destroy();
      return;
    }
    target.handleUpgrade(req, socket, head, (ws) => target.emit('connection', ws, req));
  });

  return new Promise((resolve) => {
    server.listen(opts.port, '127.0.0.1', () => {
      log(
        `[gateway] mode=${mode} listening — drones connect to ` +
          `ws://127.0.0.1:${opts.port}${mode === 'webrtc' ? signalPath : uplinkPath}`,
      );
      resolve({
        frames,
        get signalMedia() {
          return signalMedia;
        },
        send: (msg) => push(msg),
        close: () =>
          new Promise<void>((done) => {
            if (routeTimer) clearTimeout(routeTimer);
            for (const s of uplinks) s.close();
            wssUplink.close();
            wssSignal.close();
            wssDirect.close();
            server.close(() => done());
          }),
      });
    });
  });
}

/** Short-hand for the manual-takeover probe. */
export function manual(droneId: number, forward: number, yaw = 0, climb = 0): ManualControl {
  return { kind: 'manual-control', droneId, forward, yaw, climb };
}

const invoked = process.argv[1] ?? '';
if (invoked.replace(/\\/g, '/').endsWith('test/drone/fakeGateway.ts')) {
  const flag = (name: string): string | undefined =>
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split('=')[1];
  void startFakeGateway({
    port: Number(flag('port') ?? 8081),
    mode: flag('mode') === 'webrtc' ? 'webrtc' : 'relay',
  });
}
