// Stand-in for skylens_core while that component is built by someone else.
//
// It implements only what the proxy assumes of the core: a WebSocket server at
// /uplink that accepts Envelope JSON and answers WebSocket ping frames (the `ws`
// library does that automatically, which is what the proxy's health probe uses).
//
// Run two of them to exercise failover:
//   npx tsx src/test/proxy/fakeCore.ts 8080 primary
//   npx tsx src/test/proxy/fakeCore.ts 8180 standby
//
// Env:
//   FAKE_CORE_QUIET=1        do not log link-status frames
//   FAKE_CORE_HANG_AFTER_MS  after this long, block the event loop …
//   FAKE_CORE_HANG_FOR_MS    … for this long (default 6000).
// The hang is the interesting failure: the TCP socket stays open, so nothing
// closes — but the process stops answering WebSocket pings. That is exactly the
// "line is nominally up, far end is dead" case the proxy's health probe exists
// for, and it cannot be reproduced by killing the process.

import http from 'node:http';
import process from 'node:process';
import { WebSocketServer } from 'ws';

const port = Number(process.argv[2] ?? 8080);
const name = process.argv[3] ?? `core:${port}`;
const quiet = process.env.FAKE_CORE_QUIET === '1';
const hangAfterMs = Number(process.env.FAKE_CORE_HANG_AFTER_MS ?? 0);
const hangForMs = Number(process.env.FAKE_CORE_HANG_FOR_MS ?? 6000);

let frames = 0;
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ component: 'fake-core', name, port, frames, path: req.url }));
});

const wss = new WebSocketServer({ server, path: '/uplink' });

wss.on('connection', (ws) => {
  console.log(`[${name}] proxy connected`);
  ws.on('message', (data) => {
    frames += 1;
    let env: { seq?: number; from?: string; payload?: { kind?: string }; path?: unknown[] };
    try {
      env = JSON.parse(data.toString());
    } catch {
      console.log(`[${name}] non-JSON frame`);
      return;
    }
    const kind = env.payload?.kind ?? '?';
    if (quiet && kind === 'link-status') return;
    const hops = Array.isArray(env.path)
      ? (env.path as Array<{ at: string; via?: string }>)
          .map((h) => (h.via ? `${h.at}(${h.via})` : h.at))
          .join('->')
      : '-';
    console.log(`[${name}] #${frames} kind=${kind} seq=${env.seq} from=${env.from} path=${hops}`);
  });
  ws.on('close', () => console.log(`[${name}] proxy disconnected`));
});

server.listen(port, '127.0.0.1', () => {
  console.log(`[${name}] listening ws://127.0.0.1:${port}/uplink`);
});

if (hangAfterMs > 0) {
  setTimeout(() => {
    console.log(`[${name}] HANGING for ${hangForMs}ms — socket stays open, no pongs`);
    const until = Date.now() + hangForMs;
    // Deliberate busy wait: sleeping would still service the ping.
    while (Date.now() < until) {
      /* block the event loop */
    }
    console.log(`[${name}] responsive again`);
  }, hangAfterMs);
}
