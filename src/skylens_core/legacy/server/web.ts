// 관제탑 화면 서버 (COMPONENTS.md §3.4-1).
//
// The control tower UI lives at the component root (control.ts · controlview/ ·
// control/ · ui/) and is built by Vite. This module is only the delivery path:
//
//   dev  — reverse-proxy everything that is not an API/ws path to the Vite dev
//          server, HMR socket included. One origin for the operator, so the UI
//          talks to the core with a relative URL and no CORS.
//   prod — serve the built directory.
//   off  — serve nothing; the core is then a headless orchestrator and the UI is
//          reached through Vite (5173) or skylens_client directly.
//
// Nothing here touches the UI source: this component owns server/ only.

import http from 'node:http';
import type { Duplex } from 'node:stream';
import type { Express, Request, Response } from 'express';
import express from 'express';
import type { CoreConfig } from './config.ts';

export interface WebSurface {
  describe(): string;
  /** True when the upgrade was consumed (dev-mode HMR). */
  upgrade(req: http.IncomingMessage, socket: Duplex, head: Buffer): boolean;
}

export function mountWeb(app: Express, cfg: CoreConfig): WebSurface {
  if (cfg.webMode === 'off') {
    return { describe: () => 'off (headless)', upgrade: () => false };
  }

  if (cfg.webMode === 'prod') {
    app.use(express.static(cfg.webDist));
    return { describe: () => `prod (static ${cfg.webDist})`, upgrade: () => false };
  }

  const target = new URL(cfg.webTarget);
  app.use((req: Request, res: Response) => proxy(req, res, target));
  return {
    describe: () => `dev (reverse proxy → ${cfg.webTarget})`,
    upgrade: (req, socket, head) => upgradeTo(target, req, socket, head),
  };
}

function proxy(req: Request, res: Response, target: URL): void {
  const upstream = http.request(
    {
      host: target.hostname,
      port: target.port === '' ? 80 : Number(target.port),
      method: req.method,
      path: req.originalUrl,
      headers: { ...req.headers, host: target.host },
    },
    (up) => {
      res.writeHead(up.statusCode ?? 502, up.headers);
      up.pipe(res);
    },
  );
  upstream.on('error', (err) => {
    if (res.headersSent) {
      res.end();
      return;
    }
    res.status(502).type('text/plain').send(
      `[core] 관제탑 UI 개발 서버(${target.origin})에 닿지 못했습니다: ${err.message}\n` +
        'npm run dev 를 함께 띄우거나 SKYLENS_CORE_WEB_MODE=off 로 실행하십시오.\n',
    );
  });
  req.pipe(upstream);
}

/**
 * Vite's HMR socket. Piped raw — the core never reads a byte of it.
 *
 * The two buffers handed over at the upgrade are leftovers each side had
 * already sent, and they belong to the OTHER socket's outbound direction. They
 * have to be written across, not unshifted: unshift() puts bytes into a
 * socket's own READ buffer, so the pipes promptly delivered Vite's frames back
 * to Vite. A server frame is unmasked, the ws receiver rejects it, and Vite
 * died with WS_ERR_EXPECTED_MASK — taking every component down with it and
 * leaving browsers on a stale page.
 */
function upgradeTo(
  target: URL,
  req: http.IncomingMessage,
  socket: Duplex,
  head: Buffer,
): boolean {
  const upstream = http.request({
    host: target.hostname,
    port: target.port === '' ? 80 : Number(target.port),
    path: req.url ?? '/',
    headers: { ...req.headers, host: target.host },
  });
  upstream.end();
  // A browser closing a tab resets these sockets, and an unhandled 'error' on a
  // raw socket takes the whole process down with it. The core has to outlive its
  // viewers, so every socket in the pair gets a handler.
  const drop = (): void => {
    socket.destroy();
  };
  socket.on('error', drop);
  upstream.on('upgrade', (upRes, upSocket, upHead) => {
    upSocket.on('error', drop);
    const lines = Object.entries(upRes.headers).map(([k, v]) => `${k}: ${String(v)}`);
    socket.write(`HTTP/1.1 101 Switching Protocols\r\n${lines.join(`\r\n`)}\r\n\r\n`);
    if (upHead.length > 0) socket.write(upHead);
    if (head.length > 0) upSocket.write(head);
    upSocket.pipe(socket);
    socket.pipe(upSocket);
  });
  upstream.on('error', drop);
  return true;
}
