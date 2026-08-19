/// <reference types="node" />
// Serving the situation board web app itself.
//
//   dev   → reverse-proxy everything we don't own to the Vite dev server (5173),
//           HMR websocket included, so the board is edited with HMR while still
//           being loaded from :8090 — which matters, because same-origin is what
//           lets the board find its relay socket with no configuration.
//   prod  → serve the built dist/ produced by `npm run build`.
//
// When Vite isn't up, the proxy answers with a Korean waiting page that retries
// on its own instead of a bare ECONNREFUSED — the same "never a frozen screen"
// rule the upstream link follows.

import http from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Duplex } from 'node:stream';
import path from 'node:path';
import fs from 'node:fs';
import express from 'express';
import type { Express } from 'express';
import type { ClientServerConfig } from './config.ts';

export interface WebApp {
  /** Forward a non-relay upgrade (Vite HMR). Prod mode just closes it. */
  upgrade(req: IncomingMessage, socket: Duplex, head: Buffer): void;
  describe(): { mode: string; target: string; ok: boolean; detail: string };
}

function waitingPage(title: string, detail: string): string {
  return `<!doctype html><html lang="ko"><head><meta charset="utf-8" />
<title>SkyLens · 현황판 대기</title>
<meta http-equiv="refresh" content="2" />
<style>
 body{margin:0;height:100vh;display:grid;place-items:center;background:#0d0f14;color:#c8d2e0;
      font-family:system-ui,'Malgun Gothic',sans-serif}
 .box{text-align:center;max-width:34rem;padding:2rem}
 h1{font-size:1.1rem;font-weight:600;letter-spacing:.02em;margin:0 0 .6rem}
 p{font-size:.85rem;line-height:1.7;color:#8a93a6;margin:.2rem 0}
 code{color:#9fe8ff}
 .dot{width:.5rem;height:.5rem;border-radius:50%;background:#ffd27f;display:inline-block;
      margin-right:.5rem;animation:p 1.2s ease-in-out infinite}
 @keyframes p{0%,100%{opacity:.25}50%{opacity:1}}
</style></head><body><div class="box">
<h1><span class="dot"></span>${title}</h1><p>${detail}</p>
<p>2초마다 자동으로 다시 시도합니다.</p></div></body></html>`;
}

/** dev: reverse proxy to Vite. */
function mountDevProxy(app: Express, cfg: ClientServerConfig): WebApp {
  const target = new URL(cfg.viteUrl);
  const host = target.hostname;
  const port = Number(target.port || 80);
  let lastError = '';
  let reachable = false;

  app.use((req: IncomingMessage & { url?: string }, res: ServerResponse) => {
    const proxyReq = http.request(
      {
        host,
        port,
        method: (req as { method?: string }).method ?? 'GET',
        path: req.url ?? '/',
        headers: { ...req.headers, host: `${host}:${port}` },
      },
      (proxyRes) => {
        reachable = true;
        lastError = '';
        res.writeHead(proxyRes.statusCode ?? 502, proxyRes.headers);
        proxyRes.pipe(res);
      },
    );
    proxyReq.on('error', (err: Error) => {
      reachable = false;
      lastError = err.message;
      res.writeHead(503, { 'content-type': 'text/html; charset=utf-8' });
      res.end(
        waitingPage(
          'Vite 개발 서버 대기 중',
          `현황판 웹앱을 <code>${cfg.viteUrl}</code> 에서 가져오지 못했습니다 — <code>npm run dev</code> 를 실행하세요.`,
        ),
      );
    });
    req.pipe(proxyReq);
  });

  return {
    upgrade(req, socket, head): void {
      // Vite's HMR socket. Replay the client's handshake against Vite and splice
      // the two sockets together.
      const proxyReq = http.request({
        host,
        port,
        method: 'GET',
        path: req.url ?? '/',
        headers: { ...req.headers, host: `${host}:${port}` },
      });
      proxyReq.on('error', () => socket.destroy());
      proxyReq.on('upgrade', (proxyRes, proxySocket, proxyHead) => {
        const lines = [`HTTP/1.1 ${proxyRes.statusCode ?? 101} ${proxyRes.statusMessage ?? 'Switching Protocols'}`];
        for (const [k, v] of Object.entries(proxyRes.headers)) {
          if (Array.isArray(v)) for (const one of v) lines.push(`${k}: ${one}`);
          else if (v !== undefined) lines.push(`${k}: ${v}`);
        }
        socket.write(`${lines.join('\r\n')}\r\n\r\n`);
        if (proxyHead?.length) proxySocket.unshift(proxyHead);
        if (head?.length) proxyReq.write(head);
        proxySocket.on('error', () => socket.destroy());
        socket.on('error', () => proxySocket.destroy());
        proxySocket.pipe(socket).pipe(proxySocket);
      });
      proxyReq.end();
    },
    describe() {
      return {
        mode: 'dev (vite proxy)',
        target: cfg.viteUrl,
        ok: reachable,
        detail: reachable ? 'vite 응답 확인됨' : lastError || '아직 요청 없음',
      };
    },
  };
}

/** prod: serve the built bundle. */
function mountStatic(app: Express, cfg: ClientServerConfig): WebApp {
  const root = path.resolve(cfg.distDir);
  const exists = fs.existsSync(root);
  app.use(express.static(root, { index: false }));
  app.get('/', (_req, res) => {
    res.redirect('/res/static/status.html');
  });
  app.use((_req, res) => {
    res.status(exists ? 404 : 503).type('html').send(
      waitingPage(
        exists ? '페이지를 찾을 수 없습니다' : '빌드 산출물 없음',
        exists
          ? `<code>${root}</code> 에 해당 파일이 없습니다.`
          : `<code>${root}</code> 가 없습니다 — <code>npm run build</code> 를 먼저 실행하세요.`,
      ),
    );
  });

  return {
    upgrade(_req, socket): void {
      socket.destroy();
    },
    describe() {
      return {
        mode: 'prod (static)',
        target: root,
        ok: exists,
        detail: exists ? '빌드 산출물 확인됨' : 'dist 없음',
      };
    },
  };
}

export function mountWebApp(app: Express, cfg: ClientServerConfig): WebApp {
  return cfg.mode === 'prod' ? mountStatic(app, cfg) : mountDevProxy(app, cfg);
}
