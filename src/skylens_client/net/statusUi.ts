// Small shared helpers for the networked pages: room selection + a corner badge
// showing the WebRTC connection status.

import type { PeerRole, PeerStatus, Transport } from './peer.ts';

/** Room token from `?room=`, so multiple demos don't collide on the public broker. */
export function roomFromQuery(): string {
  if (typeof window === 'undefined') return 'default';
  const r = new URLSearchParams(window.location.search).get('room');
  return r && r.trim() ? r.trim() : 'default';
}

const LABEL: Record<PeerStatus, string> = {
  idle: '대기',
  connecting: '연결 중…',
  connected: '연결됨',
  disconnected: '연결 끊김',
  error: '오류',
};

const DOT: Record<PeerStatus, string> = {
  idle: '#8a93a6',
  connecting: '#ffd27f',
  connected: '#39d98a',
  disconnected: '#ff9a4d',
  error: '#ff4d4d',
};

/** Mount a live connection badge into #net-status. */
export function mountNetBadge(transport: Transport, role: PeerRole): void {
  const host = document.getElementById('net-status');
  if (!host) return;

  const dot = document.createElement('span');
  dot.className = 'net-badge__dot';
  const text = document.createElement('span');

  host.classList.add('net-badge');
  host.append(dot, text);

  const peer = role === 'sim' ? 'RECON' : 'SIM';

  transport.onStatus((s, detail) => {
    dot.style.background = DOT[s];
    text.textContent =
      s === 'connected'
        ? `${peer} 연결됨`
        : `${peer} ${LABEL[s]}${detail ? ` · ${detail}` : ''}`;
  });
}
