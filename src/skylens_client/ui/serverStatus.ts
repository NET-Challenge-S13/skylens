// Server-status panel — shows the RECON server data source's connection
// state (waiting / receiving), chunk + detection counts, and latency.
// Pure DOM, subscribes to serverSource.onStatus.

import type { ServerSource } from '../server/serverSource.ts';
import type { ServerStatus } from '../../skylens_core/protocol.ts';

export interface ServerStatusPanel {
  dispose(): void;
}

function noop(): ServerStatusPanel {
  return { dispose() {} };
}

type PanelState = 'waiting' | 'connected' | 'receiving';

const STATE_LABEL: Record<PanelState, string> = {
  waiting: '서버 · 대기 중',
  connected: '서버 · 연결됨',
  receiving: '서버 · 수신 중',
};

/** Mount into `#server-status` (see recon.html). */
export function mountServerStatus(source: ServerSource): ServerStatusPanel {
  const host = document.getElementById('server-status');
  if (!host) return noop();

  host.classList.add('server-status');

  const dot = document.createElement('span');
  dot.className = 'server-status__dot';

  const body = document.createElement('div');
  body.className = 'server-status__body';

  const text = document.createElement('span');
  text.className = 'server-status__text';

  const detail = document.createElement('span');
  detail.className = 'server-status__detail';

  body.append(text, detail);
  host.append(dot, body);

  function render(s: ServerStatus): void {
    const panelState: PanelState = s.receiving ? 'receiving' : s.connected ? 'connected' : 'waiting';
    dot.dataset.state = panelState;
    text.textContent = STATE_LABEL[panelState];

    const bits: string[] = [`청크 ${s.chunks}`, `탐지 ${s.detections}`];
    if (s.latencyMs != null) bits.push(`${s.latencyMs}ms`);
    detail.textContent = bits.join(' · ');
  }

  source.onStatus(render);

  return {
    dispose(): void {
      host.classList.remove('server-status');
      host.replaceChildren();
    },
  };
}
