// Corner badge for the board's data path.
//
// It replaces the old peer-to-peer badge, which reported whether the CONTROL
// TOWER was reachable — a link the board no longer has, and must not have
// (COMPONENTS.md §8). What matters now is the chain the data actually travels:
//
//   현황판 ── ws ──> skylens_client(8090) ── ws ──> skylens_core(8080)
//
// Both hops are shown, because they fail differently and the operator's next
// action differs: a dead relay means this component is down, a dead upstream
// means the core is. Reusing the existing .net-badge classes keeps the styling
// owned by style.css.

import type { FeedStatus, RelayClient } from '../sources/relayClient.ts';

/** Worst-of the two hops decides the colour. */
function severity(s: FeedStatus): 'ok' | 'warn' | 'down' {
  if (s.relay !== 'online') return s.relay === 'connecting' ? 'warn' : 'down';
  if (s.upstream !== 'online') return s.upstream === 'connecting' ? 'warn' : 'down';
  return 'ok';
}

const DOT: Record<'ok' | 'warn' | 'down', string> = {
  ok: '#39d98a',
  warn: '#ffd27f',
  down: '#ff4d4d',
};

export function mountRelayBadge(source: RelayClient): void {
  const host = document.getElementById('net-status');
  if (!host) return;

  const dot = document.createElement('span');
  dot.className = 'net-badge__dot';
  const text = document.createElement('span');

  host.classList.add('net-badge');
  host.replaceChildren(dot, text);

  source.onStatus((s) => {
    dot.style.background = DOT[severity(s)];
    const boards = s.boards > 1 ? ` · 현황판 ${s.boards}대` : '';
    text.textContent = `${s.detail}${boards}`;
  });
}
