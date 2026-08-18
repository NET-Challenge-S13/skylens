// Server-status panel — shows the board's server data source: connection state
// (waiting / receiving), chunk + detection counts, latency, and the DELAY-PATTERN
// ladder (which capture segment sits at which refinement level right now).
// Pure DOM, subscribes to serverSource.onStatus.

import type { ServerSource } from '../../skylens_core/server/serverSource.ts';
import type { SegmentStatus, ServerStatus } from '../../skylens_core/protocol.ts';

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

/** Mount into `#server-status` (see status.html). */
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

  // Delay-pattern ladder: one row per capture segment, filled pips for the
  // levels already delivered. A segment keeps refining while the next one starts,
  // so several rows sit at different levels at the same time.
  const ladder = document.createElement('div');
  ladder.className = 'server-status__ladder';
  host.append(ladder);

  const rows = new Map<number, { pips: HTMLSpanElement[]; note: HTMLSpanElement }>();

  function segmentRow(seg: SegmentStatus): { pips: HTMLSpanElement[]; note: HTMLSpanElement } {
    const existing = rows.get(seg.index);
    if (existing) return existing;

    const row = document.createElement('div');
    row.className = 'segment-row';

    const name = document.createElement('span');
    name.className = 'segment-row__name';
    name.textContent = `구간 ${seg.index + 1}`;

    const bar = document.createElement('span');
    bar.className = 'segment-row__bar';
    const pips: HTMLSpanElement[] = [];
    for (let i = 0; i < seg.levels; i++) {
      const pip = document.createElement('span');
      pip.className = 'segment-row__pip';
      bar.append(pip);
      pips.push(pip);
    }

    const note = document.createElement('span');
    note.className = 'segment-row__note';

    row.append(name, bar, note);
    ladder.append(row);

    const made = { pips, note };
    rows.set(seg.index, made);
    return made;
  }

  function render(s: ServerStatus): void {
    const panelState: PanelState = s.receiving ? 'receiving' : s.connected ? 'connected' : 'waiting';
    dot.dataset.state = panelState;
    text.textContent = STATE_LABEL[panelState];

    const bits: string[] = [`청크 ${s.chunks}`, `탐지 ${s.detections}`];
    if (s.latencyMs != null) bits.push(`${s.latencyMs}ms`);
    detail.textContent = bits.join(' · ');

    for (const seg of s.segments) {
      const row = segmentRow(seg);
      row.pips.forEach((pip, i) => {
        pip.dataset.on = i < seg.level ? '1' : '0';
      });
      row.note.textContent =
        seg.level === 0
          ? '복원 중'
          : `수준 ${seg.level} · ${seg.steps.toLocaleString('ko-KR')}스텝${
              seg.label ? ` · ${seg.label}` : ''
            }`;
      row.note.dataset.refining = seg.level > 0 && seg.level < seg.levels ? '1' : '0';
    }
  }

  source.onStatus(render);

  return {
    dispose(): void {
      host.classList.remove('server-status');
      host.replaceChildren();
      rows.clear();
    },
  };
}
