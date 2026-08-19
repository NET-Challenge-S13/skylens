// Server-status panel — the board's honest account of where its data is coming
// from: the two link states (browser→중계 서버, 중계 서버→코어), the mission line
// the core is broadcasting, chunk/detection counters, and the DELAY-PATTERN
// ladder.
//
// The ladder is rendered from `SegmentStatus[]` EXACTLY as received. Level count,
// step count, and label are the core's (skylens_core/server/ladder.ts); this file
// draws pips and nothing else. When the core has sent no ladder yet, the panel
// says so rather than inventing rows — an empty ladder is information.

import type { SegmentStatus } from '../../shared/protocol.ts';
import type { FeedStatus, RelayClient } from '../sources/relayClient.ts';

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
export function mountServerStatus(source: RelayClient): ServerStatusPanel {
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

  const link = document.createElement('span');
  link.className = 'server-status__detail';

  body.append(text, detail, link);
  host.append(dot, body);

  // Delay-pattern ladder: one row per capture segment, filled pips for the
  // levels already delivered. A segment keeps refining while the next one starts,
  // so several rows sit at different levels at the same time.
  const ladder = document.createElement('div');
  ladder.className = 'server-status__ladder';
  host.append(ladder);

  const rows = new Map<number, { row: HTMLElement; pips: HTMLSpanElement[]; note: HTMLSpanElement }>();

  /** Rebuild a row's pips when the core revises the ladder height. */
  function segmentRow(seg: SegmentStatus): { pips: HTMLSpanElement[]; note: HTMLSpanElement } {
    const existing = rows.get(seg.index);
    if (existing && existing.pips.length === seg.levels) return existing;

    const row = existing?.row ?? document.createElement('div');
    row.className = 'segment-row';
    row.replaceChildren();

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
    if (!existing) ladder.append(row);

    const made = { row, pips, note };
    rows.set(seg.index, made);
    return made;
  }

  function render(s: FeedStatus): void {
    const server = s.server;
    const panelState: PanelState = server.receiving
      ? 'receiving'
      : server.connected
        ? 'connected'
        : 'waiting';
    dot.dataset.state = panelState;
    text.textContent = STATE_LABEL[panelState];

    const bits: string[] = [`청크 ${server.chunks}`, `탐지 ${server.detections}`];
    if (server.latencyMs != null) bits.push(`${Math.round(server.latencyMs)}ms`);
    detail.textContent = bits.join(' · ');

    // The waiting state has to name WHICH hop is down, and the mission line the
    // core broadcasts takes precedence once the stream is healthy.
    const mission = s.mission;
    link.textContent =
      panelState === 'receiving' && mission
        ? `${mission.message} · 드론 ${mission.dronesOnline}대`
        : s.detail;

    for (const seg of server.segments) {
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
