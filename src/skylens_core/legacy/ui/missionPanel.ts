// Mission phase + core link state, in one panel.
//
// These two belong together because the operator's real question is a single
// one: "is anything happening, and if not, whose fault is it?" A mission stuck
// at 대기 중 because no route was assigned and a mission stuck because the core
// is unreachable look identical unless the screen says which.
//
// Phase wording comes from the CORE (MissionStatus.message is already
// localized — see server/mission.ts), so the tower renders it rather than
// keeping a second copy that can drift. The fallback strings here are only for
// the state BEFORE any mission-status has arrived.

import type { MissionPhase, MissionStatus } from '../../shared/protocol.ts';
import type { CoreLinkState } from '../coreLink.ts';

/** Ordered for the stepper — this is the demo scenario of COMPONENTS.md §5.2. */
const PHASES: ReadonlyArray<{ id: MissionPhase; label: string }> = [
  { id: 'idle', label: '대기' },
  { id: 'assigned', label: '태스크 지정' },
  { id: 'awaiting-drone', label: '드론 연결 대기' },
  { id: 'active', label: '임무 수행' },
];

const FALLBACK_MESSAGE: Record<MissionPhase, string> = {
  idle: '대기 중 — 관제탑에서 경로를 지정하십시오',
  assigned: '태스크 지정 완료',
  'awaiting-drone': '드론이 현장에 도착해 연결되면 작업이 시작됩니다',
  active: '드론 연결됨 — 임무 수행 중',
};

export interface MissionPanel {
  setMission(status: MissionStatus): void;
  setLink(state: CoreLinkState, detail: string): void;
  dispose(): void;
}

export function createMissionPanel(mount: HTMLElement): MissionPanel {
  const root = document.createElement('div');
  root.className = 'sl-surface sl-surface--panel mission-panel';

  // --- core link row ---
  const link = document.createElement('div');
  link.className = 'mission-panel__link is-disconnected';
  const linkDot = document.createElement('span');
  linkDot.className = 'mission-panel__dot';
  const linkText = document.createElement('span');
  linkText.className = 'mission-panel__link-text';
  linkText.textContent = '코어 연결 대기 중';
  link.append(linkDot, linkText);

  // --- phase stepper ---
  const steps = document.createElement('ol');
  steps.className = 'mission-panel__steps';
  const stepEls = new Map<MissionPhase, HTMLLIElement>();
  for (const p of PHASES) {
    const li = document.createElement('li');
    li.className = 'mission-panel__step';
    li.textContent = p.label;
    stepEls.set(p.id, li);
    steps.appendChild(li);
  }

  // --- operator-facing line ---
  const message = document.createElement('div');
  message.className = 'mission-panel__message';
  message.textContent = FALLBACK_MESSAGE.idle;

  const sub = document.createElement('div');
  sub.className = 'mission-panel__sub';

  root.append(link, steps, message, sub);
  mount.appendChild(root);

  let phase: MissionPhase = 'idle';
  let etaTimer: ReturnType<typeof setInterval> | null = null;
  let etaDeadline = 0;
  let dronesOnline = 0;

  const renderSub = (): void => {
    const parts: string[] = [];
    if (phase === 'awaiting-drone' && etaDeadline > 0) {
      const left = Math.max(0, Math.round((etaDeadline - Date.now()) / 1000));
      parts.push(`예상 ${left}초`);
    }
    parts.push(`드론 ${dronesOnline}대 연결됨`);
    sub.textContent = parts.join(' · ');
  };

  const stopEta = (): void => {
    if (etaTimer) clearInterval(etaTimer);
    etaTimer = null;
  };

  return {
    setMission(status: MissionStatus): void {
      phase = status.phase;
      dronesOnline = status.dronesOnline;
      message.textContent = status.message || FALLBACK_MESSAGE[status.phase];

      const activeIdx = PHASES.findIndex((p) => p.id === status.phase);
      PHASES.forEach((p, i) => {
        const el = stepEls.get(p.id);
        if (!el) return;
        el.classList.toggle('is-done', i < activeIdx);
        el.classList.toggle('is-active', i === activeIdx);
      });

      // The core sends eta as a snapshot, not a stream, so the panel ticks it
      // down locally from a deadline — a viewer joining mid-wait still lands on
      // the right remaining time because the deadline is recomputed on each push.
      stopEta();
      if (status.phase === 'awaiting-drone' && status.etaSeconds != null) {
        etaDeadline = Date.now() + status.etaSeconds * 1000;
        etaTimer = setInterval(renderSub, 500);
      } else {
        etaDeadline = 0;
      }
      renderSub();
    },

    setLink(state: CoreLinkState, detail: string): void {
      // Rendered directly from the argument — the panel keeps no link state
      // of its own, so it cannot disagree with coreLink about the truth.
      link.classList.toggle('is-connected', state === 'connected');
      link.classList.toggle('is-connecting', state === 'connecting');
      link.classList.toggle('is-disconnected', state === 'disconnected');
      linkText.textContent = detail;

      // Disconnected is not a mission phase — it is the absence of information
      // about one. Say that instead of leaving a stale phase looking live.
      root.classList.toggle('is-stale', state !== 'connected');
      if (state !== 'connected') {
        stopEta();
        message.textContent =
          state === 'connecting'
            ? '코어에 연결하는 중입니다'
            : '코어와 연결되어 있지 않습니다 — 표시된 임무 상태는 최신이 아닐 수 있습니다';
        sub.textContent =
          state === 'connecting'
            ? '연결되면 임무 상태와 드론 위치가 표시됩니다'
            : '연결이 복구될 때까지 드론 위치와 임무 상태를 갱신할 수 없습니다';
      } else {
        renderSub();
      }
    },

    dispose(): void {
      stopEta();
      root.remove();
    },
  };
}
