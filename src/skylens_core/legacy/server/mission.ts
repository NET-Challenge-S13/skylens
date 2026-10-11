// Mission state machine: idle → assigned → awaiting-drone → active.
//
// The demo scenario (COMPONENTS.md §5.2) is expressed entirely through these
// phases — there is no separate "demo script" anywhere in the core:
//
//   1. 데모 시작        phase=idle          시스템 정지
//   2. 관제탑 경로 지정  phase=assigned      "태스크 지정 완료"
//   3. 드론 연결 대기    phase=awaiting-drone "드론이 현장에 도착해 연결되면 …" + eta
//   4. 드론 연결        phase=active        "드론 연결됨"
//
// etaSeconds is derived, not counted down by a timer: the status push recomputes
// it, so a viewer that joins mid-wait sees the right remaining time.

import type { MissionPhase, MissionStatus } from '../../shared/protocol.ts';

const MESSAGES: Record<MissionPhase, string> = {
  idle: '대기 중 — 관제탑에서 경로를 지정하십시오',
  assigned: '태스크 지정 완료',
  'awaiting-drone': '드론이 현장에 도착해 연결되면 작업이 시작됩니다',
  active: '드론 연결됨 — 임무 수행 중',
};

export interface MissionOptions {
  assignedHoldMs: number;
  droneEtaSeconds: number;
  /** Drones connected right now — the machine asks instead of tracking. */
  dronesOnline: () => number;
  onChange: (status: MissionStatus) => void;
}

export class Mission {
  private opts: MissionOptions;
  private currentPhase: MissionPhase = 'idle';
  private awaitingSince = 0;
  private holdTimer: ReturnType<typeof setTimeout> | null = null;
  private history: Array<{ at: number; phase: MissionPhase }> = [
    { at: Date.now(), phase: 'idle' },
  ];

  constructor(opts: MissionOptions) {
    this.opts = opts;
  }

  get phase(): MissionPhase {
    return this.currentPhase;
  }

  get phaseHistory(): Array<{ at: number; phase: MissionPhase }> {
    return this.history;
  }

  status(): MissionStatus {
    return {
      kind: 'mission-status',
      phase: this.currentPhase,
      message: MESSAGES[this.currentPhase],
      dronesOnline: this.opts.dronesOnline(),
      etaSeconds: this.etaSeconds(),
    };
  }

  /** Control tower assigned a route: idle → assigned. After a short dwell so the
   *  operator actually reads "태스크 지정 완료", → awaiting-drone. */
  routeAssigned(): void {
    this.to('assigned');
    this.clearHold();
    this.holdTimer = setTimeout(() => {
      this.holdTimer = null;
      if (this.currentPhase !== 'assigned') return;
      this.to('awaiting-drone');
      // A drone that is already online means the wait is over before it began.
      if (this.opts.dronesOnline() > 0) this.to('active');
    }, this.opts.assignedHoldMs);
  }

  /** awaiting-drone → active. Ignored while idle: a drone without a task does
   *  not start a mission. Status is re-broadcast even when the phase holds:
   *  it carries dronesOnline, and a phase-change-only broadcast froze the
   *  count at whatever it was when the FIRST drone flipped the mission to
   *  active — the panel then said 1대 with three aircraft flying. */
  droneConnected(): void {
    if (this.currentPhase === 'awaiting-drone') this.to('active');
    else this.opts.onChange(this.status());
  }

  /** Last drone dropped mid-mission: back to waiting, eta restarts. The
   *  count re-broadcast matters here too — 3대 → 2대 is information. */
  droneGone(): void {
    if (this.currentPhase === 'active' && this.opts.dronesOnline() === 0) {
      this.to('awaiting-drone');
    } else {
      this.opts.onChange(this.status());
    }
  }

  reset(): void {
    this.clearHold();
    this.to('idle');
  }

  stop(): void {
    this.clearHold();
  }

  private etaSeconds(): number | null {
    if (this.currentPhase !== 'awaiting-drone') return null;
    const elapsed = (Date.now() - this.awaitingSince) / 1000;
    return Math.max(0, Math.round(this.opts.droneEtaSeconds - elapsed));
  }

  private to(phase: MissionPhase): void {
    if (phase === this.currentPhase) return;
    this.currentPhase = phase;
    if (phase === 'awaiting-drone') this.awaitingSince = Date.now();
    this.history.push({ at: Date.now(), phase });
    console.log(`[core] mission → ${phase} (${MESSAGES[phase]})`);
    this.opts.onChange(this.status());
  }

  private clearHold(): void {
    if (this.holdTimer) clearTimeout(this.holdTimer);
    this.holdTimer = null;
  }
}
