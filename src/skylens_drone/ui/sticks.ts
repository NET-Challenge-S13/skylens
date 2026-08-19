// Local stick input, so the manual-control path can be exercised without the
// control tower being up.
//
// This produces the SAME ManualControl message the core sends down, and feeds it
// through the app's control handler — there is no second code path for "local"
// input. Keys: W/S forward, A/D yaw, R/F climb. Releasing every key sends the
// zero message, which is what makes the drone rejoin its route.

import type { ManualControl } from '../../shared/protocol.ts';

const KEYS: Record<string, [axis: 'forward' | 'yaw' | 'climb', value: number]> = {
  w: ['forward', 1],
  s: ['forward', -1],
  a: ['yaw', -1],
  d: ['yaw', 1],
  r: ['climb', 1],
  f: ['climb', -1],
  arrowup: ['forward', 1],
  arrowdown: ['forward', -1],
  arrowleft: ['yaw', -1],
  arrowright: ['yaw', 1],
};

export class StickPad {
  private root: HTMLElement;
  private droneId: number;
  private emit: (msg: ManualControl) => void;
  private held = new Set<string>();
  private readout: HTMLElement;
  private detach: Array<() => void> = [];

  constructor(root: HTMLElement, droneId: number, emit: (msg: ManualControl) => void) {
    this.root = root;
    this.droneId = droneId;
    this.emit = emit;
    this.root.className = 'sticks';

    const head = document.createElement('header');
    head.className = 'sticks__head';
    head.textContent = '수동 조종 (W/S 전후 · A/D 회전 · R/F 상승)';
    this.readout = document.createElement('div');
    this.readout.className = 'sticks__out';
    this.readout.textContent = 'forward 0.0 · yaw 0.0 · climb 0.0';
    this.root.append(head, this.readout);

    const down = (ev: KeyboardEvent) => this.onKey(ev, true);
    const up = (ev: KeyboardEvent) => this.onKey(ev, false);
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    this.detach.push(() => window.removeEventListener('keydown', down));
    this.detach.push(() => window.removeEventListener('keyup', up));
  }

  dispose(): void {
    for (const off of this.detach) off();
    this.detach = [];
  }

  private onKey(ev: KeyboardEvent, pressed: boolean): void {
    const key = ev.key.toLowerCase();
    if (!(key in KEYS)) return;
    ev.preventDefault();
    const before = this.held.size;
    if (pressed) this.held.add(key);
    else this.held.delete(key);
    // Repeat events would otherwise resend an identical message every frame.
    if (pressed && before === this.held.size) return;
    this.send();
  }

  private send(): void {
    const msg: ManualControl = { kind: 'manual-control', droneId: this.droneId, forward: 0, yaw: 0, climb: 0 };
    for (const key of this.held) {
      const [axis, value] = KEYS[key];
      msg[axis] = Math.max(-1, Math.min(1, msg[axis] + value));
    }
    this.readout.textContent = `forward ${msg.forward.toFixed(1)} · yaw ${msg.yaw.toFixed(1)} · climb ${msg.climb.toFixed(1)}`;
    this.emit(msg);
  }
}
