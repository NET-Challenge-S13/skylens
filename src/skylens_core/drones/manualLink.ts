// Keyboard → `manual-control` on the wire.
//
// The tower does not move the drone; it asks the core to. Holding a key streams
// stick values at a fixed rate, and RELEASING streams one final zeroed frame so
// a dropped keyup can never leave a drone flying on a stale command. If the
// core link is down the keys do nothing visible — which is correct, because
// nothing is actually flying.

import { CONFIG } from '../../shared/viewer/config.ts';
import { state } from '../../shared/viewer/store.ts';
import { createManualInput } from './manualControl.ts';
import type { ManualControl } from '../../shared/protocol.ts';

export interface ManualLink {
  /** True while the operator is holding a stick key. */
  readonly active: boolean;
  dispose(): void;
}

export interface ManualLinkOptions {
  send(msg: ManualControl): boolean;
  /** Notified when the operator takes over or lets go, for the HUD. */
  onActiveChange?(active: boolean): void;
}

export function createManualLink(opts: ManualLinkOptions): ManualLink {
  const input = createManualInput();
  const periodMs = 1000 / CONFIG.control.manualSendHz;
  let wasActive = false;

  const frame = (): ManualControl => ({
    kind: 'manual-control',
    droneId: state.activeDroneId,
    forward: input.move.z,
    yaw: input.move.x,
    climb: input.move.y,
  });

  const timer = setInterval(() => {
    const active = input.active;
    if (active) {
      opts.send(frame());
    } else if (wasActive) {
      // Exactly one neutral frame on release — repeating it forever would
      // pin the drone under manual authority it is no longer being given.
      opts.send({
        kind: 'manual-control',
        droneId: state.activeDroneId,
        forward: 0,
        yaw: 0,
        climb: 0,
      });
    }
    if (active !== wasActive) {
      wasActive = active;
      opts.onActiveChange?.(active);
    }
  }, periodMs);

  return {
    get active() {
      return input.active;
    },
    dispose(): void {
      clearInterval(timer);
      input.dispose();
    },
  };
}
