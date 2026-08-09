// Keyboard input helper for manually flying the active drone. §4.2
import { state, emit } from '../../skylens_core/store.ts';

export interface ManualInput {
  /** Control axes, each in [-1,1]: x=yaw (turn L/R), y=up/down, z=forward/back. */
  readonly move: { x: number; y: number; z: number };
  /** True if any movement key is currently down. */
  readonly active: boolean;
  dispose(): void;
}

const MOVE_KEYS = new Set([
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'KeyQ',
  'KeyE',
]);

export function createManualInput(): ManualInput {
  const down = new Set<string>();
  const move = { x: 0, y: 0, z: 0 };

  function recompute(): void {
    move.x = (down.has('ArrowRight') ? 1 : 0) - (down.has('ArrowLeft') ? 1 : 0);
    move.z = (down.has('ArrowUp') ? 1 : 0) - (down.has('ArrowDown') ? 1 : 0);
    move.y = (down.has('KeyE') ? 1 : 0) - (down.has('KeyQ') ? 1 : 0);
  }

  // active-drone is authoritative here: update the shared state, THEN notify.
  function switchActive(id: number): void {
    if (!state.drones.some((d) => d.id === id)) return;
    if (state.activeDroneId === id) return;
    state.activeDroneId = id;
    emit({ type: 'active-drone', id });
  }

  function onKeyDown(e: KeyboardEvent): void {
    if (MOVE_KEYS.has(e.code)) {
      down.add(e.code);
      recompute();
    }
    if (e.code === 'Digit1') switchActive(1);
    else if (e.code === 'Digit2') switchActive(2);
    else if (e.code === 'Digit3') switchActive(3);
    else if (e.code === 'Tab') {
      e.preventDefault();
      const ids = state.drones.map((d) => d.id);
      if (ids.length > 0) {
        const idx = ids.indexOf(state.activeDroneId);
        switchActive(ids[(idx + 1) % ids.length]);
      }
    }
  }

  function onKeyUp(e: KeyboardEvent): void {
    if (MOVE_KEYS.has(e.code)) {
      down.delete(e.code);
      recompute();
    }
  }

  if (typeof window !== 'undefined') {
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
  }

  return {
    move,
    get active() {
      return down.size > 0;
    },
    dispose(): void {
      if (typeof window !== 'undefined') {
        window.removeEventListener('keydown', onKeyDown);
        window.removeEventListener('keyup', onKeyUp);
      }
      down.clear();
    },
  };
}
