// Central shared state + tiny pub/sub. Single-app strategy from PROJECT.md §8.2:
// both viewers reference the same object, so "sync" is just shared reads.
// The publish/subscribe layer lets the UI react to discrete events (detection
// revealed, camera state change) without polling. Swapping this for a
// BroadcastChannel later only touches this file.

import type { AppState, CameraSyncState } from './types';

export type AppEvent =
  | { type: 'detection-revealed'; id: string }
  | { type: 'detection-focused'; id: string }
  | { type: 'detection-confirmed'; id: string }
  | { type: 'camera-state'; state: CameraSyncState }
  | { type: 'active-drone'; id: number }
  | { type: 'reset' };

type Listener = (e: AppEvent) => void;

export const state: AppState = {
  time: 0,
  running: true,
  drones: [],
  visited: [],
  detections: [],
  activeDroneId: 1,
  cameraSync: 'SYNCED',
  focusedDetectionId: null,
};

const listeners = new Set<Listener>();

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function emit(e: AppEvent): void {
  for (const fn of listeners) fn(e);
}

/** Convenience helper to change camera state + notify in one call. */
export function setCameraState(next: CameraSyncState): void {
  if (state.cameraSync === next) return;
  state.cameraSync = next;
  emit({ type: 'camera-state', state: next });
}
