// DOM UI overlay layer — HUD (viewer 1) + detection card / camera-state
// indicator (viewer 2). No Three.js here; pure DOM built against the shared
// store (core/store.ts) and styled by the classes already in style.css.
// PROJECT.md §5.3 (detection reveal/confirm flow), §8.3 (camera sync).

import type { AppEvent } from '../../skylens_core/store.ts';
import { state, emit, subscribe } from '../../skylens_core/store.ts';
import { DRONE_TINTS } from '../../skylens_core/config.ts';
import { showToast } from '../../skylens_core/ui/toast.ts';

export interface UI {
  update(): void;
  dispose(): void;
}

const CAM_STATE_LABEL: Record<string, string> = {
  SYNCED: 'CAM · SYNCED',
  FOCUSING: 'CAM · FOCUSING',
  LOCKED: 'CAM · LOCKED',
  RETURNING: 'CAM · RETURNING',
};

const KICKER_LABEL: Record<string, string> = {
  person: '생존자 감지',
  danger: '위험구역 감지',
};

export function initUI(): UI {
  const hudRoot = document.getElementById('hud-sim');
  const overlayRoot = document.getElementById('overlay-recon');
  // toast-root is handled entirely by toast.ts.

  // --- HUD (viewer 1) ---
  const droneRows = new Map<number, { row: HTMLElement; dot: HTMLElement; label: HTMLElement; mode: HTMLElement }>();
  let lastActiveDroneId: number | null = null;
  let hudBuilt = false;

  function buildHud(): void {
    if (!hudRoot || hudBuilt || state.drones.length === 0) return;
    hudRoot.replaceChildren();
    droneRows.clear();

    for (const drone of state.drones) {
      const row = document.createElement('div');
      row.className = 'hud__drone';

      const dot = document.createElement('span');
      dot.className = 'hud__dot';
      dot.style.background = `#${(DRONE_TINTS[drone.id - 1] ?? 0xffffff).toString(16).padStart(6, '0')}`;

      const label = document.createElement('span');
      label.className = 'hud__label';

      const mode = document.createElement('span');
      mode.className = 'hud__mode';

      row.append(dot, label, mode);
      hudRoot.appendChild(row);
      droneRows.set(drone.id, { row, dot, label, mode });
    }

    const hint = document.createElement('div');
    hint.className = 'hud__hint';
    hint.textContent = 'Tab: 드론 전환 · WASD/화살표: 이동 · Q/E: 고도 · 조작 중단 시 자동 복귀';
    hudRoot.appendChild(hint);

    hudBuilt = true;
  }

  function refreshHud(): void {
    if (!hudRoot) return;
    if (!hudBuilt) buildHud();
    if (!hudBuilt) return;

    if (state.activeDroneId !== lastActiveDroneId) {
      for (const [id, refs] of droneRows) {
        refs.row.classList.toggle('is-active', id === state.activeDroneId);
      }
      lastActiveDroneId = state.activeDroneId;
    }

    for (const drone of state.drones) {
      const refs = droneRows.get(drone.id);
      if (!refs) continue;
      const labelText = drone.zone;
      if (refs.label.textContent !== labelText) refs.label.textContent = labelText;
      if (refs.mode.textContent !== drone.mode) refs.mode.textContent = drone.mode;
    }
  }

  // --- Detection card + camera-state indicator (viewer 2) ---
  let card: HTMLElement | null = null;
  let kicker: HTMLElement | null = null;
  let cardLabel: HTMLElement | null = null;
  let conf: HTMLElement | null = null;
  let barFill: HTMLElement | null = null;
  let btn: HTMLButtonElement | null = null;
  let camState: HTMLElement | null = null;
  let lastCamStateText: string | null = null;
  let openDetectionId: string | null = null;

  function onConfirmClick(): void {
    if (!openDetectionId) return;
    const det = state.detections.find((d) => d.id === openDetectionId);
    if (det) det.confirmed = true;
    const id = openDetectionId;
    closeCard();
    emit({ type: 'detection-confirmed', id });
  }

  function buildOverlayDom(): void {
    if (!overlayRoot) return;

    card = document.createElement('div');
    card.className = 'detect-card';

    kicker = document.createElement('div');
    kicker.className = 'detect-card__kicker';

    cardLabel = document.createElement('div');
    cardLabel.className = 'detect-card__label';

    conf = document.createElement('div');
    conf.className = 'detect-card__conf';

    const bar = document.createElement('div');
    bar.className = 'detect-card__bar';
    barFill = document.createElement('span');
    bar.appendChild(barFill);

    btn = document.createElement('button');
    btn.className = 'detect-card__btn';
    btn.type = 'button';
    btn.textContent = '탐지 확인';
    btn.addEventListener('click', onConfirmClick);

    card.append(kicker, cardLabel, conf, bar, btn);
    overlayRoot.appendChild(card);

    camState = document.createElement('div');
    camState.className = 'cam-state';
    overlayRoot.appendChild(camState);
  }

  function openCard(id: string): void {
    const det = state.detections.find((d) => d.id === id);
    if (!det || !card || !kicker || !cardLabel || !conf || !barFill) return;

    openDetectionId = id;
    card.classList.remove('kind-person', 'kind-danger');
    card.classList.add(`kind-${det.kind}`);
    kicker.textContent = KICKER_LABEL[det.kind] ?? '';
    cardLabel.textContent = det.label;
    const pct = Math.round(det.confidence * 100);
    conf.textContent = `확신도 ${pct}%`;
    barFill.style.width = `${pct}%`;
    card.classList.add('is-open');
  }

  function closeCard(): void {
    openDetectionId = null;
    card?.classList.remove('is-open');
  }

  function refreshCamState(): void {
    if (!camState) return;
    const text = CAM_STATE_LABEL[state.cameraSync] ?? `CAM · ${state.cameraSync}`;
    if (text !== lastCamStateText) {
      camState.textContent = text;
      lastCamStateText = text;
    }
  }

  buildOverlayDom();

  // --- Store subscription ---
  function onEvent(e: AppEvent): void {
    switch (e.type) {
      case 'detection-revealed': {
        const det = state.detections.find((d) => d.id === e.id);
        if (det) showToast(det.label, det.kind);
        break;
      }
      case 'detection-focused':
        openCard(e.id);
        break;
      case 'camera-state':
        refreshCamState();
        if (e.state === 'SYNCED') closeCard();
        break;
      case 'reset':
        closeCard();
        hudBuilt = false;
        lastActiveDroneId = null;
        break;
      default:
        break;
    }
  }

  const unsubscribe = subscribe(onEvent);

  return {
    update(): void {
      refreshHud();
      refreshCamState();
    },
    dispose(): void {
      unsubscribe();
      btn?.removeEventListener('click', onConfirmClick);
      hudRoot?.replaceChildren();
      overlayRoot?.replaceChildren();
      droneRows.clear();
      card = kicker = cardLabel = conf = barFill = btn = camState = null;
    },
  };
}
