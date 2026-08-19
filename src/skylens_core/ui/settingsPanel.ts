// 설정 — the operator's display-option page (COMPONENTS.md §4).
//
// Three options over ONE scene. Picking one is a live material swap in the
// viewer, so there is no "apply" button and no reload: the map changes under
// the operator's cursor. The choice is written to localStorage immediately, so
// the tower comes back the way it was left.
//
// The panel also STATES WHERE THE DATA CAME FROM. PROJECT.md §7 draws a hard
// line between staging and reality, and "이 건물들은 진짜인가" is exactly the
// question an evaluator asks — so VWorld vs. stand-in is on screen, not buried
// in a console warning.

import { DISPLAY_MODES } from '../settings.ts';
import type { DisplayMode, SettingsStore } from '../settings.ts';
import type { BuildingSourceKind } from '../../shared/viewer/sources/buildingSource.ts';

export interface SettingsPanelOptions {
  settings: SettingsStore;
  /** Applied live; the panel never reloads the scene. */
  onDisplayChange(mode: DisplayMode): void;
  /** False when no aerial mosaic loaded — 실사 항공뷰 is then unavailable. */
  aerialAvailable: boolean;
  buildingSource: BuildingSourceKind;
  footprints: number;
  /** Human label of the loaded operating area. */
  areaLabel: string;
}

export interface SettingsPanel {
  open(): void;
  close(): void;
  toggle(): void;
  dispose(): void;
}

const SOURCE_LABEL: Record<BuildingSourceKind, string> = {
  vworld: 'VWorld 실측 건물 데이터',
  synthetic: '대체 블록 (데모 대체 데이터 · 실측 아님)',
  none: '건물 데이터 없음',
};

export function createSettingsPanel(opts: SettingsPanelOptions): SettingsPanel {
  const overlay = document.createElement('div');
  overlay.className = 'settings-overlay is-hidden';

  const panel = document.createElement('div');
  panel.className = 'sl-surface sl-surface--modal settings-panel';

  const title = document.createElement('div');
  title.className = 'settings-panel__title';
  title.textContent = '설정 · Display';

  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'settings-panel__close';
  closeBtn.textContent = '×';
  closeBtn.setAttribute('aria-label', '설정 닫기');

  const sectionLabel = document.createElement('div');
  sectionLabel.className = 'settings-panel__section';
  sectionLabel.textContent = '건물 표시 방식';

  const list = document.createElement('div');
  list.className = 'settings-panel__options';
  list.setAttribute('role', 'radiogroup');
  list.setAttribute('aria-label', '건물 표시 방식');

  const buttons = new Map<DisplayMode, HTMLButtonElement>();

  for (const mode of DISPLAY_MODES) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'settings-option';
    btn.dataset.mode = mode.id;
    btn.setAttribute('role', 'radio');

    const name = document.createElement('span');
    name.className = 'settings-option__name';
    name.textContent = mode.label;
    if (mode.id === 'black') {
      const badge = document.createElement('span');
      badge.className = 'settings-option__badge';
      badge.textContent = '기본값';
      name.appendChild(badge);
    }

    const hint = document.createElement('span');
    hint.className = 'settings-option__hint';
    hint.textContent = mode.hint;

    btn.append(name, hint);

    // An option that cannot be honored is disabled and says why, rather than
    // silently rendering something else when clicked.
    if (mode.id === 'aerial' && !opts.aerialAvailable) {
      btn.disabled = true;
      hint.textContent = '위성 영상을 불러올 수 없어 사용할 수 없습니다 (VWorld 키 확인)';
    }

    btn.addEventListener('click', () => {
      if (btn.disabled) return;
      opts.settings.setDisplay(mode.id);
    });

    buttons.set(mode.id, btn);
    list.appendChild(btn);
  }

  const meta = document.createElement('dl');
  meta.className = 'settings-panel__meta';
  const addMeta = (k: string, v: string, cls?: string): void => {
    const dt = document.createElement('dt');
    dt.textContent = k;
    const dd = document.createElement('dd');
    dd.textContent = v;
    if (cls) dd.className = cls;
    meta.append(dt, dd);
  };
  addMeta('작전 구역', opts.areaLabel);
  addMeta(
    '건물 출처',
    `${SOURCE_LABEL[opts.buildingSource]}${
      opts.footprints > 0 ? ` · ${opts.footprints.toLocaleString('ko-KR')}동` : ''
    }`,
    opts.buildingSource === 'synthetic' ? 'is-warn' : undefined,
  );
  addMeta('위성 영상', opts.aerialAvailable ? '사용 가능' : '사용 불가');

  const note = document.createElement('div');
  note.className = 'settings-panel__note';
  note.textContent = opts.settings.overridden
    ? 'URL(?display=)로 지정된 표시 방식입니다 — 저장된 설정은 변경되지 않습니다'
    : '선택은 이 브라우저에 저장되어 다음 접속 때 그대로 복원됩니다';

  panel.append(closeBtn, title, sectionLabel, list, meta, note);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  // Reflect the store, don't shadow it: the panel has no state of its own, so a
  // change from anywhere (URL override, another panel) still shows up here.
  const unsubscribe = opts.settings.subscribe((s) => {
    for (const [mode, btn] of buttons) {
      const on = mode === s.display;
      btn.classList.toggle('is-active', on);
      btn.setAttribute('aria-checked', String(on));
    }
    opts.onDisplayChange(s.display);
  });

  const close = (): void => overlay.classList.add('is-hidden');
  const open = (): void => overlay.classList.remove('is-hidden');

  closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  const onKey = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') close();
  };
  window.addEventListener('keydown', onKey);

  return {
    open,
    close,
    toggle(): void {
      overlay.classList.toggle('is-hidden');
    },
    dispose(): void {
      unsubscribe();
      window.removeEventListener('keydown', onKey);
      overlay.remove();
    },
  };
}
