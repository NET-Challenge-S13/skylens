// Control-tower operator settings (COMPONENTS.md §4).
//
// The tower has ONE scene — VWorld terrain plus real building data — and the
// operator chooses how the buildings on it are DRAWN. That choice is a display
// option, not a different scene, so nothing here reloads data: the settings
// panel writes a mode, the viewer swaps materials, and the same geometry keeps
// rendering. Persisted so the screen comes back the way the operator left it.

/**
 * 점        — VWorld 건물을 포인트로 표시
 * black     — 검정 텍스처 건물 (기본값)
 * aerial    — 실사 항공뷰 (위성/항공 영상을 건물에도 그대로 적용)
 */
export type DisplayMode = 'points' | 'black' | 'aerial';

export const DISPLAY_MODES: ReadonlyArray<{
  id: DisplayMode;
  label: string;
  hint: string;
}> = [
  { id: 'points', label: '점', hint: 'VWorld 건물을 포인트로 표시' },
  { id: 'black', label: '검정 텍스처 건물', hint: '건물에 검정 텍스처를 적용해 표시' },
  { id: 'aerial', label: '실사 항공뷰', hint: '위성/항공 영상을 그대로 적용' },
];

export interface ControlSettings {
  display: DisplayMode;
}

const KEY = 'skylens.control.settings.v1';
const DEFAULT: ControlSettings = { display: 'black' };

function isDisplayMode(v: unknown): v is DisplayMode {
  return v === 'points' || v === 'black' || v === 'aerial';
}

/**
 * `?display=` wins over the stored value for one load (demo/test override) and
 * is NOT persisted — a URL used to show one thing must not silently rewrite the
 * operator's saved preference.
 */
function fromQuery(): DisplayMode | null {
  if (typeof window === 'undefined') return null;
  const q = new URLSearchParams(window.location.search).get('display');
  return isDisplayMode(q) ? q : null;
}

function read(): ControlSettings {
  if (typeof localStorage === 'undefined') return { ...DEFAULT };
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT };
    const parsed: unknown = JSON.parse(raw);
    const display = (parsed as Partial<ControlSettings> | null)?.display;
    return { display: isDisplayMode(display) ? display : DEFAULT.display };
  } catch {
    // Corrupt/blocked storage must never stop the tower from starting.
    return { ...DEFAULT };
  }
}

export interface SettingsStore {
  readonly value: ControlSettings;
  /** Applied for this load only; does NOT overwrite what is stored. */
  readonly overridden: boolean;
  setDisplay(mode: DisplayMode): void;
  /** Fires on every change, including the first subscribe (current value). */
  subscribe(fn: (s: ControlSettings) => void): () => void;
}

export function createSettings(): SettingsStore {
  const override = fromQuery();
  const current: ControlSettings = override ? { display: override } : read();
  const listeners = new Set<(s: ControlSettings) => void>();

  const persist = (): void => {
    if (typeof localStorage === 'undefined') return;
    try {
      localStorage.setItem(KEY, JSON.stringify(current));
    } catch {
      // Private mode / quota — the session still works, it just won't be restored.
    }
  };

  return {
    get value() {
      return { ...current };
    },
    overridden: override != null,

    setDisplay(mode: DisplayMode): void {
      if (current.display === mode) return;
      current.display = mode;
      persist();
      for (const fn of listeners) fn({ ...current });
    },

    subscribe(fn: (s: ControlSettings) => void): () => void {
      listeners.add(fn);
      fn({ ...current });
      return () => listeners.delete(fn);
    },
  };
}
