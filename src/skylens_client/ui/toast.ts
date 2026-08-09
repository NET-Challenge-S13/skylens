// Transient toast notifications mounted at #toast-root. PROJECT.md UI overlay
// layer — pure DOM, no Three.js. Stacks multiple toasts; each removes itself
// after the CSS-driven lifetime elapses.

const TOAST_LIFETIME_MS = 3200;

export type ToastKind = 'person' | 'danger' | 'info';

/** Append a toast to #toast-root and auto-remove it after ~3.2s. */
export function showToast(message: string, kind: ToastKind = 'info'): void {
  const root = document.getElementById('toast-root');
  if (!root) return;

  const el = document.createElement('div');
  el.className = 'toast';
  if (kind === 'person' || kind === 'danger') {
    el.classList.add(`kind-${kind}`);
  }
  el.textContent = message;
  root.appendChild(el);

  setTimeout(() => {
    el.remove();
  }, TOAST_LIFETIME_MS);
}
