// Full-screen loading overlay shown while the scene (splat) downloads + builds.

export interface LoadingScreen {
  progress(percent: number, label?: string): void;
  done(): void;
  fail(message: string): void;
}

export function createLoadingScreen(title: string): LoadingScreen {
  const el = document.createElement('div');
  el.className = 'loading-screen';

  const box = document.createElement('div');
  box.className = 'loading-screen__box';

  const h = document.createElement('div');
  h.className = 'loading-screen__title';
  h.textContent = title;

  const bar = document.createElement('div');
  bar.className = 'loading-screen__bar';
  const fill = document.createElement('span');
  bar.appendChild(fill);

  const txt = document.createElement('div');
  txt.className = 'loading-screen__text';

  box.append(h, bar, txt);
  el.appendChild(box);
  document.body.appendChild(el);

  const remove = (): void => {
    el.classList.add('is-hidden');
    setTimeout(() => el.remove(), 400);
  };

  return {
    progress(percent, label) {
      const p = Math.max(0, Math.min(100, percent));
      fill.style.width = `${p}%`;
      txt.textContent = label ?? `${Math.round(p)}%`;
    },
    done: remove,
    fail(message) {
      h.textContent = '장면 로드 실패';
      txt.textContent = message;
      el.classList.add('is-error');
      setTimeout(remove, 2500);
    },
  };
}
