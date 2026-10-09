// Board toggle for the point layer ("점 · 고도색").
//
// Shows how much of the point layer has arrived even while it is off, so the
// operator can tell "nothing delivered yet" from "delivered, not shown".

import type { StatusViewer } from '../statusview/statusViewer.ts';

export function mountPointToggle(status: StatusViewer): { update(): void } {
  const pane = document.querySelector('.pane');
  if (!(pane instanceof HTMLElement)) return { update() {} };

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.id = 'point-toggle';
  btn.className = 'point-toggle';
  const label = document.createElement('span');
  label.className = 'point-toggle__label';
  label.textContent = '점 · 고도색';
  const meta = document.createElement('span');
  meta.className = 'point-toggle__meta';
  btn.append(label, meta);
  pane.appendChild(btn);

  btn.addEventListener('click', () => {
    status.setPointLayerEnabled(!status.pointLayerEnabled);
    update();
  });

  let last = '';
  function update(): void {
    const st = status.pointStats;
    const on = status.pointLayerEnabled;
    const segs = st?.segments ?? [];
    const delivered = segs.filter((s) => s.shown > 0 || s.wanted > 0 || s.loading > 0).length;
    const refined = segs.filter((s) => s.shown >= 2).length;
    let text: string;
    if (segs.length === 0) text = '수신 대기';
    else if (!on) text = `${delivered}구간 수신`;
    else {
      const loading = segs.some((s) => s.loading > 0) ? ' · 로딩' : '';
      const pts = st ? `${(st.points / 10_000).toFixed(0)}만 점` : '';
      text = `${delivered}구간 · 정밀 ${refined} · ${pts}${loading}`;
    }
    const key = `${on}|${text}`;
    if (key === last) return;
    last = key;
    meta.textContent = text;
    btn.classList.toggle('is-on', on);
    btn.setAttribute('aria-pressed', String(on));
  }
  update();
  return { update };
}
