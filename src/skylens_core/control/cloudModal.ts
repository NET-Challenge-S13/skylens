// A small control-tower modal that lists the DJI Fly cloud missions and lets
// the operator delete one. It talks to the local mission bridge, which holds
// the account credentials; the browser never sees them.

import { showToast } from '../../shared/viewer/ui/toast.ts';
import { type CloudMission, deleteCloudMission, listCloudMissions } from '../missionBridge.ts';

export interface CloudModal {
  open(): void;
  close(): void;
  destroy(): void;
}

export function createCloudModal(): CloudModal {
  const overlay = document.createElement('div');
  overlay.className = 'route-modal-overlay is-hidden';

  const modal = document.createElement('div');
  modal.className = 'route-modal';
  overlay.append(modal);

  const title = document.createElement('h2');
  title.className = 'route-modal__title';
  title.textContent = '클라우드 미션';
  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'route-modal__close';
  closeBtn.textContent = '✕';
  closeBtn.setAttribute('aria-label', '닫기');
  title.append(closeBtn);

  const hint = document.createElement('p');
  hint.className = 'route-modal__hint';
  hint.textContent = 'DJI Fly 계정 클라우드의 웨이포인트 미션. 원격 RC 2가 동기화하면 받아간다.';

  const list = document.createElement('div');
  list.className = 'route-modal__list';

  const actions = document.createElement('div');
  actions.className = 'route-modal__actions';
  const refreshBtn = document.createElement('button');
  refreshBtn.type = 'button';
  refreshBtn.className = 'route-modal__btn';
  refreshBtn.textContent = '새로고침';
  actions.append(refreshBtn);

  modal.append(title, hint, list, actions);
  document.body.append(overlay);

  let busy = false;

  function renderEmpty(message: string): void {
    list.replaceChildren();
    const row = document.createElement('div');
    row.className = 'route-modal__item';
    row.textContent = message;
    list.append(row);
  }

  function renderMissions(missions: CloudMission[]): void {
    if (missions.length === 0) {
      renderEmpty('클라우드에 미션이 없습니다.');
      return;
    }
    list.replaceChildren();
    for (const mission of missions) {
      const row = document.createElement('div');
      row.className = 'route-modal__item';

      const label = document.createElement('span');
      const points = mission.waypointCount ?? '?';
      const dist = mission.distanceM != null ? ` · ${Math.round(mission.distanceM)}m` : '';
      label.textContent = `${mission.name || '(이름 없음)'} · ${points} WP${dist}`;

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'route-modal__remove';
      remove.textContent = '삭제';
      remove.addEventListener('click', () => void onDelete(mission, remove));

      row.append(label, remove);
      list.append(row);
    }
  }

  async function refresh(): Promise<void> {
    if (busy) return;
    busy = true;
    refreshBtn.disabled = true;
    renderEmpty('불러오는 중…');
    try {
      renderMissions(await listCloudMissions());
    } catch (error) {
      renderEmpty('불러오지 못했습니다.');
      showToast(`클라우드 미션 조회 실패 · ${String(error)}`, 'danger');
    } finally {
      busy = false;
      refreshBtn.disabled = false;
    }
  }

  async function onDelete(mission: CloudMission, button: HTMLButtonElement): Promise<void> {
    if (busy) return;
    busy = true;
    button.disabled = true;
    let removed = false;
    try {
      removed = await deleteCloudMission(mission.uuid);
      showToast(
        removed ? `삭제됨 · ${mission.name}` : `삭제 요청됨(미확인) · ${mission.name}`,
        removed ? 'info' : 'danger',
      );
    } catch (error) {
      showToast(`삭제 실패 · ${String(error)}`, 'danger');
      button.disabled = false;
    } finally {
      busy = false;
    }
    if (removed) await refresh();
  }

  function open(): void {
    overlay.classList.remove('is-hidden');
    void refresh();
  }
  function close(): void {
    overlay.classList.add('is-hidden');
  }
  function destroy(): void {
    overlay.remove();
  }

  refreshBtn.addEventListener('click', () => void refresh());
  closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', (event) => {
    if (event.target === overlay) close();
  });

  return { open, close, destroy };
}
