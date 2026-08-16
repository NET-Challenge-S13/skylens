// Control-tower route planner. A DOM modal where the operator builds a GPS
// route (lat/lon/alt waypoints) for the leader drone and assigns it — the
// leader then flies the assigned route (see pathFollower.ts setLeaderRoute)
// and the swarm clusters around it.

import type { Gps } from '../geo.ts';

export interface AssignedRoute {
  droneId: number;
  waypoints: Gps[];
}

export interface RouteModalOptions {
  /** Anchor used to prefill a sensible default route nearby. */
  anchor: Gps;
  /** Drone id the route is assigned to (the leader / active drone). */
  getLeaderId: () => number;
  onAssign(route: AssignedRoute): void;
}

export interface RouteModal {
  open(): void;
  close(): void;
  dispose(): void;
}

/** A small default loop ~40-60m around the anchor, one click to try. */
function defaultWaypoints(anchor: Gps): Gps[] {
  const dLat = 0.00045; // ~50m
  const dLon = 0.00055; // ~50m at mid latitudes
  return [
    { lat: anchor.lat, lon: anchor.lon, alt: anchor.alt + 20 },
    { lat: anchor.lat + dLat, lon: anchor.lon, alt: anchor.alt + 20 },
    { lat: anchor.lat + dLat, lon: anchor.lon + dLon, alt: anchor.alt + 22 },
    { lat: anchor.lat, lon: anchor.lon + dLon, alt: anchor.alt + 20 },
    { lat: anchor.lat, lon: anchor.lon, alt: anchor.alt + 20 },
  ];
}

export function createRouteModal(opts: RouteModalOptions): RouteModal {
  let waypoints: Gps[] = defaultWaypoints(opts.anchor);

  const overlay = document.createElement('div');
  overlay.className = 'route-modal-overlay is-hidden';

  const modal = document.createElement('div');
  modal.className = 'route-modal';

  const title = document.createElement('div');
  title.className = 'route-modal__title';
  title.textContent = '경로 계획 · Route Planner';

  const form = document.createElement('div');
  form.className = 'route-modal__form';

  const latInput = document.createElement('input');
  latInput.type = 'number';
  latInput.step = 'any';
  latInput.placeholder = 'lat';
  latInput.value = String(opts.anchor.lat);

  const lonInput = document.createElement('input');
  lonInput.type = 'number';
  lonInput.step = 'any';
  lonInput.placeholder = 'lon';
  lonInput.value = String(opts.anchor.lon);

  const altInput = document.createElement('input');
  altInput.type = 'number';
  altInput.step = 'any';
  altInput.placeholder = 'alt (m)';
  altInput.value = String(opts.anchor.alt + 20);

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.className = 'route-modal__btn';
  addBtn.textContent = '웨이포인트 추가';

  form.append(latInput, lonInput, altInput, addBtn);

  const list = document.createElement('ol');
  list.className = 'route-modal__list';

  const actions = document.createElement('div');
  actions.className = 'route-modal__actions';

  const clearBtn = document.createElement('button');
  clearBtn.type = 'button';
  clearBtn.className = 'route-modal__btn route-modal__btn--ghost';
  clearBtn.textContent = 'Clear';

  const assignBtn = document.createElement('button');
  assignBtn.type = 'button';
  assignBtn.className = 'route-modal__btn route-modal__btn--primary';
  assignBtn.textContent = 'Assign';

  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'route-modal__close';
  closeBtn.textContent = '×';

  actions.append(clearBtn, assignBtn);
  modal.append(closeBtn, title, form, list, actions);
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  function renderList(): void {
    list.innerHTML = '';
    waypoints.forEach((wp, i) => {
      const li = document.createElement('li');
      li.className = 'route-modal__item';

      const label = document.createElement('span');
      label.textContent = `#${i + 1}  ${wp.lat.toFixed(5)}, ${wp.lon.toFixed(5)}  · ${wp.alt.toFixed(0)}m`;

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'route-modal__remove';
      removeBtn.textContent = '삭제';
      removeBtn.addEventListener('click', () => {
        waypoints = waypoints.filter((_, idx) => idx !== i);
        renderList();
      });

      li.append(label, removeBtn);
      list.appendChild(li);
    });
  }
  renderList();

  addBtn.addEventListener('click', () => {
    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);
    const alt = parseFloat(altInput.value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(alt)) return;
    waypoints = [...waypoints, { lat, lon, alt }];
    renderList();
  });

  clearBtn.addEventListener('click', () => {
    waypoints = [];
    renderList();
  });

  function close(): void {
    overlay.classList.add('is-hidden');
  }

  assignBtn.addEventListener('click', () => {
    if (waypoints.length < 2) return;
    opts.onAssign({ droneId: opts.getLeaderId(), waypoints: [...waypoints] });
    close();
  });

  closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });

  return {
    open(): void {
      overlay.classList.remove('is-hidden');
    },
    close,
    dispose(): void {
      overlay.remove();
    },
  };
}
