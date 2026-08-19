// Control-tower route planner. A DOM modal with a small top-down MAP: the
// operator clicks points on the map to drop GPS waypoints for the leader drone
// and assigns the route.
//
// The route leaves here as GPS and goes straight onto the wire as `assign-route`
// (coreLink.ts). The tower does NOT fly the drone itself — the core owns the
// flight, and the result comes back as telemetry. This modal therefore produces
// coordinates, not motion.
//
// The map shows a VWorld satellite backdrop when the dev proxy has a key
// (/vworld/wmts/{z}/{y}/{x}.jpeg, same source as the ?map terrain scene); with
// no key it degrades to a schematic grid. Clicks are converted to GPS from a
// linear lon/lat frame centered on the anchor, so waypoint coordinates are
// exact regardless of whether the satellite layer loaded.

import type { Gps } from '../../shared/geo.ts';

export interface AssignedRoute {
  droneId: number;
  waypoints: Gps[];
  /** Fly the route back and forth (데모 시나리오 §5.2 "지정 경로 왕복 반복").
   *  Sent EXPLICITLY on the wire: the core defaults a missing `loop` to true,
   *  but it should never have to guess what the operator meant. */
  loop: boolean;
}

export interface RouteModalOptions {
  /** Anchor used to center the map and prefill a sensible altitude. */
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

interface Waypoint extends Gps {}

// Map drawing buffer (device px). CSS scales it to fit the modal.
const MAP_PX = 460;
// Selectable ground spans (meters, full width of the map).
const SPANS = [300, 600, 1200, 2500];
const HIT_RADIUS = 12; // px — click within this of a marker removes it.

// --- Web Mercator tile math (matches terrainSource.ts) ---
const p2 = (z: number): number => Math.pow(2, z);
const lonToTileX = (lon: number, z: number): number => ((lon + 180) / 360) * p2(z);
const latToTileY = (lat: number, z: number): number => {
  const r = (lat * Math.PI) / 180;
  return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * p2(z);
};
const tileXToLon = (x: number, z: number): number => (x / p2(z)) * 360 - 180;
const tileYToLat = (y: number, z: number): number => {
  const n = Math.PI - (2 * Math.PI * y) / p2(z);
  return (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
};
const SAT_MAX_ZOOM = 17;
/** VWorld serves tilematrix 6..19 only (see terrainSource.ts). */
const SAT_MIN_ZOOM = 6;

interface Bounds {
  west: number;
  east: number;
  south: number;
  north: number;
}
interface LoadedTile {
  img: HTMLImageElement;
  z: number;
  tx: number;
  ty: number;
}

export function createRouteModal(opts: RouteModalOptions): RouteModal {
  const anchor = opts.anchor;
  const cosLat = Math.cos((anchor.lat * Math.PI) / 180) || 1;
  let waypoints: Waypoint[] = [];
  let loop = true;
  let spanM = SPANS[1];
  let altitude = Math.round(anchor.alt + 20);
  let satTiles: LoadedTile[] = [];
  let satToken = 0; // invalidates stale tile loads on span change / close.

  // --- geo <-> pixel mapping (linear lon/lat frame around the anchor) ---
  function bounds(): Bounds {
    const dLat = spanM / 111320;
    const dLon = spanM / (111320 * cosLat);
    return {
      west: anchor.lon - dLon / 2,
      east: anchor.lon + dLon / 2,
      south: anchor.lat - dLat / 2,
      north: anchor.lat + dLat / 2,
    };
  }
  function gpsToPx(lat: number, lon: number): [number, number] {
    const b = bounds();
    return [
      ((lon - b.west) / (b.east - b.west)) * MAP_PX,
      ((b.north - lat) / (b.north - b.south)) * MAP_PX,
    ];
  }
  function pxToGps(x: number, y: number): Gps {
    const b = bounds();
    return {
      lat: b.north - (y / MAP_PX) * (b.north - b.south),
      lon: b.west + (x / MAP_PX) * (b.east - b.west),
      alt: altitude,
    };
  }

  // ---------- DOM ----------
  const overlay = document.createElement('div');
  overlay.className = 'route-modal-overlay is-hidden';

  const modal = document.createElement('div');
  modal.className = 'route-modal';

  const title = document.createElement('div');
  title.className = 'route-modal__title';
  title.textContent = '경로 계획 · Route Planner';

  const hint = document.createElement('div');
  hint.className = 'route-modal__hint';
  hint.textContent = '지도를 클릭해 웨이포인트를 추가하세요 · 마커를 클릭하면 삭제됩니다';

  // toolbar: span (zoom) presets + altitude slider
  const toolbar = document.createElement('div');
  toolbar.className = 'route-modal__toolbar';

  const spanGroup = document.createElement('div');
  spanGroup.className = 'route-modal__spans';
  const spanBtns: HTMLButtonElement[] = SPANS.map((s) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'route-modal__span';
    b.textContent = s >= 1000 ? `${s / 1000}km` : `${s}m`;
    b.addEventListener('click', () => {
      spanM = s;
      loadSatellite();
      syncSpanBtns();
      draw();
    });
    spanGroup.appendChild(b);
    return b;
  });
  function syncSpanBtns(): void {
    spanBtns.forEach((b, i) => b.classList.toggle('is-active', SPANS[i] === spanM));
  }

  const altWrap = document.createElement('label');
  altWrap.className = 'route-modal__alt';
  const altText = document.createElement('span');
  const altInput = document.createElement('input');
  altInput.type = 'range';
  altInput.min = String(Math.round(anchor.alt + 5));
  altInput.max = String(Math.round(anchor.alt + 150));
  altInput.step = '1';
  altInput.value = String(altitude);
  function syncAltText(): void {
    altText.textContent = `고도 ${altitude}m`;
  }
  altInput.addEventListener('input', () => {
    altitude = parseInt(altInput.value, 10);
    syncAltText();
  });
  altWrap.append(altText, altInput);

  const loopWrap = document.createElement('label');
  loopWrap.className = 'route-modal__loop';
  const loopInput = document.createElement('input');
  loopInput.type = 'checkbox';
  loopInput.checked = loop;
  const loopText = document.createElement('span');
  loopText.textContent = '왕복 반복';
  loopInput.addEventListener('change', () => {
    loop = loopInput.checked;
  });
  loopWrap.append(loopInput, loopText);

  toolbar.append(spanGroup, altWrap, loopWrap);

  // map canvas
  const mapWrap = document.createElement('div');
  mapWrap.className = 'route-modal__map';
  const canvas = document.createElement('canvas');
  canvas.className = 'route-modal__canvas';
  canvas.width = MAP_PX;
  canvas.height = MAP_PX;
  mapWrap.appendChild(canvas);
  const ctx = canvas.getContext('2d');

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
  actions.append(clearBtn, assignBtn);

  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'route-modal__close';
  closeBtn.textContent = '×';

  modal.append(closeBtn, title, hint, toolbar, mapWrap, list, actions);
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  // ---------- satellite backdrop ----------
  function pickZoom(b: Bounds): number {
    for (let z = SAT_MAX_ZOOM; z > SAT_MIN_ZOOM; z--) {
      const cols = Math.floor(lonToTileX(b.east, z)) - Math.floor(lonToTileX(b.west, z)) + 1;
      const rows = Math.floor(latToTileY(b.south, z)) - Math.floor(latToTileY(b.north, z)) + 1;
      if (cols * rows <= 12) return z;
    }
    return SAT_MIN_ZOOM;
  }
  function loadSatellite(): void {
    const token = ++satToken;
    satTiles = [];
    const b = bounds();
    const z = pickZoom(b);
    const tx0 = Math.floor(lonToTileX(b.west, z));
    const tx1 = Math.floor(lonToTileX(b.east, z));
    const ty0 = Math.floor(latToTileY(b.north, z)); // north → smaller y
    const ty1 = Math.floor(latToTileY(b.south, z));
    for (let tx = tx0; tx <= tx1; tx++) {
      for (let ty = ty0; ty <= ty1; ty++) {
        const img = new Image();
        img.addEventListener('load', () => {
          if (token !== satToken) return; // span changed / closed
          satTiles.push({ img, z, tx, ty });
          draw();
        });
        // onerror (no VWorld key → proxy 404): silently keep the grid.
        img.src = `/vworld/wmts/${z}/${ty}/${tx}.jpeg`;
      }
    }
  }

  // ---------- drawing ----------
  function drawGrid(): void {
    if (!ctx) return;
    ctx.fillStyle = '#0d1522';
    ctx.fillRect(0, 0, MAP_PX, MAP_PX);
    ctx.strokeStyle = 'rgba(120,170,220,0.14)';
    ctx.lineWidth = 1;
    const step = MAP_PX / 8;
    for (let i = 1; i < 8; i++) {
      ctx.beginPath();
      ctx.moveTo(i * step, 0);
      ctx.lineTo(i * step, MAP_PX);
      ctx.moveTo(0, i * step);
      ctx.lineTo(MAP_PX, i * step);
      ctx.stroke();
    }
  }
  function drawSatellite(): void {
    if (!ctx) return;
    for (const t of satTiles) {
      const nwLon = tileXToLon(t.tx, t.z);
      const nwLat = tileYToLat(t.ty, t.z);
      const seLon = tileXToLon(t.tx + 1, t.z);
      const seLat = tileYToLat(t.ty + 1, t.z);
      const [x0, y0] = gpsToPx(nwLat, nwLon);
      const [x1, y1] = gpsToPx(seLat, seLon);
      ctx.drawImage(t.img, x0, y0, x1 - x0, y1 - y0);
    }
  }
  function drawScaleAndCompass(): void {
    if (!ctx) return;
    // scale bar: quarter of the map width
    const meters = spanM / 4;
    const barPx = MAP_PX / 4;
    const x = 14;
    const y = MAP_PX - 18;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + barPx, y);
    ctx.moveTo(x, y - 4);
    ctx.lineTo(x, y + 4);
    ctx.moveTo(x + barPx, y - 4);
    ctx.lineTo(x + barPx, y + 4);
    ctx.stroke();
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(`${meters >= 1000 ? meters / 1000 + 'km' : Math.round(meters) + 'm'}`, x, y - 7);
    // compass N
    ctx.fillText('N', MAP_PX - 18, 18);
    ctx.beginPath();
    ctx.moveTo(MAP_PX - 14, 22);
    ctx.lineTo(MAP_PX - 14, 34);
    ctx.stroke();
  }
  function drawRoute(): void {
    if (!ctx) return;
    // anchor cross (map center)
    const [ax, ay] = gpsToPx(anchor.lat, anchor.lon);
    ctx.strokeStyle = 'rgba(120,200,255,0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ax - 6, ay);
    ctx.lineTo(ax + 6, ay);
    ctx.moveTo(ax, ay - 6);
    ctx.lineTo(ax, ay + 6);
    ctx.stroke();

    // polyline
    if (waypoints.length > 1) {
      ctx.strokeStyle = 'rgba(74,222,128,0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      waypoints.forEach((wp, i) => {
        const [x, y] = gpsToPx(wp.lat, wp.lon);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    // numbered markers
    waypoints.forEach((wp, i) => {
      const [x, y] = gpsToPx(wp.lat, wp.lon);
      ctx.fillStyle = i === 0 ? '#4ade80' : '#ffcc55';
      ctx.strokeStyle = 'rgba(0,0,0,0.7)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#0a0f18';
      ctx.font = 'bold 11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), x, y);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';
    });
  }
  function draw(): void {
    if (!ctx) return;
    drawGrid();
    if (satTiles.length) drawSatellite();
    // faint grid over satellite for readability
    if (satTiles.length) {
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      const step = MAP_PX / 8;
      for (let i = 1; i < 8; i++) {
        ctx.beginPath();
        ctx.moveTo(i * step, 0);
        ctx.lineTo(i * step, MAP_PX);
        ctx.moveTo(0, i * step);
        ctx.lineTo(MAP_PX, i * step);
        ctx.stroke();
      }
    }
    drawScaleAndCompass();
    drawRoute();
  }

  function renderList(): void {
    list.innerHTML = '';
    waypoints.forEach((wp, i) => {
      const li = document.createElement('li');
      li.className = 'route-modal__item';
      const label = document.createElement('span');
      label.textContent = `#${i + 1}  ${wp.lat.toFixed(5)}, ${wp.lon.toFixed(5)} · ${wp.alt.toFixed(0)}m`;
      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'route-modal__remove';
      removeBtn.textContent = '삭제';
      removeBtn.addEventListener('click', () => {
        waypoints = waypoints.filter((_, idx) => idx !== i);
        renderList();
        draw();
      });
      li.append(label, removeBtn);
      list.appendChild(li);
    });
  }

  // ---------- interaction ----------
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * MAP_PX;
    const y = ((e.clientY - rect.top) / rect.height) * MAP_PX;
    // click on an existing marker → remove it
    for (let i = 0; i < waypoints.length; i++) {
      const [wx, wy] = gpsToPx(waypoints[i].lat, waypoints[i].lon);
      if (Math.hypot(wx - x, wy - y) <= HIT_RADIUS) {
        waypoints = waypoints.filter((_, idx) => idx !== i);
        renderList();
        draw();
        return;
      }
    }
    // otherwise add a waypoint at the clicked point
    waypoints = [...waypoints, pxToGps(x, y)];
    renderList();
    draw();
  });

  clearBtn.addEventListener('click', () => {
    waypoints = [];
    renderList();
    draw();
  });

  function close(): void {
    satToken++; // cancel any in-flight tile loads
    overlay.classList.add('is-hidden');
  }

  assignBtn.addEventListener('click', () => {
    // The core refuses a route with fewer than 2 waypoints and answers with a
    // MissionStatus. Catching it here means the operator gets the reason
    // immediately instead of watching the modal close on a rejected task.
    if (waypoints.length < 2) {
      hint.textContent = '경로에는 최소 2개의 웨이포인트가 필요합니다';
      hint.classList.add('is-warn');
      return;
    }
    opts.onAssign({ droneId: opts.getLeaderId(), waypoints: [...waypoints], loop });
    close();
  });

  closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });

  syncSpanBtns();
  syncAltText();
  renderList();

  return {
    open(): void {
      hint.textContent = '지도를 클릭해 웨이포인트를 추가하세요 · 마커를 클릭하면 삭제됩니다';
      hint.classList.remove('is-warn');
      overlay.classList.remove('is-hidden');
      loadSatellite();
      draw();
    },
    close,
    dispose(): void {
      satToken++;
      overlay.remove();
    },
  };
}
