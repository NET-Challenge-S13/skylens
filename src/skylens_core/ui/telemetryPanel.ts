// Per-drone telemetry readout — in GPS, because that is what the tower thinks in.
//
// COMPONENTS.md §8: "관제탑의 좌표는 GPS다". These numbers are printed straight
// off `DroneTelemetry.gps` with no scene coordinate anywhere in the path, so
// what the operator reads is exactly what the drone reported and what a route
// waypoint would be compared against.

import { state, emit } from '../../shared/viewer/store.ts';
import { stationLabel } from './stationLabel.ts';
import { droneTint, markOperatorSelection } from '../drones/telemetryFleet.ts';
import type { FleetDrone } from '../drones/telemetryFleet.ts';

export interface TelemetryPanel {
  /** Re-render from the current fleet snapshot. */
  render(drones: FleetDrone[]): void;
  dispose(): void;
}

const hex = (n: number): string => `#${n.toString(16).padStart(6, '0')}`;

/** Signed decimal degrees at ~1 m resolution (5 dp ≈ 1.1 m at this latitude). */
const deg = (v: number): string => v.toFixed(5);

export function createTelemetryPanel(mount: HTMLElement): TelemetryPanel {
  const root = document.createElement('div');
  root.className = 'sl-surface sl-surface--panel telemetry-panel';

  const empty = document.createElement('div');
  empty.className = 'telemetry-panel__empty';
  empty.textContent = '연결된 드론이 없습니다';

  const listEl = document.createElement('ul');
  listEl.className = 'telemetry-panel__list';

  root.append(empty, listEl);
  mount.appendChild(root);

  /** Rows are rebuilt only when the drone SET changes; values update in place
   *  so a click target never moves under the operator's cursor. */
  interface Row {
    li: HTMLLIElement;
    name: HTMLElement;
    vals: HTMLElement;
  }
  const rows = new Map<number, Row>();

  const ensureRow = (d: FleetDrone): Row => {
    const existing = rows.get(d.id);
    if (existing) return existing;

    const li = document.createElement('li');
    li.className = 'telemetry-row';
    li.tabIndex = 0;

    const head = document.createElement('div');
    head.className = 'telemetry-row__head';
    const dot = document.createElement('span');
    dot.className = 'telemetry-row__dot';
    dot.style.background = hex(droneTint(d.id));
    const name = document.createElement('span');
    name.className = 'telemetry-row__name';
    name.textContent = stationLabel(d.station);
    const stale = document.createElement('span');
    stale.className = 'telemetry-row__stale';
    stale.textContent = '수신 끊김';
    head.append(dot, name, stale);

    const vals = document.createElement('div');
    vals.className = 'telemetry-row__vals';

    li.append(head, vals);
    // Selecting a drone is what the chase camera and manual control follow.
    const select = (): void => {
      // Even re-picking the same aircraft counts as a choice: it tells the
      // fleet to stop steering the selection back to the centre.
      markOperatorSelection();
      if (state.activeDroneId === d.id) return;
      state.activeDroneId = d.id;
      emit({ type: 'active-drone', id: d.id });
    };
    li.addEventListener('click', select);
    li.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        select();
      }
    });

    listEl.appendChild(li);
    const row: Row = { li, name, vals };
    rows.set(d.id, row);
    return row;
  };

  return {
    render(drones: FleetDrone[]): void {
      empty.classList.toggle('is-hidden', drones.length > 0);

      const live = new Set(drones.map((d) => d.id));
      for (const [id, row] of rows) {
        if (live.has(id)) continue;
        row.li.remove();
        rows.delete(id);
      }

      for (const d of drones) {
        const row = ensureRow(d);
        // The station can only change if the aircraft is reconfigured, but the
        // row is built once and the name would otherwise go stale.
        row.name.textContent = stationLabel(d.station);
        row.li.classList.toggle('is-active', d.id === state.activeDroneId);
        row.li.classList.toggle('is-stale', d.stale);
        row.vals.textContent =
          `${deg(d.gps.lat)}, ${deg(d.gps.lon)} · ${d.gps.alt.toFixed(0)}m · ` +
          `${d.headingDeg.toFixed(0)}° · ${d.speed.toFixed(1)}m/s · ${d.batteryPct.toFixed(0)}%`;
      }

      // Keep the list in the fleet's order — left, centre, right. Rows are
      // created when an aircraft FIRST reports, so on their own they read in
      // announcement order, and the wingmen usually announce before the centre:
      // 중앙 드론 ended up at the bottom of a list whose whole point is to read
      // like the formation looks.
      //
      // Only when it actually differs. Re-appending a row moves the node, which
      // drops keyboard focus and restarts transitions, and this runs every
      // frame.
      const desired = drones.map((d) => rows.get(d.id)?.li).filter((li) => li !== undefined);
      const current = [...listEl.children];
      const inOrder =
        desired.length === current.length && desired.every((li, i) => li === current[i]);
      if (!inOrder) for (const li of desired) listEl.appendChild(li);
    },

    dispose(): void {
      rows.clear();
      root.remove();
    },
  };
}
