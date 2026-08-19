// One place that turns a formation station into the words on screen.
//
// The operator identifies aircraft by where they fly, not by an id: "왼쪽 드론"
// is something you can point at over the map. The numeric droneId stays as the
// routing address on the wire and never reaches the screen.

import type { DroneStation } from '../../shared/protocol.ts';

const KOREAN: Record<DroneStation, string> = {
  left: '좌측 드론',
  center: '중앙 드론',
  right: '우측 드론',
};

const CAM: Record<DroneStation, string> = {
  left: 'LEFT CAM',
  center: 'CENTER CAM',
  right: 'RIGHT CAM',
};

/** Fleet list / telemetry readout. */
export function stationLabel(station: DroneStation): string {
  return KOREAN[station] ?? KOREAN.center;
}

/** Camera panel header — names the feed the operator is looking at. */
export function stationCamLabel(station: DroneStation): string {
  return CAM[station] ?? CAM.center;
}
