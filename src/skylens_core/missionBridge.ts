// Browser client for the localhost DJI Fly mission bridge.
//
// A browser cannot write into an Android/RC MTP directory. The local bridge
// validates the route, creates a Lito-identified KMZ, and either returns it as
// a download or installs it to one explicitly configured DJI Fly mission file.

import type { Gps } from '../shared/geo.ts';

export interface MissionExportRoute {
  droneId: number;
  waypoints: Gps[];
  loop: boolean;
  /** Operator-entered height above ground from the route planner. */
  aglM: number;
  /** Also upload to the DJI Fly account cloud for remote RC pickup. */
  cloud?: boolean;
}

/** Cloud upload outcome, present only when cloud delivery was requested. */
export interface CloudDelivery {
  missionUuid: string;
  verified: boolean;
}

export interface MissionExportResult {
  fileName: string;
  downloadUrl: string;
  waypointCount: number;
  routeDistanceM: number;
  takeoffAltMsl: number;
  installed: boolean;
  installTarget: string | null;
  warnings: string[];
  cloud?: CloudDelivery;
}

function bridgeUrl(): string {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8091';
  const override = new URLSearchParams(window.location.search).get('missionBridge');
  return (override || 'http://127.0.0.1:8091').replace(/\/+$/, '');
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { error?: unknown };
    if (typeof body.error === 'string') return body.error;
  } catch {
    // Status below is still more useful than hiding an unexpected response.
  }
  return `Mission Bridge HTTP ${response.status}`;
}

async function download(url: string, fileName: string): Promise<void> {
  const response = await fetch(url, { signal: AbortSignal.timeout(10_000) });
  if (!response.ok) throw new Error(await errorMessage(response));
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement('a');
    anchor.href = objectUrl;
    anchor.download = fileName;
    anchor.style.display = 'none';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  }
}

export interface FormationMember {
  station: 'left' | 'center' | 'right';
  fileName: string;
  downloadUrl: string;
  waypointCount: number;
  routeDistanceM: number;
  takeoffAltMsl: number;
  warnings: string[];
  cloud?: CloudDelivery;
}

const STATION_KO: Record<FormationMember['station'], string> = {
  left: '앞·왼쪽',
  right: '앞·오른쪽',
  center: '뒤·중앙',
};

/**
 * Fan one planned route into three DJI Fly missions — the equilateral-triangle
 * formation (front-left / front-right / rear-centre) the crew used to build by
 * hand in a spreadsheet. The planned route is the formation centroid; the bridge
 * offsets each member and returns three KMZ, which we download in station order.
 */
export async function exportFormation(
  route: MissionExportRoute & { spacingM: number },
): Promise<FormationMember[]> {
  const endpoint = bridgeUrl();
  let response: Response;
  try {
    response = await fetch(`${endpoint}/formation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'skylens-formation',
        waypoints: route.waypoints,
        loop: route.loop,
        flight: {
          aglM: route.aglM,
          speedMps: 2,
          gimbalPitchDeg: -30,
          rthHeightM: Math.max(30, route.aglM),
        },
        formation: { spacingM: route.spacingM },
        deliver: route.cloud ? 'cloud' : undefined,
      }),
      signal: AbortSignal.timeout(route.cloud ? 90_000 : 15_000),
    });
  } catch (error) {
    throw new Error(
      `Mission Bridge에 연결할 수 없습니다 (${endpoint}). npm run mission-bridge를 실행하세요. ` +
        String(error),
    );
  }
  if (!response.ok) throw new Error(await errorMessage(response));
  const result = (await response.json()) as { members?: FormationMember[] };
  const members = result.members ?? [];
  if (members.length === 0) throw new Error('Mission Bridge 응답에 편대 미션이 없습니다');
  // Sequential downloads: three anchor clicks fired in one tick can drop files.
  for (const member of members) {
    await download(member.downloadUrl, `${member.fileName}`);
  }
  return members;
}

/** The Korean formation-station name for operator toasts and logs. */
export function stationName(station: FormationMember['station']): string {
  return STATION_KO[station] ?? station;
}

export async function exportDjiMission(route: MissionExportRoute): Promise<MissionExportResult> {
  const endpoint = bridgeUrl();
  let response: Response;
  try {
    response = await fetch(`${endpoint}/missions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: `skylens-drone-${route.droneId}`,
        droneId: route.droneId,
        waypoints: route.waypoints,
        loop: route.loop,
        flight: {
          aglM: route.aglM,
          speedMps: 2,
          gimbalPitchDeg: -30,
          rthHeightM: Math.max(30, route.aglM),
        },
        deliver: route.cloud ? 'cloud' : undefined,
      }),
      // Cloud upload (STS + S3 + verify) takes longer than a local build.
      signal: AbortSignal.timeout(route.cloud ? 90_000 : 10_000),
    });
  } catch (error) {
    throw new Error(
      `Mission Bridge에 연결할 수 없습니다 (${endpoint}). npm run mission-bridge를 실행하세요. ` +
        String(error),
    );
  }
  if (!response.ok) throw new Error(await errorMessage(response));
  const result = (await response.json()) as MissionExportResult & { ok?: boolean };
  if (!result.fileName || !result.downloadUrl) {
    throw new Error('Mission Bridge 응답에 KMZ 정보가 없습니다');
  }
  if (!result.installed) await download(result.downloadUrl, result.fileName);
  return result;
}
