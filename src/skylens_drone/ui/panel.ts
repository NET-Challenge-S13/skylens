// Operator panel: everything the pilot needs to trust the link.
//
// Reads a DroneSnapshot and nothing else, so the Tauri window and a plain
// browser tab render identically. Five blocks, in the order the demo exercises
// them: connection · mode · assigned route · telemetry · capture/transmit.

import type { DroneSnapshot } from '../core/drone.ts';
import type { LinkState } from '../core/link.ts';

const PHASE_LABEL: Record<DroneSnapshot['phase'], string> = {
  offline: '오프라인',
  idle: '대기',
  transit: '현장 이동',
  flying: '경로 비행',
  manual: '수동 조종',
  holding: '지점 대기',
};

const LINK_LABEL: Record<LinkState['phase'], string> = {
  offline: '끊김',
  connecting: '접속 중',
  punching: '홀펀칭 중',
  connected: '연결됨',
  reconnecting: '재접속 중',
};

const CARRIER_LABEL: Record<LinkState['carrier'], string> = {
  none: '경로 없음',
  ws: '게이트웨이 릴레이',
  direct: '프록시 직결',
};

function el(tag: string, cls?: string, text?: string): HTMLElement {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** One label/value row. */
function row(label: string, value: string, mod?: string): HTMLElement {
  const wrap = el('div', `row${mod ? ` row--${mod}` : ''}`);
  wrap.append(el('span', 'row__k', label), el('span', 'row__v', value));
  return wrap;
}

function card(title: string, badge?: { text: string; tone: string }): HTMLElement {
  const node = el('section', 'card');
  const head = el('header', 'card__head');
  head.append(el('h2', 'card__title', title));
  if (badge) head.append(el('span', `pill pill--${badge.tone}`, badge.text));
  node.append(head);
  return node;
}

function fmtMeters(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(0)} m`;
}

function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} kB`;
  return `${b} B`;
}

function linkTone(link: LinkState): string {
  if (link.phase === 'connected') return 'ok';
  if (link.phase === 'punching' || link.phase === 'connecting' || link.phase === 'reconnecting') {
    return 'warn';
  }
  return 'bad';
}

export class OperatorPanel {
  private root: HTMLElement;

  constructor(root: HTMLElement) {
    this.root = root;
  }

  render(snap: DroneSnapshot): void {
    const frag = document.createDocumentFragment();
    frag.append(this.header(snap));
    frag.append(this.connection(snap));
    frag.append(this.mission(snap));
    frag.append(this.telemetry(snap));
    frag.append(this.capture(snap));
    this.root.replaceChildren(frag);
  }

  private header(snap: DroneSnapshot): HTMLElement {
    const head = el('header', 'drone__head');
    head.append(el('h1', 'drone__title', `드론 #${snap.droneId}`));
    head.append(el('span', 'drone__model', snap.model));
    head.append(el('span', `pill pill--${snap.phase === 'flying' ? 'ok' : 'idle'}`, PHASE_LABEL[snap.phase]));
    head.append(el('p', 'drone__note', snap.note));
    return head;
  }

  private connection(snap: DroneSnapshot): HTMLElement {
    const link = snap.link;
    const node = card('연결', { text: LINK_LABEL[link.phase], tone: linkTone(link) });
    node.append(row('접속 모드', link.mode === 'relay' ? 'relay (게이트웨이 릴레이)' : 'webrtc (홀펀칭 직결)'));
    node.append(row('게이트웨이', link.url));
    node.append(row('현재 반송로', CARRIER_LABEL[link.carrier]));
    if (link.mode === 'webrtc') {
      node.append(row('펀치 세션', link.sessionId ?? '—'));
      node.append(row('직결 종단', link.directUrl ?? '— (아직 협상 중)'));
    }
    node.append(row('DroneHello', snap.announced ? '전송 완료' : '미전송'));
    node.append(row('업링크 프레임', `${link.sent} 전송 · ${link.dropped} 유실 · 재접속 ${link.attempts}회`));
    if (link.lastError) node.append(row('최근 오류', link.lastError, 'bad'));
    return node;
  }

  private mission(snap: DroneSnapshot): HTMLElement {
    const node = card('지정 경로');
    const route = snap.route;
    if (!route) {
      node.append(el('p', 'empty', '경로 미지정 — 관제탑의 지정을 기다리는 중'));
      return node;
    }
    node.append(row('경유점', `${route.waypoints.length}개`));
    node.append(row('왕복 반복', route.loop ? '예 (loop)' : '아니오 (편도)'));
    node.append(row('편도 길이', fmtMeters(snap.routeLengthM)));
    node.append(
      row('누적 비행', `${fmtMeters(snap.odometerM)} · ${snap.lap + 1}구간 ${snap.direction === 'forward' ? '순방향' : '역방향'}`),
    );
    if (snap.etaSeconds !== null) node.append(row('현장 도착까지', `${snap.etaSeconds}초`, 'warn'));

    const bar = el('div', 'bar');
    const fill = el('div', 'bar__fill');
    fill.style.width = `${Math.round(snap.progress * 100)}%`;
    bar.append(fill);
    node.append(bar);

    const list = el('ol', 'wp');
    for (const [i, w] of route.waypoints.entries()) {
      list.append(el('li', 'wp__i', `${i + 1}. ${w.lat.toFixed(5)}, ${w.lon.toFixed(5)} · ${w.alt.toFixed(0)} m`));
    }
    node.append(list);
    return node;
  }

  private telemetry(snap: DroneSnapshot): HTMLElement {
    const t = snap.telemetry;
    const node = card('텔레메트리', { text: `${snap.telemetrySent} 프레임`, tone: t ? 'ok' : 'idle' });
    if (!t) {
      node.append(el('p', 'empty', '아직 포즈 없음'));
      return node;
    }
    const grid = el('div', 'grid');
    const cell = (k: string, v: string) => {
      const c = el('div', 'cell');
      c.append(el('span', 'cell__k', k), el('span', 'cell__v', v));
      return c;
    };
    grid.append(
      cell('위도', t.gps.lat.toFixed(6)),
      cell('경도', t.gps.lon.toFixed(6)),
      cell('고도', `${t.gps.alt.toFixed(1)} m`),
      cell('기수', `${t.headingDeg.toFixed(1)}°`),
      cell('속도', `${t.speed.toFixed(1)} m/s`),
      cell('배터리', `${t.batteryPct.toFixed(1)} %`),
    );
    node.append(grid);
    return node;
  }

  private capture(snap: DroneSnapshot): HTMLElement {
    const node = card('촬영 · 전송', {
      text: snap.capturing ? '전송 중' : '정지',
      tone: snap.capturing ? 'ok' : 'idle',
    });
    node.append(
      row(
        '촬영 소스',
        snap.captureKind === 'demo'
          ? '데모 영상 (res/static/video/h265 · 사전 인코딩된 실제 HEVC)'
          : '카메라 실시간 H.265 인코딩 (WebCodecs)',
      ),
    );
    node.append(row('전송 세그먼트', `${snap.segments.length ? snap.segments[0].seq + 1 : 0}개`));
    if (snap.segments.length === 0) {
      node.append(el('p', 'empty', '아직 구간을 통과하지 않음'));
      return node;
    }
    const list = el('ul', 'segs');
    for (const s of snap.segments) {
      const item = el('li', 'segs__i');
      item.append(el('span', 'segs__n', `#${s.seq}`));
      item.append(el('span', 'segs__u', s.uri.split('/').pop() ?? s.uri));
      item.append(
        el('span', 'segs__m', `${s.codec} · ${fmtBytes(s.bytes)} · ${(s.durationMs / 1000).toFixed(1)}s · ${s.poses.length} poses`),
      );
      list.append(item);
    }
    node.append(list);
    return node;
  }
}
