// The drone application itself. Runtime-agnostic: no DOM, no node: imports.
// The Tauri window, the plain browser page and the headless demo runner all
// construct this same class and only differ in what they inject (socket factory,
// capture source) and how they render the snapshot.
//
// Responsibilities, in the order the demo exercises them:
//   1. connect to the gateway and announce with DroneHello, honouring LinkMode
//   2. sit IDLE until the core assigns a route
//   3. on AssignRoute, transit to the site (~10 s, COMPONENTS.md §5.2 step 4)
//   4. fly the route, ping-ponging when loop=true, streaming DroneTelemetry
//   5. every time a SLICE of the route has been covered, ship a VideoSegment
//      carrying the poses for that slice — the trigger is distance flown, not a
//      wall clock, which is what lets the core segment on drone movement

import type {
  AssignRoute,
  ControlMessage,
  DroneHello,
  DroneTelemetry,
  ManualControl,
  VideoSegment,
} from '../../shared/protocol.ts';
import type { Gps } from '../../shared/geo.ts';
import type { DroneConfig } from './config.ts';
import type { CaptureSource } from './capture.ts';
import { GatewayLink, type LinkState, type SocketFactory } from './link.ts';
import {
  deriveHome,
  foldOdometer,
  groundDistance,
  interpolateGps,
  planRoute,
  sampleRoute,
  stepManual,
  type FlightDirection,
  type Pose,
  type RoutePlan,
} from './flight.ts';

export type DronePhase = 'offline' | 'idle' | 'transit' | 'flying' | 'manual' | 'holding';

export interface DroneSnapshot {
  droneId: number;
  model: string;
  phase: DronePhase;
  /** Operator-facing Korean line for the panel. */
  note: string;
  link: LinkState;
  announced: boolean;
  captureKind: 'demo' | 'live';
  capturing: boolean;
  route: AssignRoute | null;
  routeLengthM: number;
  odometerM: number;
  /** 0..1 along the current one-way traverse. */
  progress: number;
  direction: FlightDirection;
  /** Completed one-way traverses. */
  lap: number;
  /** Seconds until the drone is on station, while in transit. */
  etaSeconds: number | null;
  telemetry: DroneTelemetry | null;
  telemetrySent: number;
  segments: VideoSegment[];
}

export interface DroneAppOptions {
  config: DroneConfig;
  capture: CaptureSource;
  socketFactory?: SocketFactory;
  onUpdate?: (snap: DroneSnapshot) => void;
  onLog?: (line: string) => void;
  /** Injectable clock, for tests. */
  now?: () => number;
}

const MANUAL_SPEED = 8;
const MANUAL_CLIMB = 5;
const MANUAL_YAW_DEG = 55;
/** Keep the panel's segment list bounded; the core is the archive, not us. */
const SEGMENT_HISTORY = 12;

export class DroneApp {
  private cfg: DroneConfig;
  private capture: CaptureSource;
  private link: GatewayLink;
  private now: () => number;
  private onUpdate?: (snap: DroneSnapshot) => void;
  private onLog?: (line: string) => void;

  private timer: ReturnType<typeof setInterval> | null = null;
  private lastTick = 0;

  private phase: DronePhase = 'offline';
  private announced = false;
  private route: AssignRoute | null = null;
  private plan: RoutePlan | null = null;
  private home: Gps | null = null;
  private transitStartedAt = 0;
  private transitDistanceM = 0;

  private odometerM = 0;
  private pose: Pose | null = null;
  private speed = 0;
  private battery = 100;
  private direction: FlightDirection = 'forward';
  private lap = 0;

  private manual: ManualControl | null = null;
  private manualAt = 0;

  private poseBuffer: DroneTelemetry[] = [];
  private sliceStartedAt = 0;
  private slicesEmitted = 0;
  private cutting = false;
  private videoSeq = 0;
  private telemetrySent = 0;
  private segments: VideoSegment[] = [];
  private linkState: LinkState;

  constructor(opts: DroneAppOptions) {
    this.cfg = opts.config;
    this.capture = opts.capture;
    this.now = opts.now ?? (() => Date.now());
    this.onUpdate = opts.onUpdate;
    this.onLog = opts.onLog;
    this.link = new GatewayLink({
      url: this.cfg.gatewayUrl,
      mode: this.cfg.mode,
      droneId: this.cfg.droneId,
      socketFactory: opts.socketFactory,
      reconnectMinMs: this.cfg.reconnectMinMs,
      reconnectMaxMs: this.cfg.reconnectMaxMs,
      onControl: (msg) => this.onControl(msg),
      onState: (s) => this.onLinkState(s),
      onLog: (line) => this.log(line),
    });
    this.linkState = this.link.snapshot;
  }

  // -------------------------------------------------------------------------
  // lifecycle
  // -------------------------------------------------------------------------

  async start(): Promise<void> {
    await this.capture.start();
    this.log(
      `capture source: ${this.capture.kind}${this.capture.kind === 'demo' ? ' (res/static/video/h265 footage, pre-encoded HEVC)' : ''}`,
    );
    this.phase = 'idle';
    this.lastTick = this.now();
    this.link.connect();
    const period = Math.max(20, Math.round(1000 / this.cfg.telemetryHz));
    this.timer = setInterval(() => this.tick(), period);
    this.publish();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.capture.stop();
    this.link.close();
    this.phase = 'offline';
    this.publish();
  }

  /** Feed a control message in directly (Tauri command, UI button, tests). */
  handleControl(msg: ControlMessage): void {
    this.onControl(msg);
  }

  get snapshot(): DroneSnapshot {
    const total = this.plan?.totalM ?? 0;
    const fold = this.plan ? foldOdometer(this.plan, this.odometerM) : null;
    return {
      droneId: this.cfg.droneId,
      model: this.cfg.model,
      phase: this.phase,
      note: this.note(),
      link: this.linkState,
      announced: this.announced,
      captureKind: this.capture.kind,
      capturing: this.phase === 'flying' || this.phase === 'manual',
      route: this.route,
      routeLengthM: total,
      odometerM: this.odometerM,
      progress: total > 0 && fold ? fold.s / total : 0,
      direction: this.direction,
      lap: this.lap,
      etaSeconds: this.etaSeconds(),
      telemetry: this.pose ? this.buildTelemetry(this.now()) : null,
      telemetrySent: this.telemetrySent,
      segments: this.segments,
    };
  }

  // -------------------------------------------------------------------------
  // link
  // -------------------------------------------------------------------------

  private onLinkState(state: LinkState): void {
    const wasConnected = this.linkState.phase === 'connected';
    this.linkState = state;
    if (state.phase === 'connected' && !wasConnected) {
      if (!this.cfg.helloOnArrival) this.announce();
    } else if (state.phase !== 'connected' && wasConnected) {
      this.announced = false;
    }
    this.publish();
  }

  private announce(): void {
    const hello: DroneHello = {
      kind: 'drone-hello',
      droneId: this.cfg.droneId,
      model: this.cfg.model,
      mode: this.cfg.mode,
    };
    if (this.link.send(hello)) {
      this.announced = true;
      this.log(`hello sent — drone ${hello.droneId} "${hello.model}" mode=${hello.mode}`);
    }
  }

  private onControl(msg: ControlMessage): void {
    if (msg.droneId !== this.cfg.droneId && msg.droneId >= 0) return;
    if (msg.kind === 'assign-route') this.assignRoute(msg);
    else if (msg.kind === 'manual-control') this.applyManual(msg);
  }

  // -------------------------------------------------------------------------
  // control
  // -------------------------------------------------------------------------

  assignRoute(msg: AssignRoute): void {
    if (msg.waypoints.length < 2) {
      this.log('assign-route ignored: needs at least two waypoints');
      return;
    }
    const plan = planRoute(msg.waypoints, msg.loop);
    this.route = msg;
    this.plan = plan;
    this.home = this.cfg.home ?? deriveHome(plan, this.cfg.homeOffsetM);
    this.odometerM = 0;
    this.slicesEmitted = 0;
    this.videoSeq = 0;
    this.segments = [];
    this.poseBuffer = [];
    this.lap = 0;
    this.direction = 'forward';
    this.manual = null;

    const t = this.now();
    this.sliceStartedAt = t;

    if (this.cfg.demo) {
      this.transitStartedAt = t;
      this.transitDistanceM = groundDistance(this.home, plan.waypoints[0], plan.anchor);
      this.phase = 'transit';
      this.pose = { gps: this.home, headingDeg: 0 };
      this.log(
        `route assigned: ${msg.waypoints.length} waypoints, ${plan.totalM.toFixed(0)} m, loop=${msg.loop}` +
          ` — 현장으로 이동 시작 (도착 예정 ${(this.cfg.arrivalMs / 1000).toFixed(0)}초)`,
      );
    } else {
      this.phase = 'flying';
      this.pose = sampleRoute(plan, 0, 'forward');
      this.log(`route assigned: ${msg.waypoints.length} waypoints, ${plan.totalM.toFixed(0)} m, loop=${msg.loop}`);
    }
    this.publish();
  }

  private applyManual(msg: ManualControl): void {
    if (!this.plan || this.phase === 'transit') return;
    const idle = msg.forward === 0 && msg.yaw === 0 && msg.climb === 0;
    this.manual = msg;
    if (!idle) {
      this.manualAt = this.now();
      if (this.phase !== 'manual') {
        this.phase = 'manual';
        this.log('manual control taken over');
      }
    }
  }

  // -------------------------------------------------------------------------
  // tick
  // -------------------------------------------------------------------------

  private tick(): void {
    const t = this.now();
    const dt = Math.min(1, (t - this.lastTick) / 1000);
    this.lastTick = t;
    if (dt <= 0) return;

    switch (this.phase) {
      case 'transit':
        this.stepTransit(t);
        break;
      case 'flying':
        this.stepFlying(dt);
        break;
      case 'manual':
        this.stepManualPhase(t, dt);
        break;
      default:
        this.speed = 0;
        break;
    }

    if (this.phase !== 'idle' && this.phase !== 'offline') {
      this.battery = Math.max(5, this.battery - (this.cfg.batteryDrainPerMin / 60) * dt);
    }

    if (this.pose) {
      const telemetry = this.buildTelemetry(t);
      if (this.link.send(telemetry)) this.telemetrySent++;
      if (this.phase === 'flying' || this.phase === 'manual') this.poseBuffer.push(telemetry);
    }

    this.maybeCutSlice(t);
    this.publish();
  }

  private stepTransit(t: number): void {
    const plan = this.plan;
    const home = this.home;
    if (!plan || !home) return;
    const frac = Math.min(1, (t - this.transitStartedAt) / this.cfg.arrivalMs);
    this.pose = interpolateGps(home, plan.waypoints[0], frac, plan.anchor);
    this.speed = this.transitDistanceM / (this.cfg.arrivalMs / 1000);
    if (frac >= 1) {
      this.phase = 'flying';
      this.pose = sampleRoute(plan, 0, 'forward');
      this.sliceStartedAt = t;
      this.poseBuffer = [];
      if (this.cfg.helloOnArrival) this.announce();
      this.log('현장 도착 — 드론 연결됨, 경로 비행 시작');
    }
  }

  private stepFlying(dt: number): void {
    const plan = this.plan;
    if (!plan) return;
    this.speed = this.cfg.cruiseSpeed;
    this.odometerM += this.cfg.cruiseSpeed * dt;
    const fold = foldOdometer(plan, this.odometerM);
    if (fold.done) {
      this.phase = 'holding';
      this.speed = 0;
      this.log('route complete (loop=false) — holding at the last waypoint');
    }
    if (fold.lap !== this.lap) {
      this.lap = fold.lap;
      this.log(`turnaround — leg ${fold.lap + 1} (${fold.direction})`);
    }
    this.direction = fold.direction;
    this.pose = sampleRoute(plan, fold.s, fold.direction);
  }

  private stepManualPhase(t: number, dt: number): void {
    const plan = this.plan;
    const input = this.manual;
    if (!plan || !this.pose) return;
    if (input) {
      this.pose = stepManual(
        this.pose,
        input,
        dt,
        { speed: MANUAL_SPEED, climbSpeed: MANUAL_CLIMB, yawRateDeg: MANUAL_YAW_DEG },
        plan.anchor,
      );
      // Movement still advances the odometer, so slicing stays keyed to how far
      // the drone actually flew rather than to which mode it flew in.
      const moved = Math.abs(input.forward) * MANUAL_SPEED * dt;
      this.odometerM += moved;
      this.speed = Math.abs(input.forward) * MANUAL_SPEED;
    }
    if ((t - this.manualAt) / 1000 > this.cfg.manualIdleReturn) {
      this.phase = 'flying';
      this.manual = null;
      this.log('manual input idle — rejoining the assigned route');
    }
  }

  // -------------------------------------------------------------------------
  // slicing
  // -------------------------------------------------------------------------

  private sliceLengthM(): number {
    const total = this.plan?.totalM ?? 0;
    return total > 0 ? total / this.cfg.slicesPerLeg : 0;
  }

  private maybeCutSlice(t: number): void {
    const plan = this.plan;
    const sliceLen = this.sliceLengthM();
    if (!plan || sliceLen <= 0 || this.cutting) return;
    if (this.phase !== 'flying' && this.phase !== 'manual') return;
    if (this.odometerM < (this.slicesEmitted + 1) * sliceLen) return;

    const index = this.slicesEmitted;
    this.slicesEmitted++;
    const midM = (index + 0.5) * sliceLen;
    const at = foldOdometer(plan, midM);
    const fraction = plan.totalM > 0 ? at.s / plan.totalM : 0;
    const poses = this.poseBuffer;
    this.poseBuffer = [];
    const startedAt = poses[0]?.t ?? this.sliceStartedAt;
    const durationMs = Math.max(1, t - startedAt);
    this.sliceStartedAt = t;

    this.cutting = true;
    void this.capture
      .cutSlice({ index, fraction, direction: at.direction, startedAt, durationMs })
      .then((result) => {
        const segment: VideoSegment = {
          kind: 'video-segment',
          droneId: this.cfg.droneId,
          seq: this.videoSeq++,
          // Per-slice, from whatever actually produced the bytes.
          codec: result.codec,
          startedAt,
          durationMs,
          uri: result.uri,
          bytes: result.bytes,
          poses,
        };
        this.link.send(segment);
        this.segments = [segment, ...this.segments].slice(0, SEGMENT_HISTORY);
        this.log(
          `video-segment #${segment.seq} ${result.uri} — ${(this.odometerM).toFixed(0)} m flown,` +
            ` ${poses.length} poses, ${durationMs} ms, ${result.note}`,
        );
      })
      .catch((err) => this.log(`slice ${index} failed: ${String(err)}`))
      .finally(() => {
        this.cutting = false;
        this.publish();
      });
  }

  // -------------------------------------------------------------------------
  // helpers
  // -------------------------------------------------------------------------

  private buildTelemetry(t: number): DroneTelemetry {
    const pose = this.pose ?? { gps: { lat: 0, lon: 0, alt: 0 }, headingDeg: 0 };
    return {
      kind: 'telemetry',
      droneId: this.cfg.droneId,
      gps: pose.gps,
      headingDeg: Math.round(pose.headingDeg * 10) / 10,
      speed: Math.round(this.speed * 100) / 100,
      batteryPct: Math.round(this.battery * 10) / 10,
      t,
    };
  }

  private etaSeconds(): number | null {
    if (this.phase !== 'transit') return null;
    const left = this.cfg.arrivalMs - (this.now() - this.transitStartedAt);
    return Math.max(0, Math.ceil(left / 1000));
  }

  private note(): string {
    switch (this.phase) {
      case 'offline':
        return '게이트웨이 연결 대기';
      case 'idle':
        if (this.linkState.phase === 'connected') return '대기 중 — 관제탑의 경로 지정을 기다립니다';
        return this.linkState.phase === 'punching'
          ? '홀펀칭 중 — 프록시 직결 경로 협상'
          : '게이트웨이 연결 중';
      case 'transit':
        return `현장 이동 중 — 도착까지 ${this.etaSeconds() ?? 0}초`;
      case 'flying':
        return `경로 비행 중 · ${this.direction === 'forward' ? '순방향' : '역방향'} ${this.lap + 1}구간 · 촬영 전송 중`;
      case 'manual':
        return '수동 조종 중';
      case 'holding':
        return '경로 종료 — 최종 지점 대기';
    }
  }

  private log(line: string): void {
    this.onLog?.(line);
  }

  private publish(): void {
    this.onUpdate?.(this.snapshot);
  }
}
