// Drone path-following + mode state machine. PROJECT.md §4.2.
// Owns state.drones and advances them each frame: AUTO -> MANUAL -> RETURNING -> AUTO.
import * as THREE from 'three';
import { state } from '../../shared/viewer/store.ts';
import { CONFIG, droneViewScale } from '../../shared/viewer/config.ts';
import { samplePath, pathDuration, dampFactor, easeInOut, clamp } from '../../shared/viewer/math.ts';
import { createManualInput } from './manualControl.ts';
import type { DroneRuntime, DronePath } from '../../shared/viewer/types';

export interface DroneController {
  update(dt: number): void;
  dispose(): void;
  /** Replace a drone's path at runtime (used by the route modal to assign a
   *  freshly planned GPS route to the leader) and start flying it from t0. */
  setLeaderRoute(path: DronePath): void;
}

const WORLD_UP = new THREE.Vector3(0, 1, 0);

/** Formation slots for follower drones, in the leader's heading frame:
 *  (right, up, forward). They sit behind-left / behind-right / high-ahead. */
const FORMATION = [
  new THREE.Vector3(-9, 2, -6),
  new THREE.Vector3(9, -2, -6),
  new THREE.Vector3(0, 3, 9),
];

/** How often (control seconds) we record a visited sample per drone. */
const VISITED_INTERVAL = 0.15;
/** Minimum world-unit movement before a new visited sample is recorded. */
const VISITED_MIN_DIST = 1.0;

/** Per-drone bookkeeping that isn't part of the shared DroneRuntime contract. */
interface DroneLocal {
  /** World-up reference used to build a stable look-orientation. */
  idleTime: number;
  /** Control-time accumulator gating visited-buffer pushes. */
  visitedTimer: number;
  lastVisitedPos: THREE.Vector3 | null;
  /** Snapshot of position when a RETURNING tween begins. */
  returnStartPos: THREE.Vector3 | null;
  returnStartQuat: THREE.Quaternion | null;
  /** Target pathTime (on the preset path) the RETURNING tween is homing onto. */
  returnTargetTime: number;
  /** Elapsed seconds within the current RETURNING tween. */
  returnElapsed: number;
}

/** Build a quaternion so the drone's -Z axis points along `forward`. */
function quatFromForward(forward: THREE.Vector3): THREE.Quaternion {
  const dir = forward.lengthSq() < 1e-8 ? new THREE.Vector3(0, 0, 1) : forward;
  const m = new THREE.Matrix4();
  const eye = new THREE.Vector3(0, 0, 0);
  const target = dir.clone().normalize();
  m.lookAt(eye, target, new THREE.Vector3(0, 1, 0));
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

export function createDroneController(paths: DronePath[]): DroneController {
  const input = createManualInput();
  const locals = new Map<number, DroneLocal>();
  const pathsById = new Map<number, DronePath>(paths.map((p) => [p.id, p]));
  // Formation spread scales with the drone view scale — a swarm 9 units wide
  // around a 0.15-scale drone would scatter offscreen from the close-in camera.
  const formationScale = droneViewScale();

  // Initialize state.drones from the preset paths, one per path, at t=0.
  state.drones = paths.map((path): DroneRuntime => {
    const { pos, forward } = samplePath(path.waypoints, path.waypoints[0]?.t ?? 0);
    locals.set(path.id, {
      idleTime: 0,
      visitedTimer: 0,
      lastVisitedPos: null,
      returnStartPos: null,
      returnStartQuat: null,
      returnTargetTime: 0,
      returnElapsed: 0,
    });
    return {
      id: path.id,
      zone: path.zone,
      pos: pos.clone(),
      quat: quatFromForward(forward),
      forward: forward.clone(),
      mode: 'AUTO',
      pathTime: path.waypoints[0]?.t ?? 0,
    };
  });

  function pathFor(id: number) {
    return pathsById.get(id);
  }

  function recordVisited(drone: DroneRuntime, local: DroneLocal, dt: number): void {
    local.visitedTimer += dt;
    if (local.visitedTimer < VISITED_INTERVAL) return;
    local.visitedTimer = 0;
    if (local.lastVisitedPos && local.lastVisitedPos.distanceTo(drone.pos) < VISITED_MIN_DIST) {
      return;
    }
    local.lastVisitedPos = drone.pos.clone();
    state.visited.push({ pos: [drone.pos.x, drone.pos.y, drone.pos.z], t: state.time });
  }

  function updateAuto(drone: DroneRuntime, local: DroneLocal, dt: number): void {
    const path = pathFor(drone.id);
    if (!path) return;
    const duration = pathDuration(path.waypoints);
    const t0 = path.waypoints[0]?.t ?? 0;
    if (duration > 1e-6) {
      let t = drone.pathTime + dt - t0;
      t = ((t % duration) + duration) % duration;
      drone.pathTime = t0 + t;
    }
    const { pos, forward } = samplePath(path.waypoints, drone.pathTime);
    const k = dampFactor(6, dt);
    drone.pos.lerp(pos, k);
    drone.forward.lerp(forward, k).normalize();
    const targetQuat = quatFromForward(forward);
    drone.quat.slerp(targetQuat, k);
    local.idleTime = 0;
  }

  function updateManual(drone: DroneRuntime, local: DroneLocal, dt: number): void {
    if (input.active) {
      local.idleTime = 0;

      // Left/right = gradual YAW (steering), so the view turns a little per press
      // instead of the drone snapping to face a sideways strafe direction.
      if (input.move.x !== 0) {
        drone.forward
          .applyAxisAngle(WORLD_UP, -input.move.x * CONFIG.drone.manualYawRate * dt)
          .normalize();
      }

      // Travel along the heading, flattened to the XZ plane; Q/E change altitude.
      const fwd = new THREE.Vector3(drone.forward.x, 0, drone.forward.z);
      if (fwd.lengthSq() < 1e-8) fwd.set(0, 0, 1);
      fwd.normalize();

      const delta = new THREE.Vector3();
      delta.addScaledVector(fwd, input.move.z * CONFIG.drone.manualSpeed * dt);
      delta.y += input.move.y * CONFIG.drone.manualAltitudeSpeed * dt;
      drone.pos.add(delta);

      drone.quat.slerp(quatFromForward(drone.forward), dampFactor(10, dt));
    } else {
      local.idleTime += dt;
      if (local.idleTime > CONFIG.drone.manualIdleReturn) {
        beginReturning(drone, local);
      }
    }
  }

  /** Non-active drones cluster around the leader in formation (swarm). */
  function updateFollow(
    drone: DroneRuntime,
    leader: DroneRuntime,
    slot: number,
    local: DroneLocal,
    dt: number,
  ): void {
    local.idleTime = 0;
    const fwd = new THREE.Vector3(leader.forward.x, 0, leader.forward.z);
    if (fwd.lengthSq() < 1e-8) fwd.set(0, 0, 1);
    fwd.normalize();
    const right = new THREE.Vector3().crossVectors(WORLD_UP, fwd).normalize();

    const off = FORMATION[slot % FORMATION.length];
    const target = leader.pos
      .clone()
      .addScaledVector(right, off.x * formationScale)
      .addScaledVector(WORLD_UP, off.y * formationScale)
      .addScaledVector(fwd, off.z * formationScale);

    drone.pos.lerp(target, dampFactor(3.5, dt));
    drone.forward.lerp(leader.forward, dampFactor(3.5, dt)).normalize();
    drone.quat.slerp(quatFromForward(drone.forward), dampFactor(5, dt));
  }

  function beginReturning(drone: DroneRuntime, local: DroneLocal): void {
    const path = pathFor(drone.id);
    if (!path) return;
    drone.mode = 'RETURNING';
    local.returnStartPos = drone.pos.clone();
    local.returnStartQuat = drone.quat.clone();
    local.returnElapsed = 0;
    // Home onto the nearest future preset waypoint sample (pathTime + returnTween).
    const t0 = path.waypoints[0]?.t ?? 0;
    const tEnd = path.waypoints[path.waypoints.length - 1]?.t ?? 0;
    let target = drone.pathTime + CONFIG.drone.returnTween;
    if (target > tEnd) target = t0 + (target - t0) % Math.max(1e-6, tEnd - t0);
    local.returnTargetTime = target;
  }

  function updateReturning(drone: DroneRuntime, local: DroneLocal, dt: number): void {
    const path = pathFor(drone.id);
    if (!path || !local.returnStartPos || !local.returnStartQuat) {
      drone.mode = 'AUTO';
      return;
    }
    local.returnElapsed += dt;
    const t = clamp(local.returnElapsed / CONFIG.drone.returnTween, 0, 1);
    const e = easeInOut(t);
    const { pos, forward } = samplePath(path.waypoints, local.returnTargetTime);
    drone.pos.lerpVectors(local.returnStartPos, pos, e);
    drone.quat.copy(local.returnStartQuat).slerp(quatFromForward(forward), e);
    drone.forward.lerp(forward, e).normalize();

    if (t >= 1) {
      drone.pathTime = local.returnTargetTime;
      drone.mode = 'AUTO';
      local.returnStartPos = null;
      local.returnStartQuat = null;
    }
  }

  return {
    update(dt: number): void {
      const leader = state.drones.find((d) => d.id === state.activeDroneId) ?? state.drones[0];
      let slot = 0;

      for (const drone of state.drones) {
        const local = locals.get(drone.id);
        if (!local) continue;

        if (leader && drone.id === leader.id) {
          // Leader: the active drone flies its path or is steered.
          if (input.active && drone.mode !== 'MANUAL') {
            drone.mode = 'MANUAL';
            local.idleTime = 0;
          }
          switch (drone.mode) {
            case 'MANUAL':
              updateManual(drone, local, dt);
              break;
            case 'RETURNING':
              updateReturning(drone, local, dt);
              break;
            default:
              updateAuto(drone, local, dt);
          }
        } else if (leader) {
          // Followers cluster around the leader (swarm).
          drone.mode = 'AUTO';
          updateFollow(drone, leader, slot++, local, dt);
        } else {
          updateAuto(drone, local, dt);
        }

        recordVisited(drone, local, dt);
      }
    },
    dispose(): void {
      input.dispose();
    },
    setLeaderRoute(path: DronePath): void {
      pathsById.set(path.id, path);
      const drone = state.drones.find((d) => d.id === path.id);
      const local = locals.get(path.id);
      if (drone) {
        drone.pathTime = path.waypoints[0]?.t ?? 0;
        drone.mode = 'AUTO';
      }
      if (local) local.idleTime = 0;
    },
  };
}
