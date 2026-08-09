// Camera state machine for viewer 2 (§8.3): SYNCED -> FOCUSING -> LOCKED ->
// RETURNING -> SYNCED. The ReconViewer calls step() and applies getPose().
//
// SYNCED is a slow OVERVIEW ORBIT around the whole reconstructed scene, framed
// from outside so the building is clearly visible as a 3D object (RECON has no
// drone model to chase). On a detection it tweens in to focus, holds until the
// operator confirms, then tweens back to the orbit. Drives shared store state so
// the UI interoperates purely through state.cameraSync / focusedDetectionId.

import * as THREE from 'three';
import type { DetectionRuntime } from '../../skylens_core/types';
import { state, emit, setCameraState } from '../../skylens_core/store';
import { CONFIG } from '../../skylens_core/config';
import { easeInOut, toVector3 } from '../../skylens_core/math';

const ORBIT_SPEED = 0.12; // rad/s

export class CameraSync {
  private readonly camPos = new THREE.Vector3();
  private readonly camTarget = new THREE.Vector3();

  private readonly center = new THREE.Vector3();
  private readonly dist: number;
  private readonly elevation: number;
  private angle = Math.PI * 0.25;
  private initialized = false;

  // Tween bookkeeping for FOCUSING / RETURNING. Timed in REAL seconds (dt), not
  // the sim clock, so camera animation plays at a consistent speed regardless of
  // sim playback speed or a throttled/paused sim.
  private tweenElapsed = 0;
  private tweenFrom = { pos: new THREE.Vector3(), target: new THREE.Vector3() };
  private tweenTo = { pos: new THREE.Vector3(), target: new THREE.Vector3() };

  constructor(bounds: THREE.Box3) {
    bounds.getCenter(this.center);
    const sphere = new THREE.Sphere();
    bounds.getBoundingSphere(sphere);
    const radius = Math.max(1, sphere.radius);
    // Distance to fit the sphere in a ~50° vertical FOV, with headroom.
    this.dist = radius * 2.4;
    this.elevation = this.dist * 0.35;
  }

  getPose(): { pos: THREE.Vector3; target: THREE.Vector3 } {
    return { pos: this.camPos, target: this.camTarget };
  }

  /** Overview-orbit pose at the current angle. */
  private orbitPose(): { pos: THREE.Vector3; target: THREE.Vector3 } {
    const pos = new THREE.Vector3(
      this.center.x + Math.cos(this.angle) * this.dist,
      this.center.y + this.elevation,
      this.center.z + Math.sin(this.angle) * this.dist,
    );
    return { pos, target: this.center.clone() };
  }

  step(dt: number, detections: DetectionRuntime[]): void {
    // State-driven confirmation: once the focused detection is acknowledged
    // (overlay sets confirmed=true), the camera returns. Reading the flag rather
    // than a one-shot event means a confirmation can't be dropped.
    if (
      state.focusedDetectionId !== null &&
      (state.cameraSync === 'LOCKED' || state.cameraSync === 'FOCUSING')
    ) {
      const det = detections.find((d) => d.id === state.focusedDetectionId);
      if (det && det.confirmed) this.beginReturning();
    }

    switch (state.cameraSync) {
      case 'SYNCED':
        this.stepSynced(dt);
        break;
      case 'FOCUSING':
        this.stepTween(dt, () => setCameraState('LOCKED'));
        break;
      case 'LOCKED':
        break; // hold on the detection
      case 'RETURNING':
        this.stepTween(dt, () => {
          state.focusedDetectionId = null;
          setCameraState('SYNCED');
        });
        break;
    }

    // Trigger a focus when something newly revealed and nothing is focused.
    if (state.cameraSync === 'SYNCED' && state.focusedDetectionId === null) {
      const candidate = detections.find((d) => d.revealed && !d.confirmed);
      if (candidate) {
        state.focusedDetectionId = candidate.id;
        this.beginFocusing(candidate);
        emit({ type: 'detection-focused', id: candidate.id });
      }
    }
  }

  private stepSynced(dt: number): void {
    // Angle only advances in SYNCED, so a RETURN lands exactly where the orbit
    // resumes (no jump).
    this.angle += ORBIT_SPEED * dt;
    const pose = this.orbitPose();
    this.camPos.copy(pose.pos);
    this.camTarget.copy(pose.target);
    this.initialized = true;
  }

  private beginFocusing(det: DetectionRuntime): void {
    if (!this.initialized) {
      const pose = this.orbitPose();
      this.camPos.copy(pose.pos);
      this.camTarget.copy(pose.target);
      this.initialized = true;
    }
    this.tweenFrom.pos.copy(this.camPos);
    this.tweenFrom.target.copy(this.camTarget);

    const detPos = toVector3(det.pos);
    // Approach from the current orbit direction so the detection isn't occluded.
    const dir = detPos.clone().sub(this.camPos);
    if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
    dir.normalize();
    const camPos = detPos.clone().sub(dir.multiplyScalar(CONFIG.camera.focusDistance));
    camPos.y += CONFIG.camera.focusDistance * 0.35;

    this.tweenTo.pos.copy(camPos);
    this.tweenTo.target.copy(detPos);
    this.tweenElapsed = 0;
    setCameraState('FOCUSING');
  }

  private beginReturning(): void {
    this.tweenFrom.pos.copy(this.camPos);
    this.tweenFrom.target.copy(this.camTarget);
    const pose = this.orbitPose(); // angle is frozen during focus -> resume point
    this.tweenTo.pos.copy(pose.pos);
    this.tweenTo.target.copy(pose.target);
    this.tweenElapsed = 0;
    setCameraState('RETURNING');
  }

  private stepTween(dt: number, onDone: () => void): void {
    this.tweenElapsed += dt;
    const t = this.tweenElapsed / Math.max(1e-6, CONFIG.camera.tweenSeconds);
    const e = easeInOut(t);
    this.camPos.lerpVectors(this.tweenFrom.pos, this.tweenTo.pos, e);
    this.camTarget.lerpVectors(this.tweenFrom.target, this.tweenTo.target, e);
    if (t >= 1) onDone();
  }
}
