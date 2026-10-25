// Package mission is the mission state machine:
//
//	idle -> assigned -> awaiting-drone -> active
//
// The demo scenario (COMPONENTS.md §4) is expressed entirely through these
// phases:
//
//  1. system start          idle            nothing happens
//  2. route assigned        assigned        "태스크 지정 완료"
//  3. waiting for a drone   awaiting-drone  "드론이 현장에 도착해 연결되면 ..." + eta
//  4. drone_hello received  active          "드론 연결됨"
//
// A drone counts as arrived only on drone_hello. Telemetry from a drone still
// flying to the site must never end the wait, so this package has no telemetry
// entry point at all.
//
// The eta is derived from the time the wait started, not counted down by a
// timer, so a viewer that joins mid-wait sees the right remaining time.
package mission

import (
	"log"
	"math"
	"sort"
	"sync"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
)

// Operator-facing lines, already localized (MissionStatus.message).
var messages = map[viewerv1.MissionPhase]string{
	viewerv1.MissionPhase_MISSION_PHASE_IDLE:           "대기 중 — 관제탑에서 경로를 지정하십시오",
	viewerv1.MissionPhase_MISSION_PHASE_ASSIGNED:       "태스크 지정 완료",
	viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE: "드론이 현장에 도착해 연결되면 작업이 시작됩니다",
	viewerv1.MissionPhase_MISSION_PHASE_ACTIVE:         "드론 연결됨 — 임무 수행 중",
}

// Message returns the operator line for a phase.
func Message(p viewerv1.MissionPhase) string { return messages[p] }

// Timer is the part of *time.Timer the machine needs.
type Timer interface{ Stop() bool }

// Clock lets tests drive time.
type Clock interface {
	Now() time.Time
	AfterFunc(d time.Duration, f func()) Timer
}

type realClock struct{}

func (realClock) Now() time.Time                            { return time.Now() }
func (realClock) AfterFunc(d time.Duration, f func()) Timer { return time.AfterFunc(d, f) }

// Options configure a Mission.
type Options struct {
	// AssignedHold is how long "태스크 지정 완료" stays before the wait starts,
	// so the operator actually reads it.
	AssignedHold time.Duration
	// DroneETA is the expected wait for a drone; the status carries the
	// remaining part as eta_seconds. Zero means no eta.
	DroneETA time.Duration
	// OnChange receives every status the machine publishes, one at a time and
	// in the order the changes happened. The state lock is not held, so it may
	// read the mission (Status, Phase); it must not change it.
	OnChange func(*viewerv1.MissionStatus)
	// Clock defaults to the wall clock.
	Clock Clock
}

// Mission is safe for concurrent use.
type Mission struct {
	opts Options

	// pub orders publishing: a change and the statuses it publishes go out
	// before the next change starts. Taken before mu, held across OnChange.
	// Without it two drones saying hello at once could publish their counts in
	// the wrong order and leave screens on the smaller one.
	pub           sync.Mutex
	mu            sync.Mutex
	phase         viewerv1.MissionPhase
	awaitingSince time.Time
	hold          Timer
	holdGen       uint64
	drones        map[uint32]modelv1.DroneStation
}

// New returns a mission in the idle phase.
func New(opts Options) *Mission {
	if opts.Clock == nil {
		opts.Clock = realClock{}
	}
	return &Mission{
		opts:   opts,
		phase:  viewerv1.MissionPhase_MISSION_PHASE_IDLE,
		drones: map[uint32]modelv1.DroneStation{},
	}
}

// Phase returns the current phase.
func (m *Mission) Phase() viewerv1.MissionPhase {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.phase
}

// Online is the number of drones that said hello and are still connected.
func (m *Mission) Online() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.drones)
}

// Drones lists the ids of drones on station, ascending.
func (m *Mission) Drones() []uint32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	ids := make([]uint32, 0, len(m.drones))
	for id := range m.drones {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// Status is the current MissionStatus.
func (m *Mission) Status() *viewerv1.MissionStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.statusLocked()
}

// RouteAssigned moves to assigned. After AssignedHold it moves on to
// awaiting-drone, and straight to active if a drone is already on station.
// Assigning again restarts the sequence.
func (m *Mission) RouteAssigned() {
	m.pub.Lock()
	defer m.pub.Unlock()
	m.mu.Lock()
	var out []*viewerv1.MissionStatus
	m.stopHoldLocked()
	if m.phase == viewerv1.MissionPhase_MISSION_PHASE_ASSIGNED {
		// Same phase, but tell the screen again: it may be the one that asked.
		out = append(out, m.statusLocked())
	} else {
		out = m.toLocked(viewerv1.MissionPhase_MISSION_PHASE_ASSIGNED, out)
	}
	m.holdGen++
	gen := m.holdGen
	m.hold = m.opts.Clock.AfterFunc(m.opts.AssignedHold, func() { m.holdElapsed(gen) })
	m.mu.Unlock()
	m.emit(out)
}

func (m *Mission) holdElapsed(gen uint64) {
	m.pub.Lock()
	defer m.pub.Unlock()
	m.mu.Lock()
	if gen != m.holdGen || m.phase != viewerv1.MissionPhase_MISSION_PHASE_ASSIGNED {
		m.mu.Unlock()
		return
	}
	m.hold = nil
	out := m.toLocked(viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE, nil)
	if len(m.drones) > 0 {
		// A drone already on station means the wait is over before it began.
		out = m.toLocked(viewerv1.MissionPhase_MISSION_PHASE_ACTIVE, out)
	}
	m.mu.Unlock()
	m.emit(out)
}

// Hello records a drone_hello. It ends the wait when the mission is waiting;
// in any other phase it only updates the count. A drone without a task does
// not start a mission. The status is re-published even when the phase holds,
// because it carries drones_online.
func (m *Mission) Hello(droneID uint32, station modelv1.DroneStation) {
	m.pub.Lock()
	defer m.pub.Unlock()
	m.mu.Lock()
	m.drones[droneID] = station
	var out []*viewerv1.MissionStatus
	if m.phase == viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE {
		out = m.toLocked(viewerv1.MissionPhase_MISSION_PHASE_ACTIVE, out)
	} else {
		out = append(out, m.statusLocked())
	}
	m.mu.Unlock()
	m.emit(out)
}

// Gone records that a drone's uplink closed. The last drone leaving an active
// mission sends it back to waiting, with the eta restarted.
func (m *Mission) Gone(droneID uint32) {
	m.pub.Lock()
	defer m.pub.Unlock()
	m.mu.Lock()
	if _, ok := m.drones[droneID]; !ok {
		m.mu.Unlock()
		return
	}
	delete(m.drones, droneID)
	var out []*viewerv1.MissionStatus
	if m.phase == viewerv1.MissionPhase_MISSION_PHASE_ACTIVE && len(m.drones) == 0 {
		out = m.toLocked(viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE, out)
	} else {
		out = append(out, m.statusLocked())
	}
	m.mu.Unlock()
	m.emit(out)
}

// Reset goes back to idle.
func (m *Mission) Reset() {
	m.pub.Lock()
	defer m.pub.Unlock()
	m.mu.Lock()
	m.stopHoldLocked()
	out := m.toLocked(viewerv1.MissionPhase_MISSION_PHASE_IDLE, nil)
	m.mu.Unlock()
	m.emit(out)
}

// Stop cancels the pending hold timer.
func (m *Mission) Stop() {
	m.mu.Lock()
	m.stopHoldLocked()
	m.mu.Unlock()
}

func (m *Mission) stopHoldLocked() {
	if m.hold != nil {
		m.hold.Stop()
		m.hold = nil
	}
	m.holdGen++
}

func (m *Mission) toLocked(p viewerv1.MissionPhase, out []*viewerv1.MissionStatus) []*viewerv1.MissionStatus {
	if p == m.phase {
		return out
	}
	m.phase = p
	if p == viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE {
		m.awaitingSince = m.opts.Clock.Now()
	}
	log.Printf("[core] mission -> %s (%s)", p, messages[p])
	return append(out, m.statusLocked())
}

func (m *Mission) statusLocked() *viewerv1.MissionStatus {
	s := &viewerv1.MissionStatus{
		Phase:        m.phase,
		Message:      messages[m.phase],
		DronesOnline: uint32(len(m.drones)),
	}
	if m.phase == viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE && m.opts.DroneETA > 0 {
		left := (m.opts.DroneETA - m.opts.Clock.Now().Sub(m.awaitingSince)).Seconds()
		eta := math.Max(0, math.Round(left))
		s.EtaSeconds = &eta
	}
	return s
}

func (m *Mission) emit(out []*viewerv1.MissionStatus) {
	if m.opts.OnChange == nil {
		return
	}
	for _, s := range out {
		m.opts.OnChange(s)
	}
}
