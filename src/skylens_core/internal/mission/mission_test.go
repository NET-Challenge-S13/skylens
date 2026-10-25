package mission

import (
	"sync"
	"testing"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
)

const (
	idle     = viewerv1.MissionPhase_MISSION_PHASE_IDLE
	assigned = viewerv1.MissionPhase_MISSION_PHASE_ASSIGNED
	awaiting = viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE
	active   = viewerv1.MissionPhase_MISSION_PHASE_ACTIVE
	center   = modelv1.DroneStation_DRONE_STATION_CENTER
)

type fakeTimer struct {
	at      time.Time
	f       func()
	stopped bool
	fired   bool
}

func (t *fakeTimer) Stop() bool {
	was := !t.stopped && !t.fired
	t.stopped = true
	return was
}

type fakeClock struct {
	mu     sync.Mutex
	now    time.Time
	timers []*fakeTimer
}

func newFakeClock() *fakeClock { return &fakeClock{now: time.Unix(1_700_000_000, 0)} }

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) AfterFunc(d time.Duration, f func()) Timer {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := &fakeTimer{at: c.now.Add(d), f: f}
	c.timers = append(c.timers, t)
	return t
}

// Advance moves time forward and fires due timers outside the clock lock.
func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	var due []*fakeTimer
	for _, t := range c.timers {
		if !t.stopped && !t.fired && !t.at.After(c.now) {
			t.fired = true
			due = append(due, t)
		}
	}
	c.mu.Unlock()
	for _, t := range due {
		t.f()
	}
}

type recorder struct {
	mu  sync.Mutex
	got []*viewerv1.MissionStatus
}

func (r *recorder) on(s *viewerv1.MissionStatus) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.got = append(r.got, s)
}

func (r *recorder) phases() []viewerv1.MissionPhase {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]viewerv1.MissionPhase, len(r.got))
	for i, s := range r.got {
		out[i] = s.GetPhase()
	}
	return out
}

func (r *recorder) last() *viewerv1.MissionStatus {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.got) == 0 {
		return nil
	}
	return r.got[len(r.got)-1]
}

func newMission() (*Mission, *fakeClock, *recorder) {
	clk := newFakeClock()
	rec := &recorder{}
	m := New(Options{
		AssignedHold: 2 * time.Second,
		DroneETA:     10 * time.Second,
		OnChange:     rec.on,
		Clock:        clk,
	})
	return m, clk, rec
}

func samePhases(t *testing.T, got, want []viewerv1.MissionPhase) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("phases: got %v want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("phases: got %v want %v", got, want)
		}
	}
}

func TestStartsIdle(t *testing.T) {
	m, _, _ := newMission()
	s := m.Status()
	if s.GetPhase() != idle {
		t.Fatalf("phase %v", s.GetPhase())
	}
	if s.GetMessage() == "" {
		t.Fatal("idle has no operator message")
	}
	if s.EtaSeconds != nil {
		t.Fatal("idle must not carry an eta")
	}
}

func TestFullScenario(t *testing.T) {
	m, clk, rec := newMission()

	m.RouteAssigned()
	if m.Phase() != assigned {
		t.Fatalf("after assign: %v", m.Phase())
	}
	if got := rec.last().GetMessage(); got != "태스크 지정 완료" {
		t.Fatalf("assigned message %q", got)
	}

	clk.Advance(1999 * time.Millisecond)
	if m.Phase() != assigned {
		t.Fatalf("hold ended early: %v", m.Phase())
	}
	clk.Advance(time.Millisecond)
	if m.Phase() != awaiting {
		t.Fatalf("after hold: %v", m.Phase())
	}
	if eta := rec.last().EtaSeconds; eta == nil || *eta != 10 {
		t.Fatalf("eta at wait start: %v", eta)
	}

	clk.Advance(4 * time.Second)
	if eta := m.Status().EtaSeconds; eta == nil || *eta != 6 {
		t.Fatalf("eta mid-wait: %v", eta)
	}
	clk.Advance(30 * time.Second)
	if eta := m.Status().EtaSeconds; eta == nil || *eta != 0 {
		t.Fatalf("eta must not go negative: %v", eta)
	}

	m.Hello(1, center)
	if m.Phase() != active {
		t.Fatalf("after hello: %v", m.Phase())
	}
	last := rec.last()
	if last.GetDronesOnline() != 1 || last.EtaSeconds != nil {
		t.Fatalf("active status: %+v", last)
	}
	samePhases(t, rec.phases(), []viewerv1.MissionPhase{assigned, awaiting, active})
}

func TestHelloWhileIdleDoesNotStartMission(t *testing.T) {
	m, clk, rec := newMission()
	m.Hello(7, center)
	if m.Phase() != idle {
		t.Fatalf("hello without a route moved the mission to %v", m.Phase())
	}
	if rec.last().GetDronesOnline() != 1 {
		t.Fatal("count update must still be published")
	}

	// A drone already on station turns the wait into active immediately.
	m.RouteAssigned()
	clk.Advance(2 * time.Second)
	if m.Phase() != active {
		t.Fatalf("drone on station before the route: %v", m.Phase())
	}
	samePhases(t, rec.phases(), []viewerv1.MissionPhase{idle, assigned, awaiting, active})
}

func TestHelloDuringAssignedHoldWaitsForHold(t *testing.T) {
	m, clk, _ := newMission()
	m.RouteAssigned()
	m.Hello(1, center)
	if m.Phase() != assigned {
		t.Fatalf("hello cut the hold short: %v", m.Phase())
	}
	clk.Advance(2 * time.Second)
	if m.Phase() != active {
		t.Fatalf("after hold with a drone online: %v", m.Phase())
	}
}

func TestLastDroneGoneReturnsToWaiting(t *testing.T) {
	m, clk, rec := newMission()
	m.RouteAssigned()
	clk.Advance(2 * time.Second)
	m.Hello(1, center)
	m.Hello(2, modelv1.DroneStation_DRONE_STATION_LEFT)
	if m.Online() != 2 {
		t.Fatalf("online %d", m.Online())
	}

	m.Gone(1)
	if m.Phase() != active || rec.last().GetDronesOnline() != 1 {
		t.Fatalf("one of two gone: phase %v online %d", m.Phase(), rec.last().GetDronesOnline())
	}

	clk.Advance(5 * time.Second)
	m.Gone(2)
	if m.Phase() != awaiting {
		t.Fatalf("last drone gone: %v", m.Phase())
	}
	if eta := rec.last().EtaSeconds; eta == nil || *eta != 10 {
		t.Fatalf("eta must restart: %v", eta)
	}

	// Unknown drones are ignored.
	n := len(rec.phases())
	m.Gone(99)
	if len(rec.phases()) != n {
		t.Fatal("gone for an unknown drone published a status")
	}
}

func TestReassignRestartsHold(t *testing.T) {
	m, clk, _ := newMission()
	m.RouteAssigned()
	clk.Advance(1500 * time.Millisecond)
	m.RouteAssigned()
	clk.Advance(1500 * time.Millisecond)
	if m.Phase() != assigned {
		t.Fatalf("old hold timer fired after reassign: %v", m.Phase())
	}
	clk.Advance(500 * time.Millisecond)
	if m.Phase() != awaiting {
		t.Fatalf("new hold did not elapse: %v", m.Phase())
	}
}

func TestResetGoesIdleAndCancelsHold(t *testing.T) {
	m, clk, _ := newMission()
	m.RouteAssigned()
	m.Reset()
	clk.Advance(5 * time.Second)
	if m.Phase() != idle {
		t.Fatalf("after reset: %v", m.Phase())
	}
}

func TestNoEtaWhenDisabled(t *testing.T) {
	clk := newFakeClock()
	m := New(Options{AssignedHold: time.Second, Clock: clk})
	m.RouteAssigned()
	clk.Advance(time.Second)
	if m.Phase() != awaiting || m.Status().EtaSeconds != nil {
		t.Fatalf("phase %v eta %v", m.Phase(), m.Status().EtaSeconds)
	}
}

func TestDronesSorted(t *testing.T) {
	m, _, _ := newMission()
	m.Hello(3, center)
	m.Hello(1, center)
	m.Hello(2, center)
	got := m.Drones()
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Fatalf("drones %v", got)
	}
}

// Two drones saying hello at the same time: the status that leaves last must be
// the one with both drones. Statuses used to be computed under the lock and
// published after it, so the count of the first hello could reach the screens
// after the count of the second and stay there.
func TestConcurrentHellosPublishInOrder(t *testing.T) {
	var (
		mu      sync.Mutex
		calls   int
		counts  []uint32
		blocked = make(chan struct{})
		release = make(chan struct{})
	)
	m := New(Options{OnChange: func(s *viewerv1.MissionStatus) {
		mu.Lock()
		calls++
		first := calls == 1
		mu.Unlock()
		if first {
			close(blocked)
			<-release // the first publish is slow; the second hello runs meanwhile
		}
		mu.Lock()
		counts = append(counts, s.GetDronesOnline())
		mu.Unlock()
	}})
	done := make(chan struct{})
	go func() { m.Hello(1, modelv1.DroneStation_DRONE_STATION_LEFT); close(done) }()
	<-blocked
	second := make(chan struct{})
	go func() { m.Hello(2, modelv1.DroneStation_DRONE_STATION_CENTER); close(second) }()
	time.Sleep(50 * time.Millisecond)
	close(release)
	<-done
	<-second
	mu.Lock()
	defer mu.Unlock()
	if len(counts) != 2 || counts[0] != 1 || counts[1] != 2 {
		t.Fatalf("published drone counts %v, want [1 2]", counts)
	}
}
