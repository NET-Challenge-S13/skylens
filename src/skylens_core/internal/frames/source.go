// Package frames turns the drones' live video into FrameSets for the model
// server.
//
// A Source produces still frames of one camera (one drone). The default
// source is an ffmpeg subprocess that reads the drone's stream from the local
// MediaMTX and writes JPEG frames at a fixed rate, but anything that calls
// emit works: the demo, a test, or a future in-process decoder. The Manager
// runs one Source per drone on station, and the Grouper bundles frames of the
// formation cameras taken at the same moment into one FrameSet keyed
// (source_id, sequence, revision).
package frames

import (
	"context"
	"log"
	"sort"
	"sync"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
)

// Frame is one still image of one camera.
type Frame struct {
	DroneID    uint32
	Camera     string // LiveImage.camera
	Format     modelv1.ImageFormat
	Data       []byte
	Width      uint32 // required for RGB8 only
	Height     uint32
	CapturedAt time.Time
	Gps        *modelv1.Gps // where the drone was, if known
}

// Source produces frames until ctx is done or it fails for good. It calls emit
// from any goroutine; emit does not block for long.
type Source interface {
	Run(ctx context.Context, emit func(Frame)) error
}

// SourceFunc adapts a function to Source.
type SourceFunc func(ctx context.Context, emit func(Frame)) error

// Run calls f.
func (f SourceFunc) Run(ctx context.Context, emit func(Frame)) error { return f(ctx, emit) }

// Factory builds the source of one drone. Returning nil means no source.
type Factory func(droneID uint32, camera string) Source

// Manager runs one Source per drone.
type Manager struct {
	factory Factory
	sink    func(Frame)

	mu      sync.Mutex
	running map[uint32]*running
}

type running struct {
	camera string
	cancel context.CancelFunc
	done   chan struct{}
}

// NewManager returns a manager feeding sink. factory may be nil, in which case
// only Inject delivers frames.
func NewManager(factory Factory, sink func(Frame)) *Manager {
	return &Manager{factory: factory, sink: sink, running: map[uint32]*running{}}
}

// Start runs the drone's source unless it already runs. It reports whether a
// source was started.
func (m *Manager) Start(parent context.Context, droneID uint32, camera string) bool {
	if m.factory == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if r, ok := m.running[droneID]; ok && r.camera == camera {
		return false
	} else if ok {
		r.cancel()
		delete(m.running, droneID)
	}
	src := m.factory(droneID, camera)
	if src == nil {
		return false
	}
	ctx, cancel := context.WithCancel(parent)
	r := &running{camera: camera, cancel: cancel, done: make(chan struct{})}
	m.running[droneID] = r
	go func() {
		defer close(r.done)
		err := src.Run(ctx, func(f Frame) {
			f.DroneID = droneID
			if f.Camera == "" {
				f.Camera = camera
			}
			m.Inject(f)
		})
		if err != nil && ctx.Err() == nil {
			log.Printf("[core] frame source of drone %d (%s) stopped: %v", droneID, camera, err)
		}
		m.mu.Lock()
		if m.running[droneID] == r {
			delete(m.running, droneID)
		}
		m.mu.Unlock()
	}()
	return true
}

// Stop ends the drone's source.
func (m *Manager) Stop(droneID uint32) {
	m.mu.Lock()
	r, ok := m.running[droneID]
	delete(m.running, droneID)
	m.mu.Unlock()
	if ok {
		r.cancel()
	}
}

// StopAll ends every source and waits for them.
func (m *Manager) StopAll() {
	m.mu.Lock()
	rs := make([]*running, 0, len(m.running))
	for id, r := range m.running {
		rs = append(rs, r)
		delete(m.running, id)
	}
	m.mu.Unlock()
	for _, r := range rs {
		r.cancel()
		<-r.done
	}
}

// Cameras lists the cameras of running sources, sorted. The Grouper waits for
// these before a set is complete.
func (m *Manager) Cameras() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, 0, len(m.running))
	for _, r := range m.running {
		out = append(out, r.camera)
	}
	sort.Strings(out)
	return out
}

// Inject delivers a frame that did not come from a managed source (the demo,
// tests).
func (m *Manager) Inject(f Frame) {
	if f.CapturedAt.IsZero() {
		f.CapturedAt = time.Now()
	}
	if m.sink != nil {
		m.sink(f)
	}
}
