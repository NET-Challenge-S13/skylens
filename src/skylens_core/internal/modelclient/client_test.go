package modelclient

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/backoff"
)

var errDown = errors.New("model server down")

// fakeModel is a model server that can be taken down and brought back.
type fakeModel struct {
	mu      sync.Mutex
	up      bool
	chunks  []*modelv1.Chunk // the server's whole output, by sequence
	afters  []uint64         // after_sequence of every subscribe call
	streams []*fakeChunkStream
	got     []*modelv1.FrameSet // every set received, in order
	pushes  int
	failAt  int // fail the n-th Send across all streams (1-based), 0 = never
	sends   int
}

func (m *fakeModel) setUp(up bool) {
	m.mu.Lock()
	m.up = up
	streams := m.streams
	m.mu.Unlock()
	if !up {
		for _, s := range streams {
			s.kill()
		}
	}
}

type fakeChunkStream struct {
	ch   chan *modelv1.Chunk
	dead chan struct{}
	once sync.Once
}

func (s *fakeChunkStream) kill() { s.once.Do(func() { close(s.dead) }) }

func (s *fakeChunkStream) Recv() (*modelv1.Chunk, error) {
	select {
	case c := <-s.ch:
		return c, nil
	case <-s.dead:
		return nil, errDown
	}
}

func (m *fakeModel) SubscribeChunks(ctx context.Context, after uint64) (ChunkStream, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.afters = append(m.afters, after)
	if !m.up {
		return nil, errDown
	}
	s := &fakeChunkStream{ch: make(chan *modelv1.Chunk, 64), dead: make(chan struct{})}
	for _, c := range m.chunks {
		if c.GetSequence() > after {
			s.ch <- c
		}
	}
	m.streams = append(m.streams, s)
	go func() {
		<-ctx.Done()
		s.kill()
	}()
	return s, nil
}

type fakeFrameStream struct {
	m *fakeModel
}

func (f *fakeFrameStream) Send(fs *modelv1.FrameSet) error {
	f.m.mu.Lock()
	defer f.m.mu.Unlock()
	f.m.sends++
	if !f.m.up || (f.m.failAt > 0 && f.m.sends == f.m.failAt) {
		return io.EOF
	}
	f.m.got = append(f.m.got, fs)
	return nil
}

func (f *fakeFrameStream) CloseAndRecv() (*modelv1.PushFramesSummary, error) {
	f.m.mu.Lock()
	defer f.m.mu.Unlock()
	if !f.m.up {
		return nil, errDown
	}
	return &modelv1.PushFramesSummary{Ingested: uint64(len(f.m.got))}, nil
}

func (m *fakeModel) PushFrames(context.Context) (FrameStream, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pushes++
	if !m.up {
		return nil, errDown
	}
	return &fakeFrameStream{m: m}, nil
}

func (m *fakeModel) Health(context.Context) (*modelv1.HealthResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.up {
		return nil, errDown
	}
	return &modelv1.HealthResponse{Ok: true}, nil
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

var fast = backoff.Backoff{Min: 5 * time.Millisecond, Max: 20 * time.Millisecond}

func TestSubscriberResumesAfterLastStored(t *testing.T) {
	m := &fakeModel{up: false}
	for i := uint64(1); i <= 5; i++ {
		m.chunks = append(m.chunks, &modelv1.Chunk{Sequence: i})
	}

	var mu sync.Mutex
	var got []uint64
	var states []bool
	last := func() uint64 {
		mu.Lock()
		defer mu.Unlock()
		if len(got) == 0 {
			return 0
		}
		return got[len(got)-1]
	}
	s := &Subscriber{
		API:   m,
		After: last,
		OnChunk: func(c *modelv1.Chunk) {
			mu.Lock()
			got = append(got, c.GetSequence())
			mu.Unlock()
		},
		OnState: func(c bool, _ error) {
			mu.Lock()
			states = append(states, c)
			mu.Unlock()
		},
		Backoff: fast,
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)

	// Down at start: keeps retrying.
	waitFor(t, "retries while down", func() bool {
		m.mu.Lock()
		defer m.mu.Unlock()
		return len(m.afters) >= 2
	})

	m.setUp(true)
	waitFor(t, "first five chunks", func() bool { return last() == 5 })

	// Model restarts holding more output; the subscriber resumes after 5.
	m.setUp(false)
	m.mu.Lock()
	m.chunks = append(m.chunks, &modelv1.Chunk{Sequence: 6}, &modelv1.Chunk{Sequence: 7})
	m.mu.Unlock()
	m.setUp(true)
	waitFor(t, "resumed chunks", func() bool { return last() == 7 })

	mu.Lock()
	defer mu.Unlock()
	want := []uint64{1, 2, 3, 4, 5, 6, 7}
	if len(got) != len(want) {
		t.Fatalf("chunks %v want %v (no duplicates, no gaps)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("chunks %v want %v", got, want)
		}
	}
	m.mu.Lock()
	lastAfter := m.afters[len(m.afters)-1]
	m.mu.Unlock()
	if lastAfter != 5 {
		t.Fatalf("resume after_sequence %d, want 5", lastAfter)
	}
	if len(states) < 3 || !states[0] || states[1] || !states[2] {
		t.Fatalf("state transitions %v", states)
	}
}

func fs(seq uint64, rev uint32) *modelv1.FrameSet {
	return &modelv1.FrameSet{Key: &modelv1.FrameKey{SourceId: "formation", Sequence: seq, Revision: rev}}
}

func TestPusherQueuesWhileDownAndRedeclaresAfterReconnect(t *testing.T) {
	m := &fakeModel{up: false}
	p := &Pusher{API: m, MaxQueued: 3, Redeclare: 2, Backoff: fast}

	// Queued while down; the oldest is dropped past MaxQueued.
	for i := uint64(1); i <= 4; i++ {
		p.Push(fs(i, 0))
	}
	if st := p.Stats(); st.Queued != 3 || st.Dropped != 1 {
		t.Fatalf("stats while down %+v", st)
	}
	// A newer revision replaces the queued one instead of growing the queue.
	p.Push(fs(3, 1))
	if st := p.Stats(); st.Queued != 3 {
		t.Fatalf("revision grew the queue: %+v", st)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go p.Run(ctx)

	m.setUp(true)
	waitFor(t, "queued sets sent", func() bool { return p.Stats().Sent == 3 })
	m.mu.Lock()
	first := append([]*modelv1.FrameSet(nil), m.got...)
	m.mu.Unlock()
	if first[0].GetKey().GetSequence() != 2 || first[1].GetKey().GetRevision() != 1 {
		t.Fatalf("first session sent %v", first)
	}

	// The connection breaks on the next send: the set is kept, and after the
	// reconnect the two most recent sets are declared again before it.
	m.mu.Lock()
	m.failAt = m.sends + 1
	m.mu.Unlock()
	p.Push(fs(5, 0))
	waitFor(t, "re-declared and new set sent", func() bool {
		m.mu.Lock()
		defer m.mu.Unlock()
		return len(m.got) == 6
	})
	m.mu.Lock()
	tail := m.got[3:]
	m.mu.Unlock()
	if tail[0].GetKey().GetSequence() != 3 || tail[1].GetKey().GetSequence() != 4 || tail[2].GetKey().GetSequence() != 5 {
		t.Fatalf("after reconnect sent %v", tail)
	}
	if st := p.Stats(); st.Redeclare != 2 {
		t.Fatalf("redeclare count %+v", st)
	}
}
