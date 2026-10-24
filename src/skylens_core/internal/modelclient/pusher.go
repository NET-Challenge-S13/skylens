package modelclient

import (
	"context"
	"errors"
	"io"
	"log"
	"sync"
	"sync/atomic"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/backoff"
)

// PusherStats is a snapshot of the push counters.
type PusherStats struct {
	Queued    int    // sets waiting to be sent
	Sent      uint64 // Send calls that succeeded (including re-declarations)
	Dropped   uint64 // sets dropped because the queue was full
	Redeclare uint64 // sets declared again after a reconnect
	Connected bool
	Last      *modelv1.PushFramesSummary // summary of the last closed stream
}

// Pusher keeps one PushFrames stream open.
type Pusher struct {
	API API
	// MaxQueued bounds the queue while the model is away; the oldest sets go
	// first. Zero means 256.
	MaxQueued int
	// Redeclare is how many recently sent sets are sent again after a
	// reconnect. Zero means 32.
	Redeclare int
	// OnState reports connect / disconnect transitions.
	OnState func(connected bool, err error)
	Backoff backoff.Backoff

	mu      sync.Mutex
	queue   []*modelv1.FrameSet
	recent  []*modelv1.FrameSet
	notify  chan struct{}
	once    sync.Once
	last    *modelv1.PushFramesSummary
	connect atomic.Bool

	sent      atomic.Uint64
	dropped   atomic.Uint64
	redeclare atomic.Uint64
}

func (p *Pusher) init() {
	p.once.Do(func() {
		p.notify = make(chan struct{}, 1)
		if p.MaxQueued <= 0 {
			p.MaxQueued = 256
		}
		if p.Redeclare <= 0 {
			p.Redeclare = 32
		}
	})
}

func sameSet(a, b *modelv1.FrameSet) bool {
	return a.GetKey().GetSourceId() == b.GetKey().GetSourceId() &&
		a.GetKey().GetSequence() == b.GetKey().GetSequence()
}

// Push queues a set without blocking. A queued set with the same
// (source_id, sequence) and a lower revision is replaced: the newer revision
// supersedes it anyway.
func (p *Pusher) Push(fs *modelv1.FrameSet) {
	p.init()
	p.mu.Lock()
	replaced := false
	for i, q := range p.queue {
		if sameSet(q, fs) {
			if fs.GetKey().GetRevision() >= q.GetKey().GetRevision() {
				p.queue[i] = fs
			}
			replaced = true
			break
		}
	}
	if !replaced {
		p.queue = append(p.queue, fs)
		if over := len(p.queue) - p.MaxQueued; over > 0 {
			p.queue = append([]*modelv1.FrameSet(nil), p.queue[over:]...)
			p.dropped.Add(uint64(over))
		}
	}
	p.mu.Unlock()
	select {
	case p.notify <- struct{}{}:
	default:
	}
}

// Stats returns the push counters.
func (p *Pusher) Stats() PusherStats {
	p.init()
	p.mu.Lock()
	defer p.mu.Unlock()
	return PusherStats{
		Queued:    len(p.queue),
		Sent:      p.sent.Load(),
		Dropped:   p.dropped.Load(),
		Redeclare: p.redeclare.Load(),
		Connected: p.connect.Load(),
		Last:      p.last,
	}
}

// Run blocks until ctx is done.
func (p *Pusher) Run(ctx context.Context) {
	p.init()
	for ctx.Err() == nil {
		err := p.session(ctx)
		if ctx.Err() != nil {
			return
		}
		if p.connect.Swap(false) && p.OnState != nil {
			p.OnState(false, err)
		}
		if !p.Backoff.Wait(ctx) {
			return
		}
	}
}

// session runs one stream: re-declare recent sets, then drain the queue as it
// fills. It returns when a Send fails or ctx ends.
func (p *Pusher) session(ctx context.Context) error {
	sctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := p.API.PushFrames(sctx)
	if err != nil {
		return err
	}

	p.mu.Lock()
	// Sets sent on the previous stream may never have been processed: put
	// them back in front of the queue (newer revisions already queued win).
	if len(p.recent) > 0 {
		var back []*modelv1.FrameSet
		for _, r := range p.recent {
			dup := false
			for _, q := range p.queue {
				if sameSet(q, r) {
					dup = true
					break
				}
			}
			if !dup {
				back = append(back, r)
			}
		}
		p.redeclare.Add(uint64(len(back)))
		p.queue = append(back, p.queue...)
		p.recent = nil
	}
	p.mu.Unlock()

	for {
		p.mu.Lock()
		batch := p.queue
		p.queue = nil
		p.mu.Unlock()

		for i, fs := range batch {
			if err := stream.Send(fs); err != nil {
				// Put back what was not confirmed sent.
				p.mu.Lock()
				p.queue = append(append([]*modelv1.FrameSet(nil), batch[i:]...), p.queue...)
				p.mu.Unlock()
				return p.closeStream(stream, err)
			}
			if !p.connect.Swap(true) {
				log.Printf("[core] model frame stream open")
				if p.OnState != nil {
					p.OnState(true, nil)
				}
				p.Backoff.Reset()
			}
			p.sent.Add(1)
			p.mu.Lock()
			p.recent = append(p.recent, fs)
			if over := len(p.recent) - p.Redeclare; over > 0 {
				p.recent = p.recent[over:]
			}
			p.mu.Unlock()
		}

		select {
		case <-ctx.Done():
			_ = p.closeStream(stream, nil)
			return ctx.Err()
		case <-p.notify:
		}
	}
}

// closeStream half-closes and collects the summary. For a broken stream the
// real error comes from CloseAndRecv, not from the Send that hit io.EOF.
func (p *Pusher) closeStream(stream FrameStream, sendErr error) error {
	sum, err := stream.CloseAndRecv()
	if sum != nil {
		p.mu.Lock()
		p.last = sum
		p.mu.Unlock()
		log.Printf("[core] model frame stream closed: ingested %d, ignored %d, pending %d",
			sum.GetIngested(), sum.GetIgnored(), sum.GetPending())
	}
	if err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	if sendErr != nil {
		return sendErr
	}
	return errors.New("model server closed the frame stream")
}
