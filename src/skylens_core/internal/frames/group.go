package frames

import (
	"context"
	"sort"
	"sync"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
)

// Grouper bundles frames of the formation cameras taken at the same moment
// into FrameSets.
//
// Time is cut into buckets of Period (one bucket per extracted frame). The
// bucket index is the FrameSet sequence. A bucket is sent as revision 0 as
// soon as every expected camera has a frame in it, or after MaxWait with the
// cameras it has. A camera that shows up late for a bucket already sent makes
// the bucket go out again with the next revision; the model server waits for
// missing cameras and treats a newer revision as superseding the old one.
type Grouper struct {
	SourceID string        // FrameKey.source_id
	Period   time.Duration // bucket width, normally 1 / fps
	MaxWait  time.Duration // how long a partial bucket waits for the rest
	// Expected lists the cameras that should appear in every set.
	Expected func() []string
	// Emit receives each set. It must not block for long.
	Emit func(*modelv1.FrameSet)
	// Now defaults to time.Now.
	Now func() time.Time

	mu      sync.Mutex
	buckets map[uint64]*bucket
}

type bucket struct {
	firstSeen time.Time
	frames    map[string]Frame
	sent      bool
	revision  uint32
	sentCount int
}

func (g *Grouper) now() time.Time {
	if g.Now != nil {
		return g.Now()
	}
	return time.Now()
}

func (g *Grouper) period() time.Duration {
	if g.Period <= 0 {
		return 500 * time.Millisecond
	}
	return g.Period
}

func (g *Grouper) maxWait() time.Duration {
	if g.MaxWait <= 0 {
		return 3 * g.period()
	}
	return g.MaxWait
}

// Sequence is the bucket index of a capture time.
func (g *Grouper) Sequence(t time.Time) uint64 {
	ms := g.period().Milliseconds()
	if ms < 1 {
		ms = 1
	}
	return uint64(t.UnixMilli()) / uint64(ms)
}

// Add files a frame into its bucket and sends the bucket if that completes it.
func (g *Grouper) Add(f Frame) {
	g.mu.Lock()
	if g.buckets == nil {
		g.buckets = map[uint64]*bucket{}
	}
	seq := g.Sequence(f.CapturedAt)
	b := g.buckets[seq]
	if b == nil {
		b = &bucket{firstSeen: g.now(), frames: map[string]Frame{}}
		g.buckets[seq] = b
	}
	var out *modelv1.FrameSet
	if _, had := b.frames[f.Camera]; had && b.sent {
		// Already in a sent set; a second frame of the same camera in the
		// same bucket adds nothing.
		g.mu.Unlock()
		return
	}
	b.frames[f.Camera] = f // keep the latest while unsent
	switch {
	case !b.sent && g.complete(b):
		out = g.buildLocked(seq, b)
	case b.sent && len(b.frames) > b.sentCount:
		b.revision++
		out = g.buildLocked(seq, b)
	}
	g.mu.Unlock()
	if out != nil && g.Emit != nil {
		g.Emit(out)
	}
}

// Flush sends partial buckets that waited MaxWait and forgets old ones.
func (g *Grouper) Flush() {
	now := g.now()
	g.mu.Lock()
	var out []*modelv1.FrameSet
	keep := 4 * g.maxWait()
	for seq, b := range g.buckets {
		age := now.Sub(b.firstSeen)
		if !b.sent && age >= g.maxWait() && len(b.frames) > 0 {
			out = append(out, g.buildLocked(seq, b))
		}
		if age >= keep {
			delete(g.buckets, seq)
		}
	}
	g.mu.Unlock()
	sort.Slice(out, func(i, j int) bool { return out[i].GetKey().GetSequence() < out[j].GetKey().GetSequence() })
	if g.Emit != nil {
		for _, fs := range out {
			g.Emit(fs)
		}
	}
}

// Run flushes every Period until ctx is done.
func (g *Grouper) Run(ctx context.Context) {
	t := time.NewTicker(g.period())
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			g.Flush()
		}
	}
}

func (g *Grouper) complete(b *bucket) bool {
	var want []string
	if g.Expected != nil {
		want = g.Expected()
	}
	if len(want) == 0 {
		return len(b.frames) > 0
	}
	for _, c := range want {
		if _, ok := b.frames[c]; !ok {
			return false
		}
	}
	return true
}

func (g *Grouper) buildLocked(seq uint64, b *bucket) *modelv1.FrameSet {
	b.sent = true
	b.sentCount = len(b.frames)
	cams := make([]string, 0, len(b.frames))
	for c := range b.frames {
		cams = append(cams, c)
	}
	sort.Strings(cams)
	fs := &modelv1.FrameSet{
		Key: &modelv1.FrameKey{SourceId: g.SourceID, Sequence: seq, Revision: b.revision},
	}
	var earliest time.Time
	for _, c := range cams {
		f := b.frames[c]
		img := &modelv1.LiveImage{
			Camera: c,
			Format: f.Format,
			Data:   f.Data,
			Width:  f.Width,
			Height: f.Height,
			Gps:    f.Gps,
		}
		fs.Images = append(fs.Images, img)
		if earliest.IsZero() || f.CapturedAt.Before(earliest) {
			earliest = f.CapturedAt
		}
	}
	fs.CapturedAtMs = earliest.UnixMilli()
	return fs
}
