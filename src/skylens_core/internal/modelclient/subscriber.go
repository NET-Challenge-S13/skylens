package modelclient

import (
	"context"
	"errors"
	"io"
	"log"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/backoff"
)

// Subscriber keeps SubscribeChunks open with resume.
type Subscriber struct {
	API API
	// After returns the resume point: the last sequence already stored.
	After func() uint64
	// OnChunk receives every chunk in stream order.
	OnChunk func(*modelv1.Chunk)
	// OnState reports connect / disconnect transitions (err is nil on connect).
	OnState func(connected bool, err error)
	// Backoff between reconnects; the zero value is 500 ms .. 10 s.
	Backoff backoff.Backoff
}

// Run blocks until ctx is done.
func (s *Subscriber) Run(ctx context.Context) {
	connected := false
	setState := func(c bool, err error) {
		if c == connected {
			return
		}
		connected = c
		if s.OnState != nil {
			s.OnState(c, err)
		}
	}
	for ctx.Err() == nil {
		after := uint64(0)
		if s.After != nil {
			after = s.After()
		}
		err := s.once(ctx, after, func() { setState(true, nil) })
		if ctx.Err() != nil {
			return
		}
		setState(false, err)
		if !s.Backoff.Wait(ctx) {
			return
		}
	}
}

// once runs one stream until it fails. A stream that delivered anything resets
// the backoff, so a long healthy session reconnects quickly.
func (s *Subscriber) once(ctx context.Context, after uint64, up func()) error {
	sctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := s.API.SubscribeChunks(sctx, after)
	if err != nil {
		return err
	}
	up()
	log.Printf("[core] model chunks subscribed (after_sequence=%d)", after)
	opened := time.Now()
	for {
		c, err := stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				err = errors.New("model server ended the chunk stream")
			}
			if time.Since(opened) > 5*time.Second {
				s.Backoff.Reset()
			}
			return err
		}
		s.Backoff.Reset()
		if s.OnChunk != nil {
			s.OnChunk(c)
		}
	}
}
