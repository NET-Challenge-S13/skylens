// Package backoff is a tiny exponential backoff shared by every reconnect loop
// in the core (model streams, ffmpeg restarts).
package backoff

import (
	"context"
	"time"
)

// Backoff doubles its delay after every failure, up to Max. Reset puts it back
// at Min after a success. The zero value uses 500 ms .. 10 s.
type Backoff struct {
	Min time.Duration
	Max time.Duration

	cur time.Duration
}

func (b *Backoff) bounds() (time.Duration, time.Duration) {
	lo, hi := b.Min, b.Max
	if lo <= 0 {
		lo = 500 * time.Millisecond
	}
	if hi < lo {
		hi = 10 * time.Second
		if hi < lo {
			hi = lo
		}
	}
	return lo, hi
}

// Next returns the delay to wait before the next attempt and advances.
func (b *Backoff) Next() time.Duration {
	lo, hi := b.bounds()
	if b.cur < lo {
		b.cur = lo
		return b.cur
	}
	b.cur *= 2
	if b.cur > hi {
		b.cur = hi
	}
	return b.cur
}

// Reset is called after a success.
func (b *Backoff) Reset() { b.cur = 0 }

// Wait sleeps for the next delay or until ctx is done. It reports false when
// ctx ended the wait.
func (b *Backoff) Wait(ctx context.Context) bool {
	t := time.NewTimer(b.Next())
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
