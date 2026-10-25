package core

import (
	"context"
	"log"
	"sync"
	"time"
)

// inputWatch decides when the live input of a run is over: the run has had
// frames, and no frame from any feed has come for idle. A drone that stays
// connected after its footage ends (the demo's --no-loop) never closes its
// uplink, so the end is read off the frames, not off the drones.
//
// The model only closes the last zone and sends the final message when told, so
// without this a run that ends between zones keeps its last zone "in progress"
// for good.
type inputWatch struct {
	idle time.Duration // zero turns the watch off

	mu    sync.Mutex
	last  time.Time // the last frame, or the last failed attempt to end
	had   bool      // a frame came in this run
	ended bool      // EndInput was asked for this run
}

// seen records a frame. It reports whether the frame came after the run was
// ended: the model ignores it until the next mission.
func (w *inputWatch) seen(now time.Time) (late bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.last, w.had = now, true
	return w.ended
}

// due reports, once per run, that the input is over. queued is the number of
// frame sets still waiting to go to the model: the model runs slower than the
// video, so when the feeds go quiet much of the run is often still queued, and
// ending before it is sent would throw it away. The caller then asks the model
// to end the run.
func (w *inputWatch) due(now time.Time, queued int) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.idle <= 0 || !w.had || w.ended || queued > 0 || now.Sub(w.last) < w.idle {
		return false
	}
	w.ended = true
	return true
}

// failed undoes an end the model did not take; it is asked again after another
// idle period.
func (w *inputWatch) failed(now time.Time) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.ended, w.last = false, now
}

// reset starts a new run: no frames yet, not ended.
func (w *inputWatch) reset() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.had, w.ended = false, false
}

// noteFrameSet passes a grouped frame set on to the model, noting that input
// came. A set after the run was ended is still sent (the model ignores it), but
// the log says why the screens do not change.
func (c *Core) noteFrameSet(late bool) {
	if !late {
		return
	}
	c.mu.Lock()
	first := !c.lateLogged
	c.lateLogged = true
	c.mu.Unlock()
	if first {
		log.Printf("[core] frames came after the input of this run was ended; the model ignores them until a new route is assigned")
	}
}

// inputLoop ends the model's run once the input has been idle (inputWatch).
func (c *Core) inputLoop(ctx context.Context) {
	t := time.NewTicker(time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-t.C:
			if c.input.due(now, c.pusher.Stats().Queued) {
				go c.endInput(ctx)
			}
		}
	}
}

func (c *Core) endInput(ctx context.Context) {
	log.Printf("[core] no frame for %s: ending the input of this run so the model closes the last zone", c.input.idle)
	call, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	res, err := c.model.EndInput(call)
	if err != nil {
		c.input.failed(time.Now())
		log.Printf("[core] the model did not end the run (%v); trying again after another %s without frames", err, c.input.idle)
		return
	}
	if res.GetAlreadyEnded() {
		log.Printf("[core] the model had already ended this run")
	}
}
