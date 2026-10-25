package core

import (
	"testing"
	"time"
)

func TestInputEndsAfterIdleOnlyOnceFramesCame(t *testing.T) {
	t0 := time.Unix(1000, 0)
	w := inputWatch{idle: 20 * time.Second}

	// No frame yet: a mission waiting for its drones is not over.
	if w.due(t0.Add(time.Hour), 0) {
		t.Fatal("ended a run that never had input")
	}
	if late := w.seen(t0); late {
		t.Fatal("first frame counted as late")
	}
	if w.due(t0.Add(19*time.Second), 0) {
		t.Fatal("ended before the idle time")
	}
	if !w.due(t0.Add(20*time.Second), 0) {
		t.Fatal("did not end after the idle time")
	}
	// Once only.
	if w.due(t0.Add(time.Minute), 0) {
		t.Fatal("ended twice")
	}
	// A frame after the end is late; the run stays ended.
	if late := w.seen(t0.Add(time.Minute)); !late {
		t.Fatal("frame after the end not reported as late")
	}
	if w.due(t0.Add(2*time.Minute), 0) {
		t.Fatal("ended again after a late frame")
	}
}

func TestInputFramesKeepTheRunOpen(t *testing.T) {
	t0 := time.Unix(1000, 0)
	w := inputWatch{idle: 20 * time.Second}
	for s := 0; s <= 60; s += 10 {
		w.seen(t0.Add(time.Duration(s) * time.Second))
		if w.due(t0.Add(time.Duration(s+15)*time.Second), 0) {
			t.Fatalf("ended at %ds with a frame every 10s", s+15)
		}
	}
}

func TestInputNewMissionStartsOver(t *testing.T) {
	t0 := time.Unix(1000, 0)
	w := inputWatch{idle: 20 * time.Second}
	w.seen(t0)
	if !w.due(t0.Add(30*time.Second), 0) {
		t.Fatal("did not end")
	}
	w.reset()
	if w.due(t0.Add(time.Hour), 0) {
		t.Fatal("a new mission without frames ended")
	}
	if late := w.seen(t0.Add(time.Hour)); late {
		t.Fatal("first frame of the new mission counted as late")
	}
	if !w.due(t0.Add(time.Hour+20*time.Second), 0) {
		t.Fatal("the new mission did not end")
	}
}

func TestInputFailedEndIsTriedAgainAfterAnotherIdle(t *testing.T) {
	t0 := time.Unix(1000, 0)
	w := inputWatch{idle: 20 * time.Second}
	w.seen(t0)
	if !w.due(t0.Add(20*time.Second), 0) {
		t.Fatal("did not end")
	}
	w.failed(t0.Add(21 * time.Second))
	if w.due(t0.Add(30*time.Second), 0) {
		t.Fatal("retried before another idle period")
	}
	if !w.due(t0.Add(41*time.Second), 0) {
		t.Fatal("did not retry")
	}
}

func TestInputWaitsForQueuedFrames(t *testing.T) {
	t0 := time.Unix(1000, 0)
	w := inputWatch{idle: 20 * time.Second}
	w.seen(t0)
	// The feeds went quiet long ago but sets are still on their way to the model.
	if w.due(t0.Add(5*time.Minute), 40) {
		t.Fatal("ended with frame sets still queued")
	}
	if !w.due(t0.Add(6*time.Minute), 0) {
		t.Fatal("did not end once the queue was empty")
	}
}

func TestInputIdleZeroNeverEnds(t *testing.T) {
	t0 := time.Unix(1000, 0)
	w := inputWatch{}
	w.seen(t0)
	if w.due(t0.Add(24*time.Hour), 0) {
		t.Fatal("ended with the watch off")
	}
}
