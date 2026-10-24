package backoff

import (
	"testing"
	"time"
)

func TestBackoffDoublesAndCaps(t *testing.T) {
	b := Backoff{Min: 100 * time.Millisecond, Max: 350 * time.Millisecond}
	want := []time.Duration{100, 200, 350, 350}
	for i, w := range want {
		if got := b.Next(); got != w*time.Millisecond {
			t.Fatalf("step %d: got %v want %v", i, got, w*time.Millisecond)
		}
	}
	b.Reset()
	if got := b.Next(); got != 100*time.Millisecond {
		t.Fatalf("after reset: got %v", got)
	}
}

func TestBackoffZeroValue(t *testing.T) {
	var b Backoff
	if got := b.Next(); got != 500*time.Millisecond {
		t.Fatalf("zero value min: got %v", got)
	}
	for i := 0; i < 10; i++ {
		b.Next()
	}
	if got := b.Next(); got != 10*time.Second {
		t.Fatalf("zero value max: got %v", got)
	}
}
