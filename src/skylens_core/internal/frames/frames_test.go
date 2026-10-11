package frames

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
)

func jpeg(body ...byte) []byte {
	out := []byte{0xFF, 0xD8}
	out = append(out, body...)
	return append(out, 0xFF, 0xD9)
}

func TestSplitJPEG(t *testing.T) {
	a := jpeg(0x01, 0xFF, 0x00, 0xD9, 0x02) // stuffed FF 00 followed by D9 is data
	b := jpeg(0xFF, 0xE0, 0x00, 0x10)
	var stream bytes.Buffer
	stream.Write([]byte{0x00, 0x12}) // junk before the first SOI
	stream.Write(a)
	stream.Write(b)
	stream.Write([]byte{0xFF, 0xD8, 0x05}) // truncated tail

	var got [][]byte
	if err := SplitJPEG(&stream, 0, func(img []byte) { got = append(got, img) }); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || !bytes.Equal(got[0], a) || !bytes.Equal(got[1], b) {
		t.Fatalf("split %x", got)
	}
}

func TestSplitJPEGSizeCap(t *testing.T) {
	big := jpeg(bytes.Repeat([]byte{0x01}, 100)...)
	err := SplitJPEG(bytes.NewReader(big), 50, func([]byte) {})
	if !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("want size cap error, got %v", err)
	}
}

func TestFFmpegArgs(t *testing.T) {
	f := &FFmpeg{URL: "srt://127.0.0.1:10890?streamid=read:drone1", FPS: 2.5, Quality: 4}
	args := strings.Join(f.Args(), " ")
	for _, want := range []string{"-i srt://127.0.0.1:10890?streamid=read:drone1", "fps=2.5", "-q:v 4", "-f image2pipe pipe:1"} {
		if !strings.Contains(args, want) {
			t.Fatalf("args %q lack %q", args, want)
		}
	}
}

type clock struct{ t time.Time }

func (c *clock) now() time.Time { return c.t }

func newGrouper(expected []string) (*Grouper, *clock, *[]*modelv1.FrameSet) {
	clk := &clock{t: time.UnixMilli(1_000_000)}
	var out []*modelv1.FrameSet
	g := &Grouper{
		SourceID: "formation",
		Period:   500 * time.Millisecond,
		MaxWait:  time.Second,
		Expected: func() []string { return expected },
		Emit:     func(fs *modelv1.FrameSet) { out = append(out, fs) },
		Now:      clk.now,
	}
	return g, clk, &out
}

func frame(cam string, at time.Time) Frame {
	return Frame{Camera: cam, Format: modelv1.ImageFormat_IMAGE_FORMAT_JPEG, Data: []byte(cam), CapturedAt: at}
}

func cams(fs *modelv1.FrameSet) string {
	var s []string
	for _, img := range fs.GetImages() {
		s = append(s, img.GetCamera())
	}
	return strings.Join(s, ",")
}

func TestGrouperCompleteSetGoesOutImmediately(t *testing.T) {
	g, clk, out := newGrouper([]string{"center", "left", "right"})
	at := clk.t.Add(100 * time.Millisecond)
	g.Add(frame("left", at))
	g.Add(frame("center", at.Add(50*time.Millisecond)))
	if len(*out) != 0 {
		t.Fatal("incomplete set sent")
	}
	g.Add(frame("right", at.Add(80*time.Millisecond)))
	if len(*out) != 1 {
		t.Fatalf("sets %d", len(*out))
	}
	fs := (*out)[0]
	if cams(fs) != "center,left,right" || fs.GetKey().GetRevision() != 0 ||
		fs.GetKey().GetSourceId() != "formation" || fs.GetKey().GetSequence() != g.Sequence(at) {
		t.Fatalf("set %v", fs)
	}
	if fs.GetCapturedAtMs() != at.UnixMilli() {
		t.Fatalf("captured_at %d want earliest %d", fs.GetCapturedAtMs(), at.UnixMilli())
	}
}

func TestGrouperPartialAfterWaitThenRevision(t *testing.T) {
	g, clk, out := newGrouper([]string{"center", "left"})
	at := clk.t
	g.Add(frame("center", at))
	clk.t = clk.t.Add(999 * time.Millisecond)
	g.Flush()
	if len(*out) != 0 {
		t.Fatal("partial set sent before MaxWait")
	}
	clk.t = clk.t.Add(time.Millisecond)
	g.Flush()
	if len(*out) != 1 || cams((*out)[0]) != "center" || (*out)[0].GetKey().GetRevision() != 0 {
		t.Fatalf("partial %v", *out)
	}
	g.Flush()
	if len(*out) != 1 {
		t.Fatal("partial set sent twice")
	}

	// The late camera re-sends the same sequence as revision 1.
	g.Add(frame("left", at.Add(10*time.Millisecond)))
	if len(*out) != 2 {
		t.Fatalf("sets %d", len(*out))
	}
	rev := (*out)[1]
	if cams(rev) != "center,left" || rev.GetKey().GetRevision() != 1 ||
		rev.GetKey().GetSequence() != (*out)[0].GetKey().GetSequence() {
		t.Fatalf("revision %v", rev)
	}

	// A duplicate frame of a camera already sent adds nothing.
	g.Add(frame("left", at.Add(20*time.Millisecond)))
	if len(*out) != 2 {
		t.Fatal("duplicate camera re-sent the set")
	}
}

func TestGrouperForgetsOldBuckets(t *testing.T) {
	g, clk, _ := newGrouper([]string{"a"})
	g.Add(frame("a", clk.t))
	clk.t = clk.t.Add(4 * time.Second)
	g.Flush()
	g.mu.Lock()
	n := len(g.buckets)
	g.mu.Unlock()
	if n != 0 {
		t.Fatalf("%d bucket(s) kept", n)
	}
}

func TestManagerRunsSourcesAndInjects(t *testing.T) {
	var mu sync.Mutex
	var got []Frame
	started := make(chan uint32, 4)
	factory := func(id uint32, cam string) Source {
		return SourceFunc(func(ctx context.Context, emit func(Frame)) error {
			started <- id
			emit(Frame{Data: []byte{1}})
			<-ctx.Done()
			return nil
		})
	}
	m := NewManager(factory, func(f Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if !m.Start(ctx, 1, "left") || !m.Start(ctx, 2, "right") {
		t.Fatal("start failed")
	}
	if m.Start(ctx, 1, "left") {
		t.Fatal("second start of the same drone")
	}
	<-started
	<-started
	if c := m.Cameras(); len(c) != 2 || c[0] != "left" || c[1] != "right" {
		t.Fatalf("cameras %v", c)
	}
	m.Inject(Frame{DroneID: 9, Camera: "demo"})
	m.Stop(1)
	if c := m.Cameras(); len(c) != 1 || c[0] != "right" {
		t.Fatalf("cameras after stop %v", c)
	}
	m.StopAll()

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 3 {
		t.Fatalf("frames %d", len(got))
	}
	for _, f := range got {
		if f.CapturedAt.IsZero() || f.Camera == "" || f.DroneID == 0 {
			t.Fatalf("frame not filled in: %+v", f)
		}
	}
}
