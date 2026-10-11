package ws

import (
	"context"
	"errors"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"google.golang.org/protobuf/proto"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
)

const core = viewerv1.ComponentId_COMPONENT_ID_CORE

func chunkEnv(tag string) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_Chunk{Chunk: &modelv1.Chunk{Payload: []byte(tag)}}}
}

func decode(t *testing.T, frame []byte) *viewerv1.Envelope {
	t.Helper()
	env := &viewerv1.Envelope{}
	if err := proto.Unmarshal(frame, env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return env
}

func next(t *testing.T, p *Peer) *viewerv1.Envelope {
	t.Helper()
	done := make(chan struct{})
	timer := time.AfterFunc(time.Second, func() { close(done) })
	defer timer.Stop()
	f, err := p.Next(done)
	if err != nil {
		t.Fatalf("next: %v", err)
	}
	return decode(t, f)
}

func TestFramePrependsSeqWithoutReencodingBody(t *testing.T) {
	env := chunkEnv("payload")
	env.OriginTsMs = 1234
	body, err := Encode(env, core)
	if err != nil {
		t.Fatal(err)
	}
	got := decode(t, Frame(42, body))
	if got.GetSeq() != 42 || got.GetFrom() != core || got.GetOriginTsMs() != 1234 {
		t.Fatalf("envelope %+v", got)
	}
	if string(got.GetChunk().GetPayload()) != "payload" {
		t.Fatalf("payload %q", got.GetChunk().GetPayload())
	}
	// Encode must not touch the caller's envelope.
	if env.GetFrom() != viewerv1.ComponentId_COMPONENT_ID_UNSPECIFIED || env.GetSeq() != 0 {
		t.Fatalf("caller envelope mutated: %+v", env)
	}
}

func TestEncodeKeepsOriginAndSender(t *testing.T) {
	env := chunkEnv("x")
	env.From = viewerv1.ComponentId_COMPONENT_ID_DRONE
	body, _ := Encode(env, core)
	got := decode(t, Frame(1, body))
	if got.GetFrom() != viewerv1.ComponentId_COMPONENT_ID_DRONE {
		t.Fatalf("from %v", got.GetFrom())
	}
	if got.GetOriginTsMs() == 0 {
		t.Fatal("origin_ts_ms not filled in")
	}
}

func TestReplayThenLiveWithPerPeerSeq(t *testing.T) {
	h := NewHub(core)
	early := NewPeer("early", 0)
	h.Attach(early, nil)
	h.Broadcast(chunkEnv("live1"))

	late := NewPeer("late", 0)
	h.Attach(late, func() []*viewerv1.Envelope {
		return []*viewerv1.Envelope{chunkEnv("replay1"), chunkEnv("replay2")}
	})
	h.Broadcast(chunkEnv("live2"))

	for i, want := range []string{"replay1", "replay2", "live2"} {
		env := next(t, late)
		if env.GetSeq() != uint64(i+1) || string(env.GetChunk().GetPayload()) != want {
			t.Fatalf("late frame %d: seq %d payload %q", i, env.GetSeq(), env.GetChunk().GetPayload())
		}
	}
	for i, want := range []string{"live1", "live2"} {
		env := next(t, early)
		if env.GetSeq() != uint64(i+1) || string(env.GetChunk().GetPayload()) != want {
			t.Fatalf("early frame %d: seq %d payload %q", i, env.GetSeq(), env.GetChunk().GetPayload())
		}
	}
}

func TestPublishIsAtomicWithAttach(t *testing.T) {
	h := NewHub(core)
	var mu sync.Mutex
	state := []string{}

	var wg sync.WaitGroup
	const n = 200
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < n; i++ {
			tag := string(rune('a' + i%26))
			h.Publish(func() []*viewerv1.Envelope {
				mu.Lock()
				state = append(state, tag)
				mu.Unlock()
				return []*viewerv1.Envelope{chunkEnv(tag)}
			})
		}
	}()
	time.Sleep(time.Millisecond)
	p := NewPeer("joiner", 0)
	h.Attach(p, func() []*viewerv1.Envelope {
		mu.Lock()
		defer mu.Unlock()
		out := make([]*viewerv1.Envelope, len(state))
		for i, s := range state {
			out[i] = chunkEnv(s)
		}
		return out
	})
	wg.Wait()

	// Replay plus live must be exactly the full history, no gap, no duplicate.
	if got := p.Backlog(); got != n {
		t.Fatalf("joiner saw %d frames, want %d", got, n)
	}
}

func TestOverflowDropsLivePeerButNotReplay(t *testing.T) {
	h := NewHub(core)
	p := NewPeer("slow", 64)
	big := chunkEnv(strings.Repeat("x", 100))
	h.Attach(p, func() []*viewerv1.Envelope { return []*viewerv1.Envelope{big, big} })
	if p.Backlog() != 2 {
		t.Fatal("replay must not count against the backlog limit")
	}
	h.Broadcast(big)
	if _, err := p.Next(nil); !errors.Is(err, ErrOverflow) {
		t.Fatalf("want overflow, got %v", err)
	}
}

func TestDetachCloses(t *testing.T) {
	h := NewHub(core)
	p := NewPeer("x", 0)
	h.Attach(p, nil)
	h.Detach(p)
	if h.Len() != 0 {
		t.Fatal("peer still attached")
	}
	if _, err := p.Next(nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("want closed, got %v", err)
	}
}

type echoHandler struct {
	got chan *viewerv1.Envelope
}

func (e *echoHandler) Snapshot(*Peer) []*viewerv1.Envelope {
	return []*viewerv1.Envelope{{Body: &viewerv1.Envelope_MissionStatus{MissionStatus: &viewerv1.MissionStatus{Message: "hi"}}}}
}
func (e *echoHandler) Attached(*Peer)                         {}
func (e *echoHandler) Message(_ *Peer, env *viewerv1.Envelope) { e.got <- env }
func (e *echoHandler) Detached(*Peer)                         {}

func TestRouteOverRealSocket(t *testing.T) {
	h := NewHub(core)
	eh := &echoHandler{got: make(chan *viewerv1.Envelope, 1)}
	srv := httptest.NewServer(Route(h, eh, Options{Name: "test"}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.CloseNow()

	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if env := decode(t, data); env.GetSeq() != 1 || env.GetMissionStatus().GetMessage() != "hi" {
		t.Fatalf("replay frame %+v", env)
	}

	out, _ := proto.Marshal(&viewerv1.Envelope{Body: &viewerv1.Envelope_ManualControl{ManualControl: &viewerv1.ManualControl{DroneId: 3}}})
	if err := conn.Write(ctx, websocket.MessageBinary, out); err != nil {
		t.Fatal(err)
	}
	select {
	case env := <-eh.got:
		if env.GetManualControl().GetDroneId() != 3 {
			t.Fatalf("inbound %+v", env)
		}
	case <-ctx.Done():
		t.Fatal("inbound frame never arrived")
	}

	h.Broadcast(chunkEnv("live"))
	_, data, err = conn.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if env := decode(t, data); env.GetSeq() != 2 || string(env.GetChunk().GetPayload()) != "live" {
		t.Fatalf("live frame %+v", env)
	}
}
