package core

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/config"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/frames"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/mission"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/ws"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/terrain"
)

// ---- fakes ---------------------------------------------------------------------

type manualTimer struct {
	f    func()
	stop bool
}

func (t *manualTimer) Stop() bool {
	t.stop = true
	return true
}

// manualClock fires timers only when told to.
type manualClock struct {
	mu     sync.Mutex
	timers []*manualTimer
}

func (c *manualClock) Now() time.Time { return time.Unix(1_700_000_000, 0) }

func (c *manualClock) AfterFunc(_ time.Duration, f func()) mission.Timer {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := &manualTimer{f: f}
	c.timers = append(c.timers, t)
	return t
}

func (c *manualClock) fire() {
	c.mu.Lock()
	ts := c.timers
	c.timers = nil
	c.mu.Unlock()
	for _, t := range ts {
		if !t.stop {
			t.f()
		}
	}
}

type fakeTerrain struct {
	mu      sync.Mutex
	anchors []*modelv1.Gps
	areas   []terrain.Area
	styles  []terrain.Style
}

func (f *fakeTerrain) build(anchor *modelv1.Gps) TerrainStreamer {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.anchors = append(f.anchors, anchor)
	return f
}

func (f *fakeTerrain) Stream(ctx context.Context, area terrain.Area, center *modelv1.Gps, style terrain.Style) <-chan *modelv1.Chunk {
	f.mu.Lock()
	f.areas = append(f.areas, area)
	f.styles = append(f.styles, style)
	f.mu.Unlock()
	ch := make(chan *modelv1.Chunk, 2)
	ch <- &modelv1.Chunk{Kind: modelv1.ChunkKind_CHUNK_KIND_TERRAIN, BlockId: []int32{0, 0}, Payload: []byte("ground")}
	ch <- &modelv1.Chunk{Kind: modelv1.ChunkKind_CHUNK_KIND_BUILDINGS, BlockId: []int32{0, 0}, Payload: []byte("houses")}
	close(ch)
	return ch
}

// ---- helpers -------------------------------------------------------------------

func testConfig(t *testing.T) config.Config {
	t.Helper()
	cfg, err := config.Load(func(k string) string {
		return map[string]string{
			"SKYLENS_CORE_FRAME_SOURCE": "off",
		}[k]
	}, "")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func attach(h *ws.Hub, handler ws.Handler) *ws.Peer {
	p := ws.NewPeer("test", 0)
	h.Attach(p, func() []*viewerv1.Envelope { return handler.Snapshot(p) })
	return p
}

// drain returns every frame queued for p, decoded.
func drain(t *testing.T, p *ws.Peer) []*viewerv1.Envelope {
	t.Helper()
	var out []*viewerv1.Envelope
	for p.Backlog() > 0 {
		f, err := p.Next(nil)
		if err != nil {
			t.Fatal(err)
		}
		env := &viewerv1.Envelope{}
		if err := proto.Unmarshal(f, env); err != nil {
			t.Fatal(err)
		}
		out = append(out, env)
	}
	return out
}

func kinds(envs []*viewerv1.Envelope) []string {
	out := make([]string, len(envs))
	for i, e := range envs {
		switch b := e.GetBody().(type) {
		case *viewerv1.Envelope_Chunk:
			out[i] = "chunk:" + string(b.Chunk.GetPayload())
		default:
			out[i] = typeName(b)
		}
	}
	return out
}

func typeName(v any) string {
	switch v.(type) {
	case *viewerv1.Envelope_MissionStatus:
		return "MissionStatus"
	case *viewerv1.Envelope_AssignRoute:
		return "AssignRoute"
	case *viewerv1.Envelope_Telemetry:
		return "Telemetry"
	case *viewerv1.Envelope_CameraFeed:
		return "CameraFeed"
	case *viewerv1.Envelope_LinkStatus:
		return "LinkStatus"
	case *viewerv1.Envelope_ServerStatus:
		return "ServerStatus"
	case *viewerv1.Envelope_ManualControl:
		return "ManualControl"
	case *viewerv1.Envelope_DroneHello:
		return "DroneHello"
	}
	return "?"
}

func has(envs []*viewerv1.Envelope, kind string) bool {
	for _, k := range kinds(envs) {
		if k == kind {
			return true
		}
	}
	return false
}

func waitUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func route() *viewerv1.Envelope {
	return &viewerv1.Envelope{
		OriginTsMs: 42,
		From:       viewerv1.ComponentId_COMPONENT_ID_CLIENT,
		Body: &viewerv1.Envelope_AssignRoute{AssignRoute: &viewerv1.AssignRoute{
			DroneId: 1,
			Waypoints: []*modelv1.Gps{
				{Lat: 36.3700, Lon: 127.3600, Alt: 60},
				{Lat: 36.3720, Lon: 127.3650, Alt: 60},
			},
			Loop: true,
		}},
	}
}

// ---- tests ---------------------------------------------------------------------

func TestScenarioRouteThenDrone(t *testing.T) {
	clk := &manualClock{}
	ter := &fakeTerrain{}
	c := New(Options{Config: testConfig(t), Clock: clk, NewTerrain: ter.build})
	defer c.Close()

	viewer := attach(c.Viewers, c.ViewerHandler())
	replay := drain(t, viewer)
	if k := kinds(replay); len(k) != 2 || k[0] != "MissionStatus" || k[1] != "ServerStatus" {
		t.Fatalf("empty-core replay %v", k)
	}
	if replay[0].GetMissionStatus().GetPhase() != viewerv1.MissionPhase_MISSION_PHASE_IDLE {
		t.Fatal("core does not start idle")
	}

	uplink := attach(c.Uplinks, c.UplinkHandler())
	if got := drain(t, uplink); len(got) != 0 {
		t.Fatalf("uplink got %v before any route", kinds(got))
	}

	// 1. Control tower assigns a route.
	c.ViewerHandler().Message(viewer, route())
	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_ASSIGNED {
		t.Fatalf("phase %v", c.Mission.Phase())
	}
	up := drain(t, uplink)
	if len(up) != 1 || up[0].GetAssignRoute().GetDroneId() != 1 || up[0].GetOriginTsMs() != 42 {
		t.Fatalf("uplink did not get the route: %v", kinds(up))
	}

	// Terrain for the route area is streamed to viewers and stored.
	waitUntil(t, "terrain chunks stored", func() bool { return len(c.Store.Chunks()) == 2 })
	if len(ter.anchors) != 1 || ter.anchors[0].GetLat() != 36.37 {
		t.Fatalf("terrain anchor %v", ter.anchors)
	}
	if a := ter.areas[0]; !(a.MinLat < 36.37 && a.MaxLat > 36.372 && a.MinLon < 127.36 && a.MaxLon > 127.365) {
		t.Fatalf("terrain area %+v does not cover the route", a)
	}
	if ter.styles[0] != terrain.StyleBlack {
		t.Fatal("default terrain style is not black")
	}

	got := drain(t, viewer)
	for _, want := range []string{"AssignRoute", "MissionStatus", "chunk:ground", "chunk:houses"} {
		if !has(got, want) {
			t.Fatalf("viewer missing %s in %v", want, kinds(got))
		}
	}

	// 2. Hold elapses: waiting for a drone.
	clk.fire()
	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE {
		t.Fatalf("phase %v", c.Mission.Phase())
	}

	// 3. Telemetry alone does not count as arrival.
	tel := &viewerv1.Envelope{Body: &viewerv1.Envelope_Telemetry{Telemetry: &modelv1.DroneTelemetry{
		DroneId: 1, Gps: &modelv1.Gps{Lat: 36.371, Lon: 127.361},
	}}}
	c.UplinkHandler().Message(uplink, tel)
	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE {
		t.Fatalf("telemetry ended the wait: %v", c.Mission.Phase())
	}
	if !has(drain(t, viewer), "Telemetry") {
		t.Fatal("telemetry not forwarded to viewers")
	}

	// 4. drone_hello: active, and the screen learns where the video is.
	hello := &viewerv1.Envelope{Body: &viewerv1.Envelope_DroneHello{DroneHello: &viewerv1.DroneHello{
		DroneId: 1, Station: modelv1.DroneStation_DRONE_STATION_CENTER, Model: "Lito X1",
	}}}
	c.UplinkHandler().Message(uplink, hello)
	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_ACTIVE {
		t.Fatalf("hello did not activate: %v", c.Mission.Phase())
	}
	var feed *viewerv1.CameraFeed
	for _, e := range drain(t, viewer) {
		if f := e.GetCameraFeed(); f != nil {
			feed = f
		}
	}
	if feed == nil || feed.GetWhepUrl() != "/live/drone1/whep" || feed.GetStation() != modelv1.DroneStation_DRONE_STATION_CENTER {
		t.Fatalf("camera feed %v", feed)
	}

	// A late viewer gets the whole state replayed first.
	late := attach(c.Viewers, c.ViewerHandler())
	k := kinds(drain(t, late))
	want := []string{"MissionStatus", "AssignRoute", "Telemetry", "CameraFeed", "chunk:ground", "chunk:houses", "ServerStatus"}
	if strings.Join(k, ",") != strings.Join(want, ",") {
		t.Fatalf("late replay %v want %v", k, want)
	}

	// Manual control goes to the drone side only.
	c.ViewerHandler().Message(viewer, &viewerv1.Envelope{Body: &viewerv1.Envelope_ManualControl{ManualControl: &viewerv1.ManualControl{DroneId: 1, Forward: 0.5}}})
	if up := drain(t, uplink); len(up) != 1 || up[0].GetManualControl().GetForward() != 0.5 {
		t.Fatalf("manual control not forwarded: %v", kinds(up))
	}
	if has(drain(t, viewer), "ManualControl") {
		t.Fatal("manual control echoed to viewers")
	}

	// 5. The uplink drops: back to waiting.
	c.Uplinks.Detach(uplink)
	c.UplinkHandler().Detached(uplink)
	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_AWAITING_DRONE {
		t.Fatalf("phase after the drone left: %v", c.Mission.Phase())
	}

	// A new uplink gets the assigned route right away.
	again := attach(c.Uplinks, c.UplinkHandler())
	if up := drain(t, again); len(up) != 1 || up[0].GetAssignRoute() == nil {
		t.Fatalf("reconnecting uplink did not get the route: %v", kinds(up))
	}
}

func TestShortRouteIsRejected(t *testing.T) {
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}})
	defer c.Close()
	viewer := attach(c.Viewers, c.ViewerHandler())
	drain(t, viewer)

	env := route()
	env.GetAssignRoute().Waypoints = env.GetAssignRoute().Waypoints[:1]
	c.ViewerHandler().Message(viewer, env)
	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_IDLE || c.Store.Route() != nil {
		t.Fatal("one-waypoint route accepted")
	}
	got := drain(t, viewer)
	if len(got) != 1 || got[0].GetMissionStatus() == nil {
		t.Fatalf("sender should get the unchanged status back, got %v", kinds(got))
	}
}

func TestModelChunksFollowReplacementRules(t *testing.T) {
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}})
	defer c.Close()
	viewer := attach(c.Viewers, c.ViewerHandler())
	drain(t, viewer)

	c.PutChunk(&modelv1.Chunk{Sequence: 1, Kind: modelv1.ChunkKind_CHUNK_KIND_POINTS, Level: modelv1.Level_LEVEL_REFINED, Generation: 2, Payload: []byte("g2")})
	c.PutChunk(&modelv1.Chunk{Sequence: 2, Kind: modelv1.ChunkKind_CHUNK_KIND_POINTS, Level: modelv1.Level_LEVEL_PREVIEW, Payload: []byte("stale")})
	if k := kinds(drain(t, viewer)); strings.Join(k, ",") != "chunk:g2" {
		t.Fatalf("viewer got %v; stale chunks must not be forwarded", k)
	}
	st := c.ServerStatus()
	if st.GetChunks() != 1 || st.GetLastSeq() != 2 || len(st.GetSegments()) != 1 {
		t.Fatalf("server status %+v", st)
	}
}

func TestFramesAreGroupedAndQueuedForTheModel(t *testing.T) {
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}})
	defer c.Close()
	c.Store.PutTelemetry(&modelv1.DroneTelemetry{DroneId: 4, Gps: &modelv1.Gps{Lat: 1, Lon: 2}})
	c.InjectFrame(frames.Frame{DroneID: 4, Camera: "center", Format: modelv1.ImageFormat_IMAGE_FORMAT_JPEG, Data: []byte{1}})
	// No managed sources: any frame completes its set.
	if st := c.pusher.Stats(); st.Queued != 1 {
		t.Fatalf("pusher %+v", st)
	}
}

func TestRouteArea(t *testing.T) {
	a := RouteArea([]*modelv1.Gps{{Lat: 36.0, Lon: 127.0}, {Lat: 36.01, Lon: 127.02}}, 111.32)
	if d := 36.0 - a.MinLat; d < 0.00099 || d > 0.00101 {
		t.Fatalf("lat margin %v", d)
	}
	if a.MaxLon <= 127.02 || a.MinLon >= 127.0 {
		t.Fatalf("lon margin %+v", a)
	}
	if (RouteArea(nil, 10) != terrain.Area{}) {
		t.Fatal("empty route has an area")
	}
}

func TestHealthNeverShowsTheKey(t *testing.T) {
	cfg := testConfig(t)
	cfg.VWorldKey = "super-secret-key"
	c := New(Options{Config: cfg, Clock: &manualClock{}})
	defer c.Close()
	rec := httptest.NewRecorder()
	c.HealthHandler().ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	body := rec.Body.String()
	if strings.Contains(body, "super-secret-key") {
		t.Fatal("health leaks the VWorld key")
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(body), &parsed); err != nil || parsed["ok"] != true {
		t.Fatalf("health body %s (%v)", body, err)
	}
}
