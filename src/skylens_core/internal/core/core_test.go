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

// fakeTerrain streams one terrain and one building chunk per call, in a cell
// of its own, so stored chunks never replace each other.
type fakeTerrain struct {
	mu      sync.Mutex
	areas   []terrain.Area
	centers []*modelv1.Gps
	status  terrain.Status
}

func (f *fakeTerrain) Stream(ctx context.Context, area terrain.Area, center *modelv1.Gps) <-chan *modelv1.Chunk {
	f.mu.Lock()
	f.areas = append(f.areas, area)
	f.centers = append(f.centers, center)
	cell := int32(len(f.areas))
	f.mu.Unlock()
	ch := make(chan *modelv1.Chunk, 2)
	ch <- &modelv1.Chunk{Kind: modelv1.ChunkKind_CHUNK_KIND_TERRAIN, BlockId: []int32{cell, 0, 0}, Payload: []byte("ground")}
	ch <- &modelv1.Chunk{Kind: modelv1.ChunkKind_CHUNK_KIND_BUILDINGS, BlockId: []int32{cell, 0, 0}, Payload: []byte("houses")}
	close(ch)
	return ch
}

func (f *fakeTerrain) Status() terrain.Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.status
}

func (f *fakeTerrain) setStatus(s terrain.Status) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.status = s
}

func (f *fakeTerrain) calls() ([]terrain.Area, []*modelv1.Gps) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]terrain.Area(nil), f.areas...), append([]*modelv1.Gps(nil), f.centers...)
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
	case *viewerv1.Envelope_Site:
		return "Site"
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

func hasBaseMap(envs []*viewerv1.Envelope) bool {
	for _, e := range envs {
		switch e.GetChunk().GetKind() {
		case modelv1.ChunkKind_CHUNK_KIND_TERRAIN, modelv1.ChunkKind_CHUNK_KIND_BUILDINGS:
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

func TestBaseMapAtStartupGoesToTheControlTowerOnly(t *testing.T) {
	ter := &fakeTerrain{}
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}, Terrain: ter})
	defer c.Close()

	control := attach(c.Controls, c.ControlHandler())
	viewer := attach(c.Viewers, c.ViewerHandler())
	for _, p := range []*ws.Peer{control, viewer} {
		k := kinds(drain(t, p))
		if strings.Join(k, ",") != "Site,MissionStatus,ServerStatus" {
			t.Fatalf("empty-core replay %v", k)
		}
	}

	// No route needed: the site area streams as soon as the core starts.
	c.StartBaseMap()
	// Two cells, then the Site again: the base map is now ready, with its counts.
	waitUntil(t, "base map sent", func() bool { return control.Backlog() == 3 })
	areas, centers := ter.calls()
	if len(areas) != 1 || areas[0] != c.SiteArea() {
		t.Fatalf("streamed areas %+v, want the site %+v", areas, c.SiteArea())
	}
	if centers[0].GetLat() != 36.3685 || centers[0].GetLon() != 127.3475 {
		t.Fatalf("stream centre %v, want the anchor", centers[0])
	}

	got := drain(t, control)
	if !has(got, "chunk:ground") || !has(got, "chunk:houses") {
		t.Fatalf("control tower did not get the base map: %v", kinds(got))
	}
	if got := drain(t, viewer); hasBaseMap(got) {
		t.Fatalf("status board got the base map: %v", kinds(got))
	}

	// Late screens: the tower replays the map, the board never sees it.
	if k := kinds(drain(t, attach(c.Controls, c.ControlHandler()))); strings.Join(k, ",") != "Site,MissionStatus,chunk:ground,chunk:houses,ServerStatus" {
		t.Fatalf("late control replay %v", k)
	}
	if k := kinds(drain(t, attach(c.Viewers, c.ViewerHandler()))); strings.Join(k, ",") != "Site,MissionStatus,ServerStatus" {
		t.Fatalf("late viewer replay %v", k)
	}
}

func TestRouteExtendsTheBaseMap(t *testing.T) {
	ter := &fakeTerrain{}
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}, Terrain: ter})
	defer c.Close()
	c.StartBaseMap()
	waitUntil(t, "site streamed", func() bool { return len(c.Store.Chunks()) == 2 })

	// A route that reaches outside the site.
	env := route()
	env.GetAssignRoute().Waypoints = append(env.GetAssignRoute().Waypoints, &modelv1.Gps{Lat: 36.40, Lon: 127.40, Alt: 60})
	c.ControlHandler().Message(attach(c.Controls, c.ControlHandler()), env)
	waitUntil(t, "route area streamed", func() bool { return len(c.Store.Chunks()) == 4 })

	areas, centers := ter.calls()
	if len(areas) != 2 {
		t.Fatalf("%d streams, want 2", len(areas))
	}
	site, grown := c.SiteArea(), areas[1]
	if grown.MinLat > site.MinLat || grown.MinLon > site.MinLon || grown.MaxLat < 36.40 || grown.MaxLon < 127.40 {
		t.Fatalf("route stream area %+v does not cover the site %+v and the route", grown, site)
	}
	if centers[1].GetLat() != 36.37 || centers[1].GetLon() != 127.36 {
		t.Fatalf("route stream centre %v, want the first waypoint", centers[1])
	}
	// The anchor did not move with the route.
	if a := c.Site().GetAnchor(); a.GetLat() != 36.3685 || a.GetLon() != 127.3475 || a.GetAlt() != 30 {
		t.Fatalf("anchor moved to %v", a)
	}
}

func TestScenarioRouteThenDrone(t *testing.T) {
	clk := &manualClock{}
	c := New(Options{Config: testConfig(t), Clock: clk, Terrain: &fakeTerrain{}})
	defer c.Close()

	control := attach(c.Controls, c.ControlHandler())
	viewer := attach(c.Viewers, c.ViewerHandler())
	replay := drain(t, control)
	if replay[0].GetSite().GetName() != "daejeon" {
		t.Fatalf("first envelope %v, want the site", kinds(replay))
	}
	if replay[1].GetMissionStatus().GetPhase() != viewerv1.MissionPhase_MISSION_PHASE_IDLE {
		t.Fatal("core does not start idle")
	}
	drain(t, viewer)

	uplink := attach(c.Uplinks, c.UplinkHandler())
	if got := drain(t, uplink); len(got) != 0 {
		t.Fatalf("uplink got %v before any route", kinds(got))
	}

	// 1. Control tower assigns a route.
	c.ControlHandler().Message(control, route())
	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_ASSIGNED {
		t.Fatalf("phase %v", c.Mission.Phase())
	}
	up := drain(t, uplink)
	if len(up) != 1 || up[0].GetAssignRoute().GetDroneId() != 1 || up[0].GetOriginTsMs() != 42 {
		t.Fatalf("uplink did not get the route: %v", kinds(up))
	}
	// Route, mission status, then the route area's terrain and buildings.
	waitUntil(t, "route area sent", func() bool { return control.Backlog() >= 4 })

	got := drain(t, control)
	for _, want := range []string{"AssignRoute", "MissionStatus", "chunk:ground", "chunk:houses"} {
		if !has(got, want) {
			t.Fatalf("control missing %s in %v", want, kinds(got))
		}
	}
	got = drain(t, viewer)
	if !has(got, "AssignRoute") || !has(got, "MissionStatus") || hasBaseMap(got) {
		t.Fatalf("status board got %v; want the route and mission, no base map", kinds(got))
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
	if !has(drain(t, control), "Telemetry") || !has(drain(t, viewer), "Telemetry") {
		t.Fatal("telemetry not forwarded to both screens")
	}

	// 4. drone_hello: active, and the screens learn where the video is.
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
	drain(t, control)

	// Late screens get the whole state replayed first, the site leading.
	k := kinds(drain(t, attach(c.Controls, c.ControlHandler())))
	want := []string{"Site", "MissionStatus", "AssignRoute", "Telemetry", "CameraFeed", "chunk:ground", "chunk:houses", "ServerStatus"}
	if strings.Join(k, ",") != strings.Join(want, ",") {
		t.Fatalf("late control replay %v want %v", k, want)
	}
	k = kinds(drain(t, attach(c.Viewers, c.ViewerHandler())))
	want = []string{"Site", "MissionStatus", "AssignRoute", "Telemetry", "CameraFeed", "ServerStatus"}
	if strings.Join(k, ",") != strings.Join(want, ",") {
		t.Fatalf("late viewer replay %v want %v", k, want)
	}

	// Manual control goes to the drone side only.
	c.ControlHandler().Message(control, &viewerv1.Envelope{Body: &viewerv1.Envelope_ManualControl{ManualControl: &viewerv1.ManualControl{DroneId: 1, Forward: 0.5}}})
	if up := drain(t, uplink); len(up) != 1 || up[0].GetManualControl().GetForward() != 0.5 {
		t.Fatalf("manual control not forwarded: %v", kinds(up))
	}
	if has(drain(t, control), "ManualControl") || has(drain(t, viewer), "ManualControl") {
		t.Fatal("manual control echoed to screens")
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

func TestStatusBoardsCannotControl(t *testing.T) {
	ter := &fakeTerrain{}
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}, Terrain: ter})
	defer c.Close()
	viewer := attach(c.Viewers, c.ViewerHandler())
	control := attach(c.Controls, c.ControlHandler())
	uplink := attach(c.Uplinks, c.UplinkHandler())
	drain(t, viewer)
	drain(t, control)

	c.ViewerHandler().Message(viewer, route())
	c.ViewerHandler().Message(viewer, &viewerv1.Envelope{Body: &viewerv1.Envelope_ManualControl{ManualControl: &viewerv1.ManualControl{DroneId: 1, Forward: 1}}})

	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_IDLE || c.Store.Route() != nil {
		t.Fatal("a status board assigned a route")
	}
	if got := drain(t, uplink); len(got) != 0 {
		t.Fatalf("a status board reached the drone: %v", kinds(got))
	}
	if got := drain(t, control); len(got) != 0 {
		t.Fatalf("a status board message reached the control tower: %v", kinds(got))
	}
	if areas, _ := ter.calls(); len(areas) != 0 {
		t.Fatal("a status board route started a base map stream")
	}
}

func TestShortRouteIsRejected(t *testing.T) {
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}})
	defer c.Close()
	control := attach(c.Controls, c.ControlHandler())
	drain(t, control)

	env := route()
	env.GetAssignRoute().Waypoints = env.GetAssignRoute().Waypoints[:1]
	c.ControlHandler().Message(control, env)
	if c.Mission.Phase() != viewerv1.MissionPhase_MISSION_PHASE_IDLE || c.Store.Route() != nil {
		t.Fatal("one-waypoint route accepted")
	}
	got := drain(t, control)
	if len(got) != 1 || got[0].GetMissionStatus() == nil {
		t.Fatalf("sender should get the unchanged status back, got %v", kinds(got))
	}
}

func TestModelChunksFollowReplacementRules(t *testing.T) {
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}})
	defer c.Close()
	viewer := attach(c.Viewers, c.ViewerHandler())
	control := attach(c.Controls, c.ControlHandler())
	drain(t, viewer)
	drain(t, control)

	c.PutChunk(&modelv1.Chunk{Sequence: 1, Kind: modelv1.ChunkKind_CHUNK_KIND_POINTS, Level: modelv1.Level_LEVEL_REFINED, Generation: 2, Payload: []byte("g2")})
	c.PutChunk(&modelv1.Chunk{Sequence: 2, Kind: modelv1.ChunkKind_CHUNK_KIND_POINTS, Level: modelv1.Level_LEVEL_PREVIEW, Payload: []byte("stale")})
	for _, p := range []*ws.Peer{viewer, control} {
		if k := kinds(drain(t, p)); strings.Join(k, ",") != "chunk:g2" {
			t.Fatalf("screen got %v; stale chunks must not be forwarded", k)
		}
	}
	st := c.ServerStatus()
	if st.GetChunks() != 1 || st.GetLastSeq() != 2 || len(st.GetSegments()) != 1 {
		t.Fatalf("server status %+v", st)
	}
}

func TestSiteFollowsTheFeedAndIsRepublishedOnChange(t *testing.T) {
	ter := &fakeTerrain{}
	c := New(Options{Config: testConfig(t), Clock: &manualClock{}, Terrain: ter})
	defer c.Close()
	control := attach(c.Controls, c.ControlHandler())
	viewer := attach(c.Viewers, c.ViewerHandler())
	drain(t, control)
	drain(t, viewer)

	s := c.Site()
	if s.GetName() != "daejeon" || s.GetMinLon() != 127.33 || s.GetMaxLat() != 36.382 ||
		s.GetImagery() || s.GetBuildings() || s.GetBuildingCount() != 0 {
		t.Fatalf("site %v", s)
	}

	// Nothing changed: nothing sent.
	c.PublishSiteIfChanged()
	if got := drain(t, control); len(got) != 0 {
		t.Fatalf("unchanged site re-sent: %v", kinds(got))
	}

	ter.setStatus(terrain.Status{Imagery: true, Buildings: true, BuildingCount: 6191})
	c.PublishSiteIfChanged()
	for _, p := range []*ws.Peer{control, viewer} {
		got := drain(t, p)
		if len(got) != 1 || got[0].GetSite().GetBuildingCount() != 6191 || !got[0].GetSite().GetImagery() {
			t.Fatalf("site update %v", kinds(got))
		}
	}
}

func TestSiteHandlerKeepsTheLegacyShape(t *testing.T) {
	cfg := testConfig(t)
	cfg.VWorldKey = "super-secret-key"
	c := New(Options{Config: cfg, Clock: &manualClock{}, Terrain: &fakeTerrain{}})
	defer c.Close()
	rec := httptest.NewRecorder()
	c.SiteHandler().ServeHTTP(rec, httptest.NewRequest("GET", "/site", nil))
	body := rec.Body.String()
	if strings.Contains(body, "super-secret-key") {
		t.Fatal("/site leaks the VWorld key")
	}
	var got struct {
		Gps struct {
			Lat, Lon, Alt float64
		} `json:"gps"`
		Source string    `json:"source"`
		Name   string    `json:"name"`
		Bbox   []float64 `json:"bbox"`
	}
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatalf("%s: %v", body, err)
	}
	if got.Gps.Lat != 36.3685 || got.Gps.Lon != 127.3475 || got.Gps.Alt != 30 || got.Source != "default" || got.Name != "daejeon" {
		t.Fatalf("/site %s", body)
	}
	if len(got.Bbox) != 4 || got.Bbox[0] != 127.33 || got.Bbox[3] != 36.382 {
		t.Fatalf("/site bbox %v", got.Bbox)
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
	c := New(Options{Config: cfg, Clock: &manualClock{}, Terrain: &fakeTerrain{}})
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
