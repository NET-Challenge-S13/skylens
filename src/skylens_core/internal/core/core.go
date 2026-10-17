// Package core wires the core's parts together: the uplink and the two screen
// sockets, the mission state machine, the in-memory store, the model server
// streams, the frame pipeline and the base map feed. It is an I/O server; it
// never decodes chunk payloads (COMPONENTS.md §3.4, §9).
//
// Screens come in two kinds:
//
//	ws /control  the control tower: everything plus the base map (terrain and
//	             buildings) and, until a route is assigned, the planned
//	             route; the only socket whose assign_route and
//	             manual_control are accepted
//	ws /viewer   status boards (through the gateway): read-only, never the
//	             base map; anything they send is dropped and logged
//
// Both get the Site first on connect, then the replay, then live updates.
package core

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/config"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/frames"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/mission"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/modelclient"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/store"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/ws"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/terrain"
)

// Link hop names published as link_status.
const (
	HopModelChunks = "model->core"
	HopModelFrames = "core->model"
)

// TerrainStreamer is what the core needs from terrain.Feed.
type TerrainStreamer interface {
	Stream(ctx context.Context, area terrain.Area, center *modelv1.Gps) <-chan *modelv1.Chunk
	Status() terrain.Status
}

// Options assemble a Core.
type Options struct {
	Config config.Config
	// Model is the model server; nil runs without one (tests).
	Model modelclient.API
	// Terrain is the base map feed, anchored at Config.Site.Anchor. Nil (or
	// Config.Terrain false) runs without a base map.
	Terrain TerrainStreamer
	// Frames builds the frame source of a drone; nil means frames only come
	// in through InjectFrame.
	Frames frames.Factory
	// Clock drives the mission; nil is the wall clock.
	Clock mission.Clock
}

// Core is the running server state.
type Core struct {
	cfg     config.Config
	model   modelclient.API
	terrain TerrainStreamer

	Store    *store.Store
	Mission  *mission.Mission
	Controls *ws.Hub // ws /control: the control tower
	Viewers  *ws.Hub // ws /viewer: status boards, read-only
	Uplinks  *ws.Hub

	frames  *frames.Manager
	grouper *frames.Grouper
	pusher  *modelclient.Pusher
	sub     *modelclient.Subscriber

	ctx    context.Context
	cancel context.CancelFunc
	start  time.Time

	mu            sync.Mutex
	owners        map[uint32]*ws.Peer // drone id -> uplink that said hello
	terrainArea   terrain.Area        // everything streamed so far (site + routes)
	terrainCancel context.CancelFunc
	// baseMapStreams counts base map streams started and not finished yet.
	baseMapStreams int
	terrainChunks  uint64
	lastSite       *viewerv1.Site // last Site broadcast, to send changes only
	lastUplinkAt   time.Time
	lastFrameAt    time.Time
	framesIn       uint64
	latencyMs      *float64
	modelUp        bool
	health         *modelv1.HealthResponse
	healthErr      string
}

// New builds a core. Call Run to start its loops (and the base map).
func New(opts Options) *Core {
	ctx, cancel := context.WithCancel(context.Background())
	c := &Core{
		cfg:      opts.Config,
		model:    opts.Model,
		terrain:  opts.Terrain,
		Store:    store.New(store.Options{DetectionsPerSegment: opts.Config.DetectionsPerSegment}),
		Controls: ws.NewHub(viewerv1.ComponentId_COMPONENT_ID_CORE),
		Viewers:  ws.NewHub(viewerv1.ComponentId_COMPONENT_ID_CORE),
		Uplinks:  ws.NewHub(viewerv1.ComponentId_COMPONENT_ID_CORE),
		ctx:      ctx,
		cancel:   cancel,
		start:    time.Now(),
		owners:   map[uint32]*ws.Peer{},
	}
	c.lastSite = c.Site()
	c.Mission = mission.New(mission.Options{
		AssignedHold: opts.Config.AssignedHold,
		DroneETA:     opts.Config.DroneETA,
		Clock:        opts.Clock,
		OnChange: func(s *viewerv1.MissionStatus) {
			c.broadcast(missionEnv(s))
		},
	})

	period := time.Duration(float64(time.Second) / math.Max(opts.Config.FrameFPS, 0.01))
	c.pusher = &modelclient.Pusher{
		API:       opts.Model,
		MaxQueued: opts.Config.PushQueue,
		Redeclare: opts.Config.PushRedeclare,
		OnState:   func(up bool, err error) { c.setLink(HopModelFrames, up, err) },
	}
	c.grouper = &frames.Grouper{
		SourceID: opts.Config.FrameSourceID,
		Period:   period,
		MaxWait:  opts.Config.FrameMaxWait,
		Expected: func() []string { return c.frames.Cameras() },
		Emit:     c.pusher.Push,
	}
	c.frames = frames.NewManager(opts.Frames, c.acceptFrame)
	c.sub = &modelclient.Subscriber{
		API:     opts.Model,
		After:   c.Store.LastModelSequence,
		OnChunk: c.PutChunk,
		OnState: func(up bool, err error) {
			c.mu.Lock()
			c.modelUp = up
			c.mu.Unlock()
			c.setLink(HopModelChunks, up, err)
		},
	}
	return c
}

// Run starts the base map stream and the background loops, and blocks until
// ctx is done.
func (c *Core) Run(ctx context.Context) {
	c.StartBaseMap()

	var wg sync.WaitGroup
	spawn := func(f func(context.Context)) {
		wg.Add(1)
		go func() { defer wg.Done(); f(c.ctx) }()
	}
	if c.model != nil {
		spawn(c.sub.Run)
		spawn(c.pusher.Run)
		spawn(c.healthLoop)
	}
	spawn(c.grouper.Run)
	spawn(c.statusLoop)

	select {
	case <-ctx.Done():
	case <-c.ctx.Done():
	}
	c.Close()
	wg.Wait()
}

// Close stops every loop, frame source and terrain stream.
func (c *Core) Close() {
	c.cancel()
	c.Mission.Stop()
	c.frames.StopAll()
}

// ---- envelopes ---------------------------------------------------------------

func missionEnv(s *viewerv1.MissionStatus) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_MissionStatus{MissionStatus: s}}
}

func chunkEnv(ch *modelv1.Chunk) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_Chunk{Chunk: ch}}
}

func routeEnv(r *viewerv1.AssignRoute) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_AssignRoute{AssignRoute: r}}
}

func telemetryEnv(t *modelv1.DroneTelemetry) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_Telemetry{Telemetry: t}}
}

func feedEnv(f *viewerv1.CameraFeed) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_CameraFeed{CameraFeed: f}}
}

func linkEnv(l *viewerv1.LinkStatus) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_LinkStatus{LinkStatus: l}}
}

func serverEnv(s *viewerv1.ServerStatus) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_ServerStatus{ServerStatus: s}}
}

func siteEnv(s *viewerv1.Site) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_Site{Site: s}}
}

func plannedEnv(p *viewerv1.PlannedRoute) *viewerv1.Envelope {
	return &viewerv1.Envelope{Body: &viewerv1.Envelope_PlannedRoute{PlannedRoute: p}}
}

// ---- fan-out to screens -----------------------------------------------------------

// publish runs mutate once and sends what it returns to both screen sockets.
// It holds both hub locks (control first, always), so a screen attaching on
// either socket sees the update exactly once, in its replay or live.
func (c *Core) publish(mutate func() []*viewerv1.Envelope) {
	c.Controls.Publish(func() []*viewerv1.Envelope {
		var envs []*viewerv1.Envelope
		c.Viewers.Publish(func() []*viewerv1.Envelope {
			envs = mutate()
			return envs
		})
		return envs
	})
}

// broadcast sends envelopes to every screen.
func (c *Core) broadcast(envs ...*viewerv1.Envelope) {
	c.publish(func() []*viewerv1.Envelope { return envs })
}

// ---- chunks --------------------------------------------------------------------

// PutChunk stores a chunk (from the model or the terrain feed) and forwards it
// to the screens if the replacement rules kept it (a point slot message that
// fills a gap also forwards the held messages it releases, in order). Base map
// chunks go to the control tower only.
func (c *Core) PutChunk(ch *modelv1.Chunk) {
	keep := func() []*viewerv1.Envelope {
		kept := c.Store.PutAll(ch)
		envs := make([]*viewerv1.Envelope, 0, len(kept))
		for _, k := range kept {
			envs = append(envs, chunkEnv(k))
		}
		return envs
	}
	if store.IsBaseMap(ch.GetKind()) {
		c.Controls.Publish(keep)
		return
	}
	c.publish(keep)
}

// ---- screen sockets --------------------------------------------------------------

// ControlHandler serves ws /control (the control tower).
func (c *Core) ControlHandler() ws.Handler { return controlHandler{c} }

// ViewerHandler serves ws /viewer (status boards, read-only).
func (c *Core) ViewerHandler() ws.Handler { return viewerHandler{c} }

// snapshot replays the current state in the order a screen would have
// received it live: the site, mission, links, route, drones, video, the
// world, status. control adds what only the control tower gets: the planned
// route right after the site (until a route is assigned), and terrain and
// buildings.
func (c *Core) snapshot(control bool) []*viewerv1.Envelope {
	out := []*viewerv1.Envelope{siteEnv(c.Site())}
	if control {
		if p := c.PlannedRoute(); p != nil {
			out = append(out, plannedEnv(p))
		}
	}
	out = append(out, missionEnv(c.Mission.Status()))
	for _, l := range c.Store.Links() {
		out = append(out, linkEnv(l))
	}
	if r := c.Store.Route(); r != nil {
		out = append(out, routeEnv(r))
	}
	for _, t := range c.Store.AllTelemetry() {
		out = append(out, telemetryEnv(t))
	}
	for _, f := range c.Store.CameraFeeds() {
		out = append(out, feedEnv(f))
	}
	chunks := c.Store.ModelChunks()
	if control {
		chunks = c.Store.Chunks()
	}
	for _, ch := range chunks {
		out = append(out, chunkEnv(ch))
	}
	return append(out, serverEnv(c.ServerStatus()))
}

// PlannedRoute is the route prepared before the session
// (SKYLENS_CORE_PLANNED_ROUTE) while no route has been assigned; nil once
// one has, or when none was configured. A tower that connects after the
// assignment draws the assigned route instead.
func (c *Core) PlannedRoute() *viewerv1.PlannedRoute {
	if c.cfg.PlannedRoute == nil || c.Store.Route() != nil {
		return nil
	}
	return c.cfg.PlannedRoute
}

type controlHandler struct{ c *Core }

func (h controlHandler) Snapshot(*ws.Peer) []*viewerv1.Envelope { return h.c.snapshot(true) }
func (h controlHandler) Attached(*ws.Peer)                      {}
func (h controlHandler) Detached(*ws.Peer)                      {}

func (h controlHandler) Message(p *ws.Peer, env *viewerv1.Envelope) {
	c := h.c
	switch body := env.GetBody().(type) {
	case *viewerv1.Envelope_AssignRoute:
		c.AssignRoute(env, p)
	case *viewerv1.Envelope_ManualControl:
		// Straight through to the drone side, no mission transition: the
		// operator taking the sticks does not end the task.
		if c.Uplinks.Len() == 0 {
			log.Printf("[core] manual_control for drone %d dropped: no uplink attached", body.ManualControl.GetDroneId())
			return
		}
		c.Uplinks.Broadcast(env)
	default:
		log.Printf("[core] control peer %d sent %T; the control tower sends assign_route and manual_control", p.ID, body)
	}
}

type viewerHandler struct{ c *Core }

func (h viewerHandler) Snapshot(*ws.Peer) []*viewerv1.Envelope { return h.c.snapshot(false) }
func (h viewerHandler) Attached(*ws.Peer)                      {}
func (h viewerHandler) Detached(*ws.Peer)                      {}

// Message drops everything: status boards have no control path.
func (h viewerHandler) Message(p *ws.Peer, env *viewerv1.Envelope) {
	log.Printf("[core] viewer peer %d (%s) sent %T; dropped, status boards are read-only", p.ID, p.Remote, env.GetBody())
}

// AssignRoute handles an assign_route from the control tower. from (a
// control peer) may be nil.
func (c *Core) AssignRoute(env *viewerv1.Envelope, from *ws.Peer) {
	r := env.GetAssignRoute()
	// A route assigned from idle starts a new mission. Everything the last one
	// reconstructed is dropped first, so a second run does not paint on top of
	// the first. Reassigning mid-mission keeps what is already on the screens.
	fresh := c.Mission.Phase() == viewerv1.MissionPhase_MISSION_PHASE_IDLE
	if len(r.GetWaypoints()) < 2 {
		log.Printf("[core] assign_route ignored: %d waypoint(s)", len(r.GetWaypoints()))
		if from != nil {
			c.Controls.Send(from, missionEnv(c.Mission.Status()))
		}
		return
	}
	// Screens draw the line so the operator sees what the fleet was given.
	c.publish(func() []*viewerv1.Envelope {
		c.Store.SetRoute(r)
		return []*viewerv1.Envelope{env}
	})
	// The drone mission bridge turns it into a waypoint mission. A drone that
	// connects later gets it in its uplink snapshot.
	if c.Uplinks.Len() == 0 {
		log.Printf("[core] route for drone %d (%d waypoints) kept; no uplink attached yet", r.GetDroneId(), len(r.GetWaypoints()))
	} else {
		c.Uplinks.Broadcast(env)
	}
	c.Mission.RouteAssigned()
	if fresh {
		c.resetReconstruction()
	}
	c.extendBaseMap(r.GetWaypoints())
}

// resetReconstruction empties the store of the last mission's reconstruction.
//
// A screen holds its own copy of what it was sent, so clearing the store is not
// enough: the sockets are dropped too. Each screen clears on reconnect and takes
// the replay again (skylens_client/board.rs reconnected, control/app.rs
// LinkEvent::Connected), which now carries the same site with an empty world.
// They are back in under a second.
//
// Nothing is dropped when there was nothing to drop, so the first mission after
// the core starts runs without a blip: the screens that watched the assignment
// stay on the socket that delivered it.
func (c *Core) resetReconstruction() {
	dropped := c.Store.ResetModel()
	if dropped == 0 {
		return
	}
	for _, p := range c.Viewers.Peers() {
		p.Close()
	}
	for _, p := range c.Controls.Peers() {
		p.Close()
	}
	log.Printf("[core] new mission: dropped %d reconstruction chunk(s); screens reconnect to an empty world", dropped)
}

// ---- uplink side ---------------------------------------------------------------

// UplinkHandler serves ws /uplink.
func (c *Core) UplinkHandler() ws.Handler { return uplinkHandler{c} }

type uplinkHandler struct{ c *Core }

// Snapshot hands a newly attached uplink the route that is already assigned:
// the scenario is "route first, drone arrives later".
func (h uplinkHandler) Snapshot(*ws.Peer) []*viewerv1.Envelope {
	if r := h.c.Store.Route(); r != nil {
		return []*viewerv1.Envelope{routeEnv(r)}
	}
	return nil
}

func (h uplinkHandler) Attached(*ws.Peer) {}

func (h uplinkHandler) Detached(p *ws.Peer) {
	c := h.c
	c.mu.Lock()
	var gone []uint32
	for id, owner := range c.owners {
		if owner == p {
			gone = append(gone, id)
			delete(c.owners, id)
		}
	}
	c.mu.Unlock()
	for _, id := range gone {
		log.Printf("[core] drone %d gone (uplink closed)", id)
		c.frames.Stop(id)
		c.Mission.Gone(id)
	}
}

func (h uplinkHandler) Message(p *ws.Peer, env *viewerv1.Envelope) {
	c := h.c
	now := time.Now()
	c.mu.Lock()
	c.lastUplinkAt = now
	if ts := env.GetOriginTsMs(); ts > 0 {
		l := math.Max(0, float64(now.UnixMilli()-ts))
		c.latencyMs = &l
	}
	c.mu.Unlock()

	switch body := env.GetBody().(type) {
	case *viewerv1.Envelope_DroneHello:
		c.DroneHello(body.DroneHello, p)
	case *viewerv1.Envelope_Telemetry:
		// Never an arrival: a drone still flying to the site sends telemetry
		// too, and counting it would skip the waiting phase.
		t := body.Telemetry
		c.publish(func() []*viewerv1.Envelope {
			c.Store.PutTelemetry(t)
			return []*viewerv1.Envelope{env}
		})
	case *viewerv1.Envelope_LinkStatus:
		l := body.LinkStatus
		c.publish(func() []*viewerv1.Envelope {
			if !c.Store.PutLink(l) {
				return nil
			}
			return []*viewerv1.Envelope{env}
		})
	default:
		log.Printf("[core] uplink peer %d sent %T; uplinks send drone_hello and telemetry", p.ID, body)
	}
}

// StationName is the lower-case station, or "" when unspecified.
func StationName(s modelv1.DroneStation) string {
	switch s {
	case modelv1.DroneStation_DRONE_STATION_LEFT:
		return "left"
	case modelv1.DroneStation_DRONE_STATION_CENTER:
		return "center"
	case modelv1.DroneStation_DRONE_STATION_RIGHT:
		return "right"
	}
	return ""
}

// DroneHello is the only arrival signal. from may be nil.
func (c *Core) DroneHello(h *viewerv1.DroneHello, from *ws.Peer) {
	id := h.GetDroneId()
	station := StationName(h.GetStation())
	log.Printf("[core] drone %d hello (%s, station %q)", id, h.GetModel(), station)
	if from != nil {
		c.mu.Lock()
		c.owners[id] = from
		c.mu.Unlock()
	}

	path := config.Expand(c.cfg.StreamPath, id, station, "")
	feed := &viewerv1.CameraFeed{
		DroneId: id,
		Station: h.GetStation(),
		WhepUrl: config.Expand(c.cfg.WHEPBase, id, station, path),
	}
	c.publish(func() []*viewerv1.Envelope {
		c.Store.PutCameraFeed(feed)
		return []*viewerv1.Envelope{feedEnv(feed)}
	})

	camera := station
	if camera == "" {
		camera = fmt.Sprintf("drone%d", id)
	}
	if c.frames.Start(c.ctx, id, camera) {
		log.Printf("[core] frames of drone %d (%s) from %s", id, camera, config.Expand(c.cfg.FrameURL, id, station, path))
	}
	c.Mission.Hello(id, h.GetStation())
}

// ---- frames ----------------------------------------------------------------------

// InjectFrame feeds a frame that does not come from a managed source (the
// demo, tests).
func (c *Core) InjectFrame(f frames.Frame) { c.frames.Inject(f) }

func (c *Core) acceptFrame(f frames.Frame) {
	if f.Gps == nil {
		if t := c.Store.Telemetry(f.DroneID); t != nil && t.GetGps() != nil {
			f.Gps = proto.Clone(t.GetGps()).(*modelv1.Gps)
		}
	}
	c.mu.Lock()
	c.framesIn++
	c.lastFrameAt = time.Now()
	c.mu.Unlock()
	c.grouper.Add(f)
}

// FrameFactory builds ffmpeg sources reading from the local MediaMTX.
func FrameFactory(cfg config.Config) frames.Factory {
	if cfg.FrameSource != "ffmpeg" {
		return nil
	}
	return func(droneID uint32, camera string) frames.Source {
		station := camera
		if strings.HasPrefix(camera, "drone") {
			station = ""
		}
		path := config.Expand(cfg.StreamPath, droneID, station, "")
		return &frames.FFmpeg{
			Bin:     cfg.FFmpegBin,
			URL:     config.Expand(cfg.FrameURL, droneID, station, path),
			FPS:     cfg.FrameFPS,
			Quality: cfg.FrameQuality,
			Width:   cfg.FrameWidth,
		}
	}
}

// ---- status ----------------------------------------------------------------------

func (c *Core) setLink(hop string, up bool, err error) {
	if up {
		log.Printf("[core] %s connected", hop)
	} else {
		log.Printf("[core] %s down: %v", hop, err)
	}
	l := &viewerv1.LinkStatus{Hop: hop, Connected: up}
	c.publish(func() []*viewerv1.Envelope {
		if !c.Store.PutLink(l) {
			return nil
		}
		return []*viewerv1.Envelope{linkEnv(l)}
	})
}

// ServerStatus is the heartbeat the screens show.
func (c *Core) ServerStatus() *viewerv1.ServerStatus {
	cnt := c.Store.Counters()
	now := time.Now()
	c.mu.Lock()
	receiving := now.Sub(c.lastFrameAt) < 3*time.Second || now.Sub(c.lastUplinkAt) < 3*time.Second
	s := &viewerv1.ServerStatus{
		Connected:  c.modelUp,
		Receiving:  receiving,
		Chunks:     cnt.Model, // the base map is not reconstruction; boards never see it
		Detections: cnt.Detections,
		LastSeq:    cnt.LastModelSeq,
		// Screens warn once the model server runs on the CPU after losing its GPU.
		ModelGpuError: c.health.GetGpuError(),
	}
	if c.latencyMs != nil {
		l := *c.latencyMs
		s.LatencyMs = &l
	}
	c.mu.Unlock()
	s.Segments = c.Store.Segments()
	return s
}

func (c *Core) statusLoop(ctx context.Context) {
	every := c.cfg.StatusEvery
	if every <= 0 {
		every = time.Second
	}
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			envs := []*viewerv1.Envelope{serverEnv(c.ServerStatus())}
			// The countdown has to count down on screens too.
			if ms := c.Mission.Status(); ms.EtaSeconds != nil {
				envs = append(envs, missionEnv(ms))
			}
			c.broadcast(envs...)
			c.PublishSiteIfChanged()
		}
	}
}

func (c *Core) healthLoop(ctx context.Context) {
	probe := func() {
		hctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		h, err := c.model.Health(hctx)
		c.mu.Lock()
		defer c.mu.Unlock()
		c.health = h
		c.healthErr = ""
		if err != nil {
			c.healthErr = err.Error()
		}
	}
	probe()
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			probe()
		}
	}
}

// HealthHandler serves GET /health. It never includes the VWorld key.
func (c *Core) HealthHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cnt := c.Store.Counters()
		ps := c.pusher.Stats()
		c.mu.Lock()
		model := map[string]any{
			"target":           c.cfg.ModelGRPC,
			"chunks_stream":    c.modelUp,
			"frames_stream":    ps.Connected,
			"health_error":     c.healthErr,
			"health_ok":        c.health.GetOk(),
			"health_gpu":       c.health.GetGpu(),
			"health_gpu_error": c.health.GetGpuError(),
			"health_version":   c.health.GetVersion(),
		}
		framesIn := c.framesIn
		terrainChunks := c.terrainChunks
		c.mu.Unlock()
		ms := c.Mission.Status()
		site := c.Site()
		terrainInfo := map[string]any{
			"enabled":   c.baseMapOn(),
			"vworld":    c.cfg.VWorldKey != "",
			"chunks":    terrainChunks,
			"imagery":   site.GetImagery(),
			"buildings": site.GetBuildings(),
		}
		if c.terrain != nil {
			st := c.terrain.Status()
			terrainInfo["cells"] = st.Cells
			terrainInfo["building_count"] = st.BuildingCount
		}
		body := map[string]any{
			"ok":         true,
			"component":  "skylens_core",
			"uptime_sec": int(time.Since(c.start).Seconds()),
			"site":       siteJSON(site, c.cfg.Site.Source),
			"mission": map[string]any{
				"phase":         ms.GetPhase().String(),
				"message":       ms.GetMessage(),
				"drones_online": ms.GetDronesOnline(),
				"drones":        c.Mission.Drones(),
			},
			"model": model,
			"store": map[string]any{
				"storage":        "in-memory (SPEC §14)",
				"accepted":       cnt.Accepted,
				"rejected":       cnt.Rejected,
				"detections":     cnt.Detections,
				"last_model_seq": cnt.LastModelSeq,
				"segments":       len(c.Store.Segments()),
			},
			"frames": map[string]any{
				"source":     c.cfg.FrameSource,
				"cameras":    c.frames.Cameras(),
				"received":   framesIn,
				"sets_sent":  ps.Sent,
				"queued":     ps.Queued,
				"dropped":    ps.Dropped,
				"redeclared": ps.Redeclare,
			},
			"terrain": terrainInfo,
			"sockets": map[string]any{
				"controls":     c.Controls.Len(),
				"viewers":      c.Viewers.Len(),
				"uplinks":      c.Uplinks.Len(),
				"control_sent": c.Controls.Sent(),
				"viewer_sent":  c.Viewers.Sent(),
				"uplink_sent":  c.Uplinks.Sent(),
			},
		}
		writeJSON(w, body)
	})
}

func writeJSON(w http.ResponseWriter, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(body)
}
