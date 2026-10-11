package core

import (
	"context"
	"log"
	"math"
	"net/http"

	"google.golang.org/protobuf/proto"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/terrain"
)

// ---- site ------------------------------------------------------------------------

// Site is the operating area as screens receive it. The anchor is fixed for
// the session (config.Site.Anchor) and is the ENU origin of every chunk,
// terrain cell and building; imagery, buildings and building_count follow
// what the base map feed can offer and has sent.
func (c *Core) Site() *viewerv1.Site {
	s := c.cfg.Site
	out := &viewerv1.Site{
		MinLat: s.MinLat,
		MinLon: s.MinLon,
		MaxLat: s.MaxLat,
		MaxLon: s.MaxLon,
		Name:   s.Name,
	}
	if s.Anchor != nil {
		out.Anchor = proto.Clone(s.Anchor).(*modelv1.Gps)
	}
	if c.terrain != nil {
		st := c.terrain.Status()
		// Imagery also feeds the route planner's map through /vworld, so it
		// is offered even with the base map stream off.
		out.Imagery = st.Imagery
		if c.baseMapOn() {
			out.Buildings = st.Buildings
			out.BuildingCount = uint32(st.BuildingCount)
			c.mu.Lock()
			streams := c.baseMapStreams
			c.mu.Unlock()
			out.BaseMapReady = streams == 0 && !st.Streaming
			out.BaseMapCells = uint32(st.Cells)
			out.BaseMapCellsTotal = uint32(st.Cells + st.Pending)
			out.BaseMapChunks = uint32(c.Store.BaseMapCount())
		}
	}
	if !c.baseMapOn() {
		out.BaseMapReady = true
	}
	return out
}

// PublishSiteIfChanged sends the Site to every screen when it differs from
// the last one sent (the building count grows as the base map streams).
func (c *Core) PublishSiteIfChanged() {
	s := c.Site()
	c.mu.Lock()
	same := proto.Equal(s, c.lastSite)
	if !same {
		c.lastSite = s
	}
	c.mu.Unlock()
	if !same {
		c.broadcast(siteEnv(s))
	}
}

// siteJSON is the /site body. gps, source and label are the legacy core's
// shape (the route planner opens on gps); the rest mirrors viewer.v1.Site.
func siteJSON(s *viewerv1.Site, source string) map[string]any {
	a := s.GetAnchor()
	gps := map[string]any{"lat": a.GetLat(), "lon": a.GetLon(), "alt": a.GetAlt()}
	return map[string]any{
		"gps":            gps,
		"source":         source,
		"label":          s.GetName(),
		"name":           s.GetName(),
		"anchor":         gps,
		"bbox":           []float64{s.GetMinLon(), s.GetMinLat(), s.GetMaxLon(), s.GetMaxLat()},
		"min_lat":        s.GetMinLat(),
		"min_lon":        s.GetMinLon(),
		"max_lat":        s.GetMaxLat(),
		"max_lon":        s.GetMaxLon(),
		"imagery":        s.GetImagery(),
		"buildings":      s.GetBuildings(),
		"building_count": s.GetBuildingCount(),
	}
}

// SiteHandler serves GET /site: the Site as JSON.
func (c *Core) SiteHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A control tower served from elsewhere (a dev server) may ask too.
		w.Header().Set("Access-Control-Allow-Origin", "*")
		writeJSON(w, siteJSON(c.Site(), c.cfg.Site.Source))
	})
}

// ---- base map --------------------------------------------------------------------

func (c *Core) baseMapOn() bool { return c.cfg.Terrain && c.terrain != nil }

// SiteArea is the configured operating area.
func (c *Core) SiteArea() terrain.Area {
	s := c.cfg.Site
	return terrain.Area{MinLat: s.MinLat, MinLon: s.MinLon, MaxLat: s.MaxLat, MaxLon: s.MaxLon}
}

// RouteArea is the bounding box of the waypoints grown by margin metres.
func RouteArea(wps []*modelv1.Gps, marginM float64) terrain.Area {
	if len(wps) == 0 {
		return terrain.Area{}
	}
	a := terrain.Area{MinLat: 90, MinLon: 180, MaxLat: -90, MaxLon: -180}
	for _, w := range wps {
		a.MinLat = math.Min(a.MinLat, w.GetLat())
		a.MaxLat = math.Max(a.MaxLat, w.GetLat())
		a.MinLon = math.Min(a.MinLon, w.GetLon())
		a.MaxLon = math.Max(a.MaxLon, w.GetLon())
	}
	const mPerDegLat = 111_320.0
	midLat := (a.MinLat + a.MaxLat) / 2
	dLat := marginM / mPerDegLat
	dLon := marginM / (mPerDegLat * math.Max(math.Cos(midLat*math.Pi/180), 1e-6))
	a.MinLat, a.MaxLat = math.Max(-90, a.MinLat-dLat), math.Min(90, a.MaxLat+dLat)
	a.MinLon, a.MaxLon = math.Max(-180, a.MinLon-dLon), math.Min(180, a.MaxLon+dLon)
	return a
}

// StartBaseMap streams the operating area's terrain and buildings, nearest to
// the anchor first, so a control tower that connects gets the map at once
// (Run calls it). Every chunk is kept in the store for later replays.
func (c *Core) StartBaseMap() {
	if !c.baseMapOn() {
		return
	}
	area := c.SiteArea()
	c.mu.Lock()
	c.terrainArea = c.terrainArea.Union(area)
	area = c.terrainArea
	c.mu.Unlock()
	a := c.cfg.Site.Anchor
	log.Printf("[core] base map: %s, anchor %.6f, %.6f, %.1f", c.cfg.Site.Name, a.GetLat(), a.GetLon(), a.GetAlt())
	c.streamTerrain(area, a)
}

// extendBaseMap grows the streamed area by a route's box. The feed skips
// what it already sent, so only the new cells go out, nearest to the route's
// first waypoint first.
func (c *Core) extendBaseMap(wps []*modelv1.Gps) {
	if !c.baseMapOn() || len(wps) == 0 {
		return
	}
	c.mu.Lock()
	c.terrainArea = c.terrainArea.Union(RouteArea(wps, c.cfg.TerrainMargin))
	area := c.terrainArea
	c.mu.Unlock()
	c.streamTerrain(area, wps[0])
}

// streamTerrain replaces the running base map stream with one over area. The
// cells the previous stream already delivered stay in the store and are not
// sent again.
func (c *Core) streamTerrain(area terrain.Area, center *modelv1.Gps) {
	c.mu.Lock()
	if c.terrainCancel != nil {
		c.terrainCancel()
	}
	ctx, cancel := context.WithCancel(c.ctx)
	c.terrainCancel = cancel
	c.baseMapStreams++
	c.mu.Unlock()

	ch := c.terrain.Stream(ctx, area, center)
	go func() {
		n := 0
		for chunk := range ch {
			c.PutChunk(chunk)
			n++
		}
		c.mu.Lock()
		c.terrainChunks += uint64(n)
		c.baseMapStreams--
		c.mu.Unlock()
		if ctx.Err() == nil {
			log.Printf("[core] base map stream done: %d chunk(s)", n)
		} else {
			log.Printf("[core] base map stream superseded after %d chunk(s)", n)
		}
		c.PublishSiteIfChanged()
	}()
}
