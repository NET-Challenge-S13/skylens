// Package terrain is the control tower's terrain and building feed. It turns
// public map data into pointblitz chunks that the core streams to viewers next
// to the reconstruction chunks (COMPONENTS.md §9, SPEC.md §6):
//
//   - terrain: AWS Terrain Tiles (terrarium PNG DEM, ~30 m SRTM in Korea),
//     coloured by the VWorld WMTS satellite layer when a key is configured;
//   - buildings: VWorld WFS lt_c_bldginfo footprints extruded to roof height.
//
// Data is cut into a fixed grid of CellSizeM cells about the anchor and sent
// nearest to the viewer's centre first, then a coarser background ring.
// Positions are local ENU metres about Config.Anchor (pointblitz axes: x east,
// y north, z up), computed with the formulas of src/shared/geo.rs.
package terrain

import (
	"context"
	"log"
	"math"
	"net/http"
	"sync"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
)

// Style selects how terrain and buildings are coloured. It mirrors the control
// tower's display options (SPEC.md §6).
type Style int

const (
	// StyleBlack (default): dark buildings with cold linework over a
	// navy/teal-graded map.
	StyleBlack Style = iota
	// StyleAerial: satellite colours on terrain and buildings.
	StyleAerial
	// StylePoints: buildings as a height-ramped cyan point cloud over the
	// graded map.
	StylePoints
)

// Config configures a Feed.
type Config struct {
	// VWorldKey enables satellite colours and buildings. Empty means DEM
	// terrain only. The key is never logged, cached under, or returned.
	VWorldKey string
	// Anchor is the ENU origin. Nil means the centre of the first streamed
	// area, kept for the Feed's lifetime.
	Anchor *modelv1.Gps
	// CacheDir caches downloaded tiles and footprint pages. Empty disables it.
	CacheDir string
	// HTTPClient is used for every download. Nil means a client with a 30 s
	// timeout.
	HTTPClient *http.Client
	// VWorldDomain, when set, is sent as VWorld's domain parameter and as
	// Referer, for keys registered to a service URL.
	VWorldDomain string
	// DisableRing skips the coarse background ring around the area.
	DisableRing bool
}

// Area is a geographic box in degrees.
type Area struct {
	MinLat, MinLon, MaxLat, MaxLon float64
}

// Feed produces terrain and building chunks. It is safe for concurrent use.
type Feed struct {
	key    string
	domain string
	ring   bool
	fetch  *fetcher

	mu     sync.Mutex
	anchor *geoPoint
	// probes caches whether VWorld imagery answered for a z6 tile.
	probes map[tileKey]bool
}

// cellWorkers is how many cells are built at once; output order stays
// nearest-first regardless.
const cellWorkers = 3

// New returns a Feed for cfg.
func New(cfg Config) *Feed {
	f := &Feed{
		key:    cfg.VWorldKey,
		domain: cfg.VWorldDomain,
		ring:   !cfg.DisableRing,
		fetch:  newFetcher(cfg.HTTPClient, cfg.CacheDir, cfg.VWorldDomain),
		probes: make(map[tileKey]bool),
	}
	if cfg.Anchor != nil {
		f.anchor = &geoPoint{Lat: cfg.Anchor.GetLat(), Lon: cfg.Anchor.GetLon(), Alt: cfg.Anchor.GetAlt()}
	}
	return f
}

// Stream sends CHUNK_KIND_TERRAIN and CHUNK_KIND_BUILDINGS chunks covering
// area, cells nearest to center first, then closes the channel. Cancelling
// ctx stops it.
//
// Every chunk has BlockId [x, y, 0] (the cell index), Version 1 and Sequence
// 0; the core assigns sequences. Cells of the area are LEVEL_REFINED; cells of
// the background ring are LEVEL_PREVIEW at lower density, so a later stream
// whose area covers them replaces them. Within a cell the terrain chunk comes
// before the building chunk. Cells whose data could not be fetched are
// skipped. A nil center means the centre of area.
func (f *Feed) Stream(ctx context.Context, area Area, center *modelv1.Gps, style Style) <-chan *modelv1.Chunk {
	out := make(chan *modelv1.Chunk, 4)
	go f.run(ctx, area, center, style, out)
	return out
}

func (f *Feed) run(ctx context.Context, area Area, center *modelv1.Gps, style Style, out chan<- *modelv1.Chunk) {
	var wg sync.WaitGroup
	defer close(out)
	defer wg.Wait() // runs first: no worker outlives the stream

	if area.MinLat > area.MaxLat {
		area.MinLat, area.MaxLat = area.MaxLat, area.MinLat
	}
	if area.MinLon > area.MaxLon {
		area.MinLon, area.MaxLon = area.MaxLon, area.MinLon
	}
	if !finite(area.MinLat, area.MinLon, area.MaxLat, area.MaxLon) {
		return
	}
	cLat := (area.MinLat + area.MaxLat) / 2
	cLon := (area.MinLon + area.MaxLon) / 2
	if center != nil && finite(center.GetLat(), center.GetLon()) {
		cLat, cLon = center.GetLat(), center.GetLon()
	}
	anchor := f.anchorFor(geoPoint{Lat: (area.MinLat + area.MaxLat) / 2, Lon: (area.MinLon + area.MaxLon) / 2})

	sw := gpsToENU(geoPoint{Lat: area.MinLat, Lon: area.MinLon}, anchor)
	ne := gpsToENU(geoPoint{Lat: area.MaxLat, Lon: area.MaxLon}, anchor)
	c := gpsToENU(geoPoint{Lat: cLat, Lon: cLon}, anchor)
	jobs := planCells(enuRect{sw.E, sw.N, ne.E, ne.N}, c.E, c.N, f.ring)
	if len(jobs) == 0 {
		return
	}

	s := &session{
		feed:   f,
		anchor: anchor,
		style:  style,
		dem:    &demSampler{z: demZoom, tiles: newTileStore(f.fetch.loadDEMTile)},
		sat:    make(map[int]*satSampler),
	}
	if f.key != "" && f.imageryReachable(ctx, cLat, cLon) {
		key := f.key
		for _, p := range []*profile{&fineProfile, &coarseProfile} {
			if _, ok := s.sat[p.satZoom]; !ok {
				s.sat[p.satZoom] = &satSampler{z: p.satZoom, tiles: newTileStore(func(ctx context.Context, k tileKey) ([]uint8, error) {
					return f.fetch.loadSatTile(ctx, key, k)
				})}
			}
		}
	}

	// Ordered pipeline: up to cellWorkers cells are built ahead, and results
	// are emitted strictly in plan order. A slot is released only after its
	// chunks were sent, which bounds memory to cellWorkers cells.
	slots := make([]chan []*modelv1.Chunk, len(jobs))
	for i := range slots {
		slots[i] = make(chan []*modelv1.Chunk, 1)
	}
	sem := make(chan struct{}, cellWorkers)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range jobs {
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return
			}
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				slots[i] <- s.buildCell(ctx, jobs[i])
			}(i)
		}
	}()

	for i := range jobs {
		var chunks []*modelv1.Chunk
		select {
		case chunks = <-slots[i]:
		case <-ctx.Done():
			return
		}
		for _, ch := range chunks {
			select {
			case out <- ch:
			case <-ctx.Done():
				return
			}
		}
		<-sem
	}
}

func finite(vs ...float64) bool {
	for _, v := range vs {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

func (f *Feed) anchorFor(fallback geoPoint) geoPoint {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.anchor == nil {
		a := fallback
		f.anchor = &a
	}
	return *f.anchor
}

// imageryReachable probes one VWorld tile (z6, inside the served range) and
// remembers the verdict, so a missing or rejected key costs one request
// instead of a doomed request per tile. Cancellation is not remembered.
func (f *Feed) imageryReachable(ctx context.Context, lat, lon float64) bool {
	k := tileKey{satMinZoom, int(math.Floor(lonToTileX(lon, satMinZoom))), int(math.Floor(latToTileY(lat, satMinZoom)))}
	f.mu.Lock()
	ok, known := f.probes[k]
	f.mu.Unlock()
	if known {
		return ok
	}
	_, err := f.fetch.loadSatTile(ctx, f.key, k)
	if ctx.Err() != nil {
		return false
	}
	ok = err == nil
	if !ok {
		log.Printf("terrain: satellite imagery unavailable, colouring from the DEM instead: %v", err)
	}
	f.mu.Lock()
	f.probes[k] = ok
	f.mu.Unlock()
	return ok
}

// session is the state of one Stream call: tiles loaded once and shared by
// all of its cells.
type session struct {
	feed   *Feed
	anchor geoPoint
	style  Style
	dem    *demSampler
	sat    map[int]*satSampler // by zoom; empty without imagery
}

// cellBox is the lat/lon box of cell (x, y).
func (s *session) cellBox(x, y int) latLonBox {
	sw := enuToGPS(enu{E: float64(x) * CellSizeM, N: float64(y) * CellSizeM}, s.anchor)
	ne := enuToGPS(enu{E: float64(x+1) * CellSizeM, N: float64(y+1) * CellSizeM}, s.anchor)
	return latLonBox{minLat: sw.Lat, minLon: sw.Lon, maxLat: ne.Lat, maxLon: ne.Lon}
}

func (s *session) buildCell(ctx context.Context, j cellJob) []*modelv1.Chunk {
	if ctx.Err() != nil {
		return nil
	}
	box := s.cellBox(j.x, j.y)
	sat := s.sat[j.prof.satZoom] // nil without imagery

	// Fetch what the cell needs up front, in parallel; sampling then hits memory.
	s.dem.tiles.prefetch(ctx, tilesCovering(box.minLat, box.minLon, box.maxLat, box.maxLon, demZoom), 4)
	if sat != nil {
		sat.tiles.prefetch(ctx, tilesCovering(box.minLat, box.minLon, box.maxLat, box.maxLon, sat.z), satPrefetch)
	}

	var chunks []*modelv1.Chunk
	if pts := s.terrainPoints(ctx, j, sat); len(pts) > 0 {
		chunks = append(chunks, s.chunk(modelv1.ChunkKind_CHUNK_KIND_TERRAIN, j, pts))
	}
	if s.feed.key != "" && ctx.Err() == nil {
		if pts := s.buildingPoints(ctx, j, box, sat); len(pts) > 0 {
			chunks = append(chunks, s.chunk(modelv1.ChunkKind_CHUNK_KIND_BUILDINGS, j, pts))
		}
	}
	if ctx.Err() != nil {
		return nil
	}
	return chunks
}

func (s *session) chunk(kind modelv1.ChunkKind, j cellJob, pts []Point) *modelv1.Chunk {
	return &modelv1.Chunk{
		Kind:     kind,
		Level:    j.prof.level,
		BlockId:  []int32{int32(j.x), int32(j.y), 0},
		Version:  1,
		Sequence: 0,
		// Each chunk is the whole of its block: one generation, last chunk.
		Payload: EncodeChunk(0, 0, FlagLastInGeneration, pts),
	}
}

// terrainPoints samples the DEM on a regular grid inside the cell. Sample
// centres sit at half steps, so adjacent cells never share a point.
func (s *session) terrainPoints(ctx context.Context, j cellJob, sat *satSampler) []Point {
	step := j.prof.terrainStep
	n := int(math.Round(CellSizeM / step))
	e0 := float64(j.x) * CellSizeM
	n0 := float64(j.y) * CellSizeM
	elev := make([]float64, n*n)
	lat := make([]float64, n)
	lon := make([]float64, n)
	for i := 0; i < n; i++ {
		g := enuToGPS(enu{E: e0 + (float64(i)+0.5)*step, N: n0 + (float64(i)+0.5)*step}, s.anchor)
		lon[i] = g.Lon // column i
		lat[i] = g.Lat // row i
	}
	for r := 0; r < n; r++ {
		if ctx.Err() != nil {
			return nil
		}
		for c := 0; c < n; c++ {
			elev[r*n+c] = s.dem.elevation(ctx, lat[r], lon[c])
		}
	}
	if !sanitizeGrid(elev, n, n) {
		log.Printf("terrain: no elevation for cell %d,%d", j.x, j.y)
		return nil
	}

	pts := make([]Point, 0, n*n)
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			h := elev[r*n+c]
			// Slope shading from the east neighbour (cheap hillshade).
			east := elev[r*n+min(c+1, n-1)]
			shade := math.Min(1.25, math.Max(0.55, 1-(east-h)/step))
			col := s.terrainColour(ctx, sat, lat[r], lon[c], h, shade)
			pts = append(pts, Point{
				Pos:   [3]float32{float32(e0 + (float64(c)+0.5)*step), float32(n0 + (float64(r)+0.5)*step), float32(h - s.anchor.Alt)},
				Color: col,
			})
		}
	}
	return pts
}

// terrainColour: StyleAerial shows the photograph; the tactical styles grade
// it into the cold palette. Without imagery the DEM's hypsometric ramp stands
// in (graded the same way for the tactical styles).
func (s *session) terrainColour(ctx context.Context, sat *satSampler, lat, lon, elev, shade float64) [3]uint8 {
	var base [3]uint8
	photo := false
	if sat != nil {
		base, photo = sat.color(ctx, lat, lon)
	}
	if !photo {
		h := hypsometric(elev)
		base = toRGB([3]float64{h[0] * shade, h[1] * shade, h[2] * shade})
	}
	if s.style == StyleAerial {
		return base
	}
	graded := cyberGrade(base)
	if photo {
		// The photograph carries its own relief; keep only a hint of slope.
		return scaleRGB(graded, 0.85+0.15*shade)
	}
	return graded
}

// buildingPoints samples every footprint whose centroid lies in the cell, so
// a footprint crossing a border is drawn by exactly one cell.
func (s *session) buildingPoints(ctx context.Context, j cellJob, box latLonBox, sat *satSampler) []Point {
	blds, err := s.feed.fetch.fetchBuildings(ctx, s.feed.key, s.feed.domain, box)
	if err != nil {
		if ctx.Err() == nil {
			log.Printf("terrain: buildings for cell %d,%d unavailable: %v", j.x, j.y, err)
		}
		return nil
	}
	var pts []Point
	emit := func(p Point) { pts = append(pts, p) }
	drape := func(e, n float64) ([3]uint8, bool) {
		g := enuToGPS(enu{E: e, N: n}, s.anchor)
		return sat.color(ctx, g.Lat, g.Lon)
	}
	for _, b := range blds {
		if ctx.Err() != nil {
			return nil
		}
		for ri, ring := range b.rings {
			var cLat, cLon float64
			for _, v := range ring {
				cLon += v[0]
				cLat += v[1]
			}
			cLat /= float64(len(ring))
			cLon /= float64(len(ring))
			// Ownership is decided by the first ring's centroid.
			if ri == 0 && !box.contains(cLat, cLon) {
				break
			}
			// Ground snap at the LOWEST ground under the footprint (vertices
			// and centroid): a centroid-only snap leaves downhill edges floating.
			base := s.dem.elevation(ctx, cLat, cLon)
			ringENU := make([][2]float64, len(ring))
			for i, v := range ring {
				p := gpsToENU(geoPoint{Lat: v[1], Lon: v[0]}, s.anchor)
				ringENU[i] = [2]float64{p.E, p.N}
				if e := s.dem.elevation(ctx, v[1], v[0]); !badElevation(e) && (badElevation(base) || e < base) {
					base = e
				}
			}
			if badElevation(base) {
				continue // no ground to stand on
			}
			baseU := base - s.anchor.Alt
			sampleBuilding(ringENU, baseU, baseU+b.heightM, *j.prof, s.style, drape, emit)
		}
	}
	return pts
}
