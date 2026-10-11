// Package terrain is the control tower's base map feed. It turns public map
// data into raw source chunks that the core streams to the control tower
// next to the reconstruction chunks (COMPONENTS.md §9, SPEC.md §6):
//
//   - CHUNK_KIND_TERRAIN carries payload.v1.TerrainTile: an AWS Terrain Tiles
//     DEM height grid (terrarium PNG, ~30 m SRTM in Korea) over the cell, and
//     the VWorld WMTS satellite image of the cell as JPEG when a key is set;
//   - CHUNK_KIND_BUILDINGS carries payload.v1.BuildingSet: VWorld WFS
//     lt_c_bldginfo footprints with their ground height and extrusion height.
//
// Nothing here decides how the map looks: no colour grading, no styles, no
// vertical exaggeration. The control tower builds the look from this data.
//
// Data is cut into a fixed grid of CellSizeM cells about the anchor and sent
// nearest to the given centre first, then a coarser background ring. The feed
// remembers what it already sent, so a later stream over a larger area only
// adds the cells that are new (or upgrades ring cells to full detail).
// Positions are local ENU metres about Config.Anchor, heights are metres above
// the anchor altitude, computed with the formulas of src/shared/geo.rs.
package terrain

import (
	"context"
	"fmt"
	"log"
	"math"
	"net/http"
	"sync"

	"google.golang.org/protobuf/proto"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	payloadv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/payload/v1"
)

// Config configures a Feed.
type Config struct {
	// VWorldKey enables satellite imagery and buildings. Empty means DEM
	// terrain only. The key is never logged, cached under, or returned.
	VWorldKey string
	// Anchor is the ENU origin of every chunk. Nil means the centre of the
	// first streamed area, kept for the Feed's lifetime.
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

// Union is the smallest area covering a and b. A zero area is empty.
func (a Area) Union(b Area) Area {
	if a == (Area{}) {
		return b
	}
	if b == (Area{}) {
		return a
	}
	return Area{
		MinLat: math.Min(a.MinLat, b.MinLat),
		MinLon: math.Min(a.MinLon, b.MinLon),
		MaxLat: math.Max(a.MaxLat, b.MaxLat),
		MaxLon: math.Max(a.MaxLon, b.MaxLon),
	}
}

// Status is what the feed can offer, for Site.
type Status struct {
	// Imagery: a key is set and VWorld imagery did not fail its probe.
	Imagery bool
	// Buildings: a key is set and the footprint service did not only fail.
	Buildings bool
	// BuildingCount is the number of footprints sent so far (latest per cell).
	BuildingCount int
	// Cells is the number of cells whose terrain was sent.
	Cells int
	// Streaming: a stream is running; Pending is how many of its cells are not done yet.
	Streaming bool
	Pending   int
}

type probeState int

const (
	probeUnknown probeState = iota
	probeOK
	probeFailed
)

type cellKey struct{ x, y int }

// sentCell is what the feed already delivered for one cell.
type sentCell struct {
	terrain       modelv1.Level // LEVEL_UNSPECIFIED = not yet
	buildings     modelv1.Level
	buildingCount int
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
	probes  map[tileKey]bool
	imagery probeState
	wfsOK   bool
	wfsFail bool
	sent    map[cellKey]*sentCell
	// streams running and their cells still to deliver (loading progress).
	streams int
	owed    int
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
		sent:   make(map[cellKey]*sentCell),
	}
	if cfg.Anchor != nil {
		f.anchor = &geoPoint{Lat: cfg.Anchor.GetLat(), Lon: cfg.Anchor.GetLon(), Alt: cfg.Anchor.GetAlt()}
	}
	return f
}

// Status reports what the feed can offer and how much it has sent.
func (f *Feed) Status() Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	st := Status{
		Imagery:   f.key != "" && f.imagery != probeFailed,
		Buildings: f.key != "" && (f.wfsOK || !f.wfsFail),
	}
	for _, c := range f.sent {
		if c.terrain != modelv1.Level_LEVEL_UNSPECIFIED {
			st.Cells++
		}
		st.BuildingCount += c.buildingCount
	}
	st.Streaming = f.streams > 0
	st.Pending = f.owed
	return st
}

// Stream sends CHUNK_KIND_TERRAIN and CHUNK_KIND_BUILDINGS chunks covering
// area, cells nearest to center first, then closes the channel. Cancelling
// ctx stops it.
//
// Every chunk has BlockId [x, y, 0] (the cell index), Version 1 and Sequence
// 0; the core assigns sequences. Cells of the area are LEVEL_REFINED; cells of
// the background ring are LEVEL_PREVIEW at lower resolution, so a later stream
// whose area covers them replaces them. Within a cell the terrain chunk comes
// before the building chunk; a cell without footprints has no building chunk.
// Cells already sent at the same or a finer level are skipped. Cells whose
// data could not be fetched are skipped too (and tried again by the next
// stream). A nil center means the centre of area.
func (f *Feed) Stream(ctx context.Context, area Area, center *modelv1.Gps) <-chan *modelv1.Chunk {
	out := make(chan *modelv1.Chunk, 4)
	go f.run(ctx, area, center, out)
	return out
}

// cellResult is one built cell.
type cellResult struct {
	terrain   *modelv1.Chunk
	buildings *modelv1.Chunk
	// buildingsDone: the footprint query answered (even with nothing).
	buildingsDone bool
	buildingCount int
}

func (f *Feed) run(ctx context.Context, area Area, center *modelv1.Gps, out chan<- *modelv1.Chunk) {
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
	jobs := f.pending(planCells(enuRect{sw.E, sw.N, ne.E, ne.N}, c.E, c.N, f.ring))
	if len(jobs) == 0 {
		return
	}
	f.mu.Lock()
	f.streams++
	f.owed += len(jobs)
	f.mu.Unlock()
	left := len(jobs)
	defer func() {
		f.mu.Lock()
		f.streams--
		f.owed -= left
		f.mu.Unlock()
	}()

	s := &session{
		feed:   f,
		anchor: anchor,
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
	slots := make([]chan cellResult, len(jobs))
	for i := range slots {
		slots[i] = make(chan cellResult, 1)
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

	send := func(ch *modelv1.Chunk) bool {
		select {
		case out <- ch:
			return true
		case <-ctx.Done():
			return false
		}
	}
	for i, j := range jobs {
		var r cellResult
		select {
		case r = <-slots[i]:
		case <-ctx.Done():
			return
		}
		if ctx.Err() != nil {
			return
		}
		if r.terrain != nil {
			if !send(r.terrain) {
				return
			}
			f.markTerrain(j)
		}
		if r.buildings != nil && !send(r.buildings) {
			return
		}
		if r.buildingsDone {
			f.markBuildings(j, r.buildingCount)
		}
		f.mu.Lock()
		f.owed--
		f.mu.Unlock()
		left--
		<-sem
	}
}

// pending drops what was already sent and marks what each cell still needs.
func (f *Feed) pending(jobs []cellJob) []cellJob {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := jobs[:0]
	for _, j := range jobs {
		sc := f.sent[cellKey{j.x, j.y}]
		j.needTerrain = sc == nil || sc.terrain < j.prof.level
		j.needBuildings = f.key != "" && (sc == nil || sc.buildings < j.prof.level)
		if j.needTerrain || j.needBuildings {
			out = append(out, j)
		}
	}
	return out
}

func (f *Feed) cell(j cellJob) *sentCell {
	k := cellKey{j.x, j.y}
	sc := f.sent[k]
	if sc == nil {
		sc = &sentCell{}
		f.sent[k] = sc
	}
	return sc
}

func (f *Feed) markTerrain(j cellJob) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cell(j).terrain = j.prof.level
}

func (f *Feed) markBuildings(j cellJob, n int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	sc := f.cell(j)
	sc.buildings = j.prof.level
	sc.buildingCount = n
}

func (f *Feed) noteWFS(ok bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if ok {
		f.wfsOK = true
	} else {
		f.wfsFail = true
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
		log.Printf("terrain: satellite imagery unavailable, terrain goes out without images: %v", err)
	}
	f.mu.Lock()
	f.probes[k] = ok
	if ok {
		f.imagery = probeOK
	} else {
		f.imagery = probeFailed
	}
	f.mu.Unlock()
	return ok
}

// session is the state of one Stream call: tiles loaded once and shared by
// all of its cells.
type session struct {
	feed   *Feed
	anchor geoPoint
	dem    *demSampler
	sat    map[int]*satSampler // by zoom; empty without imagery
}

// cellBox is the lat/lon box of cell (x, y).
func (s *session) cellBox(x, y int) latLonBox {
	sw := enuToGPS(enu{E: float64(x) * CellSizeM, N: float64(y) * CellSizeM}, s.anchor)
	ne := enuToGPS(enu{E: float64(x+1) * CellSizeM, N: float64(y+1) * CellSizeM}, s.anchor)
	return latLonBox{minLat: sw.Lat, minLon: sw.Lon, maxLat: ne.Lat, maxLon: ne.Lon}
}

func (s *session) buildCell(ctx context.Context, j cellJob) cellResult {
	var r cellResult
	if ctx.Err() != nil {
		return r
	}
	box := s.cellBox(j.x, j.y)
	var sat *satSampler
	if j.needTerrain {
		sat = s.sat[j.prof.satZoom] // nil without imagery
	}

	// Fetch what the cell needs up front, in parallel; sampling then hits memory.
	s.dem.tiles.prefetch(ctx, tilesCovering(box.minLat, box.minLon, box.maxLat, box.maxLon, demZoom), 4)
	if sat != nil {
		sat.tiles.prefetch(ctx, tilesCovering(box.minLat, box.minLon, box.maxLat, box.maxLon, sat.z), satPrefetch)
	}

	if j.needTerrain {
		if tile := s.terrainTile(ctx, j, sat); tile != nil {
			r.terrain = chunk(modelv1.ChunkKind_CHUNK_KIND_TERRAIN, j, tile)
		}
	}
	if j.needBuildings && ctx.Err() == nil {
		if set, ok := s.buildingSet(ctx, j, box); ok {
			r.buildingsDone = true
			r.buildingCount = len(set.GetBuildings())
			if r.buildingCount > 0 {
				r.buildings = chunk(modelv1.ChunkKind_CHUNK_KIND_BUILDINGS, j, set)
			}
		}
	}
	if ctx.Err() != nil {
		return cellResult{}
	}
	return r
}

func chunk(kind modelv1.ChunkKind, j cellJob, msg proto.Message) *modelv1.Chunk {
	payload, err := proto.MarshalOptions{Deterministic: true}.Marshal(msg)
	if err != nil {
		log.Printf("terrain: cell %d,%d: %v", j.x, j.y, err)
		return nil
	}
	return &modelv1.Chunk{
		Kind:     kind,
		Level:    j.prof.level,
		BlockId:  []int32{int32(j.x), int32(j.y), 0},
		Version:  1,
		Sequence: 0,
		Payload:  payload,
	}
}

// terrainTile samples the DEM on a grid with a vertex on every cell edge and
// attaches the cell's satellite image.
func (s *session) terrainTile(ctx context.Context, j cellJob, sat *satSampler) *payloadv1.TerrainTile {
	step := j.prof.gridStep
	n := int(math.Round(CellSizeM / step))
	cols := n + 1
	e0 := float64(j.x) * CellSizeM
	n0 := float64(j.y) * CellSizeM
	lat := make([]float64, cols) // row r
	lon := make([]float64, cols) // column c
	for i := 0; i < cols; i++ {
		lat[i] = enuToGPS(enu{N: n0 + float64(i)*step}, s.anchor).Lat
		lon[i] = enuToGPS(enu{E: e0 + float64(i)*step}, s.anchor).Lon
	}
	elev := make([]float64, cols*cols)
	for r := 0; r < cols; r++ {
		if ctx.Err() != nil {
			return nil
		}
		for c := 0; c < cols; c++ {
			elev[r*cols+c] = s.dem.elevation(ctx, lat[r], lon[c])
		}
	}
	if !sanitizeGrid(elev, cols, cols) {
		log.Printf("terrain: no elevation for cell %d,%d", j.x, j.y)
		return nil
	}
	heights := make([]float32, len(elev))
	for i, h := range elev {
		heights[i] = float32(h - s.anchor.Alt)
	}
	return &payloadv1.TerrainTile{
		Cell:        []int32{int32(j.x), int32(j.y)},
		MinE:        e0,
		MinN:        n0,
		MaxE:        e0 + CellSizeM,
		MaxN:        n0 + CellSizeM,
		Cols:        uint32(cols),
		Rows:        uint32(cols),
		Heights:     heights,
		ImageryJpeg: cellImage(ctx, sat, s.anchor, e0, n0, CellSizeM, j.prof.imagePx),
		Ring:        j.prof.ring,
	}
}

// buildingSet collects every footprint whose centroid lies in the cell, so a
// footprint crossing a border belongs to exactly one cell. ok is false when
// the footprint service could not be reached.
func (s *session) buildingSet(ctx context.Context, j cellJob, box latLonBox) (*payloadv1.BuildingSet, bool) {
	blds, err := s.feed.fetch.fetchBuildings(ctx, s.feed.key, s.feed.domain, box)
	if err != nil {
		if ctx.Err() == nil {
			s.feed.noteWFS(false)
			log.Printf("terrain: buildings for cell %d,%d unavailable: %v", j.x, j.y, err)
		}
		return nil, false
	}
	s.feed.noteWFS(true)
	set := &payloadv1.BuildingSet{Cell: []int32{int32(j.x), int32(j.y)}}
	for _, b := range blds {
		if ctx.Err() != nil {
			return nil, false
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
			// Ground at the LOWEST point under the footprint (vertices and
			// centroid): a centroid-only snap leaves downhill edges floating.
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
			id := b.id
			if ri > 0 {
				id = fmt.Sprintf("%s#%d", b.id, ri)
			}
			set.Buildings = append(set.Buildings, &payloadv1.Building{
				Id:      id,
				Outline: outline(ringENU),
				Base:    base - s.anchor.Alt,
				Height:  b.heightM,
			})
		}
	}
	return set, true
}
