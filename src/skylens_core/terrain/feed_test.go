package terrain

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"io/fs"
	"math"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	payloadv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/payload/v1"
)

const testKey = "test-key-123"

// fakeMaps answers terrarium, VWorld WMTS and VWorld WFS requests offline.
// The DEM is flat at demM metres; imagery is one solid colour; the WFS always
// returns the same single building, whatever box is asked for.
type fakeMaps struct {
	key      string
	demM     int
	sat      color.RGBA
	building string

	mu       sync.Mutex
	requests []string
}

func (m *fakeMaps) RoundTrip(r *http.Request) (*http.Response, error) {
	m.mu.Lock()
	m.requests = append(m.requests, r.URL.String())
	m.mu.Unlock()
	switch {
	case r.URL.Host == "s3.amazonaws.com" && strings.HasPrefix(r.URL.Path, "/elevation-tiles-prod/terrarium/"):
		return respond(r, http.StatusOK, "image/png", m.demPNG()), nil
	case r.URL.Host == "api.vworld.kr" && strings.HasPrefix(r.URL.Path, "/req/wmts/1.0.0/"):
		parts := strings.Split(r.URL.Path, "/")
		if len(parts) < 9 || parts[4] != m.key || parts[5] != "Satellite" {
			return respond(r, http.StatusOK, "text/xml", []byte("<ExceptionReport/>")), nil
		}
		return respond(r, http.StatusOK, "image/jpeg", m.satJPEG()), nil
	case r.URL.Host == "api.vworld.kr" && r.URL.Path == "/req/wfs":
		if r.URL.Query().Get("key") != m.key {
			return respond(r, http.StatusOK, "text/xml", []byte("<ExceptionReport/>")), nil
		}
		return respond(r, http.StatusOK, "application/json", []byte(m.building)), nil
	}
	return respond(r, http.StatusNotFound, "text/plain", nil), nil
}

func (m *fakeMaps) count(host string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := 0
	for _, u := range m.requests {
		if strings.Contains(u, "://"+host+"/") {
			n++
		}
	}
	return n
}

func respond(r *http.Request, status int, contentType string, body []byte) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": {contentType}},
		Body:       io.NopCloser(bytes.NewReader(body)),
		Request:    r,
	}
}

func (m *fakeMaps) demPNG() []byte {
	v := m.demM + 32768
	img := image.NewNRGBA(image.Rect(0, 0, 256, 256))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = uint8(v>>8), uint8(v), 0, 255
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

func (m *fakeMaps) satJPEG() []byte {
	img := image.NewRGBA(image.Rect(0, 0, 256, 256))
	for y := 0; y < 256; y++ {
		for x := 0; x < 256; x++ {
			img.Set(x, y, m.sat)
		}
	}
	var buf bytes.Buffer
	jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95})
	return buf.Bytes()
}

// buildingAt is a WFS page holding one rectangle (in ENU metres about anchor).
func buildingAt(anchor geoPoint, e0, n0, e1, n1, height float64) string {
	c := func(e, n float64) string {
		g := enuToGPS(enu{E: e, N: n}, anchor)
		return fmt.Sprintf("[%.9f,%.9f]", g.Lon, g.Lat)
	}
	// Clockwise on purpose: the feed must turn it counter-clockwise.
	return `{"type":"FeatureCollection","features":[{"type":"Feature","id":"lt_c_bldginfo.42",` +
		`"geometry":{"type":"Polygon","coordinates":[[` +
		c(e0, n0) + "," + c(e0, n1) + "," + c(e1, n1) + "," + c(e1, n0) + "," + c(e0, n0) +
		`]]},"properties":{"height":` + fmt.Sprint(height) + `,"grnd_flr":5}}]}`
}

func testAreaAndCentre() (Area, *modelv1.Gps) {
	sw := enuToGPS(enu{E: -300, N: -300}, testAnchor)
	ne := enuToGPS(enu{E: 300, N: 300}, testAnchor)
	c := enuToGPS(enu{E: 200, N: 200}, testAnchor)
	return Area{MinLat: sw.Lat, MinLon: sw.Lon, MaxLat: ne.Lat, MaxLon: ne.Lon},
		&modelv1.Gps{Lat: c.Lat, Lon: c.Lon}
}

func newTestFeed(t *testing.T, key string, maps *fakeMaps, cacheDir string) *Feed {
	t.Helper()
	return New(Config{
		VWorldKey:   key,
		Anchor:      &modelv1.Gps{Lat: testAnchor.Lat, Lon: testAnchor.Lon, Alt: testAnchor.Alt},
		CacheDir:    cacheDir,
		HTTPClient:  &http.Client{Transport: maps},
		DisableRing: true,
	})
}

func collect(t *testing.T, ch <-chan *modelv1.Chunk) []*modelv1.Chunk {
	t.Helper()
	var out []*modelv1.Chunk
	timeout := time.After(60 * time.Second)
	for {
		select {
		case c, ok := <-ch:
			if !ok {
				return out
			}
			out = append(out, c)
		case <-timeout:
			t.Fatal("stream did not close")
		}
	}
}

func decodeTile(t *testing.T, c *modelv1.Chunk) *payloadv1.TerrainTile {
	t.Helper()
	tile := &payloadv1.TerrainTile{}
	if err := proto.Unmarshal(c.GetPayload(), tile); err != nil {
		t.Fatalf("terrain payload: %v", err)
	}
	return tile
}

func decodeBuildings(t *testing.T, c *modelv1.Chunk) *payloadv1.BuildingSet {
	t.Helper()
	set := &payloadv1.BuildingSet{}
	if err := proto.Unmarshal(c.GetPayload(), set); err != nil {
		t.Fatalf("buildings payload: %v", err)
	}
	return set
}

func TestStreamWithoutKeyIsTerrainOnly(t *testing.T) {
	maps := &fakeMaps{key: testKey, demM: 100}
	cache := t.TempDir()
	f := newTestFeed(t, "", maps, cache)
	area, centre := testAreaAndCentre()
	chunks := collect(t, f.Stream(context.Background(), area, centre))

	// Four cells, nearest to the centre (in cell 0,0) first.
	want := [][2]int32{{0, 0}, {0, -1}, {-1, 0}, {-1, -1}}
	if len(chunks) != len(want) {
		t.Fatalf("%d chunks, want %d", len(chunks), len(want))
	}
	for i, c := range chunks {
		if c.Kind != modelv1.ChunkKind_CHUNK_KIND_TERRAIN {
			t.Errorf("chunk %d kind %v, want terrain", i, c.Kind)
		}
		if c.Level != modelv1.Level_LEVEL_REFINED || c.Version != 1 || c.Sequence != 0 {
			t.Errorf("chunk %d level %v version %d sequence %d", i, c.Level, c.Version, c.Sequence)
		}
		if len(c.BlockId) != 3 || c.BlockId[0] != want[i][0] || c.BlockId[1] != want[i][1] || c.BlockId[2] != 0 {
			t.Fatalf("chunk %d block %v, want [%d %d 0]", i, c.BlockId, want[i][0], want[i][1])
		}
		tile := decodeTile(t, c)
		if len(tile.Cell) != 2 || tile.Cell[0] != want[i][0] || tile.Cell[1] != want[i][1] {
			t.Fatalf("chunk %d tile cell %v", i, tile.Cell)
		}
		e0, n0 := float64(want[i][0])*CellSizeM, float64(want[i][1])*CellSizeM
		if tile.MinE != e0 || tile.MinN != n0 || tile.MaxE != e0+CellSizeM || tile.MaxN != n0+CellSizeM {
			t.Fatalf("chunk %d rectangle %g,%g..%g,%g", i, tile.MinE, tile.MinN, tile.MaxE, tile.MaxN)
		}
		// 10 m grid with a vertex on both edges: 41 x 41.
		if tile.Cols != 41 || tile.Rows != 41 || len(tile.Heights) != 41*41 {
			t.Fatalf("chunk %d grid %dx%d with %d heights", i, tile.Cols, tile.Rows, len(tile.Heights))
		}
		for _, h := range tile.Heights {
			// Flat DEM at 100 m, anchor at 30 m: 70 m above the anchor, with
			// no exaggeration.
			if math.Abs(float64(h)-70) > 0.01 {
				t.Fatalf("chunk %d height %g, want 70", i, h)
			}
		}
		if len(tile.ImageryJpeg) != 0 || tile.Ring {
			t.Errorf("chunk %d: %d image bytes without a key, ring %v", i, len(tile.ImageryJpeg), tile.Ring)
		}
	}
	if n := maps.count("api.vworld.kr"); n != 0 {
		t.Errorf("%d VWorld requests without a key", n)
	}
	if !fileExists(t, cache, "terrarium") {
		t.Error("terrain tiles were not cached")
	}
	if st := f.Status(); st.Imagery || st.Buildings || st.BuildingCount != 0 || st.Cells != 4 {
		t.Errorf("status without a key %+v", st)
	}

	// A second feed on the same cache does not download again.
	before := maps.count("s3.amazonaws.com")
	collect(t, newTestFeed(t, "", maps, cache).Stream(context.Background(), area, centre))
	if after := maps.count("s3.amazonaws.com"); after != before {
		t.Errorf("cached run made %d new DEM requests", after-before)
	}
}

func TestStreamWithKeySendsImageAndFootprints(t *testing.T) {
	maps := &fakeMaps{
		key:      testKey,
		demM:     100,
		sat:      color.RGBA{R: 200, G: 120, B: 40, A: 255},
		building: buildingAt(testAnchor, 100, 100, 120, 110, 15),
	}
	cache := t.TempDir()
	f := newTestFeed(t, testKey, maps, cache)
	area, centre := testAreaAndCentre()
	chunks := collect(t, f.Stream(context.Background(), area, centre))

	var terrain, blds []*modelv1.Chunk
	for _, c := range chunks {
		switch c.Kind {
		case modelv1.ChunkKind_CHUNK_KIND_TERRAIN:
			terrain = append(terrain, c)
		case modelv1.ChunkKind_CHUNK_KIND_BUILDINGS:
			blds = append(blds, c)
		default:
			t.Fatalf("unexpected kind %v", c.Kind)
		}
	}
	if len(terrain) != 4 {
		t.Fatalf("%d terrain chunks, want 4", len(terrain))
	}
	// The footprint is returned for every cell's query but owned by one cell.
	if len(blds) != 1 {
		t.Fatalf("%d building chunks, want 1", len(blds))
	}
	b := blds[0]
	if b.BlockId[0] != 0 || b.BlockId[1] != 0 || b.Level != modelv1.Level_LEVEL_REFINED || b.Version != 1 {
		t.Errorf("building chunk block %v level %v version %d", b.BlockId, b.Level, b.Version)
	}
	// Within cell (0,0), terrain comes first.
	if chunks[0].Kind != modelv1.ChunkKind_CHUNK_KIND_TERRAIN || chunks[1].Kind != modelv1.ChunkKind_CHUNK_KIND_BUILDINGS {
		t.Errorf("cell 0,0 order: %v then %v", chunks[0].Kind, chunks[1].Kind)
	}
	set := decodeBuildings(t, b)
	if len(set.Cell) != 2 || set.Cell[0] != 0 || set.Cell[1] != 0 || len(set.Buildings) != 1 {
		t.Fatalf("building set %v", set)
	}
	bl := set.Buildings[0]
	if bl.Id != "lt_c_bldginfo.42" || math.Abs(bl.Base-70) > 0.01 || bl.Height != 15 {
		t.Errorf("building id %q base %g height %g, want lt_c_bldginfo.42, 70, 15", bl.Id, bl.Base, bl.Height)
	}
	// Closed counter-clockwise ring: 4 corners plus the repeated first one.
	if len(bl.Outline) != 10 || bl.Outline[0] != bl.Outline[8] || bl.Outline[1] != bl.Outline[9] {
		t.Fatalf("outline %v", bl.Outline)
	}
	if signedArea(bl.Outline) <= 0 {
		t.Errorf("outline is not counter-clockwise: %v", bl.Outline)
	}
	for i := 0; i < len(bl.Outline); i += 2 {
		e, n := bl.Outline[i], bl.Outline[i+1]
		if e < 99.9 || e > 120.1 || n < 99.9 || n > 110.1 {
			t.Fatalf("outline vertex %g,%g outside the footprint", e, n)
		}
	}

	// The image is the photograph as is: a north-up JPEG of the cell.
	tile := decodeTile(t, terrain[0])
	img, err := jpeg.Decode(bytes.NewReader(tile.ImageryJpeg))
	if err != nil {
		t.Fatalf("cell image: %v", err)
	}
	if bounds := img.Bounds(); bounds.Dx() != 512 || bounds.Dy() != 512 {
		t.Fatalf("cell image %v, want 512x512", bounds)
	}
	r, g, bb, _ := img.At(256, 256).RGBA()
	if absDiff(uint8(r>>8), 200) > 10 || absDiff(uint8(g>>8), 120) > 10 || absDiff(uint8(bb>>8), 40) > 10 {
		t.Errorf("image colour %d,%d,%d, want about (200,120,40)", r>>8, g>>8, bb>>8)
	}
	if st := f.Status(); !st.Imagery || !st.Buildings || st.BuildingCount != 1 || st.Cells != 4 {
		t.Errorf("status %+v", st)
	}

	// Nothing under the cache directory may carry the key.
	filepath.WalkDir(cache, func(path string, d fs.DirEntry, err error) error {
		if strings.Contains(path, testKey) {
			t.Errorf("cache path carries the key: %s", path)
		}
		return nil
	})
	if !fileExists(t, cache, "vworld-satellite") || !fileExists(t, cache, "vworld-wfs") {
		t.Error("VWorld responses were not cached")
	}
}

func TestStreamWithRejectedKeyFallsBackToTerrain(t *testing.T) {
	maps := &fakeMaps{key: testKey, demM: 100, building: buildingAt(testAnchor, 100, 100, 120, 110, 15)}
	cache := t.TempDir()
	f := newTestFeed(t, "wrong-key", maps, cache)
	area, centre := testAreaAndCentre()
	chunks := collect(t, f.Stream(context.Background(), area, centre))
	if len(chunks) != 4 {
		t.Fatalf("%d chunks, want 4 terrain chunks", len(chunks))
	}
	for _, c := range chunks {
		if c.Kind != modelv1.ChunkKind_CHUNK_KIND_TERRAIN {
			t.Errorf("kind %v, want terrain", c.Kind)
		}
		if len(decodeTile(t, c).ImageryJpeg) != 0 {
			t.Error("image sent although imagery failed")
		}
	}
	// Error documents are not cached.
	if fileExists(t, cache, "vworld-wfs") || fileExists(t, cache, "vworld-satellite") {
		t.Error("a VWorld error document was cached")
	}
	if st := f.Status(); st.Imagery || st.Buildings {
		t.Errorf("status after a rejected key %+v", st)
	}
}

func TestLaterStreamOnlyAddsNewCells(t *testing.T) {
	maps := &fakeMaps{key: testKey, demM: 100}
	f := newTestFeed(t, "", maps, t.TempDir())
	area, centre := testAreaAndCentre()
	if n := len(collect(t, f.Stream(context.Background(), area, centre))); n != 4 {
		t.Fatalf("first stream %d chunks", n)
	}
	// The same area again: nothing new.
	if n := len(collect(t, f.Stream(context.Background(), area, centre))); n != 0 {
		t.Fatalf("repeated stream sent %d chunks", n)
	}
	// Grown east by one cell: only the two new cells.
	ne := enuToGPS(enu{E: 700, N: 300}, testAnchor)
	grown := area
	grown.MaxLon = ne.Lon
	chunks := collect(t, f.Stream(context.Background(), grown, centre))
	if len(chunks) != 2 {
		t.Fatalf("grown stream sent %d chunks, want 2", len(chunks))
	}
	for _, c := range chunks {
		if c.BlockId[0] != 1 {
			t.Errorf("grown stream resent cell %v", c.BlockId)
		}
	}
}

func TestRingCellsAreCoarseAndUpgradedLater(t *testing.T) {
	maps := &fakeMaps{key: testKey, demM: 100}
	f := New(Config{
		Anchor:     &modelv1.Gps{Lat: testAnchor.Lat, Lon: testAnchor.Lon, Alt: testAnchor.Alt},
		HTTPClient: &http.Client{Transport: maps},
	})
	area, centre := testAreaAndCentre()
	chunks := collect(t, f.Stream(context.Background(), area, centre))
	// Area x3 = -900..900 m -> 36 cells, 4 of them the area.
	if len(chunks) != 36 {
		t.Fatalf("%d chunks, want 36", len(chunks))
	}
	found := false
	for i, c := range chunks {
		tile := decodeTile(t, c)
		if i < 4 {
			if tile.Ring || c.Level != modelv1.Level_LEVEL_REFINED {
				t.Fatalf("area cell %d is ring/preview", i)
			}
			continue
		}
		if !tile.Ring || c.Level != modelv1.Level_LEVEL_PREVIEW || tile.Cols != 11 {
			t.Fatalf("ring cell %v: ring %v level %v cols %d", c.BlockId, tile.Ring, c.Level, tile.Cols)
		}
		if c.BlockId[0] == 1 && c.BlockId[1] == 0 {
			found = true
		}
	}
	if !found {
		t.Fatal("ring cell 1,0 missing")
	}
	// Grow the area over ring cell (1,0): it comes back at full detail.
	ne := enuToGPS(enu{E: 700, N: 300}, testAnchor)
	grown := area
	grown.MaxLon = ne.Lon
	upgraded := false
	for _, c := range collect(t, f.Stream(context.Background(), grown, centre)) {
		if c.BlockId[0] == 1 && c.BlockId[1] == 0 {
			upgraded = c.Level == modelv1.Level_LEVEL_REFINED && !decodeTile(t, c).Ring
		}
	}
	if !upgraded {
		t.Fatal("ring cell 1,0 was not resent at full detail")
	}
}

func TestStreamStopsOnCancel(t *testing.T) {
	maps := &fakeMaps{key: testKey, demM: 100}
	f := newTestFeed(t, "", maps, "")
	area, centre := testAreaAndCentre()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	collect(t, f.Stream(ctx, area, centre))
	if st := f.Status(); st.Cells != 0 {
		t.Fatalf("cancelled stream marked %d cells as sent", st.Cells)
	}
}

func TestAreaUnion(t *testing.T) {
	a := Area{MinLat: 1, MinLon: 2, MaxLat: 3, MaxLon: 4}
	b := Area{MinLat: 0, MinLon: 3, MaxLat: 2, MaxLon: 5}
	if got := a.Union(b); got != (Area{MinLat: 0, MinLon: 2, MaxLat: 3, MaxLon: 5}) {
		t.Fatalf("union %+v", got)
	}
	if a.Union(Area{}) != a || (Area{}).Union(a) != a {
		t.Fatal("an empty area changed the union")
	}
}

func TestFetchErrorsNeverCarryTheKey(t *testing.T) {
	failing := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return nil, errors.New("connection refused")
	})
	fe := newFetcher(&http.Client{Transport: failing}, "", "")
	_, err := fe.loadSatTile(context.Background(), testKey, tileKey{17, 1, 2})
	if err == nil || strings.Contains(err.Error(), testKey) {
		t.Fatalf("error %v", err)
	}
	_, err = fe.fetchBuildings(context.Background(), testKey, "", latLonBox{36, 127, 36.001, 127.001})
	if err == nil || strings.Contains(err.Error(), testKey) {
		t.Fatalf("error %v", err)
	}
	direct := redact(&url.Error{Op: "Get", URL: "https://api.vworld.kr/req/wfs?key=" + testKey, Err: errors.New("boom")})
	if strings.Contains(direct.Error(), testKey) {
		t.Fatalf("redact kept the URL: %v", direct)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func fileExists(t *testing.T, root, sub string) bool {
	t.Helper()
	found := false
	filepath.WalkDir(filepath.Join(root, sub), func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			found = true
		}
		return nil
	})
	return found
}

func absDiff(a, b uint8) int {
	if a > b {
		return int(a - b)
	}
	return int(b - a)
}

// signedArea is the shoelace area of a closed e,n outline; positive means
// counter-clockwise.
func signedArea(outline []float64) float64 {
	s := 0.0
	for i := 0; i+3 < len(outline); i += 2 {
		s += outline[i]*outline[i+3] - outline[i+2]*outline[i+1]
	}
	return s / 2
}
