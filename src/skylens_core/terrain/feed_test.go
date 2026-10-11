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

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
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
	return `{"type":"FeatureCollection","features":[{"type":"Feature","id":"lt_c_bldginfo.42",` +
		`"geometry":{"type":"Polygon","coordinates":[[` +
		c(e0, n0) + "," + c(e1, n0) + "," + c(e1, n1) + "," + c(e0, n1) + "," + c(e0, n0) +
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

func TestStreamWithoutKeyIsTerrainOnly(t *testing.T) {
	maps := &fakeMaps{key: testKey, demM: 100}
	cache := t.TempDir()
	f := newTestFeed(t, "", maps, cache)
	area, centre := testAreaAndCentre()
	chunks := collect(t, f.Stream(context.Background(), area, centre, StyleBlack))

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
		pts := decodeTestPoints(t, c.Payload)
		if len(pts) != 200*200 {
			t.Fatalf("chunk %d has %d points, want 40000 (2 m grid)", i, len(pts))
		}
		x0, y0 := float32(want[i][0])*CellSizeM, float32(want[i][1])*CellSizeM
		for _, p := range pts {
			if p.Pos[0] < x0 || p.Pos[0] > x0+CellSizeM || p.Pos[1] < y0 || p.Pos[1] > y0+CellSizeM {
				t.Fatalf("chunk %d point %v outside its cell", i, p.Pos)
			}
			// Flat DEM at 100 m, anchor at 30 m: Up is 70 m.
			if math.Abs(float64(p.Pos[2])-70) > 0.01 {
				t.Fatalf("chunk %d point up %g, want 70", i, p.Pos[2])
			}
		}
	}
	if n := maps.count("api.vworld.kr"); n != 0 {
		t.Errorf("%d VWorld requests without a key", n)
	}
	if !fileExists(t, cache, "terrarium") {
		t.Error("terrain tiles were not cached")
	}

	// A second feed on the same cache does not download again.
	before := maps.count("s3.amazonaws.com")
	collect(t, newTestFeed(t, "", maps, cache).Stream(context.Background(), area, centre, StyleBlack))
	if after := maps.count("s3.amazonaws.com"); after != before {
		t.Errorf("cached run made %d new DEM requests", after-before)
	}
}

func TestStreamWithKeyAddsBuildingsAndSatelliteColour(t *testing.T) {
	maps := &fakeMaps{
		key:      testKey,
		demM:     100,
		sat:      color.RGBA{R: 200, G: 120, B: 40, A: 255},
		building: buildingAt(testAnchor, 100, 100, 120, 110, 15),
	}
	cache := t.TempDir()
	f := newTestFeed(t, testKey, maps, cache)
	area, centre := testAreaAndCentre()
	chunks := collect(t, f.Stream(context.Background(), area, centre, StyleAerial))

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
	minU, maxU := math.Inf(1), math.Inf(-1)
	for _, p := range decodeTestPoints(t, b.Payload) {
		minU, maxU = math.Min(minU, float64(p.Pos[2])), math.Max(maxU, float64(p.Pos[2]))
		if p.Pos[0] < 99.9 || p.Pos[0] > 120.1 || p.Pos[1] < 99.9 || p.Pos[1] > 110.1 {
			t.Fatalf("building point %v outside the footprint", p.Pos)
		}
	}
	if math.Abs(minU-70) > 0.01 || math.Abs(maxU-85) > 0.01 {
		t.Errorf("building spans %g..%g m up, want 70..85", minU, maxU)
	}
	// Aerial terrain carries the photograph's colour (JPEG, so roughly).
	p := decodeTestPoints(t, terrain[0].Payload)[0]
	if absDiff(p.Color[0], 200) > 10 || absDiff(p.Color[1], 120) > 10 || absDiff(p.Color[2], 40) > 10 {
		t.Errorf("terrain colour %v, want about (200,120,40)", p.Color)
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
	chunks := collect(t, f.Stream(context.Background(), area, centre, StyleBlack))
	if len(chunks) != 4 {
		t.Fatalf("%d chunks, want 4 terrain chunks", len(chunks))
	}
	for _, c := range chunks {
		if c.Kind != modelv1.ChunkKind_CHUNK_KIND_TERRAIN {
			t.Errorf("kind %v, want terrain", c.Kind)
		}
	}
	// Error documents are not cached.
	if fileExists(t, cache, "vworld-wfs") || fileExists(t, cache, "vworld-satellite") {
		t.Error("a VWorld error document was cached")
	}
}

func TestStreamStopsOnCancel(t *testing.T) {
	maps := &fakeMaps{key: testKey, demM: 100}
	f := newTestFeed(t, "", maps, "")
	area, centre := testAreaAndCentre()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	collect(t, f.Stream(ctx, area, centre, StyleBlack))
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
