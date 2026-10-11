package terrain

import (
	"math"
	"net/url"
	"strings"
	"testing"
)

func TestParseFeaturesHeightRulesAndDedupe(t *testing.T) {
	body := []byte(`{"type":"FeatureCollection","features":[
	 {"type":"Feature","id":"lt_c_bldginfo.1","geometry":{"type":"Polygon","coordinates":[[[127.0,36.0],[127.001,36.0],[127.001,36.001],[127.0,36.0]]]},"properties":{"height":21.5,"grnd_flr":3}},
	 {"type":"Feature","id":"lt_c_bldginfo.2","geometry":{"type":"MultiPolygon","coordinates":[[[[127.0,36.0,0],[127.001,36.0,0],[127.001,36.001,0],[127.0,36.001,0],[127.0,36.0,0]]]]},"properties":{"height":null,"grnd_flr":"4"}},
	 {"type":"Feature","id":3,"geometry":{"type":"Polygon","coordinates":[[[127.0,36.0],[127.001,36.0],[127.001,36.001]]]},"properties":{}},
	 {"type":"Feature","id":"lt_c_bldginfo.1","geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1]]]},"properties":{"height":99}},
	 {"type":"Feature","id":"lt_c_bldginfo.4","geometry":{"type":"Polygon","coordinates":[[[127.0,36.0],[127.001,36.0]]]},"properties":{}}
	]}`)
	out := make(map[string]building)
	rows, err := parseFeatures(body, out)
	if err != nil {
		t.Fatal(err)
	}
	if rows != 5 {
		t.Errorf("rows %d, want 5", rows)
	}
	if len(out) != 3 {
		t.Fatalf("%d buildings, want 3 (dup and degenerate dropped): %v", len(out), out)
	}
	cases := map[string]float64{"lt_c_bldginfo.1": 21.5, "lt_c_bldginfo.2": 4 * 3.3, "3": 6}
	for id, h := range cases {
		b, ok := out[id]
		if !ok {
			t.Errorf("missing %s", id)
			continue
		}
		if math.Abs(b.heightM-h) > 1e-9 {
			t.Errorf("%s height %g, want %g", id, b.heightM, h)
		}
	}
	// The closing vertex is dropped; the open triangle is kept as is.
	if n := len(out["lt_c_bldginfo.1"].rings[0]); n != 3 {
		t.Errorf("ring 1 has %d vertices, want 3", n)
	}
	if n := len(out["lt_c_bldginfo.2"].rings[0]); n != 4 {
		t.Errorf("ring 2 has %d vertices, want 4", n)
	}
}

func TestParseFeaturesRejectsErrorDocuments(t *testing.T) {
	for _, body := range []string{
		`<?xml version="1.0"?><ExceptionReport/>`,
		`{"response":{"status":"ERROR"}}`,
	} {
		if _, err := parseFeatures([]byte(body), map[string]building{}); err == nil {
			t.Errorf("accepted %q", body)
		}
	}
}

func TestWFSQueryIsLatLonAndCacheNameIsKeyFree(t *testing.T) {
	b := latLonBox{minLat: 36.1, minLon: 127.2, maxLat: 36.3, maxLon: 127.4}
	u, err := url.Parse(wfsURL("SECRET-KEY", "", b))
	if err != nil {
		t.Fatal(err)
	}
	q := u.Query()
	if got := q.Get("BBOX"); got != "36.1000000,127.2000000,36.3000000,127.4000000,EPSG:4326" {
		t.Errorf("BBOX %q", got)
	}
	if q.Get("TYPENAME") != "lt_c_bldginfo" || q.Get("key") != "SECRET-KEY" {
		t.Errorf("query %v", q)
	}
	if name := wfsCacheName(b); strings.Contains(name, "SECRET") {
		t.Errorf("cache name leaks the key: %s", name)
	}
}

func TestPointInPolygon(t *testing.T) {
	sq := [][2]float64{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	if !pointInPolygon(5, 5, sq) || pointInPolygon(15, 5, sq) || pointInPolygon(-1, -1, sq) {
		t.Fatal("square test failed")
	}
	// An L shape: the notch is outside.
	l := [][2]float64{{0, 0}, {10, 0}, {10, 4}, {4, 4}, {4, 10}, {0, 10}}
	if pointInPolygon(7, 7, l) || !pointInPolygon(2, 7, l) {
		t.Fatal("L test failed")
	}
}

func TestSampleBuildingSpansBaseToRoof(t *testing.T) {
	ring := [][2]float64{{100, 100}, {120, 100}, {120, 110}, {100, 110}}
	var pts []Point
	noDrape := func(e, n float64) ([3]uint8, bool) { return groundRGB, false }
	sampleBuilding(ring, 70, 85, fineProfile, StyleBlack, noDrape, func(p Point) { pts = append(pts, p) })
	if len(pts) == 0 {
		t.Fatal("no points")
	}
	minU, maxU := math.Inf(1), math.Inf(-1)
	roof, edge := 0, 0
	for _, p := range pts {
		u := float64(p.Pos[2])
		minU, maxU = math.Min(minU, u), math.Max(maxU, u)
		if p.Pos[0] < 100-1e-3 || p.Pos[0] > 120+1e-3 || p.Pos[1] < 100-1e-3 || p.Pos[1] > 110+1e-3 {
			t.Fatalf("point outside the footprint: %v", p.Pos)
		}
		switch p.Color {
		case blackRoof:
			roof++
		case blackEdge:
			edge++
		}
	}
	if minU != 70 || maxU != 85 {
		t.Errorf("u range %g..%g, want 70..85", minU, maxU)
	}
	// 20 m x 10 m at a 2 m grid: 10 x 5 roof points.
	if roof != 50 {
		t.Errorf("%d roof points, want 50", roof)
	}
	if edge == 0 {
		t.Error("no linework points")
	}
}

func TestSampleBuildingTinyFootprintStillHasARoof(t *testing.T) {
	ring := [][2]float64{{0.1, 0.1}, {0.6, 0.1}, {0.6, 0.6}}
	var roof int
	noDrape := func(e, n float64) ([3]uint8, bool) { return groundRGB, false }
	sampleBuilding(ring, 0, 6, fineProfile, StyleBlack, noDrape, func(p Point) {
		if p.Color == blackRoof {
			roof++
		}
	})
	if roof != 1 {
		t.Fatalf("%d roof points, want 1 (the centroid)", roof)
	}
}
