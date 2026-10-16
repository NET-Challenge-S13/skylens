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

func TestOutlineIsClosedAndCounterClockwise(t *testing.T) {
	ccw := [][2]float64{{0, 0}, {10, 0}, {10, 4}, {4, 4}, {4, 10}, {0, 10}}
	got := outline(ccw)
	want := []float64{0, 0, 10, 0, 10, 4, 4, 4, 4, 10, 0, 10, 0, 0}
	if len(got) != len(want) {
		t.Fatalf("outline %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("outline %v, want %v", got, want)
		}
	}
	// A clockwise ring is reversed.
	cw := [][2]float64{{0, 0}, {0, 10}, {10, 10}, {10, 0}}
	got = outline(cw)
	want = []float64{10, 0, 10, 10, 0, 10, 0, 0, 10, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("clockwise outline %v, want %v", got, want)
		}
	}
	if signedArea(got) <= 0 {
		t.Fatalf("outline %v is not counter-clockwise", got)
	}
}
