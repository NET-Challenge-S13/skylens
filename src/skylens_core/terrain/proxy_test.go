package terrain

import (
	"image/color"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseWMTSPath(t *testing.T) {
	k, ok := parseWMTSPath("17/50000/111000.jpeg")
	if !ok || k != (tileKey{z: 17, x: 111000, y: 50000}) {
		t.Fatalf("parsed %v %v", k, ok)
	}
	for _, bad := range []string{"5/1/1.jpeg", "20/1/1.jpeg", "7/1/1.png", "7/1.jpeg", "7/a/1.jpeg", "7/1/128.jpeg", "7/-1/1.jpeg"} {
		if _, ok := parseWMTSPath(bad); ok {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestProxyServesSatelliteTilesWithTheKeyAddedServerSide(t *testing.T) {
	maps := &fakeMaps{key: testKey, sat: color.RGBA{R: 10, G: 20, B: 30, A: 255}}
	cache := t.TempDir()
	f := New(Config{VWorldKey: testKey, CacheDir: cache, HTTPClient: &http.Client{Transport: maps}})
	h := f.VWorldHandler()

	get := func(path string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
		return rec
	}
	rec := get("/vworld/wmts/17/50000/111000.jpeg")
	if rec.Code != http.StatusOK || rec.Header().Get("Content-Type") != "image/jpeg" || rec.Body.Len() == 0 {
		t.Fatalf("tile: %d %q %d bytes", rec.Code, rec.Header().Get("Content-Type"), rec.Body.Len())
	}
	if strings.Contains(rec.Body.String(), testKey) || strings.Contains(rec.Header().Get("Location"), testKey) {
		t.Fatal("the key reached the client")
	}
	if !fileExists(t, cache, "vworld-satellite") {
		t.Fatal("tile not cached")
	}
	// The second request is served from the cache.
	before := maps.count("api.vworld.kr")
	if rec := get("/vworld/wmts/17/50000/111000.jpeg"); rec.Code != http.StatusOK {
		t.Fatalf("cached tile %d", rec.Code)
	}
	if after := maps.count("api.vworld.kr"); after != before {
		t.Fatalf("cached tile fetched again (%d requests)", after-before)
	}

	if rec := get("/vworld/wmts/3/1/1.jpeg"); rec.Code != http.StatusNotFound {
		t.Errorf("zoom 3 answered %d, want 404", rec.Code)
	}
	if rec := get("/vworld/other"); rec.Code != http.StatusNotFound {
		t.Errorf("unknown path answered %d, want 404", rec.Code)
	}
}

func TestProxyWFSAddsTheKeyAndDropsTheClients(t *testing.T) {
	maps := &fakeMaps{key: testKey, building: buildingAt(testAnchor, 0, 0, 10, 10, 9)}
	f := New(Config{VWorldKey: testKey, CacheDir: t.TempDir(), HTTPClient: &http.Client{Transport: maps}})
	rec := httptest.NewRecorder()
	f.VWorldHandler().ServeHTTP(rec, httptest.NewRequest("GET", "/vworld/wfs?SERVICE=WFS&REQUEST=GetFeature&key=evil", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "lt_c_bldginfo.42") {
		t.Fatalf("wfs: %d %s", rec.Code, rec.Body.String())
	}
	for _, u := range maps.requests {
		if strings.Contains(u, "evil") {
			t.Fatalf("client key forwarded: %s", u)
		}
	}
}

func TestProxyWithoutKeyOrWithARejectedKey(t *testing.T) {
	maps := &fakeMaps{key: testKey}
	none := New(Config{HTTPClient: &http.Client{Transport: maps}})
	rec := httptest.NewRecorder()
	none.VWorldHandler().ServeHTTP(rec, httptest.NewRequest("GET", "/vworld/wmts/17/1/1.jpeg", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("no key: %d, want 503", rec.Code)
	}
	if maps.count("api.vworld.kr") != 0 {
		t.Fatal("requested VWorld without a key")
	}

	wrong := New(Config{VWorldKey: "wrong-key", CacheDir: t.TempDir(), HTTPClient: &http.Client{Transport: maps}})
	rec = httptest.NewRecorder()
	wrong.VWorldHandler().ServeHTTP(rec, httptest.NewRequest("GET", "/vworld/wmts/17/1/1.jpeg", nil))
	if rec.Code != http.StatusBadGateway || strings.Contains(rec.Body.String(), "wrong-key") {
		t.Fatalf("rejected key: %d %q", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	wrong.VWorldHandler().ServeHTTP(rec, httptest.NewRequest("GET", "/vworld/wfs?SERVICE=WFS", nil))
	if rec.Code != http.StatusBadGateway || strings.Contains(rec.Body.String(), "wrong-key") {
		t.Fatalf("rejected key on wfs: %d %q", rec.Code, rec.Body.String())
	}
}
