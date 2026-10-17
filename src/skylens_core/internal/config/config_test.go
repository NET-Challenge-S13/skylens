package config

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func env(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func TestDefaults(t *testing.T) {
	c, err := Load(env(nil), "")
	if err != nil {
		t.Fatal(err)
	}
	if c.Addr != ":8080" || c.ModelGRPC != "127.0.0.1:8101" || c.WebDir != "src/skylens_core/control/dist" {
		t.Fatalf("defaults %+v", c)
	}
	if c.AssignedHold != 2*time.Second || c.DroneETA != 10*time.Second || c.StatusEvery != time.Second {
		t.Fatalf("mission defaults %+v", c)
	}
	if !c.Terrain || c.VWorldKey != "" {
		t.Fatalf("terrain defaults %+v", c)
	}
	// The legacy daejeon preset and its anchor.
	s := c.Site
	if s.Name != "daejeon" || s.Source != "default" ||
		s.MinLon != 127.33 || s.MinLat != 36.355 || s.MaxLon != 127.365 || s.MaxLat != 36.382 {
		t.Fatalf("site defaults %+v", s)
	}
	if a := s.Anchor; a.GetLat() != 36.3685 || a.GetLon() != 127.3475 || a.GetAlt() != 30 {
		t.Fatalf("anchor default %v", a)
	}
	if c.FrameSource != "ffmpeg" || c.FrameFPS != 2 {
		t.Fatalf("frame defaults %+v", c)
	}
}

func TestOverridesAndErrors(t *testing.T) {
	c, err := Load(env(map[string]string{
		"SKYLENS_CORE_ADDR":         ":9000",
		"SKYLENS_CORE_ANCHOR":       "36.37, 127.36, 50",
		"SKYLENS_CORE_TERRAIN":      "off",
		"SKYLENS_CORE_FRAME_FPS":    "4",
		"SKYLENS_CORE_STATUS_MS":    "250",
		"SKYLENS_CORE_FRAME_SOURCE": "OFF",
		"SKYLENS_CORE_SITE":         "Seoul",
	}), "")
	if err != nil {
		t.Fatal(err)
	}
	if c.Addr != ":9000" || c.Terrain || c.FrameFPS != 4 || c.StatusEvery != 250*time.Millisecond ||
		c.FrameSource != "off" {
		t.Fatalf("overrides %+v", c)
	}
	if c.Site.Name != "seoul" || c.Site.Source != "config" || c.Site.MinLon != 126.964 {
		t.Fatalf("site %+v", c.Site)
	}
	// SKYLENS_CORE_ANCHOR wins over the preset's anchor.
	if a := c.Site.Anchor; a == nil || a.GetLat() != 36.37 || a.GetLon() != 127.36 || a.GetAlt() != 50 {
		t.Fatalf("anchor %v", a)
	}

	_, err = Load(env(map[string]string{
		"SKYLENS_CORE_FRAME_FPS": "fast",
		"SKYLENS_CORE_ANCHOR":    "999,0",
		"SKYLENS_CORE_SITE":      "atlantis",
	}), "")
	if err == nil {
		t.Fatal("bad values accepted")
	}
	for _, want := range []string{"SKYLENS_CORE_FRAME_FPS", "SKYLENS_CORE_ANCHOR", "SKYLENS_CORE_SITE"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q does not name %s", err, want)
		}
	}
}

func TestParseSite(t *testing.T) {
	box, err := ParseSite("127.0, 36.0, 127.02, 36.01")
	if err != nil {
		t.Fatal(err)
	}
	if box.Name != "custom" || box.MinLon != 127.0 || box.MinLat != 36.0 || box.MaxLon != 127.02 || box.MaxLat != 36.01 {
		t.Fatalf("box %+v", box)
	}
	if a := box.Anchor; math.Abs(a.GetLat()-36.005) > 1e-9 || math.Abs(a.GetLon()-127.01) > 1e-9 || a.GetAlt() != 0 {
		t.Fatalf("box anchor %v", a)
	}

	// The legacy point form: the point is the anchor, the box is the size of
	// the default preset around it.
	pt, err := ParseSite("36.5,127.5,40")
	if err != nil {
		t.Fatal(err)
	}
	if a := pt.Anchor; a.GetLat() != 36.5 || a.GetLon() != 127.5 || a.GetAlt() != 40 {
		t.Fatalf("point anchor %v", a)
	}
	if !(pt.MinLat < 36.5 && pt.MaxLat > 36.5 && pt.MinLon < 127.5 && pt.MaxLon > 127.5) {
		t.Fatalf("point box %+v", pt)
	}
	if w := pt.MaxLon - pt.MinLon; w < 0.0349 || w > 0.0351 {
		t.Fatalf("point box width %g, want the daejeon width 0.035", w)
	}

	// A preset without its own anchor is anchored at its centre.
	ul, err := ParseSite("uljinup")
	if err != nil {
		t.Fatal(err)
	}
	if a := ul.Anchor; math.Abs(a.GetLat()-36.9925) > 1e-9 || math.Abs(a.GetLon()-129.4) > 1e-9 {
		t.Fatalf("preset anchor %v", a)
	}

	for _, bad := range []string{"atlantis", "1,2,3,4,5", "127.02,36.0,127.0,36.01", "x,y"} {
		if _, err := ParseSite(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestParseDotenv(t *testing.T) {
	in := "# comment\n\ufeffVWORLD_KEY=abc-123\nVWORLD_DOMAIN=\"http://localhost:5173\"\nexport A='x y'\nB=v # trailing\nnoequals\n"
	m, err := ParseDotenv(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if m["VWORLD_KEY"] != "abc-123" || m["VWORLD_DOMAIN"] != "http://localhost:5173" || m["A"] != "x y" || m["B"] != "v" {
		t.Fatalf("parsed %v", m)
	}
}

func TestVWorldKeyFromParentOfRepo(t *testing.T) {
	parent := t.TempDir()
	repo := filepath.Join(parent, "skylens")
	sub := filepath.Join(repo, "src", "skylens_core")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(repo, "go.work"), []byte("go 1.25\n"), 0o644)
	os.WriteFile(filepath.Join(parent, ".env.vworld"), []byte("VWORLD_KEY=secret\n"), 0o600)

	c, err := Load(env(nil), sub)
	if err != nil {
		t.Fatal(err)
	}
	if c.VWorldKey != "secret" || !strings.HasSuffix(c.VWorldFrom, ".env.vworld") {
		t.Fatalf("key %q from %q", c.VWorldKey, c.VWorldFrom)
	}

	// The variable wins over the file.
	c, _ = Load(env(map[string]string{"VWORLD_KEY": "direct"}), sub)
	if c.VWorldKey != "direct" {
		t.Fatalf("key %q", c.VWorldKey)
	}

	// No file: no key, no error (terrain runs without buildings and imagery).
	os.Remove(filepath.Join(parent, ".env.vworld"))
	c, err = Load(env(nil), sub)
	if err != nil || c.VWorldKey != "" {
		t.Fatalf("missing file: key %q err %v", c.VWorldKey, err)
	}
}

// Keys registered to a service URL need the domain on WFS requests, or
// VWorld answers with an XML exception instead of GeoJSON.
func TestVWorldDomain(t *testing.T) {
	parent := t.TempDir()
	repo := filepath.Join(parent, "skylens")
	sub := filepath.Join(repo, "src", "skylens_core")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(repo, "go.work"), []byte("go 1.25\n"), 0o644)
	os.WriteFile(filepath.Join(parent, ".env.vworld"),
		[]byte("VWORLD_KEY=secret\nVWORLD_DOMAIN=\"http://localhost:5173\"\n"), 0o600)

	c, err := Load(env(nil), sub)
	if err != nil {
		t.Fatal(err)
	}
	if c.VWorldDomain != "http://localhost:5173" {
		t.Fatalf("domain from file %q", c.VWorldDomain)
	}

	// The variable wins over the file, with or without a key variable.
	c, _ = Load(env(map[string]string{"VWORLD_DOMAIN": "https://example.org"}), sub)
	if c.VWorldKey != "secret" || c.VWorldDomain != "https://example.org" {
		t.Fatalf("key %q domain %q", c.VWorldKey, c.VWorldDomain)
	}
	c, _ = Load(env(map[string]string{"VWORLD_KEY": "direct", "VWORLD_DOMAIN": "https://example.org"}), sub)
	if c.VWorldKey != "direct" || c.VWorldDomain != "https://example.org" {
		t.Fatalf("key %q domain %q", c.VWorldKey, c.VWorldDomain)
	}

	// A file without a domain leaves it empty.
	os.WriteFile(filepath.Join(parent, ".env.vworld"), []byte("VWORLD_KEY=secret\n"), 0o600)
	c, err = Load(env(nil), sub)
	if err != nil || c.VWorldKey != "secret" || c.VWorldDomain != "" {
		t.Fatalf("no domain line: key %q domain %q err %v", c.VWorldKey, c.VWorldDomain, err)
	}
}

func TestPlannedRoute(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "planned-route.json")
	body := `{"drone_id": 2, "loop": true, "waypoints": [
		{"lat": 36.3657, "lon": 127.3422, "alt": 95.5},
		{"lat": 36.3651, "lon": 127.3424, "alt": 95.5}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := Load(env(map[string]string{"SKYLENS_CORE_PLANNED_ROUTE": path}), "")
	if err != nil {
		t.Fatal(err)
	}
	r := c.PlannedRoute
	if r == nil || c.PlannedRouteFrom != path {
		t.Fatalf("planned route %v from %q", r, c.PlannedRouteFrom)
	}
	if r.GetDroneId() != 2 || !r.GetLoop() || len(r.GetWaypoints()) != 2 {
		t.Fatalf("planned route %v", r)
	}
	if w := r.GetWaypoints()[1]; w.GetLat() != 36.3651 || w.GetLon() != 127.3424 || w.GetAlt() != 95.5 {
		t.Fatalf("second waypoint %v", w)
	}

	// Unset: no route, no error.
	c, err = Load(env(nil), "")
	if err != nil || c.PlannedRoute != nil {
		t.Fatalf("unset: %v, %v", c.PlannedRoute, err)
	}

	// A missing file is a configuration error that names the variable.
	_, err = Load(env(map[string]string{"SKYLENS_CORE_PLANNED_ROUTE": filepath.Join(dir, "nope.json")}), "")
	if err == nil || !strings.Contains(err.Error(), "SKYLENS_CORE_PLANNED_ROUTE") {
		t.Fatalf("missing file: %v", err)
	}

	for _, bad := range []string{
		`not json`,
		`{"waypoints": [{"lat": 36.1, "lon": 127.1}]}`,
		`{"waypoints": [{"lat": 36.1, "lon": 127.1}, {"lon": 127.2}]}`,
		`{"waypoints": [{"lat": 36.1, "lon": 127.1}, {"lat": 95, "lon": 127.2}]}`,
		`{"waypoints": [{"lat": 36.1, "lon": 127.1}, {"lat": 36.2, "lon": 181}]}`,
	} {
		if _, err := ParsePlannedRoute(strings.NewReader(bad)); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}

func TestExpand(t *testing.T) {
	got := Expand("srt://127.0.0.1:10890?streamid=read:{path}", 2, "left", Expand("drone{id}-{station}", 2, "left", ""))
	if got != "srt://127.0.0.1:10890?streamid=read:drone2-left" {
		t.Fatalf("expand %q", got)
	}
}
