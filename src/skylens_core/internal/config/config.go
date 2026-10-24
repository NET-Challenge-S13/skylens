// Package config turns environment variables (and the VWorld key file) into
// the core's settings. Nothing else in the core reads the environment.
package config

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
)

// Config is every tunable of the core.
type Config struct {
	Addr      string // SKYLENS_CORE_ADDR
	WebDir    string // SKYLENS_CORE_WEB_DIR
	ModelGRPC string // SKYLENS_MODEL_GRPC

	// Frames from the local MediaMTX.
	FrameSource   string        // SKYLENS_CORE_FRAME_SOURCE: ffmpeg | off
	FFmpegBin     string        // SKYLENS_CORE_FFMPEG
	FrameURL      string        // SKYLENS_CORE_FRAME_URL, {path} {id} {station}
	StreamPath    string        // SKYLENS_CORE_STREAM_PATH, {id} {station}
	FrameFPS      float64       // SKYLENS_CORE_FRAME_FPS
	FrameQuality  int           // SKYLENS_CORE_FRAME_QUALITY
	FrameWidth    int           // SKYLENS_CORE_FRAME_WIDTH, 0 keeps the source size
	FrameMaxWait  time.Duration // SKYLENS_CORE_FRAME_MAX_WAIT_MS
	FrameSourceID string        // SKYLENS_CORE_FRAME_SOURCE_ID
	PushQueue     int           // SKYLENS_CORE_PUSH_QUEUE
	PushRedeclare int           // SKYLENS_CORE_PUSH_REDECLARE

	// Live video for screens.
	WHEPBase string // SKYLENS_CORE_WHEP_BASE, {path} {id} {station}

	// Mission.
	AssignedHold time.Duration // SKYLENS_CORE_ASSIGNED_HOLD_MS
	DroneETA     time.Duration // SKYLENS_CORE_DRONE_ETA_SEC
	StatusEvery  time.Duration // SKYLENS_CORE_STATUS_MS

	// Store.
	DetectionsPerSegment int // SKYLENS_CORE_DETECTIONS_PER_SEGMENT

	// Operating area. Site.Anchor is the session's fixed ENU origin
	// (SKYLENS_CORE_SITE, SKYLENS_CORE_ANCHOR).
	Site Site

	// Base map (terrain and buildings, control tower only).
	Terrain       bool    // SKYLENS_CORE_TERRAIN (default on)
	TerrainMargin float64 // SKYLENS_CORE_TERRAIN_MARGIN_M, around a route
	TerrainCache  string  // SKYLENS_CORE_TERRAIN_CACHE
	VWorldKey     string  // from .env.vworld or VWORLD_KEY; never sent to clients
	VWorldFrom    string  // where the key came from, for the log (never the key)

	// PlannedRoute is a route prepared before the session (the demo's
	// recorded flight line), read from the JSON file SKYLENS_CORE_PLANNED_ROUTE
	// names. Nil when unset. The control tower gets it until a route is
	// assigned.
	PlannedRoute     *viewerv1.PlannedRoute
	PlannedRouteFrom string // the file it came from, for the log
}

// Site is the operating area: a lat/lon box and the anchor every ENU
// coordinate of the session is measured from.
type Site struct {
	Name                           string
	MinLat, MinLon, MaxLat, MaxLon float64
	Anchor                         *modelv1.Gps
	// Source says where the area came from: "default" (no SKYLENS_CORE_SITE)
	// or "config".
	Source string
}

// Preset is a named operating area. Bbox is [west, south, east, north] in
// degrees, as the legacy terrainSource.ts PRESETS.
type Preset struct {
	Bbox [4]float64
	// Anchor is the preset's ENU origin; nil means the box centre at
	// sea level.
	Anchor *modelv1.Gps
}

// DefaultSite is the preset used when SKYLENS_CORE_SITE is unset.
const DefaultSite = "daejeon"

// Presets are the legacy control tower's map presets.
var Presets = map[string]Preset{
	// Reference scene and demo flight area: Yuseong, Daejeon, around
	// Chungnam National University with KAIST to the east (~3 km). The anchor
	// is the legacy CONFIG.geo.anchor (its centre, 30 m).
	"daejeon": {Bbox: [4]float64{127.33, 36.355, 127.365, 36.382}, Anchor: &modelv1.Gps{Lat: 36.3685, Lon: 127.3475, Alt: 30}},
	// Gyeongbokgung, Seoul (~2.5 km), dense city blocks.
	"seoul": {Bbox: [4]float64{126.964, 37.5555, 126.992, 37.5775}},
	// 2022 Uljin-Samcheok wildfire area.
	"uljin": {Bbox: [4]float64{129.22, 37.0, 129.36, 37.12}},
	// Uljin town (~1.8 km).
	"uljinup": {Bbox: [4]float64{129.39, 36.985, 129.41, 37.0}},
	// 2023 Gangneung wildfire area.
	"gangneung": {Bbox: [4]float64{128.83, 37.74, 128.93, 37.82}},
}

// ParseSite reads SKYLENS_CORE_SITE:
//
//	""                         the default preset (daejeon)
//	"daejeon", "seoul", ...    a preset by name
//	"west,south,east,north"    a custom box (anchor: its centre at sea level)
//	"lat,lon[,alt]"            the legacy point form: a box the size of the
//	                           default preset centred on the point, which is
//	                           also the anchor
func ParseSite(v string) (Site, error) {
	v = strings.TrimSpace(v)
	if v == "" {
		s := presetSite(DefaultSite)
		s.Source = "default"
		return s, nil
	}
	if _, ok := Presets[strings.ToLower(v)]; ok {
		s := presetSite(strings.ToLower(v))
		s.Source = "config"
		return s, nil
	}
	parts := strings.Split(v, ",")
	nums := make([]float64, len(parts))
	for i, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return Site{}, fmt.Errorf("%q: want a preset name, west,south,east,north or lat,lon[,alt]", v)
		}
		nums[i] = f
	}
	switch len(nums) {
	case 4:
		w, s, e, n := nums[0], nums[1], nums[2], nums[3]
		if w < -180 || e > 180 || s < -90 || n > 90 || w >= e || s >= n {
			return Site{}, fmt.Errorf("%q: not a west,south,east,north box", v)
		}
		return Site{
			Name: "custom", MinLat: s, MinLon: w, MaxLat: n, MaxLon: e,
			Anchor: &modelv1.Gps{Lat: (s + n) / 2, Lon: (w + e) / 2},
			Source: "config",
		}, nil
	case 2, 3:
		g, err := ParseGps(v)
		if err != nil {
			return Site{}, err
		}
		d := Presets[DefaultSite].Bbox
		halfLon, halfLat := (d[2]-d[0])/2, (d[3]-d[1])/2
		return Site{
			Name:   "custom",
			MinLat: math.Max(-90, g.Lat-halfLat), MinLon: math.Max(-180, g.Lon-halfLon),
			MaxLat: math.Min(90, g.Lat+halfLat), MaxLon: math.Min(180, g.Lon+halfLon),
			Anchor: g,
			Source: "config",
		}, nil
	}
	return Site{}, fmt.Errorf("%q: want a preset name, west,south,east,north or lat,lon[,alt]", v)
}

func presetSite(name string) Site {
	p := Presets[name]
	s := Site{Name: name, MinLon: p.Bbox[0], MinLat: p.Bbox[1], MaxLon: p.Bbox[2], MaxLat: p.Bbox[3]}
	if p.Anchor != nil {
		s.Anchor = &modelv1.Gps{Lat: p.Anchor.GetLat(), Lon: p.Anchor.GetLon(), Alt: p.Anchor.GetAlt()}
	} else {
		s.Anchor = &modelv1.Gps{Lat: (s.MinLat + s.MaxLat) / 2, Lon: (s.MinLon + s.MaxLon) / 2}
	}
	return s
}

// Load reads the configuration. getenv is os.Getenv in production. cwd is
// where the search for the repository root starts.
func Load(getenv func(string) string, cwd string) (Config, error) {
	var errs []error
	str := func(k, def string) string {
		if v := strings.TrimSpace(getenv(k)); v != "" {
			return v
		}
		return def
	}
	num := func(k string, def float64) float64 {
		v := strings.TrimSpace(getenv(k))
		if v == "" {
			return def
		}
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s=%q: %w", k, v, err))
			return def
		}
		return f
	}
	ms := func(k string, def float64) time.Duration {
		return time.Duration(num(k, def) * float64(time.Millisecond))
	}

	c := Config{
		Addr:      str("SKYLENS_CORE_ADDR", ":8080"),
		WebDir:    str("SKYLENS_CORE_WEB_DIR", "src/skylens_core/control/dist"),
		ModelGRPC: str("SKYLENS_MODEL_GRPC", "127.0.0.1:8101"),

		FrameSource:   strings.ToLower(str("SKYLENS_CORE_FRAME_SOURCE", "ffmpeg")),
		FFmpegBin:     str("SKYLENS_CORE_FFMPEG", "ffmpeg"),
		FrameURL:      str("SKYLENS_CORE_FRAME_URL", "srt://127.0.0.1:10890?streamid=read:{path}"),
		StreamPath:    str("SKYLENS_CORE_STREAM_PATH", "drone{id}"),
		FrameFPS:      num("SKYLENS_CORE_FRAME_FPS", 2),
		FrameQuality:  int(num("SKYLENS_CORE_FRAME_QUALITY", 5)),
		FrameWidth:    int(num("SKYLENS_CORE_FRAME_WIDTH", 1920)),
		FrameMaxWait:  ms("SKYLENS_CORE_FRAME_MAX_WAIT_MS", 1500),
		FrameSourceID: str("SKYLENS_CORE_FRAME_SOURCE_ID", "formation"),
		PushQueue:     int(num("SKYLENS_CORE_PUSH_QUEUE", 256)),
		PushRedeclare: int(num("SKYLENS_CORE_PUSH_REDECLARE", 32)),

		WHEPBase: str("SKYLENS_CORE_WHEP_BASE", "/live/{path}/whep"),

		AssignedHold: ms("SKYLENS_CORE_ASSIGNED_HOLD_MS", 2000),
		DroneETA:     time.Duration(num("SKYLENS_CORE_DRONE_ETA_SEC", 10) * float64(time.Second)),
		StatusEvery:  ms("SKYLENS_CORE_STATUS_MS", 1000),

		DetectionsPerSegment: int(num("SKYLENS_CORE_DETECTIONS_PER_SEGMENT", 256)),

		Terrain:       !isOff(getenv("SKYLENS_CORE_TERRAIN")),
		TerrainMargin: num("SKYLENS_CORE_TERRAIN_MARGIN_M", 500),
		TerrainCache:  str("SKYLENS_CORE_TERRAIN_CACHE", defaultCache()),
	}

	switch c.FrameSource {
	case "ffmpeg", "off":
	default:
		errs = append(errs, fmt.Errorf("SKYLENS_CORE_FRAME_SOURCE=%q: want ffmpeg or off", c.FrameSource))
	}
	if c.FrameFPS <= 0 {
		errs = append(errs, fmt.Errorf("SKYLENS_CORE_FRAME_FPS must be positive"))
	}
	site, err := ParseSite(getenv("SKYLENS_CORE_SITE"))
	if err != nil {
		errs = append(errs, fmt.Errorf("SKYLENS_CORE_SITE: %w", err))
		site, _ = ParseSite("")
	}
	c.Site = site
	if v := strings.TrimSpace(getenv("SKYLENS_CORE_ANCHOR")); v != "" {
		a, err := ParseGps(v)
		if err != nil {
			errs = append(errs, fmt.Errorf("SKYLENS_CORE_ANCHOR: %w", err))
		} else {
			c.Site.Anchor = a
		}
	}

	if path := strings.TrimSpace(getenv("SKYLENS_CORE_PLANNED_ROUTE")); path != "" {
		r, err := LoadPlannedRoute(path)
		if err != nil {
			errs = append(errs, fmt.Errorf("SKYLENS_CORE_PLANNED_ROUTE: %w", err))
		} else {
			c.PlannedRoute, c.PlannedRouteFrom = r, path
		}
	}

	// VWorld key: an explicit variable wins, then the key file in the
	// repository's parent directory (SPEC §6).
	if k := strings.TrimSpace(getenv("VWORLD_KEY")); k != "" {
		c.VWorldKey, c.VWorldFrom = k, "VWORLD_KEY"
	} else {
		path := strings.TrimSpace(getenv("SKYLENS_VWORLD_ENV"))
		if path == "" {
			if root, ok := FindRepoRoot(cwd); ok {
				path = filepath.Join(filepath.Dir(root), ".env.vworld")
			}
		}
		if path != "" {
			k, err := ReadVWorldKey(path)
			switch {
			case err == nil && k != "":
				c.VWorldKey, c.VWorldFrom = k, path
			case err != nil && !errors.Is(err, os.ErrNotExist):
				errs = append(errs, fmt.Errorf("%s: %w", path, err))
			}
		}
	}
	return c, errors.Join(errs...)
}

func isOff(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "0", "off", "false", "no":
		return true
	}
	return false
}

func defaultCache() string {
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "skylens", "terrain")
	}
	return filepath.Join(os.TempDir(), "skylens-terrain")
}

// ParseGps parses "lat,lon" or "lat,lon,alt".
func ParseGps(s string) (*modelv1.Gps, error) {
	parts := strings.Split(s, ",")
	if len(parts) < 2 || len(parts) > 3 {
		return nil, fmt.Errorf("%q: want lat,lon[,alt]", s)
	}
	var v [3]float64
	for i, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("%q: %w", s, err)
		}
		v[i] = f
	}
	if v[0] < -90 || v[0] > 90 || v[1] < -180 || v[1] > 180 {
		return nil, fmt.Errorf("%q: out of range", s)
	}
	return &modelv1.Gps{Lat: v[0], Lon: v[1], Alt: v[2]}, nil
}

// plannedRouteFile is the JSON shape of SKYLENS_CORE_PLANNED_ROUTE:
//
//	{"drone_id": 2, "loop": true,
//	 "waypoints": [{"lat": 36.36, "lon": 127.34, "alt": 95.8}, ...]}
//
// alt is absolute (metres, the same as Gps.alt everywhere else).
type plannedRouteFile struct {
	DroneID   uint32 `json:"drone_id"`
	Loop      bool   `json:"loop"`
	Waypoints []struct {
		Lat *float64 `json:"lat"`
		Lon *float64 `json:"lon"`
		Alt float64  `json:"alt"`
	} `json:"waypoints"`
}

// ParsePlannedRoute reads a planned route. It needs at least two waypoints,
// each with a lat and lon in range.
func ParsePlannedRoute(r io.Reader) (*viewerv1.PlannedRoute, error) {
	var f plannedRouteFile
	dec := json.NewDecoder(r)
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("not a planned route: %w", err)
	}
	if len(f.Waypoints) < 2 {
		return nil, fmt.Errorf("%d waypoint(s), want at least 2", len(f.Waypoints))
	}
	out := &viewerv1.PlannedRoute{DroneId: f.DroneID, Loop: f.Loop}
	for i, w := range f.Waypoints {
		if w.Lat == nil || w.Lon == nil {
			return nil, fmt.Errorf("waypoint %d: lat and lon are required", i+1)
		}
		lat, lon := *w.Lat, *w.Lon
		if math.IsNaN(lat) || math.IsNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180 {
			return nil, fmt.Errorf("waypoint %d: %v,%v out of range", i+1, lat, lon)
		}
		out.Waypoints = append(out.Waypoints, &modelv1.Gps{Lat: lat, Lon: lon, Alt: w.Alt})
	}
	return out, nil
}

// LoadPlannedRoute reads a planned route file.
func LoadPlannedRoute(path string) (*viewerv1.PlannedRoute, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r, err := ParsePlannedRoute(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return r, nil
}

// FindRepoRoot walks up from dir to the directory holding go.work (the
// repository root). In a git worktree that is the worktree itself.
func FindRepoRoot(dir string) (string, bool) {
	if dir == "" {
		return "", false
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return "", false
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.work")); err == nil {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

// ReadVWorldKey reads VWORLD_KEY from a dotenv file.
func ReadVWorldKey(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	vars, err := ParseDotenv(f)
	if err != nil {
		return "", err
	}
	return vars["VWORLD_KEY"], nil
}

// ParseDotenv reads KEY=VALUE lines. Blank lines and # comments are skipped,
// an "export " prefix is allowed, and matching single or double quotes around
// the value are removed. An unquoted value ends at " #".
func ParseDotenv(r io.Reader) (map[string]string, error) {
	out := map[string]string{}
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(sc.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		if n := len(v); n >= 2 && (v[0] == '"' || v[0] == '\'') && v[n-1] == v[0] {
			v = v[1 : n-1]
		} else if i := strings.Index(v, " #"); i >= 0 {
			v = strings.TrimSpace(v[:i])
		}
		out[k] = v
	}
	return out, sc.Err()
}

// Expand fills {id}, {station} and {path} in a template.
func Expand(tmpl string, id uint32, station, path string) string {
	r := strings.NewReplacer(
		"{id}", strconv.FormatUint(uint64(id), 10),
		"{station}", station,
		"{path}", path,
	)
	return r.Replace(tmpl)
}
