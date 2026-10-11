// Package config turns environment variables (and the VWorld key file) into
// the core's settings. Nothing else in the core reads the environment.
package config

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
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

	// Terrain.
	Terrain       bool         // SKYLENS_CORE_TERRAIN (default on)
	TerrainStyle  string       // SKYLENS_CORE_TERRAIN_STYLE: black | aerial | points
	TerrainMargin float64      // SKYLENS_CORE_TERRAIN_MARGIN_M
	TerrainCache  string       // SKYLENS_CORE_TERRAIN_CACHE
	Anchor        *modelv1.Gps // SKYLENS_CORE_ANCHOR "lat,lon[,alt]"; nil = first route waypoint
	VWorldKey     string       // from .env.vworld or VWORLD_KEY; never sent to clients
	VWorldFrom    string       // where the key came from, for the log (never the key)
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
		WebDir:    str("SKYLENS_CORE_WEB_DIR", "src/skylens_client/dist"),
		ModelGRPC: str("SKYLENS_MODEL_GRPC", "127.0.0.1:8101"),

		FrameSource:   strings.ToLower(str("SKYLENS_CORE_FRAME_SOURCE", "ffmpeg")),
		FFmpegBin:     str("SKYLENS_CORE_FFMPEG", "ffmpeg"),
		FrameURL:      str("SKYLENS_CORE_FRAME_URL", "srt://127.0.0.1:10890?streamid=read:{path}"),
		StreamPath:    str("SKYLENS_CORE_STREAM_PATH", "drone{id}"),
		FrameFPS:      num("SKYLENS_CORE_FRAME_FPS", 2),
		FrameQuality:  int(num("SKYLENS_CORE_FRAME_QUALITY", 5)),
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
		TerrainStyle:  strings.ToLower(str("SKYLENS_CORE_TERRAIN_STYLE", "black")),
		TerrainMargin: num("SKYLENS_CORE_TERRAIN_MARGIN_M", 500),
		TerrainCache:  str("SKYLENS_CORE_TERRAIN_CACHE", defaultCache()),
	}

	switch c.FrameSource {
	case "ffmpeg", "off":
	default:
		errs = append(errs, fmt.Errorf("SKYLENS_CORE_FRAME_SOURCE=%q: want ffmpeg or off", c.FrameSource))
	}
	switch c.TerrainStyle {
	case "black", "aerial", "points":
	default:
		errs = append(errs, fmt.Errorf("SKYLENS_CORE_TERRAIN_STYLE=%q: want black, aerial or points", c.TerrainStyle))
	}
	if c.FrameFPS <= 0 {
		errs = append(errs, fmt.Errorf("SKYLENS_CORE_FRAME_FPS must be positive"))
	}
	if v := strings.TrimSpace(getenv("SKYLENS_CORE_ANCHOR")); v != "" {
		a, err := ParseGps(v)
		if err != nil {
			errs = append(errs, fmt.Errorf("SKYLENS_CORE_ANCHOR: %w", err))
		}
		c.Anchor = a
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
