package config

import (
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
	if c.Addr != ":8080" || c.ModelGRPC != "127.0.0.1:8101" || c.WebDir != "src/skylens_client/dist" {
		t.Fatalf("defaults %+v", c)
	}
	if c.AssignedHold != 2*time.Second || c.DroneETA != 10*time.Second || c.StatusEvery != time.Second {
		t.Fatalf("mission defaults %+v", c)
	}
	if !c.Terrain || c.TerrainStyle != "black" || c.Anchor != nil || c.VWorldKey != "" {
		t.Fatalf("terrain defaults %+v", c)
	}
	if c.FrameSource != "ffmpeg" || c.FrameFPS != 2 {
		t.Fatalf("frame defaults %+v", c)
	}
}

func TestOverridesAndErrors(t *testing.T) {
	c, err := Load(env(map[string]string{
		"SKYLENS_CORE_ADDR":          ":9000",
		"SKYLENS_CORE_ANCHOR":        "36.37, 127.36, 50",
		"SKYLENS_CORE_TERRAIN":       "off",
		"SKYLENS_CORE_FRAME_FPS":     "4",
		"SKYLENS_CORE_STATUS_MS":     "250",
		"SKYLENS_CORE_FRAME_SOURCE":  "OFF",
		"SKYLENS_CORE_TERRAIN_STYLE": "aerial",
	}), "")
	if err != nil {
		t.Fatal(err)
	}
	if c.Addr != ":9000" || c.Terrain || c.FrameFPS != 4 || c.StatusEvery != 250*time.Millisecond ||
		c.FrameSource != "off" || c.TerrainStyle != "aerial" {
		t.Fatalf("overrides %+v", c)
	}
	if c.Anchor == nil || c.Anchor.GetLat() != 36.37 || c.Anchor.GetLon() != 127.36 || c.Anchor.GetAlt() != 50 {
		t.Fatalf("anchor %v", c.Anchor)
	}

	_, err = Load(env(map[string]string{
		"SKYLENS_CORE_FRAME_FPS":     "fast",
		"SKYLENS_CORE_ANCHOR":        "999,0",
		"SKYLENS_CORE_TERRAIN_STYLE": "neon",
	}), "")
	if err == nil {
		t.Fatal("bad values accepted")
	}
	for _, want := range []string{"SKYLENS_CORE_FRAME_FPS", "SKYLENS_CORE_ANCHOR", "SKYLENS_CORE_TERRAIN_STYLE"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q does not name %s", err, want)
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

func TestExpand(t *testing.T) {
	got := Expand("srt://127.0.0.1:10890?streamid=read:{path}", 2, "left", Expand("drone{id}-{station}", 2, "left", ""))
	if got != "srt://127.0.0.1:10890?streamid=read:drone2-left" {
		t.Fatalf("expand %q", got)
	}
}
