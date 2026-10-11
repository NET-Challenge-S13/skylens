// Command skylens-core is the control tower server (COMPONENTS.md §3.4): it
// takes drones and screens over WebSocket, runs the mission state machine,
// pulls live frames from the local MediaMTX and pushes them to the model
// server, keeps every chunk the model streams back (in memory), adds terrain
// and buildings, and serves the control tower web build.
//
//	GET /health   state of everything (never the VWorld key)
//	ws  /uplink   drones or the demo: drone_hello, telemetry; core sends assign_route, manual_control
//	ws  /viewer   screens: core pushes; screens send assign_route, manual_control
//	/             control tower web build (SKYLENS_CORE_WEB_DIR)
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/config"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/core"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/modelclient"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/ws"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/terrain"
)

func main() {
	cwd, _ := os.Getwd()
	cfg, err := config.Load(os.Getenv, cwd)
	if err != nil {
		log.Fatalf("[core] config: %v", err)
	}

	conn, err := modelclient.Dial(cfg.ModelGRPC)
	if err != nil {
		log.Fatalf("[core] model server %s: %v", cfg.ModelGRPC, err)
	}
	defer conn.Close()

	if cfg.VWorldKey != "" {
		log.Printf("[core] VWorld key loaded from %s (satellite and buildings on)", cfg.VWorldFrom)
	} else {
		log.Printf("[core] no VWorld key: terrain runs without satellite imagery and buildings")
	}
	httpClient := &http.Client{Timeout: 30 * time.Second}

	c := core.New(core.Options{
		Config: cfg,
		Model:  modelclient.NewGRPC(conn),
		NewTerrain: func(anchor *modelv1.Gps) core.TerrainStreamer {
			return terrain.New(terrain.Config{
				VWorldKey:  cfg.VWorldKey,
				Anchor:     anchor,
				CacheDir:   cfg.TerrainCache,
				HTTPClient: httpClient,
			})
		},
		Frames: core.FrameFactory(cfg),
	})

	mux := http.NewServeMux()
	mux.Handle("GET /health", c.HealthHandler())
	mux.Handle("/uplink", ws.Route(c.Uplinks, c.UplinkHandler(), ws.Options{Name: "uplink"}))
	mux.Handle("/viewer", ws.Route(c.Viewers, c.ViewerHandler(), ws.Options{Name: "viewer"}))
	mux.Handle("/", webHandler(resolveWebDir(cfg.WebDir, cwd)))

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	done := make(chan struct{})
	go func() {
		c.Run(ctx)
		close(done)
	}()

	go func() {
		<-ctx.Done()
		log.Printf("[core] shutting down (in-memory state is discarded)")
		sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(sctx)
	}()

	log.Printf("[core] listening on %s (uplink /uplink, viewer /viewer, health /health)", cfg.Addr)
	log.Printf("[core] model server %s, frames %s from %s", cfg.ModelGRPC, cfg.FrameSource, cfg.FrameURL)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Printf("[core] http: %v", err)
		stop()
	}
	<-done
}

// resolveWebDir accepts an absolute path, a path relative to the working
// directory, or one relative to the repository root (so `go run .` inside
// src/skylens_core finds src/skylens_client/dist).
func resolveWebDir(dir, cwd string) string {
	if filepath.IsAbs(dir) {
		return dir
	}
	if st, err := os.Stat(filepath.Join(cwd, dir)); err == nil && st.IsDir() {
		return filepath.Join(cwd, dir)
	}
	if root, ok := config.FindRepoRoot(cwd); ok {
		return filepath.Join(root, dir)
	}
	return filepath.Join(cwd, dir)
}

// webHandler serves the control tower build. Until it is built, / says so
// instead of a bare 404.
func webHandler(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if st, err := os.Stat(dir); err != nil || !st.IsDir() {
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("control tower web build not found at " + dir + "\n"))
			return
		}
		files.ServeHTTP(w, r)
	})
}
