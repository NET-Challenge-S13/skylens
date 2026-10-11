// Command skylens-core is the control tower server: it stores every segment,
// schedules jobs on the model server, and serves the control tower web UI.
package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	addr := os.Getenv("SKYLENS_CORE_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"ok":true}`))
	})

	log.Printf("skylens-core listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}
