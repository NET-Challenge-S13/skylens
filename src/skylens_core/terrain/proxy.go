package terrain

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// VWorld proxy for the control tower's route planner. api.vworld.kr sends no
// CORS headers and the key must never reach a browser, so the core answers
// the same paths the legacy Vite dev proxy did (vite.config.ts) and adds the
// key server-side:
//
//	GET /vworld/wmts/{z}/{y}/{x}.jpeg  -> /req/wmts/1.0.0/KEY/Satellite/{z}/{y}/{x}.jpeg
//	GET /vworld/wfs?...                -> /req/wfs?...&key=KEY[&domain=...]
//
// Responses go through the same disk cache as the terrain feed (a satellite
// tile the planner showed is the tile the base map drapes). Error documents
// are never cached or passed on, and no response or log line carries the key.

// VWorldHandler serves /vworld/ requests.
func (f *Feed) VWorldHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		rest := strings.TrimPrefix(r.URL.Path, "/vworld/")
		switch {
		case strings.HasPrefix(rest, "wmts/"):
			f.serveWMTS(w, r, strings.TrimPrefix(rest, "wmts/"))
		case rest == "wfs":
			f.serveWFS(w, r)
		default:
			http.NotFound(w, r)
		}
	})
}

// parseWMTSPath reads "{z}/{y}/{x}.jpeg" and checks it against the zoom range
// VWorld serves.
func parseWMTSPath(p string) (tileKey, bool) {
	parts := strings.Split(p, "/")
	if len(parts) != 3 || !strings.HasSuffix(parts[2], ".jpeg") {
		return tileKey{}, false
	}
	parts[2] = strings.TrimSuffix(parts[2], ".jpeg")
	var v [3]int
	for i, s := range parts {
		n, err := strconv.Atoi(s)
		if err != nil || n < 0 {
			return tileKey{}, false
		}
		v[i] = n
	}
	z, y, x := v[0], v[1], v[2]
	if z < satMinZoom || z > satMaxZoom || x >= 1<<z || y >= 1<<z {
		return tileKey{}, false
	}
	return tileKey{z: z, x: x, y: y}, true
}

func (f *Feed) serveWMTS(w http.ResponseWriter, r *http.Request, p string) {
	k, ok := parseWMTSPath(p)
	if !ok {
		http.NotFound(w, r)
		return
	}
	if f.key == "" {
		http.Error(w, "no VWorld key configured", http.StatusServiceUnavailable)
		return
	}
	body, err := f.fetch.rawSatTile(r.Context(), f.key, k)
	if err != nil {
		f.proxyError(w, r, "satellite tile", err)
		return
	}
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Cache-Control", "public, max-age=86400")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	if r.Method == http.MethodHead {
		return
	}
	_, _ = w.Write(body)
}

// wfsProxyQuery is the client's query without any key or domain it sent,
// in canonical order (url.Values.Encode sorts), so it doubles as cache key.
func wfsProxyQuery(q url.Values) string {
	q = cloneValues(q)
	for k := range q {
		switch strings.ToLower(k) {
		case "key", "domain":
			delete(q, k)
		}
	}
	return q.Encode()
}

func cloneValues(q url.Values) url.Values {
	out := make(url.Values, len(q))
	for k, v := range q {
		out[k] = append([]string(nil), v...)
	}
	return out
}

func (f *Feed) serveWFS(w http.ResponseWriter, r *http.Request) {
	if f.key == "" {
		http.Error(w, "no VWorld key configured", http.StatusServiceUnavailable)
		return
	}
	query := wfsProxyQuery(r.URL.Query())
	target := "https://api.vworld.kr/req/wfs?" + query
	if query != "" {
		target += "&"
	}
	target += "key=" + url.QueryEscape(f.key)
	if f.domain != "" {
		target += "&domain=" + url.QueryEscape(f.domain)
	}
	sum := sha1.Sum([]byte(query))
	name := "vworld-wfs-proxy/" + hex.EncodeToString(sum[:]) + ".json"
	body, err := f.fetch.get(r.Context(), target, name, "building footprints (proxy)", true, nil)
	if err != nil {
		f.proxyError(w, r, "building footprints", err)
		return
	}
	// VWorld answers errors with HTTP 200 and an XML or JSON error document.
	var probe struct {
		Features json.RawMessage `json:"features"`
	}
	if json.Unmarshal(body, &probe) != nil || len(probe.Features) == 0 {
		f.fetch.drop(name)
		http.Error(w, "VWorld returned no feature collection", http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "public, max-age=3600")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	if r.Method == http.MethodHead {
		return
	}
	_, _ = w.Write(body)
}

// proxyError answers a failed upstream request. The fetcher's errors are
// already free of the key; the response still carries only a fixed text.
func (f *Feed) proxyError(w http.ResponseWriter, r *http.Request, what string, err error) {
	if r.Context().Err() != nil {
		return // the client went away
	}
	if errors.Is(err, errNotFound) {
		http.Error(w, what+" not found", http.StatusNotFound)
		return
	}
	log.Printf("terrain: vworld proxy: %v", err)
	http.Error(w, "VWorld unavailable", http.StatusBadGateway)
}
