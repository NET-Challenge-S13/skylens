package terrain

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	// fetchAttempts is how often one resource is tried before it is left out.
	// VWorld silently drops bursts, so a throttled tile usually succeeds later.
	fetchAttempts = 3
	// maxBodyBytes guards against a runaway response.
	maxBodyBytes = 32 << 20
)

// errNotFound is a definitive miss (HTTP 404): not retried.
var errNotFound = errors.New("not found")

// fetcher downloads resources through an on-disk cache. It never puts the
// VWorld key into an error, a log line or a cache path: callers pass a
// key-free cache name and a label, and transport errors are stripped of the
// request URL.
type fetcher struct {
	client   *http.Client
	cacheDir string
	referer  string
	// vworld bounds simultaneous VWorld requests (it drops large bursts).
	vworld chan struct{}
}

func newFetcher(client *http.Client, cacheDir, referer string) *fetcher {
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	return &fetcher{client: client, cacheDir: cacheDir, referer: referer, vworld: make(chan struct{}, 6)}
}

// get returns the body of rawURL, from the cache when present. cacheName is a
// slash-separated path under the cache directory ("" = do not cache). accept
// validates the content type before anything is cached; VWorld answers errors
// with HTTP 200 and an XML body, so the status alone is not enough.
func (f *fetcher) get(ctx context.Context, rawURL, cacheName, label string, throttled bool, accept func(contentType string) bool) ([]byte, error) {
	if body, ok := f.readCache(cacheName); ok {
		return body, nil
	}
	if throttled {
		select {
		case f.vworld <- struct{}{}:
			defer func() { <-f.vworld }()
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	var lastErr error
	for attempt := 0; attempt < fetchAttempts; attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(time.Duration(200*attempt) * time.Millisecond):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		body, err := f.once(ctx, rawURL, label, accept)
		if err == nil {
			f.writeCache(cacheName, body)
			return body, nil
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		lastErr = err
		if errors.Is(err, errNotFound) {
			break
		}
	}
	return nil, lastErr
}

func (f *fetcher) once(ctx context.Context, rawURL, label string, accept func(string) bool) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, fmt.Errorf("%s: bad request", label)
	}
	req.Header.Set("User-Agent", "skylens-core terrain feed")
	if f.referer != "" {
		req.Header.Set("Referer", f.referer)
	}
	res, err := f.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", label, redact(err))
	}
	defer res.Body.Close()
	if res.StatusCode == http.StatusNotFound {
		io.Copy(io.Discard, io.LimitReader(res.Body, 1<<16))
		return nil, fmt.Errorf("%s: %w", label, errNotFound)
	}
	if res.StatusCode != http.StatusOK {
		io.Copy(io.Discard, io.LimitReader(res.Body, 1<<16))
		return nil, fmt.Errorf("%s: HTTP %d", label, res.StatusCode)
	}
	if accept != nil && !accept(res.Header.Get("Content-Type")) {
		return nil, fmt.Errorf("%s: unexpected content type %q", label, res.Header.Get("Content-Type"))
	}
	body, err := io.ReadAll(io.LimitReader(res.Body, maxBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", label, redact(err))
	}
	return body, nil
}

// redact drops the request URL (which may carry the VWorld key) from
// transport errors and keeps only the underlying cause.
func redact(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		if ue.Timeout() {
			return errors.New("timeout")
		}
		return errors.New(scrubbed(ue.Err))
	}
	return errors.New(scrubbed(err))
}

func scrubbed(err error) string {
	if err == nil {
		return "request failed"
	}
	s := err.Error()
	if strings.Contains(s, "vworld") || strings.Contains(s, "key=") {
		return "request failed"
	}
	return s
}

func (f *fetcher) cachePath(name string) string {
	if f.cacheDir == "" || name == "" {
		return ""
	}
	return filepath.Join(f.cacheDir, filepath.FromSlash(name))
}

func (f *fetcher) readCache(name string) ([]byte, bool) {
	p := f.cachePath(name)
	if p == "" {
		return nil, false
	}
	body, err := os.ReadFile(p)
	if err != nil {
		return nil, false
	}
	return body, true
}

// writeCache stores body atomically (temp file + rename), so a reader never
// sees half a tile. Cache failures are not errors: the data is still served.
func (f *fetcher) writeCache(name string, body []byte) {
	p := f.cachePath(name)
	if p == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return
	}
	tmp, err := os.CreateTemp(filepath.Dir(p), ".part-*")
	if err != nil {
		return
	}
	_, werr := tmp.Write(body)
	cerr := tmp.Close()
	if werr != nil || cerr != nil {
		os.Remove(tmp.Name())
		return
	}
	if err := os.Rename(tmp.Name(), p); err != nil {
		os.Remove(tmp.Name())
	}
}

// drop removes a cached entry that turned out to be unusable.
func (f *fetcher) drop(name string) {
	if p := f.cachePath(name); p != "" {
		os.Remove(p)
	}
}

// tileKey addresses one 256x256 tile.
type tileKey struct{ z, x, y int }

// tileStore loads each tile at most once per stream and shares it between
// cells. A failed tile is remembered as failed (a hole, not a retry storm);
// a tile abandoned because the stream was cancelled is forgotten.
type tileStore[T any] struct {
	mu    sync.Mutex
	tiles map[tileKey]*tileEntry[T]
	load  func(ctx context.Context, k tileKey) (T, error)
}

type tileEntry[T any] struct {
	done chan struct{}
	val  T
	err  error
}

func newTileStore[T any](load func(context.Context, tileKey) (T, error)) *tileStore[T] {
	return &tileStore[T]{tiles: make(map[tileKey]*tileEntry[T]), load: load}
}

func (s *tileStore[T]) get(ctx context.Context, k tileKey) (T, error) {
	s.mu.Lock()
	e, ok := s.tiles[k]
	if !ok {
		e = &tileEntry[T]{done: make(chan struct{})}
		s.tiles[k] = e
		s.mu.Unlock()
		e.val, e.err = s.load(ctx, k)
		if e.err != nil && ctx.Err() != nil {
			s.mu.Lock()
			delete(s.tiles, k)
			s.mu.Unlock()
		}
		close(e.done)
		return e.val, e.err
	}
	s.mu.Unlock()
	select {
	case <-e.done:
		return e.val, e.err
	case <-ctx.Done():
		var zero T
		return zero, ctx.Err()
	}
}

// prefetch loads keys concurrently (at most par at a time) and waits.
func (s *tileStore[T]) prefetch(ctx context.Context, keys []tileKey, par int) {
	sem := make(chan struct{}, par)
	var wg sync.WaitGroup
	for _, k := range keys {
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			wg.Wait()
			return
		}
		wg.Add(1)
		go func(k tileKey) {
			defer wg.Done()
			defer func() { <-sem }()
			s.get(ctx, k)
		}(k)
	}
	wg.Wait()
}

// peek returns a tile only if it is already loaded; it never fetches.
func (s *tileStore[T]) peek(k tileKey) (T, bool) {
	s.mu.Lock()
	e, ok := s.tiles[k]
	s.mu.Unlock()
	var zero T
	if !ok {
		return zero, false
	}
	select {
	case <-e.done:
		if e.err != nil {
			return zero, false
		}
		return e.val, true
	default:
		return zero, false
	}
}
