package ws

import (
	"context"
	"errors"
	"log"
	"net/http"
	"time"

	"github.com/coder/websocket"
	"google.golang.org/protobuf/proto"

	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
)

// Handler is what a route (uplink or viewer) does with its peers.
type Handler interface {
	// Snapshot is queued to a new peer before any live frame.
	Snapshot(p *Peer) []*viewerv1.Envelope
	// Attached runs after the peer is registered.
	Attached(p *Peer)
	// Message handles one decoded envelope from the peer.
	Message(p *Peer, env *viewerv1.Envelope)
	// Detached runs once the peer is gone.
	Detached(p *Peer)
}

// Options tune a WebSocket route.
type Options struct {
	Name         string        // for logs, e.g. "viewer"
	ReadLimit    int64         // largest inbound frame; default 4 MiB
	BacklogLimit int           // live bytes a peer may lag behind; default 64 MiB
	WriteTimeout time.Duration // per frame, plus the frame's size at WriteRate; default 15 s
	WriteRate    int           // slowest acceptable write, bytes per second; default 1 MiB/s
	PingInterval time.Duration // default 20 s
}

func (o *Options) defaults() {
	if o.ReadLimit <= 0 {
		o.ReadLimit = 4 << 20
	}
	if o.BacklogLimit <= 0 {
		o.BacklogLimit = 64 << 20
	}
	if o.WriteTimeout <= 0 {
		o.WriteTimeout = 15 * time.Second
	}
	if o.WriteRate <= 0 {
		o.WriteRate = 1 << 20
	}
	if o.PingInterval <= 0 {
		o.PingInterval = 20 * time.Second
	}
}

// Route returns an http.Handler that upgrades to WebSocket and runs a peer on
// hub until either side goes away.
func Route(hub *Hub, h Handler, opts Options) http.Handler {
	opts.defaults()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			// Screens and drones reach the core through the bridge and the
			// proxy, under other origins. KOREN admission is the gate here.
			InsecureSkipVerify: true,
		})
		if err != nil {
			log.Printf("[core] %s upgrade failed from %s: %v", opts.Name, r.RemoteAddr, err)
			return
		}
		conn.SetReadLimit(opts.ReadLimit)
		serve(r.Context(), conn, r.RemoteAddr, hub, h, opts)
	})
}

func serve(parent context.Context, conn *websocket.Conn, remote string, hub *Hub, h Handler, opts Options) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	p := NewPeer(remote, opts.BacklogLimit)
	hub.Attach(p, func() []*viewerv1.Envelope { return h.Snapshot(p) })
	log.Printf("[core] %s attached: %s (peer %d, %d open)", opts.Name, remote, p.ID, hub.Len())
	h.Attached(p)

	readErr := make(chan error, 1)
	go func() {
		readErr <- readLoop(ctx, conn, p, h, opts.Name)
		cancel()
	}()
	go pingLoop(ctx, cancel, conn, opts.PingInterval)

	writeErr := writeLoop(ctx, conn, p, opts.WriteTimeout, opts.WriteRate)
	cancel()

	hub.Detach(p)
	h.Detached(p)

	rerr := <-readErr
	status := websocket.StatusNormalClosure
	reason := ""
	if errors.Is(writeErr, ErrOverflow) {
		status = websocket.StatusTryAgainLater
		reason = "too slow, reconnect for a fresh replay"
	}
	conn.Close(status, reason)
	log.Printf("[core] %s detached: %s (peer %d, read: %v, write: %v)", opts.Name, remote, p.ID, quiet(rerr), quiet(writeErr))
}

func readLoop(ctx context.Context, conn *websocket.Conn, p *Peer, h Handler, name string) error {
	for {
		typ, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		if typ != websocket.MessageBinary {
			log.Printf("[core] %s peer %d sent a text frame; dropped (frames are binary Envelopes)", name, p.ID)
			continue
		}
		env := &viewerv1.Envelope{}
		if err := proto.Unmarshal(data, env); err != nil {
			log.Printf("[core] %s peer %d sent an undecodable frame: %v", name, p.ID, err)
			continue
		}
		h.Message(p, env)
	}
}

// frameTimeout is the time one frame of n bytes may take: base plus n at rate
// bytes per second (a point slot message of a live diff stream event can be
// tens of MB).
func frameTimeout(base time.Duration, rate, n int) time.Duration {
	return base + time.Duration(int64(n)*int64(time.Second)/int64(rate))
}

func writeLoop(ctx context.Context, conn *websocket.Conn, p *Peer, timeout time.Duration, rate int) error {
	for {
		prefix, body, err := p.NextParts(ctx.Done())
		if err != nil {
			return err
		}
		wctx, cancel := context.WithTimeout(ctx, frameTimeout(timeout, rate, len(prefix)+len(body)))
		err = writeFrame(wctx, conn, prefix, body)
		cancel()
		if err != nil {
			return err
		}
	}
}

// writeFrame sends prefix+body as one binary message without joining them.
func writeFrame(ctx context.Context, conn *websocket.Conn, prefix, body []byte) error {
	w, err := conn.Writer(ctx, websocket.MessageBinary)
	if err != nil {
		return err
	}
	if _, err := w.Write(prefix); err != nil {
		w.Close()
		return err
	}
	if _, err := w.Write(body); err != nil {
		w.Close()
		return err
	}
	return w.Close()
}

func pingLoop(ctx context.Context, cancel context.CancelFunc, conn *websocket.Conn, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			pctx, done := context.WithTimeout(ctx, every)
			err := conn.Ping(pctx)
			done()
			if err != nil {
				// No pong in a whole interval: the peer is gone without a close.
				cancel()
				return
			}
		}
	}
}

func quiet(err error) any {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, ErrClosed) {
		return "closed"
	}
	if s := websocket.CloseStatus(err); s != -1 {
		return s
	}
	return err
}
