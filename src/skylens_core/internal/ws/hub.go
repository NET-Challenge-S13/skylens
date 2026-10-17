// Package ws carries viewerv1.Envelope frames over WebSocket (ws /uplink,
// ws /control and ws /viewer). Every frame is one binary-encoded Envelope.
//
// A Hub fans frames out to its peers. Each envelope is encoded once; the
// per-peer sequence number (Envelope.seq, "monotonic per sender so a receiver
// can spot gaps") is prepended as a separate field at write time. Protobuf
// accepts fields in any order, so a large chunk payload is never re-encoded or
// copied per peer.
//
// Joining is atomic with publishing: Attach takes a snapshot and registers the
// peer under the same lock Publish holds while it mutates state and fans out,
// so a new peer sees every update exactly once, either in its replay or live.
package ws

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"

	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
)

// ErrOverflow is reported when a peer fell too far behind and was dropped.
// It reconnects and gets a fresh replay.
var ErrOverflow = errors.New("ws: peer send queue overflow")

// ErrClosed is reported by Next once the peer was closed.
var ErrClosed = errors.New("ws: peer closed")

var peerIDs atomic.Uint64

type item struct {
	body []byte
	live bool // counted against the backlog limit
}

// Peer is one connected socket as seen by the hub. Data is free for the owner
// (for example the drone ids an uplink announced).
type Peer struct {
	ID     uint64
	Remote string

	mu      sync.Mutex
	queue   []item
	pending int // bytes of live frames queued
	limit   int
	seq     uint64
	err     error
	notify  chan struct{}

	dataMu sync.Mutex
	data   map[string]any
}

// NewPeer returns a peer whose live backlog may hold up to limitBytes before
// it is dropped. Replay frames are not counted against the limit.
func NewPeer(remote string, limitBytes int) *Peer {
	if limitBytes <= 0 {
		limitBytes = 64 << 20
	}
	return &Peer{
		ID:     peerIDs.Add(1),
		Remote: remote,
		limit:  limitBytes,
		notify: make(chan struct{}, 1),
	}
}

// Set stores owner data on the peer.
func (p *Peer) Set(key string, v any) {
	p.dataMu.Lock()
	defer p.dataMu.Unlock()
	if p.data == nil {
		p.data = map[string]any{}
	}
	p.data[key] = v
}

// Get reads owner data from the peer.
func (p *Peer) Get(key string) any {
	p.dataMu.Lock()
	defer p.dataMu.Unlock()
	return p.data[key]
}

func (p *Peer) enqueue(body []byte, replay bool) {
	p.mu.Lock()
	if p.err != nil {
		p.mu.Unlock()
		return
	}
	if !replay && p.pending+len(body) > p.limit {
		p.err = ErrOverflow
		p.queue = nil
		p.mu.Unlock()
		p.wake()
		return
	}
	p.queue = append(p.queue, item{body: body, live: !replay})
	if !replay {
		p.pending += len(body)
	}
	p.mu.Unlock()
	p.wake()
}

func (p *Peer) wake() {
	select {
	case p.notify <- struct{}{}:
	default:
	}
}

// Close stops the peer; Next returns ErrClosed (or the earlier error).
func (p *Peer) Close() {
	p.mu.Lock()
	if p.err == nil {
		p.err = ErrClosed
	}
	p.queue = nil
	p.mu.Unlock()
	p.wake()
}

// Closed reports whether the peer is finished, by Close or by an earlier error.
func (p *Peer) Closed() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.err != nil
}

// Next blocks until a frame is ready and returns it with the peer's sequence
// number prepended, or returns the error that ended the peer. done aborts the
// wait.
func (p *Peer) Next(done <-chan struct{}) ([]byte, error) {
	prefix, body, err := p.NextParts(done)
	if err != nil {
		return nil, err
	}
	return append(prefix, body...), nil
}

// NextParts is Next without the copy: the frame is prefix followed by body.
// body is shared between peers and must not be modified.
func (p *Peer) NextParts(done <-chan struct{}) (prefix, body []byte, err error) {
	for {
		p.mu.Lock()
		if p.err != nil {
			err := p.err
			p.mu.Unlock()
			return nil, nil, err
		}
		if len(p.queue) > 0 {
			it := p.queue[0]
			p.queue[0] = item{}
			p.queue = p.queue[1:]
			if it.live {
				p.pending -= len(it.body)
			}
			p.seq++
			seq := p.seq
			p.mu.Unlock()
			return SeqPrefix(seq), it.body, nil
		}
		p.mu.Unlock()
		select {
		case <-p.notify:
		case <-done:
			return nil, nil, ErrClosed
		}
	}
}

// Backlog reports how many frames wait to be written.
func (p *Peer) Backlog() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.queue)
}

// Encode marshals an envelope without its seq. From defaults to the hub's
// component and origin_ts_ms to now when unset (it is propagated unchanged
// when the envelope came from elsewhere).
func Encode(env *viewerv1.Envelope, from viewerv1.ComponentId) ([]byte, error) {
	// Shallow copy: the body (possibly a large chunk) is shared, not cloned.
	e := &viewerv1.Envelope{From: env.GetFrom(), OriginTsMs: env.GetOriginTsMs(), Body: env.Body}
	if e.From == viewerv1.ComponentId_COMPONENT_ID_UNSPECIFIED {
		e.From = from
	}
	if e.OriginTsMs == 0 {
		e.OriginTsMs = time.Now().UnixMilli()
	}
	return proto.MarshalOptions{Deterministic: true}.Marshal(e)
}

// SeqPrefix is the encoding of Envelope.seq (field 1, varint) alone.
func SeqPrefix(seq uint64) []byte {
	out := make([]byte, 0, 11)
	out = protowire.AppendTag(out, 1, protowire.VarintType)
	return protowire.AppendVarint(out, seq)
}

// Frame prepends Envelope.seq to an encoded envelope body.
func Frame(seq uint64, body []byte) []byte {
	return append(SeqPrefix(seq), body...)
}

// Hub fans envelopes out to its peers.
type Hub struct {
	from viewerv1.ComponentId

	mu    sync.Mutex
	peers map[*Peer]struct{}

	sent    atomic.Uint64
	dropped atomic.Uint64
}

// NewHub returns an empty hub whose envelopes default to from.
func NewHub(from viewerv1.ComponentId) *Hub {
	return &Hub{from: from, peers: map[*Peer]struct{}{}}
}

// Publish runs mutate under the hub lock and sends the envelopes it returns to
// every peer. Use it when state changes and its broadcast must be atomic with
// respect to Attach.
func (h *Hub) Publish(mutate func() []*viewerv1.Envelope) {
	h.mu.Lock()
	defer h.mu.Unlock()
	envs := mutate()
	if len(envs) == 0 {
		return
	}
	bodies := h.encodeAll(envs)
	for p := range h.peers {
		for _, b := range bodies {
			p.enqueue(b, false)
		}
	}
	h.sent.Add(uint64(len(bodies) * len(h.peers)))
}

// Broadcast sends envelopes to every peer.
func (h *Hub) Broadcast(envs ...*viewerv1.Envelope) {
	h.Publish(func() []*viewerv1.Envelope { return envs })
}

// Attach registers a peer and queues the snapshot as its replay, atomically
// with any Publish.
func (h *Hub) Attach(p *Peer, snapshot func() []*viewerv1.Envelope) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if snapshot != nil {
		for _, b := range h.encodeAll(snapshot()) {
			p.enqueue(b, true)
		}
	}
	h.peers[p] = struct{}{}
}

// Detach removes a peer and closes it.
func (h *Hub) Detach(p *Peer) {
	h.mu.Lock()
	delete(h.peers, p)
	h.mu.Unlock()
	p.Close()
}

// Send queues envelopes for one peer only.
func (h *Hub) Send(p *Peer, envs ...*viewerv1.Envelope) {
	for _, b := range h.encodeAll(envs) {
		p.enqueue(b, false)
	}
}

// Len is the number of attached peers.
func (h *Hub) Len() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.peers)
}

// Peers returns the attached peers.
func (h *Hub) Peers() []*Peer {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]*Peer, 0, len(h.peers))
	for p := range h.peers {
		out = append(out, p)
	}
	return out
}

// Sent counts frames queued to peers; Dropped counts envelopes that failed to
// encode.
func (h *Hub) Sent() uint64    { return h.sent.Load() }
func (h *Hub) Dropped() uint64 { return h.dropped.Load() }

func (h *Hub) encodeAll(envs []*viewerv1.Envelope) [][]byte {
	out := make([][]byte, 0, len(envs))
	for _, e := range envs {
		if e == nil {
			continue
		}
		b, err := Encode(e, h.from)
		if err != nil {
			h.dropped.Add(1)
			continue
		}
		out = append(out, b)
	}
	return out
}
