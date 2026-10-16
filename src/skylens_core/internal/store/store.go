// Package store keeps everything the core holds, in memory (SPEC §14: no DB,
// the session's data dies with the process).
//
// Chunks are kept by routing metadata only. The payload is opaque bytes that
// go to viewers unchanged; nothing here decodes it. The replacement rules are:
//
//   - points, preview: appended per segment, until the segment has a refined
//     generation; preview chunks arriving after that are stale and dropped.
//   - points, refined: a newer generation replaces the segment's preview
//     chunks and every older generation. Chunks of the same generation
//     accumulate (refined data is sent coarse first, in several chunks).
//     Older generations are dropped.
//   - mesh blocks: one per block id; replaced only by a higher version. What an
//     empty payload means is the model's business; the newest is kept as is.
//   - detections: appended in arrival order (a viewer replays them in the same
//     order it would have seen them live), capped per segment.
//   - terrain / buildings: the control tower's base map, produced by the
//     core's terrain feed; one per (kind, block id). A refined cell replaces
//     a preview (ring) cell and is never replaced by one; within a level the
//     newest version wins. Status boards never get these (ModelChunks).
//   - point slots: every message of the current slot stream, in message
//     order (version = seq 1, 2, ...), never compacted: a viewer rebuilds the
//     points by applying all of them. A message ahead of a gap is held until
//     the gap fills; a repeated one is stale. A newer stream (higher
//     generation, a new model run) drops the older stream's messages; an
//     older stream's message is stale.
//
// Rejected chunks must not reach viewers either: PutAll returns the chunks to
// forward (a slot message can release held ones), Put whether any was kept.
package store

import (
	"fmt"
	"sort"
	"sync"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
)

// Options configure a Store.
type Options struct {
	// DetectionsPerSegment caps detection chunks kept for one segment; the
	// oldest are dropped first. Zero means 256.
	DetectionsPerSegment int
}

type segmentPoints struct {
	preview    []*modelv1.Chunk
	refined    []*modelv1.Chunk
	generation uint32
	hasRefined bool
}

// slotStream holds one point slot stream: messages seq 1..len(msgs) in order,
// and the messages that arrived ahead of a gap.
type slotStream struct {
	id   uint32
	msgs []*modelv1.Chunk
	held map[uint32]*modelv1.Chunk
}

// ordered keeps keyed chunks in first-insertion order so replay is stable.
type ordered struct {
	keys []string
	m    map[string]*modelv1.Chunk
}

func newOrdered() *ordered { return &ordered{m: map[string]*modelv1.Chunk{}} }

func (o *ordered) get(k string) (*modelv1.Chunk, bool) {
	c, ok := o.m[k]
	return c, ok
}

func (o *ordered) put(k string, c *modelv1.Chunk) {
	if _, ok := o.m[k]; !ok {
		o.keys = append(o.keys, k)
	}
	o.m[k] = c
}

func (o *ordered) each(f func(*modelv1.Chunk)) {
	for _, k := range o.keys {
		f(o.m[k])
	}
}

// Counters is a snapshot of the chunk counters.
type Counters struct {
	Accepted     uint64 // chunks accepted from any producer
	Model        uint64 // accepted reconstruction chunks (points, mesh blocks, detections): what a board counts
	Rejected     uint64 // stale chunks dropped by the replacement rules
	Detections   uint64 // detection chunks accepted
	LastModelSeq uint64 // highest model chunk sequence stored (resume point)
	LastChunkAt  time.Time
}

// Store is safe for concurrent use.
type Store struct {
	opts Options

	mu         sync.RWMutex
	points     map[uint32]*segmentPoints
	mesh       *ordered
	detections []*modelv1.Chunk
	detPerSeg  map[uint32]int
	terrain    *ordered // terrain and buildings
	slots      *slotStream
	segLevel   map[uint32]modelv1.Level
	counters   Counters

	telemetry   map[uint32]*modelv1.DroneTelemetry
	cameraFeeds map[uint32]*viewerv1.CameraFeed
	route       *viewerv1.AssignRoute
	links       map[string]*viewerv1.LinkStatus
	linkOrder   []string
}

// New returns an empty store.
func New(opts Options) *Store {
	if opts.DetectionsPerSegment <= 0 {
		opts.DetectionsPerSegment = 256
	}
	return &Store{
		opts:        opts,
		points:      map[uint32]*segmentPoints{},
		mesh:        newOrdered(),
		detPerSeg:   map[uint32]int{},
		terrain:     newOrdered(),
		segLevel:    map[uint32]modelv1.Level{},
		telemetry:   map[uint32]*modelv1.DroneTelemetry{},
		cameraFeeds: map[uint32]*viewerv1.CameraFeed{},
		links:       map[string]*viewerv1.LinkStatus{},
	}
}

// IsModelKind reports whether a chunk kind comes from the model server (and so
// counts towards the SubscribeChunks resume point).
func IsModelKind(k modelv1.ChunkKind) bool {
	switch k {
	case modelv1.ChunkKind_CHUNK_KIND_POINTS,
		modelv1.ChunkKind_CHUNK_KIND_MESH_BLOCK,
		modelv1.ChunkKind_CHUNK_KIND_DETECTIONS,
		modelv1.ChunkKind_CHUNK_KIND_POINT_SLOTS:
		return true
	}
	return false
}

func blockKey(id []int32) string { return fmt.Sprint(id) }

// Put applies the replacement rules. It returns true when the chunk was kept
// and so should be forwarded to viewers (see PutAll for held slot messages).
func (s *Store) Put(c *modelv1.Chunk) bool {
	return len(s.PutAll(c)) > 0
}

// PutAll applies the replacement rules and returns the chunks to forward to
// viewers, in order: the chunk itself when kept, plus for a point slot message
// that fills a gap the held messages it releases. Nil when nothing was kept
// (or the slot message is held).
func (s *Store) PutAll(c *modelv1.Chunk) []*modelv1.Chunk {
	if c == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	var kept []*modelv1.Chunk
	held := false
	one := func(ok bool) {
		if ok {
			kept = []*modelv1.Chunk{c}
		}
	}
	switch c.GetKind() {
	case modelv1.ChunkKind_CHUNK_KIND_POINTS:
		one(s.putPointsLocked(c))
	case modelv1.ChunkKind_CHUNK_KIND_MESH_BLOCK:
		one(s.putMeshLocked(c))
	case modelv1.ChunkKind_CHUNK_KIND_DETECTIONS:
		one(s.putDetectionLocked(c))
	case modelv1.ChunkKind_CHUNK_KIND_TERRAIN, modelv1.ChunkKind_CHUNK_KIND_BUILDINGS:
		one(s.putTerrainLocked(c))
	case modelv1.ChunkKind_CHUNK_KIND_POINT_SLOTS:
		kept, held = s.putSlotsLocked(c)
	}

	// The resume point advances even for a stale chunk: the model has sent it,
	// asking for it again after a reconnect would only be rejected again.
	if IsModelKind(c.GetKind()) && c.GetSequence() > s.counters.LastModelSeq {
		s.counters.LastModelSeq = c.GetSequence()
	}
	if len(kept) == 0 {
		if !held {
			s.counters.Rejected++
		}
		return nil
	}
	for _, k := range kept {
		s.counters.Accepted++
		if IsModelKind(k.GetKind()) {
			s.counters.Model++
		}
		switch k.GetKind() {
		case modelv1.ChunkKind_CHUNK_KIND_POINTS, modelv1.ChunkKind_CHUNK_KIND_MESH_BLOCK,
			modelv1.ChunkKind_CHUNK_KIND_POINT_SLOTS:
			if cur, seen := s.segLevel[k.GetSegment()]; !seen || k.GetLevel() > cur {
				s.segLevel[k.GetSegment()] = k.GetLevel()
			}
		}
	}
	s.counters.LastChunkAt = time.Now()
	return kept
}

// putSlotsLocked keeps a point slot message (version = seq, generation =
// stream id). It returns the messages now in order (this one and any it
// releases), and whether it was held for a missing seq.
func (s *Store) putSlotsLocked(c *modelv1.Chunk) ([]*modelv1.Chunk, bool) {
	id, seq := c.GetGeneration(), c.GetVersion()
	if seq == 0 {
		return nil, false
	}
	if s.slots == nil || id > s.slots.id {
		// A new stream (a new model run) starts over at seq 1.
		s.slots = &slotStream{id: id, held: map[uint32]*modelv1.Chunk{}}
	}
	st := s.slots
	if id < st.id {
		return nil, false
	}
	next := uint32(len(st.msgs)) + 1
	switch {
	case seq < next:
		return nil, false
	case seq > next:
		if _, dup := st.held[seq]; dup {
			return nil, false
		}
		st.held[seq] = c
		return nil, true
	}
	out := []*modelv1.Chunk{c}
	st.msgs = append(st.msgs, c)
	for {
		h, ok := st.held[uint32(len(st.msgs))+1]
		if !ok {
			break
		}
		delete(st.held, h.GetVersion())
		st.msgs = append(st.msgs, h)
		out = append(out, h)
	}
	return out, false
}

// SlotMessages is the number of point slot messages kept (in order) and held.
func (s *Store) SlotMessages() (kept, held int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.slots == nil {
		return 0, 0
	}
	return len(s.slots.msgs), len(s.slots.held)
}

func (s *Store) putPointsLocked(c *modelv1.Chunk) bool {
	seg := s.points[c.GetSegment()]
	if seg == nil {
		seg = &segmentPoints{}
		s.points[c.GetSegment()] = seg
	}
	if c.GetLevel() != modelv1.Level_LEVEL_REFINED {
		// Preview (or unspecified, treated as preview): appended until refined.
		if seg.hasRefined {
			return false
		}
		seg.preview = append(seg.preview, c)
		return true
	}
	switch {
	case !seg.hasRefined || c.GetGeneration() > seg.generation:
		seg.preview = nil
		seg.refined = []*modelv1.Chunk{c}
		seg.generation = c.GetGeneration()
		seg.hasRefined = true
		return true
	case c.GetGeneration() == seg.generation:
		seg.refined = append(seg.refined, c)
		return true
	default:
		return false
	}
}

func (s *Store) putMeshLocked(c *modelv1.Chunk) bool {
	k := blockKey(c.GetBlockId())
	if old, ok := s.mesh.get(k); ok && c.GetVersion() <= old.GetVersion() {
		return false
	}
	s.mesh.put(k, c)
	return true
}

func (s *Store) putDetectionLocked(c *modelv1.Chunk) bool {
	seg := c.GetSegment()
	s.detections = append(s.detections, c)
	s.detPerSeg[seg]++
	if s.detPerSeg[seg] > s.opts.DetectionsPerSegment {
		for i, d := range s.detections {
			if d.GetSegment() == seg {
				s.detections = append(s.detections[:i], s.detections[i+1:]...)
				s.detPerSeg[seg]--
				break
			}
		}
	}
	s.counters.Detections++
	return true
}

// putTerrainLocked keeps one chunk per (kind, cell). A finer level replaces a
// coarser one (a ring cell the area grew over) and is never replaced by it;
// within a level the newest version wins.
func (s *Store) putTerrainLocked(c *modelv1.Chunk) bool {
	k := fmt.Sprintf("%d/%v", c.GetKind(), c.GetBlockId())
	if old, ok := s.terrain.get(k); ok {
		if c.GetLevel() < old.GetLevel() ||
			(c.GetLevel() == old.GetLevel() && c.GetVersion() < old.GetVersion()) {
			return false
		}
	}
	s.terrain.put(k, c)
	return true
}

// BaseMapCount is how many base map chunks (terrain and building cells) the
// store holds: exactly what a control tower's replay delivers.
func (s *Store) BaseMapCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.terrain.m)
}

// IsBaseMap reports whether a chunk kind is part of the control tower's base
// map (terrain and buildings), which status boards never receive.
func IsBaseMap(k modelv1.ChunkKind) bool {
	return k == modelv1.ChunkKind_CHUNK_KIND_TERRAIN || k == modelv1.ChunkKind_CHUNK_KIND_BUILDINGS
}

// Chunks returns every kept chunk in replay order: terrain and buildings
// first (the ground under everything), then points by segment, the point
// slot messages in seq order, mesh blocks, and detections last in arrival
// order. This is the control tower's replay.
func (s *Store) Chunks() []*modelv1.Chunk { return s.chunks(true) }

// ModelChunks is Chunks without the base map: the status boards' replay.
func (s *Store) ModelChunks() []*modelv1.Chunk { return s.chunks(false) }

func (s *Store) chunks(baseMap bool) []*modelv1.Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var out []*modelv1.Chunk
	if baseMap {
		s.terrain.each(func(c *modelv1.Chunk) {
			if c.GetKind() == modelv1.ChunkKind_CHUNK_KIND_TERRAIN {
				out = append(out, c)
			}
		})
		s.terrain.each(func(c *modelv1.Chunk) {
			if c.GetKind() == modelv1.ChunkKind_CHUNK_KIND_BUILDINGS {
				out = append(out, c)
			}
		})
	}
	segs := make([]uint32, 0, len(s.points))
	for k := range s.points {
		segs = append(segs, k)
	}
	sort.Slice(segs, func(i, j int) bool { return segs[i] < segs[j] })
	for _, k := range segs {
		p := s.points[k]
		out = append(out, p.preview...)
		out = append(out, p.refined...)
	}
	if s.slots != nil {
		out = append(out, s.slots.msgs...)
	}
	s.mesh.each(func(c *modelv1.Chunk) { out = append(out, c) })
	out = append(out, s.detections...)
	return out
}

// Counters returns a snapshot of the chunk counters.
func (s *Store) Counters() Counters {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.counters
}

// LastModelSequence is the SubscribeChunks resume point.
func (s *Store) LastModelSequence() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.counters.LastModelSeq
}

// Segments reports the highest level delivered per segment, ascending.
func (s *Store) Segments() []*viewerv1.SegmentStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()
	idx := make([]uint32, 0, len(s.segLevel))
	for k := range s.segLevel {
		idx = append(idx, k)
	}
	sort.Slice(idx, func(i, j int) bool { return idx[i] < idx[j] })
	out := make([]*viewerv1.SegmentStatus, 0, len(idx))
	for _, k := range idx {
		out = append(out, &viewerv1.SegmentStatus{Index: k, Level: s.segLevel[k]})
	}
	return out
}

// ---- telemetry, camera feeds, route, links ---------------------------------

// PutTelemetry keeps the latest telemetry per drone.
func (s *Store) PutTelemetry(t *modelv1.DroneTelemetry) {
	if t == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.telemetry[t.GetDroneId()] = t
}

// Telemetry returns the latest telemetry of one drone, or nil.
func (s *Store) Telemetry(droneID uint32) *modelv1.DroneTelemetry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.telemetry[droneID]
}

// AllTelemetry returns the latest telemetry of every drone, by drone id.
func (s *Store) AllTelemetry() []*modelv1.DroneTelemetry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*modelv1.DroneTelemetry, 0, len(s.telemetry))
	for _, t := range s.telemetry {
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].GetDroneId() < out[j].GetDroneId() })
	return out
}

// PutCameraFeed keeps the WHEP location of a drone's video.
func (s *Store) PutCameraFeed(f *viewerv1.CameraFeed) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cameraFeeds[f.GetDroneId()] = f
}

// CameraFeeds returns every known camera feed, by drone id.
func (s *Store) CameraFeeds() []*viewerv1.CameraFeed {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*viewerv1.CameraFeed, 0, len(s.cameraFeeds))
	for _, f := range s.cameraFeeds {
		out = append(out, f)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].GetDroneId() < out[j].GetDroneId() })
	return out
}

// SetRoute keeps the assigned route.
func (s *Store) SetRoute(r *viewerv1.AssignRoute) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.route = r
}

// Route returns the assigned route, or nil.
func (s *Store) Route() *viewerv1.AssignRoute {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.route
}

// PutLink keeps the latest status of one hop. It reports whether anything
// changed, so callers can publish only on change.
func (s *Store) PutLink(l *viewerv1.LinkStatus) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	old, ok := s.links[l.GetHop()]
	if !ok {
		s.linkOrder = append(s.linkOrder, l.GetHop())
	} else if old.GetConnected() == l.GetConnected() &&
		old.GetLatencyMs() == l.GetLatencyMs() && old.GetMbps() == l.GetMbps() {
		return false
	}
	s.links[l.GetHop()] = l
	return true
}

// Links returns the latest status of every hop, in first-seen order.
func (s *Store) Links() []*viewerv1.LinkStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*viewerv1.LinkStatus, 0, len(s.linkOrder))
	for _, h := range s.linkOrder {
		out = append(out, s.links[h])
	}
	return out
}
