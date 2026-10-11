package store

import (
	"testing"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	viewerv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/viewer/v1"
)

const (
	points    = modelv1.ChunkKind_CHUNK_KIND_POINTS
	mesh      = modelv1.ChunkKind_CHUNK_KIND_MESH_BLOCK
	dets      = modelv1.ChunkKind_CHUNK_KIND_DETECTIONS
	terrain   = modelv1.ChunkKind_CHUNK_KIND_TERRAIN
	buildings = modelv1.ChunkKind_CHUNK_KIND_BUILDINGS
	preview   = modelv1.Level_LEVEL_PREVIEW
	refined   = modelv1.Level_LEVEL_REFINED
)

func pt(seq uint64, seg uint32, lvl modelv1.Level, gen uint32, tag string) *modelv1.Chunk {
	return &modelv1.Chunk{Sequence: seq, Kind: points, Segment: seg, Level: lvl, Generation: gen, Payload: []byte(tag)}
}

func tags(cs []*modelv1.Chunk) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = string(c.GetPayload())
	}
	return out
}

func sameTags(t *testing.T, got []*modelv1.Chunk, want ...string) {
	t.Helper()
	g := tags(got)
	if len(g) != len(want) {
		t.Fatalf("chunks: got %v want %v", g, want)
	}
	for i := range g {
		if g[i] != want[i] {
			t.Fatalf("chunks: got %v want %v", g, want)
		}
	}
}

func TestPreviewAppendsPerSegment(t *testing.T) {
	s := New(Options{})
	for i, tag := range []string{"p0a", "p1a", "p0b"} {
		seg := uint32(0)
		if tag == "p1a" {
			seg = 1
		}
		if !s.Put(pt(uint64(i+1), seg, preview, 0, tag)) {
			t.Fatalf("preview %s rejected", tag)
		}
	}
	sameTags(t, s.Chunks(), "p0a", "p0b", "p1a")
}

func TestRefinedReplacesPreviewOfSameSegmentOnly(t *testing.T) {
	s := New(Options{})
	s.Put(pt(1, 0, preview, 0, "p0"))
	s.Put(pt(2, 1, preview, 0, "p1"))
	if !s.Put(pt(3, 0, refined, 1, "r0g1")) {
		t.Fatal("refined rejected")
	}
	sameTags(t, s.Chunks(), "r0g1", "p1")

	// A preview arriving after the segment is refined is stale.
	if s.Put(pt(4, 0, preview, 0, "late")) {
		t.Fatal("late preview accepted over refined")
	}
	sameTags(t, s.Chunks(), "r0g1", "p1")
}

func TestRefinedGenerations(t *testing.T) {
	s := New(Options{})
	s.Put(pt(1, 0, refined, 1, "g1a"))
	if !s.Put(pt(2, 0, refined, 1, "g1b")) {
		t.Fatal("same generation must accumulate")
	}
	sameTags(t, s.Chunks(), "g1a", "g1b")

	if !s.Put(pt(3, 0, refined, 2, "g2a")) {
		t.Fatal("newer generation rejected")
	}
	sameTags(t, s.Chunks(), "g2a")

	if s.Put(pt(4, 0, refined, 1, "g1c")) {
		t.Fatal("older generation accepted")
	}
	sameTags(t, s.Chunks(), "g2a")
}

func meshBlock(seq uint64, id []int32, version uint32, tag string) *modelv1.Chunk {
	return &modelv1.Chunk{Sequence: seq, Kind: mesh, Segment: 0, Level: preview, BlockId: id, Version: version, Payload: []byte(tag)}
}

func TestMeshBlockOnlyHigherVersionReplaces(t *testing.T) {
	s := New(Options{})
	if !s.Put(meshBlock(1, []int32{0, 0, 0}, 1, "a1")) {
		t.Fatal("first block rejected")
	}
	s.Put(meshBlock(2, []int32{1, 0, 0}, 1, "b1"))
	if s.Put(meshBlock(3, []int32{0, 0, 0}, 1, "a1again")) {
		t.Fatal("same version accepted")
	}
	if !s.Put(meshBlock(4, []int32{0, 0, 0}, 3, "a3")) {
		t.Fatal("higher version rejected")
	}
	if s.Put(meshBlock(5, []int32{0, 0, 0}, 2, "a2")) {
		t.Fatal("lower version accepted")
	}
	// The core keeps the newest even when its payload is empty.
	if !s.Put(meshBlock(6, []int32{1, 0, 0}, 2, "")) {
		t.Fatal("empty newest block rejected")
	}
	// Replay keeps the first-seen block order.
	sameTags(t, s.Chunks(), "a3", "")
}

func TestDetectionsAppendInOrderWithCap(t *testing.T) {
	s := New(Options{DetectionsPerSegment: 2})
	put := func(seq uint64, seg uint32, tag string) {
		if !s.Put(&modelv1.Chunk{Sequence: seq, Kind: dets, Segment: seg, Payload: []byte(tag)}) {
			t.Fatalf("detection %s rejected", tag)
		}
	}
	put(1, 0, "d0a")
	put(2, 1, "d1a")
	put(3, 0, "d0b")
	put(4, 0, "d0c")
	sameTags(t, s.Chunks(), "d1a", "d0b", "d0c")
	if got := s.Counters().Detections; got != 4 {
		t.Fatalf("detections counter %d", got)
	}
}

func TestTerrainNewestPerCellAndReplayOrder(t *testing.T) {
	s := New(Options{})
	s.Put(&modelv1.Chunk{Kind: buildings, BlockId: []int32{1, 1}, Payload: []byte("b")})
	s.Put(&modelv1.Chunk{Kind: terrain, BlockId: []int32{1, 1}, Payload: []byte("t1")})
	s.Put(pt(1, 0, preview, 0, "p"))
	s.Put(&modelv1.Chunk{Kind: terrain, BlockId: []int32{1, 1}, Payload: []byte("t1new")})
	s.Put(&modelv1.Chunk{Kind: terrain, BlockId: []int32{2, 1}, Payload: []byte("t2")})
	sameTags(t, s.Chunks(), "t1new", "t2", "b", "p")

	// Terrain never moves the model resume point.
	if got := s.LastModelSequence(); got != 1 {
		t.Fatalf("resume point %d", got)
	}
}

func TestResumePointAndCounters(t *testing.T) {
	s := New(Options{})
	s.Put(pt(5, 0, refined, 2, "r"))
	s.Put(pt(9, 0, refined, 1, "old")) // stale, but the model already sent it
	if got := s.LastModelSequence(); got != 9 {
		t.Fatalf("resume point %d", got)
	}
	c := s.Counters()
	if c.Accepted != 1 || c.Rejected != 1 {
		t.Fatalf("counters %+v", c)
	}
	if s.Put(&modelv1.Chunk{Kind: modelv1.ChunkKind_CHUNK_KIND_UNSPECIFIED}) {
		t.Fatal("unspecified kind accepted")
	}
	if s.Put(nil) {
		t.Fatal("nil accepted")
	}
}

func TestSegmentsHighestLevel(t *testing.T) {
	s := New(Options{})
	s.Put(pt(1, 2, preview, 0, "a"))
	s.Put(pt(2, 0, preview, 0, "b"))
	s.Put(pt(3, 2, refined, 1, "c"))
	s.Put(meshBlock(4, []int32{0, 0, 0}, 1, "m")) // segment 0, preview
	got := s.Segments()
	if len(got) != 2 || got[0].GetIndex() != 0 || got[0].GetLevel() != preview ||
		got[1].GetIndex() != 2 || got[1].GetLevel() != refined {
		t.Fatalf("segments %v", got)
	}
}

func TestTelemetryRouteLinks(t *testing.T) {
	s := New(Options{})
	s.PutTelemetry(&modelv1.DroneTelemetry{DroneId: 2, Speed: 1})
	s.PutTelemetry(&modelv1.DroneTelemetry{DroneId: 1, Speed: 1})
	s.PutTelemetry(&modelv1.DroneTelemetry{DroneId: 2, Speed: 5})
	all := s.AllTelemetry()
	if len(all) != 2 || all[0].GetDroneId() != 1 || all[1].GetSpeed() != 5 {
		t.Fatalf("telemetry %v", all)
	}
	if s.Telemetry(3) != nil {
		t.Fatal("unknown drone has telemetry")
	}

	if s.Route() != nil {
		t.Fatal("route before assignment")
	}
	r := &viewerv1.AssignRoute{DroneId: 1}
	s.SetRoute(r)
	if s.Route() != r {
		t.Fatal("route not kept")
	}

	if !s.PutLink(&viewerv1.LinkStatus{Hop: "core->model", Connected: true}) {
		t.Fatal("first link status is a change")
	}
	if s.PutLink(&viewerv1.LinkStatus{Hop: "core->model", Connected: true}) {
		t.Fatal("identical link status is not a change")
	}
	if !s.PutLink(&viewerv1.LinkStatus{Hop: "core->model", Connected: false}) {
		t.Fatal("disconnect is a change")
	}
	if l := s.Links(); len(l) != 1 || l[0].GetConnected() {
		t.Fatalf("links %v", l)
	}
}
