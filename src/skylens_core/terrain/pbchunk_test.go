package terrain

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// The golden cloud: two points whose box is (0,0,-6)..(2,4,0).
//
//	origin   = round(box centre) = (1, 2, -3)
//	bbox_min = (0,0,-6) - origin = (-1, -2, -3)
//	bbox_max = (2,4, 0) - origin = ( 1,  2,  3)
//	P0 rel (-1,-2, 3) -> q = (0, 0, 65535)
//	P1 rel ( 1, 2,-3) -> q = (65535, 65535, 0)
var goldenPoints = []Point{
	{Pos: [3]float32{0, 0, 0}, Color: [3]uint8{1, 2, 3}},
	{Pos: [3]float32{2, 4, -6}, Color: [3]uint8{255, 128, 0}},
}

func goldenHeader(stride byte) []byte {
	return []byte{
		'P', 'B', 'C', 'K', // magic
		0x02, 0x00, // version 2
		0x50, 0x00, // header length 80
		0x07, 0x00, 0x00, 0x00, // generation 7
		0x01, 0x00, 0x00, 0x00, // index 1
		0x02, 0x00, 0x00, 0x00, // point count 2
		0x01, 0x00, 0x00, 0x00, // flags: last in generation
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // origin x =  1.0 (f64)
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // origin y =  2.0
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xC0, // origin z = -3.0
		0x00, 0x00, 0x80, 0xBF, // bbox min x = -1 (f32)
		0x00, 0x00, 0x00, 0xC0, // bbox min y = -2
		0x00, 0x00, 0x40, 0xC0, // bbox min z = -3
		0x00, 0x00, 0x80, 0x3F, // bbox max x =  1
		0x00, 0x00, 0x00, 0x40, // bbox max y =  2
		0x00, 0x00, 0x40, 0x40, // bbox max z =  3
		stride, 0x00, // point stride
		0, 0, 0, 0, 0, 0, // reserved
	}
}

func TestEncodeChunkGoldenStride12(t *testing.T) {
	want := goldenHeader(12)
	want = append(want,
		// P0: q = (0, 0, 65535), zero pad, RGBA
		0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 1, 2, 3, 255,
		// P1: q = (65535, 65535, 0), zero pad, RGBA
		0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 255, 128, 0, 255,
	)
	got := EncodeChunk(7, 1, FlagLastInGeneration, goldenPoints)
	if len(want) != 80+2*12 {
		t.Fatalf("golden itself is %d bytes", len(want))
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("stride 12 chunk mismatch\n got % x\nwant % x", got, want)
	}
}

func TestEncodeChunkGoldenStride8(t *testing.T) {
	// rgb565(1,2,3) = 0x0000; rgb565(255,128,0) = 31<<11 | 32<<5 = 0xFC00.
	want := goldenHeader(8)
	want = append(want,
		0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00,
		0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFC,
	)
	got := EncodeChunkStride(7, 1, FlagLastInGeneration, goldenPoints, 8)
	if !bytes.Equal(got, want) {
		t.Fatalf("stride 8 chunk mismatch\n got % x\nwant % x", got, want)
	}
}

func TestEmptyChunkIsHeaderOnly(t *testing.T) {
	got := EncodeChunk(9, 0, FlagLastInGeneration, nil)
	if len(got) != 80 {
		t.Fatalf("len %d, want 80", len(got))
	}
	h := decodeTestHeader(t, got)
	if h.count != 0 || h.generation != 9 || h.flags != 1 || h.stride != 12 {
		t.Fatalf("header %+v", h)
	}
	for i := 24; i < 72; i++ {
		if got[i] != 0 {
			t.Fatalf("origin/bbox byte %d = %d, want 0", i, got[i])
		}
	}
}

func TestRGB565KeepsTheEnds(t *testing.T) {
	cases := []struct {
		r, g, b uint8
		want    uint16
	}{
		{0, 0, 0, 0}, {255, 255, 255, 0xffff}, {255, 0, 0, 0xf800}, {0, 255, 0, 0x07e0}, {0, 0, 255, 0x001f},
	}
	for _, c := range cases {
		if got := RGB565(c.r, c.g, c.b); got != c.want {
			t.Errorf("rgb565(%d,%d,%d) = %#x, want %#x", c.r, c.g, c.b, got, c.want)
		}
	}
}

func TestPositionsRoundTripWithinHalfAStep(t *testing.T) {
	// Same cloud as the Rust test: 5 km from the origin, 24.75 m by 99 m.
	var in []Point
	for i := 0; i < 100; i++ {
		in = append(in, Point{Pos: [3]float32{5000 + float32(i)*0.25, -float32(i), 10}, Color: [3]uint8{uint8(i), 2, 3}})
	}
	for _, stride := range []int{12, 8} {
		c := EncodeChunkStride(0, 0, 0, in, stride)
		h := decodeTestHeader(t, c)
		if len(c) != 80+stride*len(in) {
			t.Fatalf("stride %d: len %d", stride, len(c))
		}
		for i := 0; i < len(in); i++ {
			rec := c[80+i*stride:]
			for k := 0; k < 3; k++ {
				q := binary.LittleEndian.Uint16(rec[2*k:])
				step := (h.max[k] - h.min[k]) / 65535
				back := h.min[k] + float32(q)/65535*(h.max[k]-h.min[k])
				want := float64(in[i].Pos[k]) - h.origin[k]
				if err := math.Abs(float64(back) - want); err > float64(step)/2+1e-4 {
					t.Fatalf("stride %d point %d axis %d: err %g m", stride, i, k, err)
				}
			}
		}
	}
}

type testHeader struct {
	generation, index, count, flags uint32
	origin                          [3]float64
	min, max                        [3]float32
	stride                          uint16
}

// decodeTestHeader mirrors pointblitz_io::decode_header's checks.
func decodeTestHeader(t *testing.T, b []byte) testHeader {
	t.Helper()
	le := binary.LittleEndian
	if len(b) < 80 || string(b[:4]) != "PBCK" || le.Uint16(b[4:]) != 2 || le.Uint16(b[6:]) != 80 {
		t.Fatalf("bad chunk header")
	}
	h := testHeader{
		generation: le.Uint32(b[8:]),
		index:      le.Uint32(b[12:]),
		count:      le.Uint32(b[16:]),
		flags:      le.Uint32(b[20:]),
		stride:     le.Uint16(b[72:]),
	}
	for i := 0; i < 3; i++ {
		h.origin[i] = math.Float64frombits(le.Uint64(b[24+8*i:]))
		h.min[i] = math.Float32frombits(le.Uint32(b[48+4*i:]))
		h.max[i] = math.Float32frombits(le.Uint32(b[60+4*i:]))
	}
	if h.stride != 12 && h.stride != 8 {
		t.Fatalf("bad stride %d", h.stride)
	}
	if len(b) != 80+int(h.count)*int(h.stride) {
		t.Fatalf("length %d does not match %d points", len(b), h.count)
	}
	if h.flags&^3 != 0 {
		t.Fatalf("unknown flags %#x", h.flags)
	}
	for _, r := range b[74:80] {
		if r != 0 {
			t.Fatalf("reserved byte not zero")
		}
	}
	return h
}

// decodeTestPoints dequantises every point of a stride-12 chunk back to ENU.
func decodeTestPoints(t *testing.T, b []byte) []Point {
	t.Helper()
	h := decodeTestHeader(t, b)
	if h.stride != 12 {
		t.Fatalf("stride %d", h.stride)
	}
	out := make([]Point, h.count)
	for i := range out {
		rec := b[80+12*i:]
		for k := 0; k < 3; k++ {
			q := binary.LittleEndian.Uint16(rec[2*k:])
			v := float64(h.min[k]) + float64(q)/65535*float64(h.max[k]-h.min[k])
			out[i].Pos[k] = float32(h.origin[k] + v)
		}
		out[i].Color = [3]uint8{rec[8], rec[9], rec[10]}
	}
	return out
}
