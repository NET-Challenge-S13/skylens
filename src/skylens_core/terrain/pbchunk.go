package terrain

import (
	"encoding/binary"
	"math"
)

// PointBlitz chunk format v2, reimplemented byte for byte from
// pointblitz-io 0.2.0 (src/chunk.rs, encode_with):
//
//	offset size  field
//	     0    4  magic "PBCK"
//	     4    2  version = 2            (u16 LE)
//	     6    2  header length = 80     (u16 LE)
//	     8    4  generation             (u32 LE)
//	    12    4  index within generation (u32 LE)
//	    16    4  point count            (u32 LE)
//	    20    4  flags                  (u32 LE)  bit 0: last chunk of its generation
//	                                              bit 1: the coarse first pass is complete
//	    24   24  origin x, y, z         (f64 LE, ENU metres)
//	    48   12  bbox min x, y, z       (f32 LE, relative to origin)
//	    60   12  bbox max x, y, z       (f32 LE, relative to origin)
//	    72    2  point stride = 12 or 8 (u16 LE)
//	    74    6  reserved, zero
//	    80  stride*n points:
//	         stride 12: position u16x3, zero u16, colour u8x4 (sRGB, a = 255)
//	         stride 8:  position u16x3, colour RGB565 (u16 LE)
//
// Positions are quantised to the chunk's bounding box:
// p = origin + bbox_min + q/65535 * (bbox_max - bbox_min), per axis.
// The origin is the bbox centre rounded to whole metres.
// Axes are pointblitz ENU: x east, y north, z up.

const (
	chunkVersion   uint16 = 2
	chunkHeaderLen        = 80
	// FlagLastInGeneration marks the last chunk of a generation.
	FlagLastInGeneration uint32 = 1
	// FlagFirstPassComplete marks the chunk that completes a coarse first pass.
	FlagFirstPassComplete uint32 = 2
	// DefaultStride is the stride pointblitz-io writes unless told otherwise.
	DefaultStride = 12
)

var chunkMagic = [4]byte{'P', 'B', 'C', 'K'}

// Point is one coloured point in local ENU metres (x east, y north, z up),
// the same shape as pointblitz_io::Point: f32 position, sRGB colour.
type Point struct {
	Pos   [3]float32
	Color [3]uint8
}

// EncodeChunk encodes points into one chunk with the default stride (12).
func EncodeChunk(generation, index, flags uint32, points []Point) []byte {
	return EncodeChunkStride(generation, index, flags, points, DefaultStride)
}

// EncodeChunkStride encodes points with point stride 12 or 8. Any other
// stride panics, as the Rust encoder asserts.
func EncodeChunkStride(generation, index, flags uint32, points []Point, stride int) []byte {
	if stride != 12 && stride != 8 {
		panic("terrain: pointblitz stride must be 12 or 8")
	}
	lo, hi := chunkBounds(points)
	var origin [3]float64
	for i := 0; i < 3; i++ {
		// f64::round rounds half away from zero, as math.Round does.
		origin[i] = math.Round((lo[i] + hi[i]) / 2)
	}
	rel := func(p [3]float64) [3]float32 {
		return [3]float32{
			float32(p[0] - origin[0]),
			float32(p[1] - origin[1]),
			float32(p[2] - origin[2]),
		}
	}

	out := make([]byte, chunkHeaderLen, chunkHeaderLen+stride*len(points))
	le := binary.LittleEndian
	copy(out[0:4], chunkMagic[:])
	le.PutUint16(out[4:], chunkVersion)
	le.PutUint16(out[6:], chunkHeaderLen)
	le.PutUint32(out[8:], generation)
	le.PutUint32(out[12:], index)
	le.PutUint32(out[16:], uint32(len(points)))
	le.PutUint32(out[20:], flags)
	for i := 0; i < 3; i++ {
		le.PutUint64(out[24+8*i:], math.Float64bits(origin[i]))
	}
	var bmin, bmax [3]float32
	if len(points) > 0 {
		bmin = rel(lo)
		bmax = rel(hi)
	}
	for i := 0; i < 3; i++ {
		le.PutUint32(out[48+4*i:], math.Float32bits(bmin[i]))
		le.PutUint32(out[60+4*i:], math.Float32bits(bmax[i]))
	}
	le.PutUint16(out[72:], uint16(stride))
	// out[74:80] stays zero (reserved).

	// Quantise against the stored f32 box, exactly as the GPU undoes it.
	var scale [3]float64
	for i := 0; i < 3; i++ {
		ext := float64(bmax[i]) - float64(bmin[i])
		if ext > 0 {
			scale[i] = 65535.0 / ext
		}
	}
	var rec [12]byte
	for _, p := range points {
		r := rel([3]float64{float64(p.Pos[0]), float64(p.Pos[1]), float64(p.Pos[2])})
		for i := 0; i < 3; i++ {
			q := math.Round((float64(r[i]) - float64(bmin[i])) * scale[i])
			if q < 0 {
				q = 0
			} else if q > 65535 {
				q = 65535
			}
			le.PutUint16(rec[2*i:], uint16(q))
		}
		if stride == 12 {
			rec[6], rec[7] = 0, 0
			rec[8], rec[9], rec[10] = p.Color[0], p.Color[1], p.Color[2]
			rec[11] = 255
		} else {
			le.PutUint16(rec[6:], RGB565(p.Color[0], p.Color[1], p.Color[2]))
		}
		out = append(out, rec[:stride]...)
	}
	return out
}

// RGB565 packs sRGB bytes into RGB565 with rounding, as pointblitz-io does.
func RGB565(r, g, b uint8) uint16 {
	q := func(v uint8, bits uint32) uint16 {
		return uint16((uint32(v)*((1<<bits)-1) + 127) / 255)
	}
	return q(r, 5)<<11 | q(g, 6)<<5 | q(b, 5)
}

func chunkBounds(points []Point) (lo, hi [3]float64) {
	if len(points) == 0 {
		return lo, hi
	}
	for i := 0; i < 3; i++ {
		lo[i] = math.Inf(1)
		hi[i] = math.Inf(-1)
	}
	for _, p := range points {
		for i := 0; i < 3; i++ {
			v := float64(p.Pos[i])
			if v < lo[i] {
				lo[i] = v
			}
			if v > hi[i] {
				hi[i] = v
			}
		}
	}
	return lo, hi
}
