package terrain

import (
	"image"
	"image/color"
	"math"
)

// Web Mercator (slippy map) tile math, as in the legacy terrainSource.ts.
// Coordinates are fractional tile units at zoom z.

func lonToTileX(lon float64, z int) float64 {
	return (lon + 180) / 360 * math.Exp2(float64(z))
}

func latToTileY(lat float64, z int) float64 {
	r := lat * deg
	return (1 - math.Log(math.Tan(r)+1/math.Cos(r))/math.Pi) / 2 * math.Exp2(float64(z))
}

func tileXToLon(x float64, z int) float64 {
	return x/math.Exp2(float64(z))*360 - 180
}

func tileYToLat(y float64, z int) float64 {
	n := math.Pi - 2*math.Pi*y/math.Exp2(float64(z))
	return math.Atan(math.Sinh(n)) / deg
}

// tileSize is the pixel edge of every tile used here (terrarium and VWorld).
const tileSize = 256

// terrariumElevation decodes one terrarium pixel: (R*256 + G + B/256) - 32768 m.
func terrariumElevation(r, g, b uint8) float32 {
	return float32(float64(r)*256+float64(g)+float64(b)/256) - 32768
}

// decodeTerrarium turns a decoded terrarium PNG into elevations, row-major.
func decodeTerrarium(img image.Image) []float32 {
	out := make([]float32, tileSize*tileSize)
	for i := range out {
		out[i] = float32(math.NaN())
	}
	b := img.Bounds()
	w := min(b.Dx(), tileSize)
	h := min(b.Dy(), tileSize)
	switch m := img.(type) {
	case *image.NRGBA:
		for y := 0; y < h; y++ {
			row := m.Pix[y*m.Stride:]
			for x := 0; x < w; x++ {
				out[y*tileSize+x] = terrariumElevation(row[4*x], row[4*x+1], row[4*x+2])
			}
		}
	case *image.RGBA:
		// Terrarium tiles are opaque, so premultiplied values equal straight ones.
		for y := 0; y < h; y++ {
			row := m.Pix[y*m.Stride:]
			for x := 0; x < w; x++ {
				out[y*tileSize+x] = terrariumElevation(row[4*x], row[4*x+1], row[4*x+2])
			}
		}
	default:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				out[y*tileSize+x] = terrariumElevation(c.R, c.G, c.B)
			}
		}
	}
	return out
}

// decodeRGB turns a decoded imagery tile into packed RGB bytes, row-major.
func decodeRGB(img image.Image) []uint8 {
	out := make([]uint8, tileSize*tileSize*3)
	b := img.Bounds()
	w := min(b.Dx(), tileSize)
	h := min(b.Dy(), tileSize)
	switch m := img.(type) {
	case *image.YCbCr:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				yi := m.YOffset(b.Min.X+x, b.Min.Y+y)
				ci := m.COffset(b.Min.X+x, b.Min.Y+y)
				r, g, bb := color.YCbCrToRGB(m.Y[yi], m.Cb[ci], m.Cr[ci])
				o := (y*tileSize + x) * 3
				out[o], out[o+1], out[o+2] = r, g, bb
			}
		}
	default:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				o := (y*tileSize + x) * 3
				out[o], out[o+1], out[o+2] = c.R, c.G, c.B
			}
		}
	}
	return out
}
