package terrain

import (
	"bytes"
	"context"
	"fmt"
	"image"
	_ "image/jpeg" // VWorld satellite tiles
	_ "image/png"
	"math"
	"net/url"
	"strings"
)

// Satellite imagery: VWorld WMTS "Satellite" layer. The key goes in the path,
// so the URL is built here and never logged or used as a cache name.

const (
	// satMinZoom: VWorld serves tilematrix 6..19 only and answers anything
	// shallower with HTTP 200 + an XML ExceptionReport.
	satMinZoom = 6
	// satPrefetch is how many tiles one cell asks for at once; the fetcher's
	// VWorld semaphore bounds the total across cells.
	satPrefetch = 6
)

// groundRGB stands in for a missing imagery tile: a neutral ground tone reads
// as "imagery missing here", where black would read as a hole in the world.
var groundRGB = [3]uint8{0x4b, 0x55, 0x60}

func satelliteURL(key string, k tileKey) string {
	return fmt.Sprintf("https://api.vworld.kr/req/wmts/1.0.0/%s/Satellite/%d/%d/%d.jpeg",
		url.PathEscape(key), k.z, k.y, k.x)
}

func isImage(contentType string) bool {
	return strings.Contains(contentType, "image")
}

// loadSatTile fetches and decodes one VWorld satellite tile into packed RGB.
func (f *fetcher) loadSatTile(ctx context.Context, key string, k tileKey) ([]uint8, error) {
	name := fmt.Sprintf("vworld-satellite/%d/%d/%d.jpeg", k.z, k.y, k.x)
	label := fmt.Sprintf("satellite tile %d/%d/%d", k.z, k.y, k.x)
	body, err := f.get(ctx, satelliteURL(key, k), name, label, true, isImage)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(body))
	if err != nil {
		f.drop(name)
		return nil, fmt.Errorf("%s: not an image", label)
	}
	return decodeRGB(img), nil
}

// satSampler reads satellite colours at one zoom.
type satSampler struct {
	tiles *tileStore[[]uint8]
	z     int
}

// color is the satellite pixel containing lat/lon; ok is false when its tile
// is missing, in which case the neutral ground tone is returned.
func (s *satSampler) color(ctx context.Context, lat, lon float64) ([3]uint8, bool) {
	if s == nil {
		return groundRGB, false
	}
	n := tileSize << s.z
	gx := int(math.Floor(lonToTileX(lon, s.z) * tileSize))
	gy := int(math.Floor(latToTileY(lat, s.z) * tileSize))
	if gx < 0 || gy < 0 || gx >= n || gy >= n {
		return groundRGB, false
	}
	t, err := s.tiles.get(ctx, tileKey{s.z, gx / tileSize, gy / tileSize})
	if err != nil || t == nil {
		return groundRGB, false
	}
	o := ((gy%tileSize)*tileSize + gx%tileSize) * 3
	return [3]uint8{t[o], t[o+1], t[o+2]}, true
}

// ---- colour grading ---------------------------------------------------------

// cyberLUT maps luminance to the control tower's restrained navy/teal ramp
// (legacy terrainSource.ts cyberLut, midtone lift baked in). It is a
// background palette: even the brightest pixels stay below the UI and drone
// highlight range.
var cyberLUT = func() [256][3]uint8 {
	type stop struct {
		t float64
		c [3]float64
	}
	stops := []stop{
		{0.0, [3]float64{4, 9, 18}},
		{0.38, [3]float64{8, 27, 42}},
		{0.7, [3]float64{15, 57, 72}},
		{0.9, [3]float64{30, 87, 104}},
		{1.0, [3]float64{58, 126, 142}},
	}
	var lut [256][3]uint8
	for i := 0; i < 256; i++ {
		t := math.Pow(float64(i)/255, 0.88)
		k := 0
		for k < len(stops)-2 && t > stops[k+1].t {
			k++
		}
		s0, s1 := stops[k], stops[k+1]
		f := math.Min(1, math.Max(0, (t-s0.t)/(s1.t-s0.t)))
		for ch := 0; ch < 3; ch++ {
			// Uint8ClampedArray rounds half to even; math.RoundToEven matches.
			lut[i][ch] = clampByte(math.RoundToEven(s0.c[ch] + (s1.c[ch]-s0.c[ch])*f))
		}
	}
	return lut
}()

func luminance(c [3]uint8) int {
	return min(255, (int(c[0])*54+int(c[1])*183+int(c[2])*19)>>8)
}

func cyberGrade(c [3]uint8) [3]uint8 {
	return cyberLUT[luminance(c)]
}

// Hypsometric ramp used when there is no imagery (legacy colours).
var (
	hypsoLow  = [3]float64{0x35, 0x54, 0x3a} // valley forest green
	hypsoMid  = [3]float64{0x8a, 0x7a, 0x55} // ridge tan
	hypsoHigh = [3]float64{0xd8, 0xd2, 0xc6} // peak light grey
)

// hypsoReliefM is the elevation mapped to the top of the ramp. The legacy
// ramp was normalised per scene; a fixed range keeps adjacent cells seamless.
const hypsoReliefM = 900.0

func hypsometric(elev float64) [3]float64 {
	t := math.Min(1, math.Max(0, elev/hypsoReliefM))
	if t < 0.5 {
		return lerp3(hypsoLow, hypsoMid, t*2)
	}
	return lerp3(hypsoMid, hypsoHigh, (t-0.5)*2)
}

func lerp3(a, b [3]float64, t float64) [3]float64 {
	return [3]float64{a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t}
}

func lerpRGB(a, b [3]uint8, t float64) [3]uint8 {
	t = math.Min(1, math.Max(0, t))
	var out [3]uint8
	for i := 0; i < 3; i++ {
		out[i] = clampByte(math.Round(float64(a[i]) + (float64(b[i])-float64(a[i]))*t))
	}
	return out
}

func scaleRGB(c [3]uint8, f float64) [3]uint8 {
	return [3]uint8{
		clampByte(math.Round(float64(c[0]) * f)),
		clampByte(math.Round(float64(c[1]) * f)),
		clampByte(math.Round(float64(c[2]) * f)),
	}
}

func toRGB(c [3]float64) [3]uint8 {
	return [3]uint8{clampByte(math.Round(c[0])), clampByte(math.Round(c[1])), clampByte(math.Round(c[2]))}
}

func clampByte(v float64) uint8 {
	if v <= 0 || math.IsNaN(v) {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

func rgbHex(v uint32) [3]uint8 {
	return [3]uint8{uint8(v >> 16), uint8(v >> 8), uint8(v)}
}
