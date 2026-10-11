package terrain

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/png"
	"math"
	"net/url"
	"strings"
)

// Satellite imagery: VWorld WMTS "Satellite" layer. The key goes in the path,
// so the URL is built here and never logged or used as a cache name.
//
// The core sends the photograph as it is: each cell carries its own image,
// resampled north up onto the cell rectangle. Colour grading and display
// styles are the control tower's business.

const (
	// satMinZoom..satMaxZoom is the tilematrix range VWorld serves; anything
	// shallower is answered with HTTP 200 and an XML ExceptionReport.
	satMinZoom = 6
	satMaxZoom = 19
	// satPrefetch is how many tiles one cell asks for at once; the fetcher's
	// VWorld semaphore bounds the total across cells.
	satPrefetch = 6
	// imageQuality is the JPEG quality of cell images.
	imageQuality = 85
)

// groundRGB fills pixels whose imagery tile is missing while the rest of the
// cell has imagery: a neutral ground tone reads as "imagery missing here",
// where black would read as a hole in the world.
var groundRGB = [3]uint8{0x4b, 0x55, 0x60}

func satelliteURL(key string, k tileKey) string {
	return fmt.Sprintf("https://api.vworld.kr/req/wmts/1.0.0/%s/Satellite/%d/%d/%d.jpeg",
		url.PathEscape(key), k.z, k.y, k.x)
}

func satCacheName(k tileKey) string {
	return fmt.Sprintf("vworld-satellite/%d/%d/%d.jpeg", k.z, k.y, k.x)
}

func isImage(contentType string) bool {
	return strings.Contains(contentType, "image")
}

// rawSatTile returns the JPEG bytes of one VWorld satellite tile, from the
// disk cache when present.
func (f *fetcher) rawSatTile(ctx context.Context, key string, k tileKey) ([]byte, error) {
	label := fmt.Sprintf("satellite tile %d/%d/%d", k.z, k.y, k.x)
	return f.get(ctx, satelliteURL(key, k), satCacheName(k), label, true, isImage)
}

// loadSatTile fetches and decodes one VWorld satellite tile into packed RGB.
func (f *fetcher) loadSatTile(ctx context.Context, key string, k tileKey) ([]uint8, error) {
	body, err := f.rawSatTile(ctx, key, k)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(body))
	if err != nil {
		f.drop(satCacheName(k))
		return nil, fmt.Errorf("satellite tile %d/%d/%d: not an image", k.z, k.y, k.x)
	}
	return decodeRGB(img), nil
}

// satSampler reads satellite colours at one zoom.
type satSampler struct {
	tiles *tileStore[[]uint8]
	z     int
}

// pixel is global pixel (gx, gy) at the sampler's zoom; ok is false when its
// tile is missing.
func (s *satSampler) pixel(ctx context.Context, gx, gy int) ([3]uint8, bool) {
	n := tileSize << s.z
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

// color is the bilinear satellite colour at lat/lon, pixel centres at
// i + 0.5. When some of the four pixels are missing (a tile failed) the
// nearest present one is used; ok is false when none is.
func (s *satSampler) color(ctx context.Context, lat, lon float64) ([3]uint8, bool) {
	if s == nil {
		return groundRGB, false
	}
	gx := lonToTileX(lon, s.z)*tileSize - 0.5
	gy := latToTileY(lat, s.z)*tileSize - 0.5
	x0 := int(math.Floor(gx))
	y0 := int(math.Floor(gy))
	fx := gx - float64(x0)
	fy := gy - float64(y0)
	var px [4][3]uint8
	var ok [4]bool
	px[0], ok[0] = s.pixel(ctx, x0, y0)
	px[1], ok[1] = s.pixel(ctx, x0+1, y0)
	px[2], ok[2] = s.pixel(ctx, x0, y0+1)
	px[3], ok[3] = s.pixel(ctx, x0+1, y0+1)
	if ok[0] && ok[1] && ok[2] && ok[3] {
		w := [4]float64{(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy}
		var out [3]uint8
		for ch := 0; ch < 3; ch++ {
			v := 0.0
			for i := 0; i < 4; i++ {
				v += w[i] * float64(px[i][ch])
			}
			out[ch] = clampByte(math.Round(v))
		}
		return out, true
	}
	// Nearest present pixel, by quadrant.
	order := [4]int{0, 1, 2, 3}
	switch {
	case fx >= 0.5 && fy < 0.5:
		order = [4]int{1, 0, 3, 2}
	case fx < 0.5 && fy >= 0.5:
		order = [4]int{2, 3, 0, 1}
	case fx >= 0.5 && fy >= 0.5:
		order = [4]int{3, 2, 1, 0}
	}
	for _, i := range order {
		if ok[i] {
			return px[i], true
		}
	}
	return groundRGB, false
}

// cellImage resamples the satellite layer onto the ENU rectangle
// [e0, e0+size] x [n0, n0+size] as a size-px square JPEG, north up (row 0 is
// the north edge). It returns nil when no pixel of the rectangle had imagery.
func cellImage(ctx context.Context, sat *satSampler, anchor geoPoint, e0, n0, size float64, px int) []byte {
	if sat == nil || px <= 0 {
		return nil
	}
	// The ENU frame is equirectangular: latitude depends on north only and
	// longitude on east only, so both are computed once per row and column.
	lat := make([]float64, px)
	lon := make([]float64, px)
	for i := 0; i < px; i++ {
		t := (float64(i) + 0.5) / float64(px) * size
		lat[i] = enuToGPS(enu{N: n0 + size - t}, anchor).Lat
		lon[i] = enuToGPS(enu{E: e0 + t}, anchor).Lon
	}
	img := image.NewRGBA(image.Rect(0, 0, px, px))
	hits := 0
	for y := 0; y < px; y++ {
		if ctx.Err() != nil {
			return nil
		}
		row := img.Pix[y*img.Stride:]
		for x := 0; x < px; x++ {
			c, ok := sat.color(ctx, lat[y], lon[x])
			if ok {
				hits++
			}
			row[4*x], row[4*x+1], row[4*x+2], row[4*x+3] = c[0], c[1], c[2], 255
		}
	}
	if hits == 0 {
		return nil
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: imageQuality}); err != nil {
		return nil
	}
	return buf.Bytes()
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
