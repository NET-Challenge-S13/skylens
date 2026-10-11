package terrain

import (
	"bytes"
	"context"
	"fmt"
	"image/png"
	"math"
	"sort"
)

// DEM: AWS Open Data "Terrain Tiles" (Mapzen terrarium PNGs, SRTM ~30 m in
// Korea). Public bucket, no key.

// demZoom is the native zoom for ~30 m SRTM data at Korean latitudes.
const demZoom = 12

func terrariumURL(k tileKey) string {
	return fmt.Sprintf("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/%d/%d/%d.png", k.z, k.x, k.y)
}

// loadDEMTile fetches and decodes one terrarium tile.
func (f *fetcher) loadDEMTile(ctx context.Context, k tileKey) ([]float32, error) {
	name := fmt.Sprintf("terrarium/%d/%d/%d.png", k.z, k.x, k.y)
	label := fmt.Sprintf("terrain tile %d/%d/%d", k.z, k.x, k.y)
	body, err := f.get(ctx, terrariumURL(k), name, label, false, nil)
	if err != nil {
		return nil, err
	}
	img, err := png.Decode(bytes.NewReader(body))
	if err != nil {
		f.drop(name)
		return nil, fmt.Errorf("%s: not a PNG", label)
	}
	return decodeTerrarium(img), nil
}

// demSampler reads elevations (metres above sea level) from terrarium tiles.
type demSampler struct {
	tiles *tileStore[[]float32]
	z     int
}

// pixel is the elevation of global pixel (gx, gy), NaN when its tile failed.
func (d *demSampler) pixel(ctx context.Context, gx, gy int) float64 {
	n := tileSize << d.z
	if gx < 0 || gy < 0 || gx >= n || gy >= n {
		return math.NaN()
	}
	t, err := d.tiles.get(ctx, tileKey{d.z, gx / tileSize, gy / tileSize})
	if err != nil || t == nil {
		return math.NaN()
	}
	return float64(t[(gy%tileSize)*tileSize+gx%tileSize])
}

// elevation is the bilinear elevation at lat/lon, pixel centres at i + 0.5.
// It returns NaN when any of the four samples is missing.
func (d *demSampler) elevation(ctx context.Context, lat, lon float64) float64 {
	gx := lonToTileX(lon, d.z)*tileSize - 0.5
	gy := latToTileY(lat, d.z)*tileSize - 0.5
	x0 := int(math.Floor(gx))
	y0 := int(math.Floor(gy))
	fx := gx - float64(x0)
	fy := gy - float64(y0)
	a := d.pixel(ctx, x0, y0)
	b := d.pixel(ctx, x0+1, y0)
	c := d.pixel(ctx, x0, y0+1)
	e := d.pixel(ctx, x0+1, y0+1)
	return a*(1-fx)*(1-fy) + b*fx*(1-fy) + c*(1-fx)*fy + e*fx*fy
}

// tilesCovering lists the tiles at zoom z that cover a lat/lon box.
func tilesCovering(minLat, minLon, maxLat, maxLon float64, z int) []tileKey {
	n := 1 << z
	clamp := func(v int) int { return max(0, min(n-1, v)) }
	x0 := clamp(int(math.Floor(lonToTileX(minLon, z))))
	x1 := clamp(int(math.Floor(lonToTileX(maxLon, z))))
	y0 := clamp(int(math.Floor(latToTileY(maxLat, z)))) // north edge, smaller y
	y1 := clamp(int(math.Floor(latToTileY(minLat, z))))
	var out []tileKey
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			out = append(out, tileKey{z, x, y})
		}
	}
	return out
}

func badElevation(v float64) bool {
	return math.IsNaN(v) || math.IsInf(v, 0) || v < -1000 || v > 9500
}

// sanitizeGrid repairs SRTM voids and failed tiles in place, as the legacy
// terrain source did: a few neighbour-averaging sweeps for edges, then the
// median of the valid samples for anything left. It reports false when no
// sample at all is valid.
func sanitizeGrid(h []float64, w, ht int) bool {
	var good []float64
	for _, v := range h {
		if !badElevation(v) {
			good = append(good, v)
		}
	}
	if len(good) == 0 {
		return false
	}
	if len(good) == len(h) {
		return true
	}
	sort.Float64s(good)
	fallback := good[len(good)/2]
	for sweep := 0; sweep < 3; sweep++ {
		fixed := 0
		for y := 0; y < ht; y++ {
			for x := 0; x < w; x++ {
				i := y*w + x
				if !badElevation(h[i]) {
					continue
				}
				sum, n := 0.0, 0
				if x > 0 && !badElevation(h[i-1]) {
					sum += h[i-1]
					n++
				}
				if x < w-1 && !badElevation(h[i+1]) {
					sum += h[i+1]
					n++
				}
				if y > 0 && !badElevation(h[i-w]) {
					sum += h[i-w]
					n++
				}
				if y < ht-1 && !badElevation(h[i+w]) {
					sum += h[i+w]
					n++
				}
				if n > 0 {
					h[i] = sum / float64(n)
					fixed++
				}
			}
		}
		if fixed == 0 {
			break
		}
	}
	for i, v := range h {
		if badElevation(v) {
			h[i] = fallback
		}
	}
	return true
}
