package terrain

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"sort"
	"strconv"
	"sync"
)

// Buildings: VWorld WFS lt_c_bldginfo footprints with their height and the
// DEM ground under them. The screen extrudes them. Facts the legacy layer
// measured and this code keeps:
//   - WFS 1.1.0 + EPSG:4326 BBOX axis order is LAT,LON (lon,lat returns 0 rows).
//   - Response GeoJSON coordinates are lon,lat.
//   - maxFeatures caps at 1000 per request, so a full page is split in four.
//   - height / grnd_flr are often null: floors x 3.3 m, else 6 m.

const (
	wfsMaxFeatures = 1000
	// wfsMinCellDeg stops quad splits below ~50 m cells.
	wfsMinCellDeg = 0.0005
	// wfsMaxDepth reaches ~55 m cells; shallower truncated dense city blocks.
	wfsMaxDepth = 7
)

type lonLat [2]float64

type building struct {
	id string
	// outer rings only; holes are not sent.
	rings   [][]lonLat
	heightM float64
}

type latLonBox struct {
	minLat, minLon, maxLat, maxLon float64
}

func (b latLonBox) contains(lat, lon float64) bool {
	return lat >= b.minLat && lat <= b.maxLat && lon >= b.minLon && lon <= b.maxLon
}

func wfsQuery(b latLonBox) string {
	f := func(v float64) string { return strconv.FormatFloat(v, 'f', 7, 64) }
	// BBOX is LAT,LON order for WFS 1.1.0 + EPSG:4326.
	return "SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature" +
		"&TYPENAME=lt_c_bldginfo&SRSNAME=EPSG:4326" +
		"&BBOX=" + f(b.minLat) + "," + f(b.minLon) + "," + f(b.maxLat) + "," + f(b.maxLon) + ",EPSG:4326" +
		"&maxFeatures=" + strconv.Itoa(wfsMaxFeatures) + "&OUTPUT=application/json"
}

func wfsURL(key, domain string, b latLonBox) string {
	u := "https://api.vworld.kr/req/wfs?" + wfsQuery(b) + "&key=" + url.QueryEscape(key)
	if domain != "" {
		u += "&domain=" + url.QueryEscape(domain)
	}
	return u
}

// wfsCacheName is keyed on the key-free query only.
func wfsCacheName(b latLonBox) string {
	sum := sha1.Sum([]byte(wfsQuery(b)))
	return "vworld-wfs/" + hex.EncodeToString(sum[:]) + ".json"
}

// fetchBuildings returns every footprint intersecting b, sorted by id.
func (f *fetcher) fetchBuildings(ctx context.Context, key, domain string, b latLonBox) ([]building, error) {
	var mu sync.Mutex
	found := make(map[string]building)
	ok, err := f.fetchBuildingCell(ctx, key, domain, b, 0, &mu, found)
	if !ok {
		return nil, err
	}
	out := make([]building, 0, len(found))
	for _, bl := range found {
		out = append(out, bl)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out, nil
}

// fetchBuildingCell fetches one WFS box and quad-splits it when the page is
// full. A failed sub-box leaves a gap but keeps what the others found; ok is
// false only when nothing could be fetched at all.
func (f *fetcher) fetchBuildingCell(ctx context.Context, key, domain string, b latLonBox, depth int, mu *sync.Mutex, out map[string]building) (bool, error) {
	name := wfsCacheName(b)
	body, err := f.get(ctx, wfsURL(key, domain, b), name, "building footprints", true, nil)
	if err != nil {
		return false, err
	}
	probe := make(map[string]building)
	rows, err := parseFeatures(body, probe)
	if err != nil {
		// An error document (XML, or JSON without features) must not stay cached.
		f.drop(name)
		return false, err
	}
	canSplit := depth < wfsMaxDepth &&
		(b.maxLon-b.minLon)/2 > wfsMinCellDeg && (b.maxLat-b.minLat)/2 > wfsMinCellDeg
	if rows >= wfsMaxFeatures && canSplit {
		midLat := (b.minLat + b.maxLat) / 2
		midLon := (b.minLon + b.maxLon) / 2
		subs := []latLonBox{
			{b.minLat, b.minLon, midLat, midLon},
			{b.minLat, midLon, midLat, b.maxLon},
			{midLat, b.minLon, b.maxLat, midLon},
			{midLat, midLon, b.maxLat, b.maxLon},
		}
		var wg sync.WaitGroup
		results := make([]bool, len(subs))
		errs := make([]error, len(subs))
		for i, s := range subs {
			wg.Add(1)
			go func(i int, s latLonBox) {
				defer wg.Done()
				results[i], errs[i] = f.fetchBuildingCell(ctx, key, domain, s, depth+1, mu, out)
			}(i, s)
		}
		wg.Wait()
		for i, r := range results {
			if r {
				return true, nil
			}
			err = errs[i]
		}
		return false, err
	}
	mu.Lock()
	for id, bl := range probe {
		if _, dup := out[id]; !dup {
			out[id] = bl
		}
	}
	mu.Unlock()
	return true, nil
}

type wfsResponse struct {
	Features *[]wfsFeature `json:"features"`
}

type wfsFeature struct {
	ID         json.RawMessage `json:"id"`
	Geometry   *wfsGeometry    `json:"geometry"`
	Properties map[string]any  `json:"properties"`
}

type wfsGeometry struct {
	Type        string          `json:"type"`
	Coordinates json.RawMessage `json:"coordinates"`
}

// parseFeatures adds the footprints in a WFS GeoJSON body to out (dedup by
// id) and returns the number of rows the page held.
func parseFeatures(body []byte, out map[string]building) (int, error) {
	var res wfsResponse
	if err := json.Unmarshal(body, &res); err != nil {
		return 0, fmt.Errorf("building footprints: not GeoJSON")
	}
	if res.Features == nil {
		return 0, fmt.Errorf("building footprints: no feature collection")
	}
	feats := *res.Features
	for _, ft := range feats {
		id := featureID(ft.ID)
		if id == "" {
			continue
		}
		if _, dup := out[id]; dup {
			continue
		}
		if ft.Geometry == nil {
			continue
		}
		rings := parseRings(ft.Geometry)
		if len(rings) == 0 {
			continue
		}
		out[id] = building{id: id, rings: rings, heightM: buildingHeight(ft.Properties)}
	}
	return len(feats), nil
}

func featureID(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return string(raw) // numeric id, as written
}

func parseRings(g *wfsGeometry) [][]lonLat {
	var rings [][]lonLat
	switch g.Type {
	case "Polygon":
		var poly [][][]float64
		if json.Unmarshal(g.Coordinates, &poly) == nil && len(poly) > 0 {
			if r := toRing(poly[0]); r != nil {
				rings = append(rings, r)
			}
		}
	case "MultiPolygon":
		var multi [][][][]float64
		if json.Unmarshal(g.Coordinates, &multi) == nil {
			for _, poly := range multi {
				if len(poly) == 0 {
					continue
				}
				if r := toRing(poly[0]); r != nil {
					rings = append(rings, r)
				}
			}
		}
	}
	return rings
}

// toRing keeps lon,lat pairs and drops the closing vertex GeoJSON repeats.
func toRing(coords [][]float64) []lonLat {
	var r []lonLat
	for _, c := range coords {
		if len(c) < 2 || math.IsNaN(c[0]) || math.IsNaN(c[1]) {
			continue
		}
		r = append(r, lonLat{c[0], c[1]})
	}
	if len(r) >= 2 && r[0] == r[len(r)-1] {
		r = r[:len(r)-1]
	}
	if len(r) < 3 {
		return nil
	}
	return r
}

// buildingHeight: most Korean rows carry a null height and only a floor
// count; 3.3 m is a storey including its slab. Rows with neither get 6 m.
func buildingHeight(props map[string]any) float64 {
	h := numberProp(props, "height")
	if h > 2 {
		return h
	}
	if floors := numberProp(props, "grnd_flr"); floors > 0 {
		return floors * 3.3
	}
	return 6
}

func numberProp(props map[string]any, key string) float64 {
	switch v := props[key].(type) {
	case float64:
		return v
	case string:
		f, err := strconv.ParseFloat(v, 64)
		if err == nil {
			return f
		}
	}
	return math.NaN()
}

// ---- outlines ---------------------------------------------------------------

// outline turns a ring in local metres into the BuildingSet wire form: e,n
// pairs, counter-clockwise seen from above, closed by repeating the first
// vertex at the end.
func outline(ring [][2]float64) []float64 {
	area2 := 0.0
	for i := range ring {
		a := ring[i]
		b := ring[(i+1)%len(ring)]
		area2 += a[0]*b[1] - b[0]*a[1]
	}
	out := make([]float64, 0, 2*(len(ring)+1))
	if area2 >= 0 {
		for _, v := range ring {
			out = append(out, v[0], v[1])
		}
	} else {
		for i := len(ring) - 1; i >= 0; i-- {
			out = append(out, ring[i][0], ring[i][1])
		}
	}
	return append(out, out[0], out[1])
}
