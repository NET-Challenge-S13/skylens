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

// Buildings: VWorld WFS lt_c_bldginfo footprints, extruded to their roof
// height on the DEM. Facts the legacy layer measured and this code keeps:
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
	// outer rings only; holes are not needed for point sampling.
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

// ---- sampling ---------------------------------------------------------------

// pointInPolygon is the even-odd test on (x, y) pairs.
func pointInPolygon(x, y float64, ring [][2]float64) bool {
	inside := false
	for i, j := 0, len(ring)-1; i < len(ring); j, i = i, i+1 {
		xi, yi := ring[i][0], ring[i][1]
		xj, yj := ring[j][0], ring[j][1]
		if (yi > y) != (yj > y) && x < (xj-xi)*(y-yi)/(yj-yi)+xi {
			inside = !inside
		}
	}
	return inside
}

// Building colours per Style.
var (
	// StyleBlack: dark body, cold cyan linework (roof outline, base, uprights).
	blackWall = rgbHex(0x182128)
	blackRoof = rgbHex(0x222d35)
	blackEdge = rgbHex(0x4f9fb0)
	// Neutral daylight greys, used by StyleAerial where imagery is missing.
	dayWall = rgbHex(0x9aa2ab)
	dayRoof = rgbHex(0xdde2e7)
	// StylePoints: the legacy tactical height ramp.
	pointsLow  = rgbHex(0x2a4a8a)
	pointsHigh = rgbHex(0x3fd0e0)
)

// pointsRampM is the height above ground that reaches the top of the ramp.
const pointsRampM = 30.0

// sampleBuilding turns one footprint into points. ringENU is the outer ring
// in local metres (east, north); baseU and topU are Up in metres; colour
// samples the satellite drape at an ENU position (aerial only).
func sampleBuilding(ringENU [][2]float64, baseU, topU float64, prof profile, style Style,
	drape func(e, n float64) ([3]uint8, bool), emit func(Point)) {
	h := topU - baseU
	put := func(e, n, u float64, c [3]uint8) {
		emit(Point{Pos: [3]float32{float32(e), float32(n), float32(u)}, Color: c})
	}
	colour := func(part int, e, n, u float64) [3]uint8 {
		switch style {
		case StyleAerial:
			c, ok := drape(e, n)
			switch part {
			case partRoof, partOutline:
				if !ok {
					return dayRoof
				}
				return c
			default:
				// Walls get the ground pixel they rise from, a little darker
				// so faces separate from roofs.
				if !ok {
					return dayWall
				}
				return scaleRGB(c, 0.78)
			}
		case StylePoints:
			t := 0.0
			if h > 0 {
				t = (u - baseU) / pointsRampM
			}
			return lerpRGB(pointsLow, pointsHigh, t)
		default:
			switch part {
			case partRoof:
				return blackRoof
			case partWall:
				return blackWall
			default:
				return blackEdge
			}
		}
	}

	// Roof: a grid aligned to whole multiples of the step, so neighbouring
	// buildings and repeated runs sample identically.
	minE, minN := math.Inf(1), math.Inf(1)
	maxE, maxN := math.Inf(-1), math.Inf(-1)
	var cE, cN float64
	for _, v := range ringENU {
		minE, maxE = math.Min(minE, v[0]), math.Max(maxE, v[0])
		minN, maxN = math.Min(minN, v[1]), math.Max(maxN, v[1])
		cE += v[0]
		cN += v[1]
	}
	cE /= float64(len(ringENU))
	cN /= float64(len(ringENU))
	roofs := 0
	step := prof.roofStep
	for n := (math.Floor(minN/step) + 0.5) * step; n <= maxN; n += step {
		for e := (math.Floor(minE/step) + 0.5) * step; e <= maxE; e += step {
			if pointInPolygon(e, n, ringENU) {
				put(e, n, topU, colour(partRoof, e, n, topU))
				roofs++
			}
		}
	}
	if roofs == 0 {
		put(cE, cN, topU, colour(partRoof, cE, cN, topU))
	}

	// Walls: rows from the base up to just under the roof, spaced along each
	// edge; the roof outline closes the top. Every few corners carry an
	// upright so the volume reads (legacy capped them at about four).
	levels := max(1, int(math.Ceil(h/prof.wallRise)))
	upright := max(1, int(math.Ceil(float64(len(ringENU))/4)))
	for i := range ringENU {
		a := ringENU[i]
		b := ringENU[(i+1)%len(ringENU)]
		length := math.Hypot(b[0]-a[0], b[1]-a[1])
		if length == 0 {
			continue
		}
		along := max(1, int(math.Ceil(length/prof.wallStep)))
		for s := 0; s < along; s++ {
			t := float64(s) / float64(along)
			e, n := a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t
			for k := 0; k < levels; k++ {
				u := baseU + h*float64(k)/float64(levels)
				part := partWall
				if k == 0 || (s == 0 && i%upright == 0) {
					part = partEdge
				}
				put(e, n, u, colour(part, e, n, u))
			}
		}
		outline := max(1, int(math.Ceil(length/(prof.wallStep/2))))
		for s := 0; s < outline; s++ {
			t := float64(s) / float64(outline)
			e, n := a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t
			put(e, n, topU, colour(partOutline, e, n, topU))
		}
	}
}

const (
	partRoof = iota
	partWall
	partEdge
	partOutline
)
